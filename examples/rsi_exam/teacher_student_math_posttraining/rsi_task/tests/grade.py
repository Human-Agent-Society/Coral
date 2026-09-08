#!/usr/bin/env python3
"""Sealed exact-answer model verifier. Submitted code is never imported or executed."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path

from reward_mapping import anchored_reward

SUBMISSION = Path("/app/submission")
HELDOUT = Path("/tests/heldout/sealed.json")
TRUSTED_TOKENIZER = Path("/tests/trusted-tokenizer")
ANCHORS = Path("/tests/anchors.json")
REWARD_DIR = Path("/logs/verifier")
SEALED_SHA256 = "d9f359e4a7568138d4bda1b7dcf7af15adfb04b523e18f7b53d63a288bb92097"
ALLOWED_SUFFIXES = {".json", ".safetensors", ".jinja"}
PROMPT_PREFIX = "Please reason step by step, and put your final answer within \\boxed{}.\n\nProblem:\n"
SEED_BASE = 20260804
MAX_OUTPUT_TOKENS = 30720
ARCH = {
    "model_type": "qwen3", "hidden_size": 2048, "intermediate_size": 6144,
    "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
    "vocab_size": 151936, "max_position_embeddings": 32768,
    "head_dim": 128, "hidden_act": "silu", "attention_bias": False,
    "attention_dropout": 0.0, "initializer_range": 0.02, "max_window_layers": 28,
    "rms_norm_eps": 1e-6, "sliding_window": None, "use_sliding_window": False,
    "bos_token_id": 151643, "eos_token_id": 151643, "dtype": "bfloat16",
    "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
    "layer_types": ["full_attention"] * 28,
}


class SubmissionError(RuntimeError):
    pass


class InfrastructureError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_anchors() -> tuple[float, float]:
    try:
        info = ANCHORS.lstat()
    except OSError as exc:
        raise InfrastructureError("anchor payload unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o400:
        raise InfrastructureError("anchor payload is not a mode-0400 single-link file")
    if os.geteuid() == 0 and (info.st_uid, info.st_gid) != (0, 0):
        raise InfrastructureError("anchor payload is not root owned")
    try:
        payload = json.loads(ANCHORS.read_text())
        names = ("BASELINE", "UPPER_BOUND")
        if set(payload) != set(names):
            raise ValueError("fields")
        values = tuple(float(payload[name]) for name in names)
    except Exception as exc:
        raise InfrastructureError("anchor payload malformed") from exc
    if not all(math.isfinite(value) for value in values) or not (0 <= values[0] < values[1] == 1):
        raise InfrastructureError("anchor ordering invalid")
    return values


def expected_schema() -> dict[str, tuple[tuple[int, ...], str]]:
    schema = {
        "model.embed_tokens.weight": ((151936, 2048), "BF16"),
        "model.norm.weight": ((2048,), "BF16"),
    }
    for layer in range(28):
        prefix = f"model.layers.{layer}"
        schema.update({
            f"{prefix}.input_layernorm.weight": ((2048,), "BF16"),
            f"{prefix}.post_attention_layernorm.weight": ((2048,), "BF16"),
            f"{prefix}.self_attn.q_norm.weight": ((128,), "BF16"),
            f"{prefix}.self_attn.k_norm.weight": ((128,), "BF16"),
            f"{prefix}.self_attn.q_proj.weight": ((2048, 2048), "BF16"),
            f"{prefix}.self_attn.k_proj.weight": ((1024, 2048), "BF16"),
            f"{prefix}.self_attn.v_proj.weight": ((1024, 2048), "BF16"),
            f"{prefix}.self_attn.o_proj.weight": ((2048, 2048), "BF16"),
            f"{prefix}.mlp.gate_proj.weight": ((6144, 2048), "BF16"),
            f"{prefix}.mlp.up_proj.weight": ((6144, 2048), "BF16"),
            f"{prefix}.mlp.down_proj.weight": ((2048, 6144), "BF16"),
        })
    return schema


def validate_submission() -> None:
    from safetensors import safe_open
    from transformers import AutoTokenizer

    if SUBMISSION.is_symlink() or not SUBMISSION.is_dir():
        raise SubmissionError("missing standalone model directory")
    files = []
    total = 0
    for path in SUBMISSION.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise SubmissionError("unsafe artifact entry")
        if path.is_file():
            if path.parent != SUBMISSION or path.suffix not in ALLOWED_SUFFIXES:
                raise SubmissionError("unexpected artifact file")
            files.append(path)
            total += info.st_size
    if len(files) > 32 or not 3_000_000_000 <= total <= 5_000_000_000:
        raise SubmissionError("artifact is not a complete student model")
    if (SUBMISSION / "adapter_config.json").exists() or not (SUBMISSION / "config.json").is_file():
        raise SubmissionError("adapter-only or missing config")
    try:
        config = json.loads((SUBMISSION / "config.json").read_text())
    except Exception as exc:
        raise SubmissionError("invalid model config") from exc
    for name, expected in ARCH.items():
        if config.get(name) != expected:
            raise SubmissionError("frozen architecture changed")
    if config.get("architectures") != ["Qwen3ForCausalLM"] or config.get("tie_word_embeddings") is not True:
        raise SubmissionError("frozen model class changed")
    if any(name in config for name in ("auto_map", "quantization_config", "base_model_name_or_path")):
        raise SubmissionError("external/custom model references are forbidden")

    actual: dict[str, tuple[tuple[int, ...], str]] = {}
    weights = sorted(SUBMISSION.glob("*.safetensors"))
    if not weights:
        raise SubmissionError("safetensors weights missing")
    try:
        for path in weights:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in actual:
                        raise SubmissionError("duplicate tensor")
                    tensor = handle.get_slice(name)
                    actual[name] = (tuple(tensor.get_shape()), str(tensor.get_dtype()))
    except SubmissionError:
        raise
    except Exception as exc:
        raise SubmissionError("unsafe or malformed safetensors") from exc
    if actual != expected_schema():
        raise SubmissionError("tensor schema/dtype differs from frozen student")

    try:
        trusted = AutoTokenizer.from_pretrained(TRUSTED_TOKENIZER, local_files_only=True, trust_remote_code=False)
        candidate = AutoTokenizer.from_pretrained(SUBMISSION, local_files_only=True, trust_remote_code=False)
        if trusted.get_vocab() != candidate.get_vocab():
            raise SubmissionError("tokenizer vocabulary changed")
        probe = [{"role": "user", "content": "Solve: 2+2."}]
        trusted_ids = trusted.apply_chat_template(probe, tokenize=True, add_generation_prompt=True, enable_thinking=True)
        candidate_ids = candidate.apply_chat_template(probe, tokenize=True, add_generation_prompt=True, enable_thinking=True)
        if list(trusted_ids) != list(candidate_ids):
            raise SubmissionError("tokenizer chat protocol changed")
    except SubmissionError:
        raise
    except Exception as exc:
        raise SubmissionError("tokenizer could not be validated") from exc


def extract_integer(text: str) -> int | None:
    candidates = re.findall(r"\\boxed\s*\{\s*([+-]?\d[\d,]*)\s*\}", text)
    if not candidates:
        candidates = re.findall(
            r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*\\?\(?\s*([+-]?\d[\d,]*)",
            text, re.I,
        )
    try:
        return int(candidates[-1].replace(",", "")) if candidates else None
    except ValueError:
        return None


def evaluate(rows: list[dict[str, object]]) -> tuple[int, int]:
    """Static-batch Qwen3-recommended sampling with fixed per-question seeds."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        TRUSTED_TOKENIZER, local_files_only=True, trust_remote_code=False
    )
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT_PREFIX + str(row["problem"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        for row in rows
    ]
    sampling = [
        SamplingParams(
            temperature=0.6, top_p=0.95, top_k=20, min_p=0,
            max_tokens=MAX_OUTPUT_TOKENS, seed=SEED_BASE + index,
        )
        for index in range(len(rows))
    ]
    engine = LLM(
        model=str(SUBMISSION), tokenizer=str(TRUSTED_TOKENIZER), trust_remote_code=False,
        dtype="bfloat16", tensor_parallel_size=1, max_model_len=32768,
        gpu_memory_utilization=0.90, max_num_seqs=8, seed=0,
    )
    outputs = engine.generate(prompts, sampling)
    correct = sum(
        extract_integer(output.outputs[0].text) == int(row["answer"])
        for row, output in zip(rows, outputs)
    )
    return correct, len(rows)


def atomic_write(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            # The verifier log directory is a Docker bind mount. Container root must
            # leave outputs host-readable before Harbor parses reward.json.
            os.fchmod(handle.fileno(), 0o644)
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass


def write_result(
    metric: float | None, reward: float, correctness: bool, category: str = "",
    anchors: tuple[float, float] | None = None,
) -> None:
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        reward = 0.0
    public = {
        "correctness": correctness, "metric": "sealed_exact_integer_accuracy",
        "raw_metric": metric, "reward": reward,
    }
    if category:
        public["error_category"] = category
    details = dict(public)
    if anchors is not None:
        named = dict(zip(("baseline", "upper_bound"), anchors))
        details["anchors"] = named
        details["instances"] = [{"id": "sealed_suite", "raw_metric": metric, "anchors": named, "reward": reward}]
    atomic_write(REWARD_DIR / "reward.txt", f"{reward:.12g}\n")
    atomic_write(REWARD_DIR / "reward.json", json.dumps({"reward": reward}, sort_keys=True) + "\n")
    atomic_write(REWARD_DIR / "grade_debug.json", json.dumps(public, sort_keys=True) + "\n")
    atomic_write(REWARD_DIR / "score_details.json", json.dumps(details, sort_keys=True) + "\n")
    print(json.dumps(public, sort_keys=True))


def main() -> None:
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
        "VLLM_LOGGING_LEVEL": "WARNING", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TOKENIZERS_PARALLELISM": "false",
    })
    try:
        anchors = load_anchors()
        if sha256(HELDOUT) != SEALED_SHA256:
            raise InfrastructureError("sealed commitment mismatch")
        rows = json.loads(HELDOUT.read_text())
        if len(rows) != 60 or any(set(row) != {"id", "problem", "answer"} for row in rows):
            raise InfrastructureError("sealed schema mismatch")
        validate_submission()
        correct, total = evaluate(rows)
        metric = correct / total
        reward = anchored_reward(metric, *anchors)
        write_result(metric, reward, True, anchors=anchors)
    except SubmissionError:
        write_result(None, 0.0, False, "submission_failed")
    except Exception:
        write_result(None, 0.0, False, "verifier_failed")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        try:
            REWARD_DIR.mkdir(parents=True, exist_ok=True)
            write_result(None, 0.0, False, "verifier_failed")
        except BaseException:
            pass
        raise
