#!/opt/venvs/train/bin/python
"""Structural submission check. The sealed verifier repeats these checks independently."""
from __future__ import annotations

import argparse
import json
import stat
import subprocess
import sys
from pathlib import Path

from safetensors import safe_open
from transformers import AutoTokenizer

BASE = Path("/app/models/student")
ALLOWED = {".json", ".safetensors", ".jinja"}
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


def tensor_schema(root: Path) -> dict[str, tuple[int, ...]]:
    schema: dict[str, tuple[int, ...]] = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in schema:
                    raise RuntimeError(f"duplicate tensor {name}")
                schema[name] = tuple(handle.get_slice(name).get_shape())
    return schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", nargs="?", type=Path, default=Path("/app/submission"))
    root = parser.parse_args().submission
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("submission must be a real directory")
    files = []
    total = 0
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise RuntimeError(f"unsafe entry: {path}")
        if path.is_file():
            if path.parent != root or path.suffix not in ALLOWED:
                raise RuntimeError(f"unexpected submission file: {path.relative_to(root)}")
            files.append(path)
            total += info.st_size
    if len(files) > 32 or not 3_000_000_000 <= total <= 5_000_000_000:
        raise RuntimeError(f"unexpected full-model artifact size/count: {total} bytes, {len(files)} files")
    if (root / "adapter_config.json").exists() or not (root / "config.json").is_file():
        raise RuntimeError("adapter-only or missing model config")
    config = json.loads((root / "config.json").read_text())
    for key, expected in ARCH.items():
        if config.get(key) != expected:
            raise RuntimeError(f"frozen architecture mismatch: {key}")
    if any(name in config for name in ("auto_map", "quantization_config", "base_model_name_or_path")):
        raise RuntimeError("external/custom model references are forbidden")
    expected = tensor_schema(BASE)
    # transformers omits the duplicate tied lm_head when saving a merged checkpoint.
    if json.loads((root / "config.json").read_text()).get("tie_word_embeddings") is True:
        expected.pop("lm_head.weight", None)
    if tensor_schema(root) != expected:
        raise RuntimeError("submitted tensor names/shapes differ from the frozen student")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=False)
    submitted_tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if base_tokenizer.get_vocab() != submitted_tokenizer.get_vocab():
        raise RuntimeError("student tokenizer vocabulary changed")
    if (base_tokenizer.eos_token_id, base_tokenizer.bos_token_id) != (
        submitted_tokenizer.eos_token_id, submitted_tokenizer.bos_token_id
    ):
        raise RuntimeError("student special token IDs changed")
    print(json.dumps({"ok": True, "bytes": total, "files": len(files), "tensors": len(tensor_schema(root))}))
    if Path("/app/budget.py").exists():
        subprocess.run([sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()
