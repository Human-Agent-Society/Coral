"""Shared evaluation protocol for the counting-VQA GRPO task.

This file ships in BOTH the agent image (selfcheck.py) and the verifier image (grade.py), so the
local feedback loop and the sealed grade score IDENTICALLY:

  * prompt   : the EVAL template  ("{question} First output the thinking process ...") — note this
               differs from the trainer's *training* template ("... Output the thinking process
               ..."). The verifier mirrors the EVAL template.
  * decoding : greedy (do_sample=False), max_new_tokens=256
  * extract  : the FIRST integer inside <answer>...</answer>  (regex r'<answer>\\s*(\\d+)\\s*</answer>')
  * correct  : EXACT integer equality with the gold count (no fuzzy / keyword matching)

Keep these semantics frozen — they define the metric.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

# EVAL template. Counting answers are integers.
QUESTION_TEMPLATE = (
    "{question} First output the thinking process in <think> </think> and final answer (number) "
    "in <answer> </answer> tags."
)

MAX_NEW_TOKENS = 256
MAX_PIXELS = 401408   # cap vision tokens (matches training) -> bounded eval memory
MIN_PIXELS = 3136
ANSWER_RE = re.compile(r"<answer>\s*(\d+)\s*</answer>", re.IGNORECASE)


def extract_int_answer(text: str) -> Optional[int]:
    """Return the first integer inside <answer>...</answer>, or None."""
    if not text:
        return None
    m = ANSWER_RE.search(text)
    return int(m.group(1)) if m else None


def gold_int(gold_raw: str) -> Optional[int]:
    """Gold counting answer as an int. Handles bare '4' and '<answer> 4 </answer>' forms."""
    s = str(gold_raw)
    m = ANSWER_RE.search(s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"-?\d+", s)
    return int(m2.group()) if m2 else None


def is_correct(pred_text: str, gold_raw: str) -> bool:
    """Exact integer equality between the model's extracted answer and the gold count."""
    p = extract_int_answer(pred_text)
    g = gold_int(gold_raw)
    return p is not None and g is not None and p == g


def _build_messages(question: str, image) -> List[Dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": QUESTION_TEMPLATE.format(question=question)},
        ],
    }]


def evaluate_model(
    model_dir: str,
    dataset,
    *,
    batch_size: int = 8,
    device: str = "cuda",
    limit: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Load the model at `model_dir`, greedy-decode every example, return (accuracy_pct, details).

    `dataset` is a HF dataset with columns: image (PIL), problem (str), solution (gold count str).
    Uses attn_implementation="sdpa" so the verifier does not require flash-attn to be installed.
    """
    import time
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Robustness: on a shared box the agent's training memory may not be released the instant the
    # verifier starts (or a co-tenant is mid-spike). Wait for enough free VRAM before loading so a
    # transient contention doesn't turn a good model into a spurious OOM -> reward 0.
    if torch.cuda.is_available():
        need = float(os.environ.get("EVAL_MIN_FREE_GB", "12"))
        for _ in range(40):  # up to ~10 min
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            if free_gb >= need:
                break
            print(f"[eval] waiting for GPU memory: {free_gb:.1f} GiB free < {need} needed")
            time.sleep(15)

    processor = AutoProcessor.from_pretrained(model_dir, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device
    )
    model.eval()

    correct = 0
    total = 0
    samples: List[Dict[str, Any]] = []

    for start in range(0, len(dataset), batch_size):
        rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batch_msgs = [_build_messages(r["problem"], r["image"]) for r in rows]
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_msgs
        ]
        image_inputs, video_inputs = process_vision_info(batch_msgs)
        inputs = processor(
            text=texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
        decoded = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for r, out_text in zip(rows, decoded):
            ok = is_correct(out_text, str(r["solution"]))
            correct += int(ok)
            total += 1
            if len(samples) < 20:
                samples.append({
                    "gold": gold_int(str(r["solution"])),
                    "pred": extract_int_answer(out_text),
                    "correct": ok,
                })

    acc = 100.0 * correct / max(1, total)
    return acc, {"correct": correct, "total": total, "samples": samples}
