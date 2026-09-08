#!/usr/bin/env python
"""Local self-check: evaluate the model in /app/submission/ with the EXACT grader protocol, on your
two held-out validation sets. Free, unlimited — these are your tuning signals.

    python /app/selfcheck.py                       # evals BOTH val sets on /app/submission
    python /app/selfcheck.py --model-dir /app/models/Qwen2-VL-2B-Instruct   # baseline (untrained)
    python /app/selfcheck.py --data /app/data/ood_val --limit 200           # one split, quick

Two signals (both held out from training):
  * ood_val (/app/data/ood_val)  — OUT-of-distribution vs train. This is the one that tracks the
                                    sealed OOD TEST. TUNE FOR THIS: train a version, check ood_val,
                                    adjust the reward / hyper-parameters, retrain, keep the best.
  * val (/app/data/val)          — IN-distribution (held out from train): a training-health check
                                    (is it learning to count / under- or over-training).

The graded TEST stays sealed and is disjoint from both val sets; the verifier scores your final
/app/submission on it. Maximize ood_val without training on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/app")
from eval_common import evaluate_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/app/submission")
    ap.add_argument("--data", default=None, help="eval ONE split dir; default evals both val sets")
    ap.add_argument("--batch-size", type=int, default=8)  # 8 fits the larger OOD images; 16 OOMs the vision tower
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from datasets import load_from_disk

    # By default evaluate BOTH signals: the in-distribution val (training health) and the OOD val
    # (your generalization signal; this is the one that tracks the sealed OOD test).
    if args.data:
        splits = [(args.data, args.data)]
    else:
        splits = [
            ("/app/data/ood_val", "ood_val (OUT-of-distribution) <- tune for this"),
            ("/app/data/val", "val (in-distribution, held out from train)"),
        ]

    for path, label in splits:
        ds = load_from_disk(path)
        acc, details = evaluate_model(args.model_dir, ds, batch_size=args.batch_size, limit=args.limit)
        print(json.dumps({
            "model_dir": args.model_dir,
            "split": label,
            "accuracy_pct": round(acc, 3),
            "correct": details["correct"],
            "total": details["total"],
            "samples": details["samples"],
        }, ensure_ascii=False, indent=2))

    if Path("/app/budget.py").exists():
        import subprocess
        subprocess.run([sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()
