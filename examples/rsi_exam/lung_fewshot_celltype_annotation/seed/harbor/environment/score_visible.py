"""Replay the current method and request one visible macro-F1 score."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from visible_client import score


APP = Path("/app")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=Path, default=APP / "methods" / "main" / "solver.py")
    parser.add_argument("--socket", type=Path, default=APP / ".visible-evaluator" / "evaluator.sock")
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="cell_visible_") as temporary:
        predictions = Path(temporary) / "predictions.csv"
        command = [
            sys.executable,
            str(args.method),
            "--labeled",
            str(APP / "data" / "visible_labeled.h5ad"),
            "--unlabeled",
            str(APP / "data" / "visible_unlabeled.h5ad"),
            "--query",
            str(APP / "data" / "visible_query.h5ad"),
            "--classes",
            str(APP / "data" / "classes.txt"),
            "--output",
            str(predictions),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout, check=False)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        if completed.returncode != 0:
            raise SystemExit(f"method failed with exit code {completed.returncode}")
        result = score(predictions, args.socket)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
