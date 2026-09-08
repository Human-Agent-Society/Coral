"""Propagate the shared adapter to the standalone task copies after editing it."""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for name in json.loads((ROOT / "tasks.json").read_text()):
        task = ROOT / name
        shutil.copytree(
            ROOT / "_grader",
            task / "grader",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", "dist"),
        )
        package = ROOT / "_grader/src/rsi_exam_grader"
        shutil.copy2(package / "replay.py", task / "seed/rsi_runtime.py")
        shutil.copy2(package / "contract.py", task / "seed/rsi_contract.py")
        if name == "qlib_alpha_factor_icir":
            shutil.copy2(package / "qlib_docker.py", task / "seed/qlib_docker.py")


if __name__ == "__main__":
    main()
