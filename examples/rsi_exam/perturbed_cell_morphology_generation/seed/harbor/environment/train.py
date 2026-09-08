#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_solver():
    path = Path("/app/methods/main/solver.py")
    if not path.exists():
        path = Path(__file__).parent / "methods" / "main" / "solver.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("submission_solver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/app/data/train.npz")
    parser.add_argument("--output", default="/app/submission")
    parser.add_argument("--config-json", default="{}")
    args = parser.parse_args()
    result = load_solver().train(args.data, args.output, json.loads(args.config_json))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
