#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--solver", required=True)
parser.add_argument("--inputs", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

solver_path = Path(args.solver)
sys.path.insert(0, str(solver_path.parent))
spec = importlib.util.spec_from_file_location("submission_solver", solver_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.predict(args.inputs, args.checkpoint, args.output)
