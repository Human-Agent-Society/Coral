from __future__ import annotations

import importlib.util
import json
import math
import statistics
from pathlib import Path

import torch

import protocol


ROOT = Path("/app")


def load():
    path = ROOT / "methods" / "main" / "solver.py"
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, protocol.ENTRYPOINT)


def timed(fn, args, case):
    for _ in range(2):
        protocol.run_candidate(fn, args, case)
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record(); protocol.run_candidate(fn, args, case); end.record(); end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    fn = load()
    panel = json.loads((ROOT / "problems" / "visible_cases.json").read_text())
    baseline = json.loads((ROOT / "problems" / "visible_baseline_manifest.json").read_text())["cases"]
    scores = []
    for case in panel["cases"]:
        args = protocol.make_inputs(case, protocol.phase_seed(case, "selfcheck", 0))
        truth = protocol.run_reference(args, case)
        values = protocol.run_candidate(fn, args, case)
        checks = protocol.compare(
            [x.detach().float().cpu().numpy() for x in values],
            [x.detach().float().cpu().numpy() for x in truth], case,
        )
        if not all(row["passed"] for row in checks):
            print(json.dumps({"id": case["id"], "passed": False, "comparisons": checks}))
            raise SystemExit(1)
        candidate_ms = timed(fn, args, case)
        speedup = float(baseline[case["id"]]["baseline_ms"]) / candidate_ms
        scores.append(speedup)
        print(json.dumps({"id": case["id"], "passed": True, "candidate_ms": candidate_ms, "raw_speedup": speedup}))
    print(json.dumps({
        "all_cases_passed": True,
        "raw_speedup": math.exp(sum(math.log(x) for x in scores) / len(scores)),
        "normalization": None,
    }))


if __name__ == "__main__":
    main()
