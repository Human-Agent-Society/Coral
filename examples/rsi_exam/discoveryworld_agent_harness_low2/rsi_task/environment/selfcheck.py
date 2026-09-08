#!/usr/bin/env python3
"""Run the submitted harness on the visible DiscoveryWorld instances."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from runner import VISIBLE_CASES, evaluate


def main() -> None:
    # Visible self-check runs inside Codex's already-isolated task sandbox.
    # Avoid a second credential transition so concurrent episode launches work.
    os.environ["DISCOVERYWORLD_VISIBLE_SELF_CHECK"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="", help="comma-separated visible case ids")
    parser.add_argument("--n", type=int, default=0, help="evaluate only the first N cases")
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--harness", default="/app/methods/main/agent.py")
    parser.add_argument("--report", default="/app/selfcheck_report.json")
    args = parser.parse_args()

    wanted = {item for item in args.ids.split(",") if item}
    cases = [case for case in VISIBLE_CASES if not wanted or case["id"] in wanted]
    if args.n:
        cases = cases[: args.n]
    if not cases:
        raise SystemExit("No visible cases selected")

    started = time.time()
    report = evaluate(
        cases,
        harness_path=args.harness,
        max_steps=args.max_steps,
        log_root="/app/selfcheck_logs",
    )
    report["wall_sec"] = time.time() - started
    Path(args.report).write_text(json.dumps(report, indent=2))

    print("\nVisible scientific progress")
    for item in report["instances"]:
        print(
            f"  {item['id']}: raw={item['raw_metric']:.4f} "
            f"procedure={item['procedure']:.4f} completion={item['completion']:.0f} "
            f"knowledge={item['knowledge']:.4f} "
            f"steps={item['steps']} wall={item['wall_sec']:.1f}s"
        )
    print(
        f"mean_raw_metric = {report['mean_raw_metric']:.4f} "
        f"({len(report['instances'])} instances, {report['wall_sec']:.1f}s)"
    )
    print(
        f"components: procedure={report['mean_procedure']:.4f} "
        f"completion={report['mean_completion']:.4f} "
        f"knowledge={report['mean_knowledge']:.4f}"
    )
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
