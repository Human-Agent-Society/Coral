"""Untrusted worker: load submission, gate correctness, time it. One case per run.

Runs as a deprivileged user in an isolated process. Prints a single JSON line
with {name, ok, err?, t?} to stdout. The trusted parent computes the reward from
sealed anchors it never exposes here.
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, "/tests")
from eval import attn_eval


def main():
    case = json.loads(sys.argv[1])
    sub_path = Path(sys.argv[2])
    spec = importlib.util.spec_from_file_location("submission", sub_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # attn_eval.evaluate_case does load -> correctness gate -> timing, all here
    r = attn_eval.evaluate_case(mod.make_attention, case)
    sys.stdout.write(json.dumps({k: r[k] for k in r if k != "speedup"}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        sys.stdout.write(json.dumps({"ok": False, "err": f"{type(e).__name__}: {e}"}))
