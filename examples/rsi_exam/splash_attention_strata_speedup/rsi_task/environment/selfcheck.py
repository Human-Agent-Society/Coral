"""Free, unlimited self-check on the visible cases: correctness + speedup vs baseline."""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval import attn_eval


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    root = Path(__file__).parent
    cases = attn_eval.load_strata(root / "data" / "visible_strata.json")
    sub = _load(root / "methods" / "main" / "attention.py", "submission")
    base = _load(root / "eval" / "baseline_impl.py", "baseline")

    results, speedups = [], []
    for cfg in cases:
        bt = attn_eval.evaluate_case(base.make_attention, cfg)
        r = attn_eval.evaluate_case(sub.make_attention, cfg,
                                    baseline_t=bt.get("t"))
        results.append(r)
        if r["ok"]:
            print(f"{cfg['name']:16s} ok  t={r['t']*1e3:8.3f}ms  "
                  f"speedup={r.get('speedup', 0):6.2f}x")
            speedups.append(r.get("speedup", 0))
        else:
            print(f"{cfg['name']:16s} FAIL  {r['err']}")
            speedups.append(0.0)

    mean = sum(speedups) / len(speedups)
    print(f"\nvisible cases: {sum(r['ok'] for r in results)}/{len(results)} correct, "
          f"mean speedup {mean:.3f}x (baseline=1.0)")
    (root / "selfcheck_details.json").write_text(json.dumps(results, indent=2))

    if Path("/app/budget.py").exists():
        import subprocess
        subprocess.run([sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()
