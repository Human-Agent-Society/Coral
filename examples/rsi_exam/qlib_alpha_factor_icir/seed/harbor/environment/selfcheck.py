"""Free local self-check: train solver.py on train, predict on valid, report ICIR.

This is a PROXY only — the real grade runs your solver on a hidden test period.
A good valid ICIR is necessary but not sufficient. Run: python environment/selfcheck.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def _safe_corr(x, y):
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    import sys
    sys.path.insert(0, str(HERE / "methods" / "main"))
    import solver  # the algorithm you edit: methods/main/solver.py

    _rd = lambda n: (pd.read_parquet(DATA / f"{n}.parquet") if (DATA / f"{n}.parquet").exists()
                     else pd.read_csv(DATA / f"{n}.csv"))
    train = _rd("train_panel")
    valid = _rd("valid_panel")
    # Hold out valid as a stand-in "test": features only -> predict -> score vs its labels.
    feat_cols = [c for c in valid.columns if c.startswith("f")]
    test_features = valid[["datetime", "instrument", *feat_cols]].copy()

    scores = solver.predict(train, valid, test_features)
    merged = valid[["datetime", "instrument", "label"]].merge(
        scores[["datetime", "instrument", "score"]], on=["datetime", "instrument"], how="left")
    daily = np.array([_safe_corr(g["score"].to_numpy(), g["label"].to_numpy())
                      for _, g in merged.groupby("datetime", sort=False)])
    icir = daily.mean() / (daily.std(ddof=1) + 1e-12)
    print(f"[selfcheck on valid]  mean_ic={daily.mean():.6f}  ICIR={icir:.6f}  "
          f"(proxy only — hidden test period differs)")
    if Path("/app/budget.py").exists():
        import subprocess
        subprocess.run([sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()
