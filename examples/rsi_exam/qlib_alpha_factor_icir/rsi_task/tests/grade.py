"""Harbor verifier for qlib_alpha_factor_icir.

Re-runs the submitted solver on the hidden test period, scores cross-sectional ICIR, and
maps it to a reward: linear 0 -> 0.6 between BASELINE and FRONTIER, 1 - 0.4/x above.
Anchors come from tests/anchors.json. Any failure -> 0.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ANCHORS_PATH = Path(os.environ.get("ANCHORS_PATH", "/tests/anchors.json"))
_anchors = json.loads(ANCHORS_PATH.read_text())
BASELINE = float(_anchors["BASELINE"])
FRONTIER = float(_anchors["FRONTIER"])

# --- Paths (PHASE-0 VERIFY against Harbor's actual mounts) -----------------------------------
# Agent workspace inside the box. Harbor's Dockerfile WORKDIR is /app; the agent edits
# methods/main/solver.py there. Override with SUBMISSION_DIR if Harbor mounts elsewhere.
SUBMISSION_DIR = Path(os.environ.get("SUBMISSION_DIR", "/app/methods/main"))
# Public panels live in the agent box (also fine for the grader to re-fit on).
# PUBLIC_DATA_DIRS (colon-separated) overrides for local/offline checks.
PUBLIC_DATA = ([Path(p) for p in os.environ["PUBLIC_DATA_DIRS"].split(":")]
               if os.environ.get("PUBLIC_DATA_DIRS") else [Path("/app/data"), Path("/tests/data")])
# Held-out travels with tests/ and lands only at verify time.
HELDOUT = Path(os.environ.get("HELDOUT_DIR", "/tests/heldout"))
REWARD_DIR = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))


def _panel(name: str, dirs) -> pd.DataFrame:
    for d in dirs:
        for ext in (".parquet", ".csv"):
            p = Path(d) / f"{name}{ext}"
            if p.exists():
                return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"{name}.(parquet|csv) not found in {[str(x) for x in dirs]}")


def _load_solver(path: Path):
    import sys
    if path.is_dir():
        sys.path.insert(0, str(path))
        entry = path / "solver.py"
    else:
        entry = path
    spec = importlib.util.spec_from_file_location("submitted_solver", str(entry))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load submission at {entry}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # top-level code runs HERE — labels must already be gone
    if not hasattr(mod, "predict"):
        raise AttributeError("submission must define predict(train_df, valid_df, test_features_df)")
    return mod


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _icir(scores: pd.DataFrame, labels: pd.DataFrame) -> float:
    merged = labels.merge(scores[["datetime", "instrument", "score"]],
                          on=["datetime", "instrument"], how="left")
    if len(merged) != len(labels):
        raise ValueError("row_count_mismatch")
    if merged["score"].isna().any():
        raise ValueError("missing_predictions")
    if not np.isfinite(merged["score"].to_numpy()).all():
        raise ValueError("non_finite_scores")
    daily_ic = np.array(
        [_safe_corr(g["score"].to_numpy(), g["label"].to_numpy())
         for _, g in merged.groupby("datetime", sort=False)],
        dtype=float,
    )
    return float(daily_ic.mean() / (daily_ic.std(ddof=1) + 1e-12))


def _reward(icir: float) -> float:
    """baseline->0, frontier->0.6, soft-cap tail above (sup 1)."""
    x = (icir - BASELINE) / (FRONTIER - BASELINE)
    if x <= 0.0:
        return 0.0
    if x <= 1.0:
        return float(0.6 * x)
    return float(1.0 - 0.4 / x)


def main() -> None:
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    out = {"metric": None, "reward": 0.0, "correctness": False, "errors": []}
    try:
        train = _panel("train_panel", PUBLIC_DATA)
        valid = _panel("valid_panel", PUBLIC_DATA)
        test_feat = _panel("test_features", [HELDOUT])
        test_lab = _panel("test_labels", [HELDOUT])  # read into memory NOW...

        # Answer-key isolation: the submission's predict() runs in THIS process, where
        # the answer key sits on disk. Delete the on-disk copy BEFORE importing/running the
        # submission (its top-level code runs at import) so it cannot open() the labels.
        # Scoring below uses the in-memory `test_lab`; the math is unchanged.
        for f in [HELDOUT / f"test_labels{ext}" for ext in (".parquet", ".csv")] + [ANCHORS_PATH]:
            if f.exists():
                f.unlink()

        solver = _load_solver(SUBMISSION_DIR)
        scores = solver.predict(train, valid, test_feat)

        if not isinstance(scores, pd.DataFrame) or not {"datetime", "instrument", "score"}.issubset(scores.columns):
            raise ValueError("predict() must return a DataFrame with datetime, instrument, score")
        if scores.duplicated(["datetime", "instrument"]).any():
            raise ValueError("duplicate_datetime_instrument")

        icir = _icir(scores, test_lab)
        out = {"metric": round(icir, 6), "reward": round(_reward(icir), 6),
               "correctness": True, "errors": []}
    except Exception as exc:  # noqa: BLE001 — any failure -> reward 0, never crash
        out = {"metric": None, "reward": 0.0, "correctness": False,
               "errors": [f"{type(exc).__name__}: {exc}"]}

    # Harbor reward.json must be a flat {key: number} map (Rewards = dict[str, float|int]).
    # Keep ONLY numeric keys here; non-numeric diagnostics go to stdout + grade_debug.json.
    rewards = {"reward": float(out["reward"]),
               "icir": float(out["metric"]) if out["metric"] is not None else 0.0}
    (REWARD_DIR / "reward.txt").write_text(f"{out['reward']}\n", encoding="utf-8")
    (REWARD_DIR / "reward.json").write_text(json.dumps(rewards), encoding="utf-8")
    (REWARD_DIR / "grade_debug.json").write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
