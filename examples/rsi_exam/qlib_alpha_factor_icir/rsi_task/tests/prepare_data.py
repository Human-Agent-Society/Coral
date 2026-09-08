"""Build-time data generator for qlib_alpha_factor_icir.

Downloads the public Qlib CN daily bundle (microsoft/qlib, region=cn, frozen at 2020-09-25),
rebuilds the Alpha360 cross-sectional factor panel for the top-200 CSI300 names, rounds +
downcasts to float32, then applies a deterministic anonymization (real datetime/instrument ->
opaque d######/s##### IDs, one global map across splits so every role agrees). Reproduces the
original shipped panels byte-for-byte.

--role selects which split to emit:
  --role visible  -> <out>/{train_panel,valid_panel}.parquet + feature_catalog.csv
  --role heldout  -> <out>/{test_features,test_labels}.parquet

Usage:
    python prepare_data.py --role visible --out /app/data
    python prepare_data.py --role heldout --out /heldout
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# --- generation spec (author convention; Qlib CN data ends 2020-09-25) ----------------------
N_INST = 200
W0, W1 = "2008-01-01", "2020-09-25"          # full window
TR0, TR1 = "2008-01-01", "2013-11-30"        # train  ~6y
VA0, VA1 = "2014-01-01", "2015-06-30"        # valid  1.5y   (~1-month gap before)
TE0, TE1 = "2015-08-01", "2020-09-25"        # test   ~5y    (~1-month gap before)
FCOLS = [f"f{i:03d}" for i in range(1, 361)]
# Where the public bundle is fetched/read at build. Overridable for local verification.
PROVIDER_URI = os.environ.get("QLIB_PROVIDER_URI", "/opt/qlib_cn_data")


def _ensure_qlib_bundle() -> None:
    """Download the public Qlib CN daily bundle if not already present (build-time network)."""
    if (Path(PROVIDER_URI) / "calendars" / "day.txt").is_file():
        print(f"[prepare] qlib bundle already at {PROVIDER_URI}", flush=True)
        return
    from qlib.tests.data import GetData
    print(f"[prepare] downloading public Qlib CN bundle -> {PROVIDER_URI}", flush=True)
    GetData().qlib_data(target_dir=PROVIDER_URI, region="cn", delete_old=False, exists_skip=True)


def _build_panel_with_splits() -> pd.DataFrame:
    """Reproduce the anonymized factor panel for ALL splits, tagged by split.

    Mirrors scripts/gen_real_data.py + scripts/anonymize_ids.py exactly: build the raw
    Alpha360 panel with real ids, tag the train/valid/test split by real date, then apply
    ONE global opaque-ID map over every split (so both build sides agree), and return it.
    """
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    from qlib.contrib.data.handler import Alpha360

    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

    # 1) top-200 CSI300 names by coverage (deterministic)
    close = D.features(D.instruments("csi300"), ["$close"], start_time=W0, end_time=W1)
    cov = close.groupby(level="instrument").size().sort_values(ascending=False)
    insts = cov.index[:N_INST].tolist()
    print(f"[prepare] selected {len(insts)} instruments", flush=True)

    # 2) raw Alpha360 factors for just those names
    h = Alpha360(instruments=insts, start_time=W0, end_time=W1,
                 fit_start_time=W0, fit_end_time=W1,
                 infer_processors=[], learn_processors=[])
    df = h.fetch()
    df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    label_col = [c for c in df.columns if "LABEL" in str(c).upper()][0]
    feat_cols = [c for c in df.columns if c != label_col]
    assert len(feat_cols) == 360, len(feat_cols)
    df = df.rename(columns={label_col: "label"}).dropna(subset=["label"])

    panel = df.reset_index()
    panel["datetime"] = pd.to_datetime(panel["datetime"]).dt.strftime("%Y-%m-%d")
    panel = panel.rename(columns={f: f"f{i:03d}" for i, f in enumerate(feat_cols, 1)})
    panel[FCOLS] = panel[FCOLS].round(6)
    panel = panel[["datetime", "instrument", "label"] + FCOLS].sort_values(["datetime", "instrument"])

    # 3) tag split by REAL date, dropping the ~1-month gaps between splits
    rd = panel["datetime"]
    split = pd.Series(index=panel.index, dtype=object)
    split[(rd >= TR0) & (rd <= TR1)] = "train"
    split[(rd >= VA0) & (rd <= VA1)] = "valid"
    split[(rd >= TE0) & (rd <= TE1)] = "test"
    panel = panel[split.notna()].copy()
    panel["__split"] = split[split.notna()].values

    # 4) ONE global deterministic anonymization over every kept row
    dates = sorted(panel["datetime"].unique())
    insts_sorted = sorted(panel["instrument"].unique())
    date_map = {d: f"d{i:06d}" for i, d in enumerate(dates, 1)}
    inst_map = {s: f"s{i:05d}" for i, s in enumerate(insts_sorted, 1)}
    panel["datetime"] = panel["datetime"].map(date_map)
    panel["instrument"] = panel["instrument"].map(inst_map)
    print(f"[prepare] anonymized {len(date_map)} dates, {len(inst_map)} instruments", flush=True)
    return panel


def _f32(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype({c: "float32" for c in df.columns if str(df[c].dtype) == "float64"})


def _save(df: pd.DataFrame, path: Path) -> None:
    _f32(df).to_parquet(path, engine="pyarrow", compression="zstd", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["visible", "heldout"])
    ap.add_argument("--out", required=True, help="output dir to write this role's files into")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _ensure_qlib_bundle()
    panel = _build_panel_with_splits()
    cols_no_split = [c for c in panel.columns if c != "__split"]

    if args.role == "visible":
        train = panel[panel["__split"] == "train"][cols_no_split]
        valid = panel[panel["__split"] == "valid"][cols_no_split]
        _save(train, out / "train_panel.parquet")
        _save(valid, out / "valid_panel.parquet")
        pd.DataFrame({"feature": FCOLS,
                      "family": ["A" if i < 158 else "B" for i in range(360)]}
                     ).to_csv(out / "feature_catalog.csv", index=False)
        print(f"[prepare] wrote visible: train{train.shape} valid{valid.shape} + feature_catalog", flush=True)
    else:  # heldout (sealed answer key)
        test = panel[panel["__split"] == "test"][cols_no_split]
        _save(test.drop(columns=["label"]), out / "test_features.parquet")
        _save(test[["datetime", "instrument", "label"]], out / "test_labels.parquet")
        print(f"[prepare] wrote heldout: test_features{test.drop(columns=['label']).shape} "
              f"test_labels{test[['datetime','instrument','label']].shape}", flush=True)


if __name__ == "__main__":
    main()
