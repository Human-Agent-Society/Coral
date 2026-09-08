"""Baseline alpha-factor model — the submission entrypoint.

You SUBMIT this algorithm (not a predictions CSV). The sealed verifier copies
this file into a fresh isolated sandbox and calls :func:`predict` on a HIDDEN
test panel (instruments/dates you never see), then scores the cross-sectional
ICIR of your output against the hidden labels. So the only way to win is to
build a model that GENERALIZES — overfitting the public train/valid panels is
useless.

Contract (do not change the signature):

    predict(train_df, valid_df, test_features_df) -> pd.DataFrame
        train_df, valid_df : columns datetime, instrument, label, f001..f360 (+ optional sector)
        test_features_df   : columns datetime, instrument, f001..f360  (NO label)
        returns            : DataFrame with columns datetime, instrument, score
                             (one finite score per test (datetime, instrument) row)

This is a deliberately simple starting point fit on train only. Improve it however
you like, as long as :func:`predict` keeps its contract.

On the features: the 360 columns `f001..f360` are 6 price/volume fields, each given
as a 60-day history — i.e. six contiguous 60-column blocks, one field per block — so
the panel carries temporal structure along the 60-day axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# The panels ship 360 factor columns; the first 158 are family A, the rest are
# family B (see data/feature_catalog.csv).
FEATURE_COLS = [f"f{i:03d}" for i in range(1, 361)]
FAMILY_A = FEATURE_COLS[:158]


def _preprocess(train, test, cols, winsorize_q=0.02):
    """Median-fill + winsorise + global standardise, fit on train only."""
    medians = train[cols].median()
    lo = train[cols].quantile(winsorize_q)
    hi = train[cols].quantile(1.0 - winsorize_q)

    def _clean(df):
        out = df.copy()
        # median-fill, then 0-fill any residual (an all-NaN column has a NaN median)
        out[cols] = out[cols].fillna(medians).fillna(0.0).clip(lower=lo, upper=hi, axis=1)
        return out

    tr, te = _clean(train), _clean(test)
    scaler = StandardScaler().fit(tr[cols].to_numpy())
    tr[cols] = scaler.transform(tr[cols].to_numpy())
    te[cols] = scaler.transform(te[cols].to_numpy())
    return tr, te


def _cross_sectional_z(df):
    g = df.groupby("datetime")["label"]
    return ((df["label"] - g.transform("mean")) / (g.transform("std") + 1e-9)).to_numpy()


def predict(train_df: pd.DataFrame, valid_df: pd.DataFrame,
            test_features_df: pd.DataFrame) -> pd.DataFrame:
    cols = FAMILY_A                       # baseline: family-A subset only
    train, test = _preprocess(train_df, test_features_df, cols)

    model = MLPRegressor(
        hidden_layer_sizes=(4,), alpha=0.01,
        learning_rate_init=0.002, max_iter=200, random_state=20260615,
    )
    model.fit(train[cols].to_numpy(), _cross_sectional_z(train_df))
    scores = model.predict(test[cols].to_numpy())

    out = test_features_df[["datetime", "instrument"]].copy()
    out["score"] = np.asarray(scores, dtype=float)
    return out.sort_values(["datetime", "instrument"]).reset_index(drop=True)
