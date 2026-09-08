"""Inherited starting method: the utility's current first-pass record
check, kept simple on purpose. You may rewrite anything here.

What it does per feeder record:
  1. Replays the monitoring window through the RECORDED model (GIS
     phases, recorded conductor codes, tap 0) with OpenDSS
     (dss_forward.py) and estimates the regulator tap from the average
     voltage offset between measured and simulated meter voltages.
  2. Re-labels each metered customer's phase by grouping the meters'
     voltage traces into three groups by trace similarity and aligning
     the groups to the GIS labels by majority vote. Unmetered customers
     keep their GIS label.
  3. Leaves the recorded conductor codes and impedances untouched.

Interface (the evaluation calls exactly this):
    calibrate(record) -> {"phase": [L ints in 0..2],
                          "code":  [S catalog codes],
                          "scale": [S positive floats],
                          "tap_steps": float}
Optionally return "z_rx": [[r1_ohm, x1_ohm] per segment] to specify
positive-sequence path impedances directly; it then overrides
code/scale in the impedance part of the score.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

try:                                    # package-style or flat import
    from . import dss_forward as df
except ImportError:                     # pragma: no cover
    import dss_forward as df


def _group_phases(record):
    """Similarity grouping of metered voltage traces into 3 phase groups,
    aligned to the GIS labels by majority vote. Deterministic."""
    met = list(record["meter_load_idx"])
    v = np.array(record["meter_v"])
    head = np.array(record["head_v"])
    X = v - head[None, :]
    X = X - X.mean(axis=1, keepdims=True)
    X = X / (X.std(axis=1, keepdims=True) + 1e-12)
    Cm = X @ X.T / X.shape[1]
    w, vecs = np.linalg.eigh(Cm)
    emb = vecs[:, -3:] * np.sqrt(np.clip(w[-3:], 0, None))
    gis = np.array([record["loads"][li]["gis_phase"] for li in met])
    cent = np.stack([emb[gis == q].mean(axis=0) if (gis == q).any()
                     else emb.mean(axis=0) for q in range(3)])
    lab = np.zeros(len(met), int)
    for it in range(30):
        d = ((emb[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d, axis=1)
        if it > 0 and np.array_equal(new, lab):
            break
        lab = new
        for k in range(3):
            if (lab == k).any():
                cent[k] = emb[lab == k].mean(axis=0)
    best, bperm = -1.0, (0, 1, 2)
    for perm in permutations(range(3)):
        agree = float(np.mean([perm[l] == g for l, g in zip(lab, gis)]))
        if agree > best:
            best, bperm = agree, perm
    phases = [ld["gis_phase"] for ld in record["loads"]]
    for mi, li in enumerate(met):
        phases[li] = int(bperm[lab[mi]])
    return phases


def calibrate(record):
    segments = record["segments"]
    codes = [s["rec_code"] for s in segments]
    scales = [1.0] * len(segments)

    # 1. tap from the average level offset of the recorded-model replay
    P = df.load_profiles(record)
    model = df.FeederModel(
        record,
        phases=[ld["gis_phase"] for ld in record["loads"]],
        codes=codes, scales=scales,
        tap_steps=record["reg"]["rec_tap_steps"])
    vload, _ = model.sweep(P, np.array(record["head_v"]))
    v_sim = vload[record["meter_load_idx"]]
    v_meas = np.array(record["meter_v"])
    tap_steps = float(np.mean(v_meas - v_sim) / record["reg"]["step"])

    # 2. metered phases from voltage-trace similarity grouping
    phases = _group_phases(record)

    # 3. impedances: records as shipped
    return {"phase": phases, "code": codes, "scale": scales,
            "tap_steps": tap_steps}
