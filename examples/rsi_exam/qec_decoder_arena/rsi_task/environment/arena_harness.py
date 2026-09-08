"""Shared harness for the QEC decoder arena (the verifier's exact machinery).

The agent submits a DECODER: `decode(setting, dets) -> predicted logical
observable flips`. The verifier compares predictions against ground-truth
observable flips that were sampled together with the (hidden) detection events
— verification is a bit-array comparison, so the only way to score is to
actually decode better.

Per setting: LER = mean over shots of any-observable-mismatch. Setting score
    s = clip( (ln LER_base - ln LER_agent) / (ln LER_base - ln LER_sota), 0, CAP )
with LERs floored at 1/(2*shots). LER_base / LER_sota are the published-decoder
anchors' rates, recorded in the setting's meta at calibration time.
Metric = mean setting score: 0 = plain pymatching on the given DEM,
1.0 = the reference level recorded in each setting's meta, above 1.0 = beyond it
(capped at CAP)."""
import json
import math
import os

import numpy as np

SCORE_CAP = 2.5


def load_setting(root, name, sealed=False):
    d = os.path.join(root, name)
    meta = json.load(open(os.path.join(d, "meta.json")))
    out = dict(meta=meta, name=name)
    out["dem"] = open(os.path.join(d, "model.dem")).read()
    out["model_stim"] = open(os.path.join(d, "model.stim")).read()
    dev = np.load(os.path.join(d, "dev.npz"))
    out["dev_dets"], out["dev_obs"] = dev["dets"], dev["obs"]
    ev = np.load(os.path.join(d, "eval.npz"))
    out["eval_dets"] = ev["dets"]
    if "obs" in ev.files:              # visible settings carry truth; sealed do not
        out["eval_obs"] = ev["obs"]
    return out


def ler_of(pred, obs):
    pred = np.asarray(pred, dtype=bool)
    obs = np.asarray(obs, dtype=bool)
    assert pred.shape == obs.shape, f"prediction shape {pred.shape} != {obs.shape}"
    return float(np.mean(np.any(pred != obs, axis=1)))


def floored(ler, shots):
    return max(ler, 1.0 / (2.0 * shots))


def setting_score(ler_agent, meta, shots):
    lb = floored(meta["ler_base"], shots)
    ls = floored(meta["ler_sota"], shots)
    la = floored(ler_agent, shots)
    denom = math.log(lb) - math.log(ls)
    if denom <= 0:
        return 0.0
    s = (math.log(lb) - math.log(la)) / denom
    return float(min(max(s, 0.0), SCORE_CAP))
