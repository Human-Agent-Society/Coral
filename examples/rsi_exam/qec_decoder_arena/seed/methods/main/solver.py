"""WEAK INHERITED DECODER (metric 0 anchor) — the decoder you improve.

Contract: `decode(setting, dets) -> predictions`, a boolean array of shape
[n_shots, n_observables] giving, per shot, the predicted logical-observable
flips. The sealed grader compares your predictions against the hidden
ground-truth observable flips (a bit-array comparison): LER = fraction of shots
with ANY observable mispredicted. The only way to score is to decode better.

WHAT THIS INHERITED VERSION DOES
    Plain PyMatching (Higgott-Gidney minimum-weight perfect matching, the field
    standard) built directly from the PUBLISHED detector error model. On color
    codes the hyperedges are force-decomposed into edges
    (`ignore_decomposition_failures=True`) — the known-weak thing MWPM can do on
    a color code. This is deliberately the floor.

WHY IT IS WEAK (and where your work goes)
    * MWPM is STRUCTURALLY wrong for color codes: their stabilizers touch three
      colors, so error syndromes are hyperedges that no edge-matching graph
      represents faithfully.
    * The published noise model is DELIBERATELY MISCALIBRATED versus the true
      noise the eval shots were drawn from, so weights taken straight from
      `model.dem` are wrong. `setting["dev_dets"]` / `setting["dev_obs"]` are a
      LABELED development pack drawn from the SAME true noise as eval.

AVAILABLE TOOLS
    stim, pymatching, numpy, scipy, and the standard library ONLY. No other
    decoding package is installed and none can be added.

`setting` is the dict from `arena_harness.load_setting`, with keys:
    meta        - dict: name, family, style, distance, rounds, n_dev, n_eval,
                  decode_budget_sec, n_detectors, n_observables, ...
    name        - setting name (e.g. "color_d5")
    dem         - published detector error model text (miscalibrated)
    model_stim  - published annotated stim circuit text (miscalibrated)
    dev_dets    - [n_dev, n_detectors] bool, labeled dev syndromes
    dev_obs     - [n_dev, n_observables] bool, their true observable flips
    eval_dets   - [n_eval, n_detectors] bool, the syndromes you must decode
                  (eval_obs is withheld on graded settings)
"""
import numpy as np
import pymatching
import stim


def decode(setting, dets):
    dets = np.asarray(dets, dtype=bool)
    n_obs = int(setting["meta"].get("n_observables", 1))
    try:
        circ = stim.Circuit(setting["model_stim"])
        dem = circ.detector_error_model(decompose_errors=True,
                                        ignore_decomposition_failures=True)
        m = pymatching.Matching.from_detector_error_model(dem)
        return np.asarray(m.decode_batch(dets), dtype=bool).reshape(len(dets), n_obs)
    except Exception:
        # MWPM is structurally weak on color DEMs; a high LER is the intended
        # signal, so fall back to all-zeros rather than raising.
        return np.zeros((len(dets), n_obs), dtype=bool)
