# Beat MWPM: decode quantum error-correction color codes

You inherit a plain PyMatching (minimum-weight perfect matching) decoder and a
labeled development pack for four superdense **color-code** memory experiments.
Your decoder is then re-run by a sealed grader on four HELD-OUT color settings
(same code family, calibrated to the same anchors) that you never see. MWPM is
the field standard for surface codes but is **structurally wrong for color
codes**. Each setting is scored on a relative scale whose 1.0 is a reference
level recorded in its `meta.json`; the top of the range is for beating it.
You submit a decoder, not answers.

## Hard Constraints

- Edit `/app/methods/main/solver.py`. Expose
  `decode(setting, dets) -> predictions`, a boolean array of shape
  `[n_shots, n_observables]` — per shot, the predicted logical-observable flips.
- **`stim` + `pymatching` + `numpy` + `scipy` + standard library ONLY.** No
  other decoding package is installed, and none can be added — whatever decoder
  you submit has to be built from these.
- Predictions are compared bit-for-bit against hidden ground-truth flips: LER =
  fraction of shots with ANY observable mispredicted. The shape must match the
  truth exactly or the setting scores 0.
- Per-setting decode wall-clock budget = `setting["meta"]["decode_budget_sec"]`
  (600 s for d5, 900 s for d7/d9). Over budget = 0 for that setting.

## What You Have

- `/app/methods/main/solver.py`: the inherited weak decoder — plain PyMatching
  on the published DEM (force-decomposed hyperedges on color codes).
  **This file is graded** — improve it in place.
- `/app/arena_harness.py`: the verifier's exact scoring code (`load_setting`,
  `ler_of`, `setting_score`, `SCORE_CAP`).
- `/app/settings_visible/{color_d5,color_d5_Z,color_d7,color_d9}/`: four VISIBLE
  color-code settings. Each carries `meta.json` (with the calibrated `ler_base`
  / `ler_sota` anchors), `model.dem` and `model.stim` (the published,
  **deliberately miscalibrated** noise model), `dev.npz` (a LABELED dev pack:
  `dets` + true `obs`), and `eval.npz` (here `dets` + `obs`, so you can score
  locally). The sealed graded settings are siblings of these four.
- `/app/selfcheck.py`: free local dry-run using the grader's exact scoring.

## What You Submit

Edit `/app/methods/main/solver.py`, keeping the contract:

```python
def decode(setting, dets):
    # setting: dict from arena_harness.load_setting, keys:
    #   meta       (name, family, style, distance, rounds, n_dev, n_eval,
    #               decode_budget_sec, n_detectors, n_observables, ...)
    #   name, dem (published DEM text), model_stim (published circuit text),
    #   dev_dets [n_dev, n_detectors] bool, dev_obs [n_dev, n_observables] bool,
    #   eval_dets [n_eval, n_detectors] bool  (eval_obs is withheld when graded)
    # dets: the syndromes to decode (== setting["eval_dets"]).
    # Return predictions [n_shots, n_observables] bool.
    ...
```

Iterate against `selfcheck.py`, then leave your best `solver.py` in place.

## How It Is Judged

For each sealed color-code setting the grader runs your `decode()` in an
isolated subprocess (never seeing the ground truth), computes
`ler_agent`, and scores

    score = clip( (ln ler_base - ln ler_agent) / (ln ler_base - ln ler_sota), 0, CAP )

with LERs floored at `1/(2*shots)`. `ler_base` is plain MWPM on the published
DEM; `ler_sota` is the reference level recorded with each setting. **metric =
mean over the four sealed color settings** of that per-setting score. Higher is
better; the inherited baseline sits at 0 and the reference level at 1.0.
Each per-setting score is capped at 2.5.

Two facts about the setup, stated without a recommended approach: MWPM is the
inherited decoder and it is structurally wrong for color codes, and the published
DEM is deliberately miscalibrated versus the noise the eval shots were drawn from.
The labeled dev pack (`dev_dets` / `dev_obs`) is yours to use however you see fit.
Only `stim`, `pymatching`, `numpy` and `scipy` are available.
