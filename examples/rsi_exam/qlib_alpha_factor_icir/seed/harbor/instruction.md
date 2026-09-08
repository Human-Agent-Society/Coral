# Build a cross-sectional return-prediction model

You are given a real-market-derived **daily cross-sectional factor panel** with anonymized
identifiers (opaque `datetime`/`instrument` IDs, 360 factor columns and a forward-return
`label`). Train a method that outputs a continuous `score` per instrument-day; it is scored by
the cross-sectional rank agreement between your scores and forward returns on a hidden, later
period. Real cross-sectional alpha is weak and noisy — what matters is **generalization**, not
fitting the data you can see.

## Hard Constraints

- Submit an **algorithm** (`predict`), not a precomputed table of scores — the grader re-runs
  your code on hidden data.
- Keep the exact signature:
  `predict(train_df, valid_df, test_features_df) -> DataFrame[datetime, instrument, score]`.
- Output must cover **every** test `(datetime, instrument)` row, with **finite** scores, no
  missing rows and no duplicates — otherwise the submission scores 0.
- Do not hardcode or look up answers: `datetime`/`instrument` are opaque IDs (not real
  dates/tickers), the test period is sealed, and there is no network at any time.

## What You Have

- Data (`/app/data/`): `train_panel.parquet` (earlier period), `valid_panel.parquet` (a later,
  non-overlapping period), and `feature_catalog.csv` (factor families: `f001..f158` = family A,
  `f159..f360` = family B). **The 360 columns `f001..f360` are 6 price/volume fields, each given as a 60-day history — six contiguous 60-column blocks, one field per block — so they form a 6-field x 60-day temporal panel.** `datetime` is an opaque, chronologically-ordered day index
  (`d000001`, …); `instrument` is an opaque ID (`s00001`, …); the raw factors contain missing
  values. A later, non-overlapping test period is sealed in the grader — you never see it.
- The editable baseline `/app/methods/main/` — **this directory is what gets graded**. It is a
  deliberately weak **template** (a family-A subset + a vanilla MLP). Improve it in place, or
  rewrite the algorithm entirely. **No stronger
  reference is shipped** — going weak→strong is the task.
- **Your self-check surface** (free, unlimited): `python /app/selfcheck.py` trains your current
  `methods/main/solver.py` on `train` and reports ICIR on `valid`. This is your *only* feedback
  loop — iterate against it as much as you want. It is a **proxy**: the sealed test period
  differs, so a high `valid` ICIR is necessary but not sufficient.

## What You Submit

Edit `/app/methods/main/solver.py` to expose this exact signature:

```python
def predict(train_df, valid_df, test_features_df) -> pd.DataFrame  # columns: datetime, instrument, score
```

You may add helper modules next to `solver.py`. The runtime provides
`numpy / pandas / scikit-learn / torch / pyyaml`. **There is no submit step and no per-attempt
feedback** — work and self-check for as long as your run window allows, then leave your best
`solver.py` in place; it is graded once at the end on the hidden test period.

## How It Is Judged

After your run, the grader copies your `methods/main/` into a clean box, hands `predict` the
hidden test-period **features** (no `label`), and scores the result. The grader re-runs your
`predict` on a hidden panel of similar size, in a container with a 30-minute limit and more CPU
than your own box.

- **IC** = the same-day cross-sectional Pearson correlation between your `score` and the
  forward-return `label`;
- **ICIR** = the mean daily IC over the test period divided by its standard deviation.

The metric is **ICIR (higher is better)**.
