# FinScope coverage panel — data dictionary

The desk's data feed is a real-world fundamentals export: it is **not clean**.
Read this before trusting any column. `lib/paneldata.py` loads these three files
into the `problem` object your forecaster receives; a blank CSV cell is parsed to
`None` (a *missing observation*, NOT a zero).

## `coverage.csv` — one row per company (as-of facts + metadata)

| column | meaning | notes |
|---|---|---|
| `company_id` | anonymized id (`FS0000`, …) | the universe key |
| `set` | `calibration` or `holdout` | calibration ships its realized truth; holdout is what you are judged on |
| `sector` | GICS-style sector label | **sometimes blank** — sector classification was not assigned for every name |
| `beta_proxy` | observable risk proxy that drives the discount rate | **sometimes blank** — not every name has a clean beta estimate |
| `revenue_0` | latest annual revenue (level, $M) | as-of fact, reliable |
| `net_debt` | net debt ($M, negative = net cash) | as-of fact |
| `shares` | shares outstanding (M) | as-of fact |
| `tax_rate` | cash tax rate | as-of fact |
| `fy_end_month` | fiscal-year-end month (12 = December) | several names are off-calendar (e.g. 6, 9) |
| `ipo_year` | year the company listed | a *late* ipo_year means a short observable history |
| `first_fiscal_year`, `last_fiscal_year` | span of the history rows in `fundamentals.csv` | |

## `fundamentals.csv` — one row per company-fiscal-year (observable history)

| column | meaning | notes |
|---|---|---|
| `company_id` | join key | |
| `fiscal_year` | the year of the observation | history length **varies** by company (3–8 years; recent IPOs have fewer) |
| `rev_growth` | year-over-year revenue growth | **may be blank** (not reported that year) |
| `ebit_margin` | EBIT margin that year | **may be blank** |
| `reinvest` | reinvestment as a fraction of NOPAT | **may be blank** |
| `special_items_flag` | `1` if the year is distorted by a one-off event | see below |

### `special_items_flag`

A `1` marks a fiscal year whose `rev_growth` / `ebit_margin` are **not
representative of the underlying trend** — a large acquisition or divestiture, a
one-time charge or gain, an accounting one-timer. The reported number for that
year is real, but it reflects the event, not the company's run-rate. Your
predecessor's note: *"these wrecked my trend lines until I started dropping
them."* Treat flagged years with care when you infer a company's trajectory.

### Missing data

Missing cells are **MNAR** (missing not at random): they are more common in the
oldest reported year and in years that also carry a special-items event. A blank
is not a zero — decide deliberately how to impute, down-weight, or skip it.

### Short histories

A company that listed recently has only a few years of history. There may be too
little of its own data to fit a stable trajectory; comparable names in the same
sector are usually the only way to anchor it.

## `calibration_truth.csv` — realized drivers + truth (calibration names ONLY)

For each calibration company, the columns `rev_growth_1..5`, `ebit_margin_1..5`,
`reinvest_1..5`, `wacc`, `terminal_growth` are the company's **realized** forward
drivers, and `intrinsic_value` is the per-share value the DCF engine produces from
them (the perfect-foresight value). Use these to fit and validate your method.
Holdout companies have **no** truth in your workspace — that is what you are
forecasting.
