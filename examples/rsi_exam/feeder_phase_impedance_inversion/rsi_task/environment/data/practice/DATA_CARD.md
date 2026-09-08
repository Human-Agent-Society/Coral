# Practice fleet — data card

`instances.json` holds 12 monitored distribution feeders
(`{"instances": [record, ...]}`). Each record is one radial medium-voltage
feeder with its network records, its AMI monitoring window, and — because
this is the practice fleet — the full as-operated truth, so you can
iterate freely. `data/assessment/records.json` holds the 16 assessment
feeders in the identical schema minus the `truth` field.

## Network records (as filed; the point of the task is that they are wrong)

| field | meaning |
|---|---|
| `id` | record name |
| `kv_ll` | nominal line-to-line voltage, kV (all feeders 12.47 kV) |
| `segments` | line segments: `id`, `fb`/`tb` (from/to bus), `km` length, `rec_code` (recorded conductor class). Bus `t0` is the substation LV bus; the network is radial |
| `catalog` | conductor classes: positive/zero-sequence resistance and reactance per km (`r1`,`x1`,`r0`,`x0`, ohm/km) |
| `catalog_order` | the same classes ordered from heaviest to lightest |
| `reg` | substation regulator: `rec_tap_steps` (recorded tap, stale), `step` (pu per step), `max_steps` (tap range ±), transformer nameplate `kva`, `xhl_pct`, `loadloss_pct` |
| `loads` | one entry per customer: `bus`, `gis_phase` (0/1/2 as filed in GIS), `mean_kw` (billing average), `pf` (contracted power factor), `metered` (has an AMI meter), `pv_kw` (interconnection-record solar nameplate kW; 0 = none) |

## Monitoring window (`T` steps of 15 minutes)

| field | meaning |
|---|---|
| `meter_load_idx` | indices into `loads` of the metered customers, in measurement row order |
| `meter_v` | per metered customer: voltage magnitude series, per-unit. Magnitude only — AMI meters give no phase angle |
| `meter_p` | per metered customer: net active-power series, kW (customers with solar export at midday, i.e. negative values occur) |
| `head_v` | substation head: balanced voltage-magnitude series, per-unit, measured on the source side of the regulator |
| `head_p` | substation head: total three-phase feeder active power, kW |
| `noise` | measurement accuracy classes: `v` (std of the per-unit voltage noise), `p` (relative std of the power noise) |

Unmetered customers have no series at all — only their billing `mean_kw`.

## Truth (practice fleet only)

- `truth.phase`: the phase (0/1/2) each customer is actually connected
  to. Field crews re-phase laterals without updating GIS: a fraction of
  `gis_phase` entries are wrong.
- `truth.code`: the conductor class actually in service per segment;
  differs from `rec_code` on a few re-conductored segments.
- `truth.scale`: the actual impedance multiplier per segment relative to
  the catalog values at the recorded length (bookkeeping and temperature
  effects; study how these multipliers behave across a feeder).
- `truth.tap_steps`: the regulator tap actually in service (integer
  steps; the recorded value 0 is stale on every feeder).

The records were synthesized with a full three-phase unbalanced power-flow
engine (the same OpenDSS engine installed in this image) from the network
description above; the measurement noise is independent zero-mean
Gaussian at the stated accuracy classes.

**Provenance (2026-08-01).** These feeders are **100% synthetic**: the
networks, the as-operated truth and the AMI monitoring windows are all
generated from seeds, with **no external dataset behind them**. The
generator's parameters — record-error rate, metered fraction, PV
penetration, nominal voltage class, meter accuracy classes — are
**engineering judgement, not statistically calibrated**: they were chosen
to sit in the range these quantities plausibly occupy on a real
distribution feeder, but they were never fitted or checked against any
published distribution-system statistics, and there is no citation
behind any of them. Treat the *difficulty structure* of this inverse
problem as the thing being modelled, not the *distribution* of any
particular utility's fleet. The assessment feeders use the
same record schema and physics but come from related, deliberately
non-identical network populations: their record-error statistics,
operating conditions, AMI coverage and topology are not limited to the
ranges you can estimate from these 12 records.
