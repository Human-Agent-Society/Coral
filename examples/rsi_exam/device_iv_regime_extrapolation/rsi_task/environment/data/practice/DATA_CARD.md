# Data card — practice fleet and qualification lot

Two data files ship with this task. All currents are terminal currents in
amperes at the anode (positive under forward bias); temperatures in kelvin;
biases in volts applied to the anode.

## Device family (applies to every device in both files)

One-dimensional p+/n junction diode, qualified as a discrete lot:

- p+ region: 5e17 cm^-3 acceptors from the anode contact to the junction at
  depth 1.2 um; n region: a uniform donor level (device-to-device varying,
  not disclosed) from the junction to the cathode contact at 7 um.
- Terminal current scaling area: 1e-3 cm^2.
- Carrier mobilities and their temperature dependence, generation/
  recombination lifetimes and leakage physics vary device to device and are
  not disclosed. The practice truth is your calibration reference for any
  forward model you build (`examples/devsim_diode_demo.py` reproduces the
  architecture above and runs as shipped).

## Measurement protocol (identical for every device)

Safe probe-station window, 56 points per device:

- temperatures: 290, 300, 312, 325 K;
- forward sweep: +0.10 to +0.45 V in 0.05 V steps (8 points per T);
- reverse sweep: -0.4, -0.8, -1.4, -2.0, -2.8, -3.6 V (6 points per T).

Measurement noise: about 3% relative on each reading plus an additive
instrument floor of order 3e-14 A. Readings are independent across points
and devices; the true currents are noise-free physics.

## `practice/instances.json` — 24 development devices, WITH truth

`{"instances": [...]}`; each instance has

- `id`: `p00`..`p23`.
- `window`: list of 56 `{T, V, I}` noisy safe-window measurements.
- `extreme`: 12 `{T, V}` grid points outside the window — 385 and 400 K at
  -8/-12/-16/-20/-24/-26 V.
- `truth_logI`: 12 floats aligned with `extreme` — the TRUE (noise-free)
  log10 of |I| in amperes at each grid point. Development reference only;
  `selfcheck.py` scores your method against it. The qualification devices
  below are committed on the same grid, so the practice truth covers
  exactly the graded regime.

## `qualification/records.json` — 16 qualification devices, NO truth

`{"records": [...]}`; each record has `id` (`q00`..`q15`), `window` (same
56-point protocol), and `extreme` — the grid you must commit numbers at:
385/400 K x -8/-12/-16/-20/-24/-26 V (the same grid the practice truth
covers). No truth fields: these devices' extreme-regime behavior is what
the sealed evaluation scores. The qualification lot is a fresh, unseen
draw from the same production population as the practice fleet; part of
the lot is weighted toward the harder corner of that same population, so
per-device identification from each record's own window is what carries
over — not constants memorized off any single practice device.
