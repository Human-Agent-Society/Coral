from pathlib import Path
import numpy as np
from problem import N, grid_xy, simulate

NOISE = 0.003
OUTLIER_FRAC = 0.12   # fraction of observed gauges that are faulty
OUTLIER_MAG = 0.6     # magnitude of the gross error on a faulty gauge


def bathymetry(rng):
    xy = grid_xy()
    z = 1.0 + 0.15 * xy[:, 1]
    for _ in range(3):
        c = rng.uniform(0.1, 0.9, 2); s = rng.uniform(0.15, 0.4)
        z += rng.uniform(0.2, 0.6) * np.exp(-((xy - c) ** 2).sum(1) / (2 * s * s))
    return np.clip(z.reshape(N, N), 0.4, 2.5)


def friction_field(rng, target_std=0.85):
    xy = grid_xy()
    z = np.zeros(len(xy))
    for _ in range(8):
        c = rng.uniform(0.05, 0.95, 2); sx, sy = rng.uniform(0.06, 0.26, 2)
        ang = rng.uniform(0, np.pi)
        d = xy - c
        u = np.cos(ang) * d[:, 0] + np.sin(ang) * d[:, 1]
        v = -np.sin(ang) * d[:, 0] + np.cos(ang) * d[:, 1]
        z += rng.normal(0, 1.0) * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
    for _ in range(2):
        phase = rng.uniform(0, 2 * np.pi)
        centre = 0.3 + 0.4 * rng.random()
        channel = np.exp(-((xy[:, 1] - (centre + 0.18 * np.sin(2 * np.pi * xy[:, 0] + phase))) / 0.04) ** 2)
        z += rng.choice([-1, 1]) * rng.uniform(0.9, 1.5) * channel
    z -= z.mean()
    z *= target_std / max(z.std(), 1e-6)          # fix heterogeneity magnitude across cases
    z += -1.0                                      # background mean log-friction
    return np.clip(z.reshape(N, N), -3.0, 1.0)


def forcing_bc(rng):
    x = np.linspace(0, 1, N)
    amp = 1.0 + 0.3 * np.sin(np.pi * x + rng.uniform(0, np.pi))
    phase = rng.uniform(-0.6, 0.6) * x + rng.uniform(0, 0.4)
    return amp * np.exp(1j * (phase + rng.uniform(0, 2 * np.pi)))


def make_case(seed, n_obs=22, n_score=30):
    rng = np.random.default_rng(seed)
    depth = bathymetry(rng)
    logf = friction_field(rng)
    forcing = forcing_bc(rng)
    gauges = rng.uniform(0.04, 0.96, (n_obs + n_score, 2))
    heads = simulate(logf, gauges, depth, forcing)
    heads = heads + rng.normal(0, NOISE, heads.shape) + 1j * rng.normal(0, NOISE, heads.shape)
    obs = heads[:n_obs].copy()
    # A fraction of observed gauges are faulty; the scoring gauges are clean.
    k = max(1, int(OUTLIER_FRAC * n_obs))
    bad = rng.choice(n_obs, k, replace=False)
    obs[bad] += rng.normal(0, OUTLIER_MAG, k) + 1j * rng.normal(0, OUTLIER_MAG, k)
    return dict(
        depth=depth, forcing=forcing, gauges=gauges[:n_obs], obs_heads=obs,
        score_gauges=gauges[n_obs:], score_heads=heads[n_obs:],
    )


def main():
    out = Path("/heldout")
    out.mkdir(parents=True, exist_ok=True)
    cases = np.empty(8, dtype=object)
    for i, seed in enumerate((7717, 8831, 9949, 10177, 11299, 12421, 13553, 14669)):
        cases[i] = make_case(seed)
    np.savez_compressed(out / "cases.npz", cases=cases)


if __name__ == "__main__":
    main()
