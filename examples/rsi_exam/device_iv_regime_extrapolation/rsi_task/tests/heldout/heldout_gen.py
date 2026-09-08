"""SEALED device-lot generator for device_iv_regime_extrapolation_v1.

Ships ONLY in the verifier image (tests/ build context) and is deleted from
disk by the trusted grader parent before any submitted code runs. It carries
the SECRET evaluation seeds and the sealed lot configurations (L-028
rework: same distribution as the practice fleet — identical mechanism
menu, identical parameter support, identical qualification grid — drawn
from fresh secret seeds, with one lot weighted to the hard end of that
same support).

Derived from the Stage-C spike generator (campaigns/campaign-001/c104/spike/
generate.py) with identical physics and sampling. Because instance synthesis
runs DEVSIM (seconds per device), the sealed instances are generated ONCE at
authoring time by this module's __main__ and baked into
tests/heldout/instances_sealed.json; grade.py loads that file into memory and
never regenerates. The visible copies (practice fleet with truth,
qualification window records without truth) are emitted from the same
make_fleet() by the authoring script campaigns/campaign-001/c204/
evidence_local/gen_data.py, so the qualification records the agent sees are
byte-identical to the sealed features.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

import dd_core as dc

NOISE_REL = 0.03      # 3% multiplicative measurement noise
NOISE_FLOOR = 3e-14   # A, additive instrument floor
RNG_SALT = 20260901   # generation salt (device-level rng stream)

FAMILIES = {
    # visible practice fleet (dev family). L-028 rework note: the shipped
    # truth grid (practice_ext, 385/400 K x -8..-26 V) is now also the
    # sealed qualification grid — dev and sealed are one distribution, so
    # the practice truth is honest supervision for exactly the graded
    # regime; the fleet itself is untouched (same seed, same rng path).
    "practice": {
        "menu": dc.MENU_DEV,
        "logNd": (np.log10(4e15), np.log10(1.3e16)),
        "mu_n300": (1150.0, 1400.0), "mu_p300": (400.0, 480.0),
        "alpha_mu": (2.0, 2.4), "alpha_tau": (0.7, 1.5),
        "logIw": (np.log10(3e-11), np.log10(5e-10)),  # |I| @ (-2V, 300K)
        "dEt": (0.09, 0.16), "boost": (1.30, 1.90), "m": (2.5, 3.5),
        "grids": ["practice_ext"],
        "n_devices": 24,
    },
    # sealed lot A (L-028 rework): SAME distribution as the practice
    # fleet — identical mechanism menu, identical parameter support,
    # identical qualification grid, fresh secret seed, more devices.
    # Difficulty comes from the extrapolation skill itself, not from a
    # population shift the practice fleet never showed.
    "indist": {
        "menu": dc.MENU_DEV,
        "logNd": (np.log10(4e15), np.log10(1.3e16)),
        "mu_n300": (1150.0, 1400.0), "mu_p300": (400.0, 480.0),
        "alpha_mu": (2.0, 2.4), "alpha_tau": (0.7, 1.5),
        "logIw": (np.log10(3e-11), np.log10(5e-10)),
        "dEt": (0.09, 0.16), "boost": (1.30, 1.90), "m": (2.5, 3.5),
        "grids": ["practice_ext"],
        "n_devices": 8,
    },
    # sealed lot B (L-028 rework): hard end of the SAME practice support —
    # the two mechanisms the practice fleet shows to be hardest (srh_mid,
    # trap_shallow, both on the visible menu) and the difficult end of the
    # continuous ranges (highest doping, steepest mobility/lifetime
    # temperature laws, deepest trap level); every value stays inside the
    # practice support. In-distribution difficulty weighting, not a shift.
    "indist_hard": {
        "menu": ["srh_mid", "trap_shallow"],
        "logNd": (np.log10(8e15), np.log10(1.3e16)),
        "mu_n300": (1150.0, 1400.0), "mu_p300": (400.0, 480.0),
        "alpha_mu": (2.2, 2.4), "alpha_tau": (1.1, 1.5),
        "logIw": (np.log10(3e-11), np.log10(5e-10)),
        "dEt": (0.13, 0.16), "boost": (1.30, 1.90), "m": (2.5, 3.5),
        "grids": ["practice_ext"],
        "n_devices": 8,
    },
}

# SECRET seeds (never leave tests/). The practice seed is public in effect
# (its 24 instances ship in full, truth included); the sealed ones are not.
SEALED = {
    "practice_seed": 3701,
    "eval_seeds": {"indist": 48317, "indist_hard": 26903},
    "eval_order": ["indist", "indist_hard"],
}

GRIDS = {
    # the one qualification grid (L-028 rework): practice truth and both
    # sealed lots are scored on the same 385/400 K x -8..-26 V grid
    "practice_ext": {"T": [385.0, 400.0],
                     "V": [-8.0, -12.0, -16.0, -20.0, -24.0, -26.0]},
}


def _u(rng, lohi):
    return float(rng.uniform(lohi[0], lohi[1]))


def draw_device(rng, fam, mech):
    f = FAMILIES[fam]
    d = {
        "mech": mech,
        "Nd": 10.0 ** _u(rng, f["logNd"]),
        "mu_n300": _u(rng, f["mu_n300"]),
        "mu_p300": _u(rng, f["mu_p300"]),
        "alpha_mu": _u(rng, f["alpha_mu"]),
        "alpha_tau": _u(rng, f["alpha_tau"]),
        "Iw_target": 10.0 ** _u(rng, f["logIw"]),
    }
    if d["mech"] == "trap_shallow":
        d["dEt"] = _u(rng, f["dEt"])
    if d["mech"] == "hurkx":
        d["boost"] = _u(rng, f["boost"])
        d["m"] = _u(rng, f["m"])
    return d


def calibrate_mech(dv, st_cal, x, vol):
    """Set mechanism strength so |I(-2V, 300K)| == Iw_target (measurable),
    and for hurkx set E0 so the window-edge boost matches the drawn target.
    Exploits exact 1/tau (resp. s0) linear scaling of the mech current."""
    sel2 = (np.abs(st_cal["V"] + 2.0) < 1e-9) & \
           (np.abs(st_cal["T"] - 300.0) < 1e-9)
    sel36 = (np.abs(st_cal["V"] + 3.6) < 1e-9) & \
            (np.abs(st_cal["T"] - 300.0) < 1e-9)
    idd2 = float(st_cal["idd"][sel2][0])
    mech = dv["mech"]
    if mech == "surface":
        prm = {"s0": 1.0, "alpha_s": 0.0}
        i1 = float(dc.model_currents(st_cal, mech, prm, x, vol)[sel2][0]) \
            - idd2
        prm["s0"] = dv["Iw_target"] / abs(i1)
        return prm
    prm = {"tau300": 1e-6, "alpha_tau": dv["alpha_tau"],
           "dEt": dv.get("dEt", 0.0)}
    if mech == "hurkx":
        # bisect E0 for target boost at window edge (-3.6 V, 300 K)
        i_plain = float(dc.model_currents(
            st_cal, "srh_mid",
            {"tau300": 1e-6, "alpha_tau": dv["alpha_tau"]},
            x, vol)[sel36][0]) - float(st_cal["idd"][sel36][0])
        lo, hi = 4.5, 6.3  # log10 E0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            prm.update({"E0": 10.0 ** mid, "m": dv["m"]})
            i_h = float(dc.model_currents(st_cal, mech, prm, x,
                                          vol)[sel36][0]) \
                - float(st_cal["idd"][sel36][0])
            if i_h / i_plain > dv["boost"]:
                lo = mid  # larger E0 -> weaker boost
            else:
                hi = mid
        prm["E0"] = 10.0 ** (0.5 * (lo + hi))
        prm["m"] = dv["m"]
    i1 = float(dc.model_currents(st_cal, mech, prm, x, vol)[sel2][0]) - idd2
    prm["tau300"] = 1e-6 * abs(i1) / dv["Iw_target"]
    return prm


def build_one(args):
    """Worker: synthesize one device (fresh process; devsim state isolated)."""
    idx, seed, fam, mech = args
    cfg = FAMILIES[fam]
    rng = np.random.default_rng([seed, idx, RNG_SALT])
    dv = draw_device(rng, fam, mech)
    sim = dc.DiodeSim(f"gen_{fam}_{seed}_{idx}", Nd=dv["Nd"],
                      mu_n300=dv["mu_n300"], mu_p300=dv["mu_p300"],
                      alpha_mu=dv["alpha_mu"])
    x, vol = sim.x, sim.vol
    profs_w = dc.collect_profiles(sim, dc.WINDOW_T, dc.WINDOW_VF,
                                  dc.WINDOW_VR)
    profs_e = []
    for g in cfg["grids"]:
        profs_e += dc.collect_profiles(sim, GRIDS[g]["T"], [],
                                       GRIDS[g]["V"])
    st_w = dc.stack_profiles(profs_w)
    st_e = dc.stack_profiles(profs_e)

    prm = calibrate_mech(dv, st_w, x, vol)
    i_w = dc.model_currents(st_w, dv["mech"], prm, x, vol)
    i_e = dc.model_currents(st_e, dv["mech"], prm, x, vol)

    noise = 1.0 + NOISE_REL * rng.standard_normal(i_w.shape)
    i_meas = i_w * noise + NOISE_FLOOR * rng.standard_normal(i_w.shape)

    return {
        "window": [{"T": float(t), "V": float(v), "I": float(i)}
                   for t, v, i in zip(st_w["T"], st_w["V"], i_meas)],
        "extreme": [{"T": float(t), "V": float(v)}
                    for t, v in zip(st_e["T"], st_e["V"])],
        "truth_logI": [float(np.log10(max(abs(i), 1e-16))) for i in i_e],
        "_hidden": {"mech": dv["mech"], "Nd": dv["Nd"], "prm": prm,
                    "mu_n300": dv["mu_n300"], "mu_p300": dv["mu_p300"],
                    "alpha_mu": dv["alpha_mu"]},
    }


def make_fleet(family: str, seed: int, processes: int = 8):
    """Deterministic device fleet for one family (spike rng scheme)."""
    cfg = FAMILIES[family]
    n = int(cfg["n_devices"])
    # stratified mechanism assignment: cover the menu, then shuffle
    menu = cfg["menu"]
    mechs = [menu[i % len(menu)] for i in range(n)]
    np.random.default_rng([seed, 77]).shuffle(mechs)
    jobs = [(i, seed, family, mechs[i]) for i in range(n)]
    with mp.get_context("fork").Pool(processes=min(processes, n),
                                     maxtasksperchild=1) as pool:
        devices = pool.map(build_one, jobs)
    return devices


def features_only(dev: dict) -> dict:
    """Strip every truth field; this is all a submitted solver may see."""
    return {k: v for k, v in dev.items()
            if k not in ("truth_logI", "_hidden")}


def main() -> None:
    """Authoring-time: bake the sealed lots into instances_sealed.json."""
    out = {}
    for fam in SEALED["eval_order"]:
        out[fam] = make_fleet(fam, SEALED["eval_seeds"][fam])
    # sequential ids across families: q00..q07 (indist), q08..q15 (hard)
    k = 0
    for fam in SEALED["eval_order"]:
        for dev in out[fam]:
            dev["id"] = f"q{k:02d}"
            k += 1
    path = Path(__file__).resolve().parent / "instances_sealed.json"
    path.write_text(json.dumps({"families": out}))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
