"""SEALED physics for feeder_phase_impedance_inversion_v1: radial MV feeder
generation and OpenDSS timeseries synthesis of AMI measurements.

This module ships ONLY in the verifier image (tests/ build context) and is
deleted from disk by the trusted grader parent before any submitted code
runs. Derived from the Stage-C spike physics
(campaigns/campaign-001/c106/spike/dss_core.py) with identical sampling;
records additionally carry the feeder voltage base and the regulator
nameplate so submitted methods are self-contained.

Truth per instance (all hidden from methods, stored under "truth"):
  - phase(l)   : actual phase (0/1/2) of each single-phase load; the shipped
                 GIS record has a fraction of labels wrong (sparse flips)
  - code(s)    : actual conductor catalog code per segment; the record is
                 wrong on k segments (adjacent catalog swaps)
  - scale(s)   : actual impedance scale per segment = common feeder factor
                 (LOW-DIM bookkeeping/temperature term) x small per-segment
                 deviation
  - tap_steps  : substation regulator tap (integer steps of 0.00625,
                 |steps| in 4..8, sign random) -- recorded value is 0 (stale)
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# conductor catalog (ohms/km, sequence impedances), ordered by ampacity so
# "adjacent swap" is physically meaningful (conductor-type entry off by one)
CATALOG_ORDER = ["ACSR556", "ACSR336", "ACSR4/0", "ACSR1/0", "ACSR#2"]
CATALOG = {
    "ACSR556": {"r1": 0.117, "x1": 0.409, "r0": 0.295, "x0": 1.339},
    "ACSR336": {"r1": 0.190, "x1": 0.430, "r0": 0.368, "x0": 1.360},
    "ACSR4/0": {"r1": 0.368, "x1": 0.464, "r0": 0.546, "x0": 1.394},
    "ACSR1/0": {"r1": 0.588, "x1": 0.487, "r0": 0.766, "x0": 1.417},
    "ACSR#2":  {"r1": 0.902, "x1": 0.512, "r0": 1.080, "x0": 1.442},
}
KV_LL = 12.47
KV_LN = KV_LL / math.sqrt(3.0)      # 7.199 kV
TAP_STEP = 0.00625
TAP_MAX = 8
REG_KVA = 10000
REG_XHL_PCT = 1.0
REG_LOADLOSS_PCT = 0.05


# ---------------------------------------------------------------------------
# topology generation: trunk chain + laterals off trunk buses
def gen_topology(rng, cfg):
    n_trunk = int(rng.integers(cfg["trunk"][0], cfg["trunk"][1] + 1))
    n_lat = int(rng.integers(cfg["laterals"][0], cfg["laterals"][1] + 1))
    lat_len = cfg.get("lat_segs", [1, 2])
    seg_km = cfg["seg_km"]
    trunk_codes = CATALOG_ORDER[:3]           # heavy conductors on trunk
    lat_codes = CATALOG_ORDER[2:]             # lighter on laterals

    segments = []   # {id, fb, tb, km, rec_code}
    buses = ["t0"]

    def add_seg(fb, tb, code_pool):
        km = float(rng.uniform(seg_km[0], seg_km[1]))
        code = str(rng.choice(code_pool))
        segments.append({"id": len(segments), "fb": fb, "tb": tb,
                         "km": round(km, 3), "rec_code": code})
        buses.append(tb)

    for i in range(n_trunk):
        add_seg(f"t{i}", f"t{i+1}", trunk_codes)
    lat_at = rng.choice(np.arange(1, n_trunk + 1),
                        size=min(n_lat, n_trunk), replace=False)
    for li, ti in enumerate(sorted(int(x) for x in lat_at)):
        prev = f"t{ti}"
        n_ls = int(rng.integers(lat_len[0], lat_len[1] + 1))
        for j in range(n_ls):
            nb = f"l{li}_{j}"
            add_seg(prev, nb, lat_codes)
            prev = nb
    return segments, buses


def load_buses(rng, segments, buses, n_loads):
    """Attach loads to non-root buses; every leaf gets one, rest random."""
    cand = [b for b in buses if b != "t0"]
    leafs = set(cand) - {s["fb"] for s in segments}
    picks = list(leafs)
    while len(picks) < n_loads:
        picks.append(str(rng.choice(cand)))
    rng.shuffle(picks)
    return picks[:n_loads] if len(picks) >= n_loads else picks


# ---------------------------------------------------------------------------
# load profiles (15-min resolution)
def _ar1(rng, n, rho, sig):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(0, sig)
    return x


def profiles(rng, cfg, n_loads, T):
    t = np.arange(T)
    hod = (t % 96) / 96.0 * 24.0
    diurnal = (0.55 + 0.30 * np.exp(-0.5 * ((hod - 8.0) / 2.5) ** 2)
               + 0.55 * np.exp(-0.5 * ((hod - 19.0) / 3.0) ** 2))
    weather = 1.0 + _ar1(rng, T, 0.97, 0.015)
    common = diurnal * weather
    idio_amp = float(rng.uniform(*cfg["idio"]))
    shapes = []
    for _ in range(n_loads):
        idio = _ar1(rng, T, 0.85, 0.35)
        s = np.clip(common * (1.0 + idio_amp * idio), 0.10, None)
        shapes.append(s / s.mean())
    return np.array(shapes), hod


def pv_bell(hod, rng, T):
    cloud = np.clip(1.0 + _ar1(rng, T, 0.9, 0.12), 0.15, 1.0)
    return np.exp(-0.5 * ((hod - 12.5) / 2.6) ** 2) * cloud


# ---------------------------------------------------------------------------
# OpenDSS forward: build deck once, sweep timeseries
class DssFeeder:
    def __init__(self, segments, codes, scales, tap_steps, loads, phases,
                 mvasc=2000):
        import opendssdirect as dss
        self.dss = dss
        dss.Basic.AllowChangeDir(False)
        dss.Text.Command("clear")
        dss.Text.Command(
            f"new circuit.c106 basekv={KV_LL} pu=1.0 phases=3 bus1=srcbus "
            f"mvasc3={mvasc} mvasc1={mvasc + 100}")
        tap = 1.0 + tap_steps * TAP_STEP
        dss.Text.Command(
            "new transformer.reg phases=3 windings=2 buses=[srcbus t0] "
            f"conns=[wye wye] kvs=[{KV_LL} {KV_LL}] "
            f"kvas=[{REG_KVA} {REG_KVA}] "
            f"xhl={REG_XHL_PCT} %loadloss={REG_LOADLOSS_PCT} "
            f"taps=[1.0 {tap:.6f}] "
            "maxtap=1.10 mintap=0.90 numtaps=32")
        for s, code, sc in zip(segments, codes, scales):
            c = CATALOG[code]
            dss.Text.Command(
                f"new linecode.lc{s['id']} nphases=3 "
                f"r1={c['r1']*sc:.6f} x1={c['x1']*sc:.6f} "
                f"r0={c['r0']*sc:.6f} x0={c['x0']*sc:.6f} units=km")
            dss.Text.Command(
                f"new line.seg{s['id']} bus1={s['fb']} bus2={s['tb']} "
                f"linecode=lc{s['id']} length={s['km']} units=km phases=3")
        self.load_names = []
        for i, (ld, ph) in enumerate(zip(loads, phases)):
            nm = f"ld{i}"
            dss.Text.Command(
                f"new load.{nm} bus1={ld['bus']}.{int(ph)+1} phases=1 "
                f"kv={KV_LN:.4f} kw={ld['mean_kw']} pf={ld['pf']} model=1 "
                "vminpu=0.60 vmaxpu=1.40")
            self.load_names.append(nm)
        dss.Text.Command(f"set voltagebases=[{KV_LL}]")
        dss.Text.Command("calcvoltagebases")
        dss.Solution.Solve()
        names = dss.Circuit.AllNodeNames()
        self.node_idx = {n: i for i, n in enumerate(names)}
        self.load_node = [f"{ld['bus']}.{int(ph)+1}"
                          for ld, ph in zip(loads, phases)]
        self.head_nodes = [f"srcbus.{k}" for k in (1, 2, 3)]
        self.tanphi = np.array([math.tan(math.acos(ld["pf"]))
                                for ld in loads])

    def sweep(self, P, src_pu):
        """P: (L,T) kW; src_pu: (T,). Returns vload (L,T) pu, vhead (T,),
        p_head (T,) kW (3-phase feeder total)."""
        dss = self.dss
        L, T = P.shape
        vload = np.zeros((L, T))
        vhead = np.zeros(T)
        p_head = np.zeros(T)
        li = np.array([self.node_idx[n] for n in self.load_node])
        hi = np.array([self.node_idx[n] for n in self.head_nodes])
        for t in range(T):
            dss.Vsources.Name("source")
            dss.Vsources.PU(float(src_pu[t]))
            for i, nm in enumerate(self.load_names):
                dss.Loads.Name(nm)
                dss.Loads.kW(float(P[i, t]))
                dss.Loads.kvar(float(P[i, t] * self.tanphi[i]))
            dss.Solution.Solve()
            if not dss.Solution.Converged():
                raise RuntimeError("OpenDSS did not converge")
            v = np.array(dss.Circuit.AllBusMagPu())
            vload[:, t] = v[li]
            vhead[t] = v[hi].mean()
            pw = dss.Circuit.TotalPower()      # kW, negative = delivering
            p_head[t] = -pw[0]
        return vload, vhead, p_head


# ---------------------------------------------------------------------------
def sample_instance(rng, cfg):
    T = int(cfg["T"])
    segments, buses = gen_topology(rng, cfg)
    S = len(segments)
    n_loads = int(rng.integers(cfg["n_loads"][0], cfg["n_loads"][1] + 1))
    lb = load_buses(rng, segments, buses, n_loads)
    L = len(lb)

    loads = []
    for b in lb:
        loads.append({"bus": b,
                      "mean_kw": round(float(rng.uniform(*cfg["kw"])), 2),
                      "pf": round(float(rng.uniform(0.92, 0.98)), 3),
                      "metered": bool(rng.random() < cfg["metered_frac"])})
    # ensure a workable meter count
    while sum(ld["metered"] for ld in loads) < max(6, int(0.4 * L)):
        loads[int(rng.integers(0, L))]["metered"] = True

    # --- truth draws ------------------------------------------------------
    true_phase = rng.integers(0, 3, size=L)
    gis_phase = true_phase.copy()
    n_flip = max(1, int(round(cfg["gis_flip"] * L)))
    # ensure at least 2 flips land on metered loads (score denominator)
    met_idx = [i for i, ld in enumerate(loads) if ld["metered"]]
    unm_idx = [i for i in range(L) if i not in met_idx]
    n_flip_met = min(len(met_idx), max(2, int(round(
        n_flip * len(met_idx) / L))))
    flips = list(rng.choice(met_idx, size=n_flip_met, replace=False))
    rest = max(0, n_flip - n_flip_met)
    if rest and unm_idx:
        flips += list(rng.choice(unm_idx, size=min(rest, len(unm_idx)),
                                 replace=False))
    for i in flips:
        gis_phase[i] = (true_phase[i] + int(rng.integers(1, 3))) % 3

    true_code = [s["rec_code"] for s in segments]
    n_swap = int(rng.integers(cfg["swaps"][0], cfg["swaps"][1] + 1))
    swap_at = rng.choice(S, size=min(n_swap, S), replace=False)
    for si in swap_at:
        k = CATALOG_ORDER.index(true_code[si])
        opts = [j for j in (k - 1, k + 1) if 0 <= j < len(CATALOG_ORDER)]
        true_code[si] = CATALOG_ORDER[int(rng.choice(opts))]
    common_scale = float(np.exp(rng.normal(cfg["scale_mu"],
                                           cfg["scale_common_sig"])))
    dev = np.exp(rng.normal(0.0, cfg["scale_dev_sig"], size=S))
    true_scale = common_scale * dev

    tap_mag = int(rng.integers(4, TAP_MAX + 1))
    tap_steps = int(tap_mag * (1 if rng.random() < 0.5 else -1))

    # --- load timeseries ---------------------------------------------------
    shapes, hod = profiles(rng, cfg, L, T)
    P = shapes * np.array([ld["mean_kw"] for ld in loads])[:, None]
    has_pv = rng.random(L) < cfg["pv_frac"]
    pv_nameplate = np.zeros(L)
    for i in np.where(has_pv)[0]:
        pv_kw = float(rng.uniform(*cfg["pv_kw"])) * loads[i]["mean_kw"]
        pv_nameplate[i] = round(pv_kw, 2)     # public interconnection record
        P[i] -= pv_kw * pv_bell(hod, rng, T)

    src = 1.0 + np.clip(_ar1(rng, T, 0.995, cfg["src_sig"]), -0.02, 0.02)

    feeder = DssFeeder(segments, true_code, true_scale, tap_steps,
                       loads, true_phase)
    vload, vhead, p_head = feeder.sweep(P, src)

    sv, sp = cfg["noise_v"], cfg["noise_p"]
    met = np.array([ld["metered"] for ld in loads])
    v_meas = vload[met] + rng.normal(0, sv, size=vload[met].shape)
    p_meas = P[met] * (1.0 + rng.normal(0, sp, size=P[met].shape))
    head_v = vhead + rng.normal(0, sv / 3.0, size=T)
    head_p = p_head * (1.0 + rng.normal(0, 0.005, size=T))

    return {
        "T": T,
        "kv_ll": KV_LL,
        "segments": segments,
        "catalog": CATALOG, "catalog_order": CATALOG_ORDER,
        "loads": [{"bus": ld["bus"], "gis_phase": int(gp),
                   "mean_kw": ld["mean_kw"], "pf": ld["pf"],
                   "metered": ld["metered"],
                   "pv_kw": float(pv)}
                  for ld, gp, pv in zip(loads, gis_phase, pv_nameplate)],
        "reg": {"rec_tap_steps": 0, "step": TAP_STEP, "max_steps": TAP_MAX,
                "kva": REG_KVA, "xhl_pct": REG_XHL_PCT,
                "loadloss_pct": REG_LOADLOSS_PCT},
        "noise": {"v": sv, "p": sp},
        "meter_load_idx": [int(i) for i in np.where(met)[0]],
        "meter_v": [[round(float(x), 6) for x in row] for row in v_meas],
        "meter_p": [[round(float(x), 4) for x in row] for row in p_meas],
        "head_v": [round(float(x), 6) for x in head_v],
        "head_p": [round(float(x), 2) for x in head_p],
        "truth": {"phase": [int(p) for p in true_phase],
                  "code": list(true_code),
                  "scale": [round(float(s), 5) for s in true_scale],
                  "tap_steps": tap_steps},
    }
