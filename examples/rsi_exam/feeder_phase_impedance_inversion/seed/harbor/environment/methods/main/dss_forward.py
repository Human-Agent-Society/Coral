"""OpenDSS forward engine for one feeder record.

Builds the three-phase circuit model of a record for ANY candidate
configuration (per-load phases, per-segment conductor codes and impedance
multipliers, regulator tap) and sweeps it through a load timeseries,
returning per-load voltage magnitudes in pu. The substation source is
driven by a given per-step pu series (use the measured head voltage when
replaying the monitoring window).

Kept inside methods/main/ so the submitted method stays self-contained
(only files under methods/ are collected and re-run by the evaluation).

Demo (run from /app): python3 methods/main/dss_forward.py
replays practice feeder 0 with its recorded model (records as shipped,
tap 0) and prints how far that recorded model sits from the measured
voltages — the miscalibration this task is about.
"""
from __future__ import annotations

import math

import numpy as np


class FeederModel:
    """One OpenDSS circuit at a fixed candidate configuration.

    Parameters
    ----------
    record : dict            feeder record (see data/practice/DATA_CARD.md)
    phases : (L,) ints       candidate phase (0/1/2) per load
    codes  : list[str]       candidate conductor catalog code per segment
    scales : (S,) floats     candidate impedance multiplier per segment
    tap_steps : int          candidate regulator tap (integer steps)
    mvasc : float            source short-circuit MVA; use a stiff source
                             (default 1e6) when driving with the measured
                             head-voltage series
    """

    def __init__(self, record, phases, codes, scales, tap_steps, mvasc=1e6):
        import opendssdirect as dss
        self.dss = dss
        kv_ll = record["kv_ll"]
        kv_ln = kv_ll / math.sqrt(3.0)
        reg = record["reg"]
        dss.Basic.AllowChangeDir(False)
        dss.Text.Command("clear")
        dss.Text.Command(
            f"new circuit.feeder basekv={kv_ll} pu=1.0 phases=3 "
            f"bus1=srcbus mvasc3={mvasc} mvasc1={mvasc + 100}")
        tap = 1.0 + tap_steps * reg["step"]
        dss.Text.Command(
            "new transformer.reg phases=3 windings=2 buses=[srcbus t0] "
            f"conns=[wye wye] kvs=[{kv_ll} {kv_ll}] "
            f"kvas=[{reg['kva']} {reg['kva']}] "
            f"xhl={reg['xhl_pct']} %loadloss={reg['loadloss_pct']} "
            f"taps=[1.0 {tap:.6f}] "
            "maxtap=1.10 mintap=0.90 numtaps=32")
        for s, code, sc in zip(record["segments"], codes, scales):
            c = record["catalog"][code]
            dss.Text.Command(
                f"new linecode.lc{s['id']} nphases=3 "
                f"r1={c['r1']*sc:.6f} x1={c['x1']*sc:.6f} "
                f"r0={c['r0']*sc:.6f} x0={c['x0']*sc:.6f} units=km")
            dss.Text.Command(
                f"new line.seg{s['id']} bus1={s['fb']} bus2={s['tb']} "
                f"linecode=lc{s['id']} length={s['km']} units=km phases=3")
        self.load_names = []
        for i, (ld, ph) in enumerate(zip(record["loads"], phases)):
            nm = f"ld{i}"
            dss.Text.Command(
                f"new load.{nm} bus1={ld['bus']}.{int(ph)+1} phases=1 "
                f"kv={kv_ln:.4f} kw={ld['mean_kw']} pf={ld['pf']} model=1 "
                "vminpu=0.60 vmaxpu=1.40")
            self.load_names.append(nm)
        dss.Text.Command(f"set voltagebases=[{kv_ll}]")
        dss.Text.Command("calcvoltagebases")
        dss.Solution.Solve()
        names = dss.Circuit.AllNodeNames()
        self.node_idx = {n: i for i, n in enumerate(names)}
        self.load_node = [f"{ld['bus']}.{int(ph)+1}"
                          for ld, ph in zip(record["loads"], phases)]
        self.head_nodes = [f"srcbus.{k}" for k in (1, 2, 3)]
        self.tanphi = np.array([math.tan(math.acos(ld["pf"]))
                                for ld in record["loads"]])

    def sweep(self, P, src_pu):
        """P: (L,T) net kW per load; src_pu: (T,) source pu series.
        Returns vload (L,T) pu at every load, vhead (T,) pu at the source
        bus. One power-flow solve per step (~0.01 ms each)."""
        dss = self.dss
        L, T = P.shape
        vload = np.zeros((L, T))
        vhead = np.zeros(T)
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
        return vload, vhead


def load_profiles(record):
    """Per-load net-kW profiles for replaying the monitoring window:
    metered loads use their measured series; unmetered loads use the
    billing mean scaled by the average metered consumption shape, minus
    the interconnection-record kW times a generation shape estimated from
    metered customers that have one. Returns (L,T) kW."""
    loads = record["loads"]
    L, T = len(loads), record["T"]
    met = list(record["meter_load_idx"])
    p_meas = np.array(record["meter_p"])
    P = np.zeros((L, T))
    pv_kw = np.array([ld.get("pv_kw", 0.0) for ld in loads])
    means = p_meas.mean(axis=1)
    nopv = np.array([pv_kw[li] <= 0 for li in met])
    good = (means > 0.2 * max(np.median(means), 1e-6)) & nopv
    if good.any():
        shapes = p_meas[good] / means[good][:, None]
        shape_avg = np.clip(shapes.mean(axis=0), 0.0, None)
    else:
        shape_avg = np.ones(T)
    pvm = np.array([pv_kw[li] > 0 for li in met])
    if pvm.any():
        base_exp = (np.array([loads[li]["mean_kw"]
                              for li in met])[pvm][:, None]
                    * shape_avg[None, :])
        gen = (base_exp - p_meas[pvm]) / np.array(
            [pv_kw[li] for li in met])[pvm][:, None]
        gen_shape = np.clip(np.median(gen, axis=0), 0.0, None)
    else:
        gen_shape = np.zeros(T)
    for mi, li in enumerate(met):
        P[li] = p_meas[mi]
    for li, ld in enumerate(loads):
        if li not in met:
            P[li] = ld["mean_kw"] * shape_avg - pv_kw[li] * gen_shape
    return P


if __name__ == "__main__":
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    rec = json.loads((root / "data" / "practice" /
                      "instances.json").read_text())["instances"][0]
    feats = {k: v for k, v in rec.items() if k != "truth"}
    P = load_profiles(feats)
    model = FeederModel(
        feats,
        phases=[ld["gis_phase"] for ld in feats["loads"]],
        codes=[s["rec_code"] for s in feats["segments"]],
        scales=[1.0] * len(feats["segments"]),
        tap_steps=feats["reg"]["rec_tap_steps"])
    vload, vhead = model.sweep(P, np.array(feats["head_v"]))
    v_meas = np.array(feats["meter_v"])
    v_sim = vload[feats["meter_load_idx"]]
    resid = v_meas - v_sim
    print(f"records-as-shipped replay of practice feeder {rec['id']}:")
    print(f"  mean voltage offset {resid.mean():+.5f} pu, "
          f"rms residual {np.sqrt((resid**2).mean()):.5f} pu, "
          f"meter noise class {feats['noise']['v']:.4f} pu")
    print("  (a calibrated model should bring the rms residual toward the "
          "noise class)")
