"""DEVSIM 1D diode engine + generation-mechanism menu (SEALED side).

This module ships ONLY in the verifier image (tests/ build context) and in
the author-only solution/ tree. It is the exact forward model that defines
the world:

    I(V, T; device) = I_dd_contact(V, T)          (DEVSIM DD solve)
                    - q * Integral USRH_eq dx     (remove engine SRH)
                    + q * Integral R_mech(n,p,E; theta, T) dx
    then multiplied by avalanche M = 1/(1 - K_ii) (Chynoweth integral,
    clamped at M <= 20).

The DD solve uses a fixed, instance-independent regularization lifetime
(TAU_EQ); the per-instance leakage physics (SRH mid-gap / shallow-trap /
Hurkx field-enhanced / surface sheet) is a post-processed G-R integral over
the DEVSIM-solved (n, p, E) profiles -- the classic low-injection
perturbation. Ported from the validated Stage-C spike
(campaigns/campaign-001/c104/spike/_spike_common.py), physics unchanged.
"""
from __future__ import annotations

import math
import os

# Hard assignment, not setdefault: both Dockerfiles already set DEVSIM_MATH_LIBS to this
# same value, and assigning it here keeps the solver from depending on the caller's
# environment when that ENV is absent.
os.environ["DEVSIM_MATH_LIBS"] = "liblapack.so.3:libblas.so.3"

import numpy as np

Q = 1.6e-19
KB_EV = 8.617e-5           # eV/K
EPS_SI = 11.1 * 8.85e-14   # F/cm
AREA_CM2 = 1.0e-3          # device area scale factor
TAU_EQ = 1.0e-5            # engine (regularization) lifetime, s -- fixed
# universal Si impact-ionization (Chynoweth) engine constants: avalanche
# multiplication M = 1/(1 - K), K = Integral alpha(|E|) dx.  Invisible in
# the safe window (K ~ 1e-3), decisive at deep reverse bias.
ALPHA_A = 7.03e5           # 1/cm
ALPHA_B = 1.231e6          # V/cm
K_CAP = 0.95               # clamp: M <= 20 (near-breakdown ceiling)
XJ_CM = 1.2e-4             # junction depth (p+ side thickness), cm
L_CM = 7.0e-4              # total device length, cm
NA_P = 5.0e17              # p+ doping, fixed engine constant


def ni_si(T: float) -> float:
    """Intrinsic carrier density of Si, cm^-3 (standard empirical fit)."""
    return 5.29e19 * (T / 300.0) ** 2.54 * math.exp(-6726.0 / T)


def _quiet_fds():
    """Redirect C-level stdout of devsim solves to /dev/null (keep stderr)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(devnull, 1)
    os.close(devnull)
    return saved


def _restore_fd(saved):
    os.dup2(saved, 1)
    os.close(saved)


class DiodeSim:
    """One DEVSIM 1D p+/n diode. Unique `name` per instance in a process."""

    def __init__(self, name: str, Nd: float, mu_n300: float = 1300.0,
                 mu_p300: float = 450.0, alpha_mu: float = 2.2,
                 quiet: bool = True):
        import devsim as ds
        self.ds = ds
        self.name = name
        self.Nd = Nd
        self.mu_n300 = mu_n300
        self.mu_p300 = mu_p300
        self.alpha_mu = alpha_mu
        self.quiet = quiet
        self.T = None
        self.cur_V = 0.0
        self._build()

    # ---------------------------------------------------------------- build
    def _build(self):
        ds = self.ds
        from devsim.python_packages.simple_physics import (
            SetSiliconParameters, CreateSiliconPotentialOnly,
            CreateSiliconPotentialOnlyContact, CreateSiliconDriftDiffusion,
            CreateSiliconDriftDiffusionAtContact, GetContactBiasName)
        from devsim.python_packages.model_create import (
            CreateSolution, CreateNodeModel, CreateEdgeModel)

        dev = self.name
        mesh = "m_" + dev
        reg = "bulk"
        ds.create_1d_mesh(mesh=mesh)
        # p+ contact end / junction / cathode end (positions in cm)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=2.0e-6, tag="anode_t")
        ds.add_1d_mesh_line(mesh=mesh, pos=XJ_CM, ps=6.0e-7, tag="jct")
        ds.add_1d_mesh_line(mesh=mesh, pos=4.0e-4, ps=6.0e-6, tag="mid")
        ds.add_1d_mesh_line(mesh=mesh, pos=L_CM, ps=2.0e-5, tag="cathode_t")
        ds.add_1d_contact(mesh=mesh, name="anode", tag="anode_t",
                          material="metal")
        ds.add_1d_contact(mesh=mesh, name="cathode", tag="cathode_t",
                          material="metal")
        ds.add_1d_region(mesh=mesh, material="Si", region=reg,
                         tag1="anode_t", tag2="cathode_t")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        self.region = reg

        SetSiliconParameters(dev, reg, 300.0)
        self._set_params(300.0)

        CreateNodeModel(dev, reg, "Acceptors",
                        "{0:g}*step({1:g}-x)".format(NA_P, XJ_CM))
        CreateNodeModel(dev, reg, "Donors",
                        "{0:g}*step(x-{1:g})".format(self.Nd, XJ_CM))
        CreateNodeModel(dev, reg, "NetDoping", "Donors-Acceptors")

        CreateSiliconPotentialOnly(dev, reg)
        for c in ("anode", "cathode"):
            ds.set_parameter(device=dev, name=GetContactBiasName(c),
                             value=0.0)
            CreateSiliconPotentialOnlyContact(dev, reg, c)

        self._solve()

        CreateSolution(dev, reg, "Electrons")
        CreateSolution(dev, reg, "Holes")
        ds.set_node_values(device=dev, region=reg, name="Electrons",
                           init_from="IntrinsicElectrons")
        ds.set_node_values(device=dev, region=reg, name="Holes",
                           init_from="IntrinsicHoles")
        CreateSiliconDriftDiffusion(dev, reg)
        for c in ("anode", "cathode"):
            CreateSiliconDriftDiffusionAtContact(dev, reg, c)
        self._solve()

        # edge midpoint helper (for mapping edge field -> nodes)
        ds.edge_from_node_model(device=dev, region=reg, node_model="x")
        CreateEdgeModel(dev, reg, "xmid", "0.5*(x@n0+x@n1)")

        # cache sorted geometry
        x = np.array(ds.get_node_model_values(device=dev, region=reg,
                                              name="x"))
        self._nsort = np.argsort(x)
        self.x = x[self._nsort]
        vol = np.array(ds.get_node_model_values(device=dev, region=reg,
                                                name="NodeVolume"))
        self.vol = vol[self._nsort]
        xm = np.array(ds.get_edge_model_values(device=dev, region=reg,
                                               name="xmid"))
        self._esort = np.argsort(xm)
        self.n_nodes = len(self.x)

    # ------------------------------------------------------------- helpers
    def _set_params(self, T: float):
        ds = self.ds
        dev, reg = self.name, self.region
        ni = ni_si(T)
        tf = (T / 300.0)
        for nm, val in (("T", T), ("kT", 1.3806503e-23 * T),
                        ("V_t", 1.3806503e-23 * T / Q), ("n_i", ni),
                        ("mu_n", self.mu_n300 * tf ** (-self.alpha_mu)),
                        ("mu_p", self.mu_p300 * tf ** (-self.alpha_mu)),
                        ("taun", TAU_EQ), ("taup", TAU_EQ),
                        ("n1", ni), ("p1", ni)):
            ds.set_parameter(device=dev, region=reg, name=nm, value=val)
        self.T = T
        self.ni = ni

    def _solve(self, iters: int = 40):
        saved = _quiet_fds() if self.quiet else None
        try:
            self.ds.solve(type="dc", absolute_error=1e10,
                          relative_error=1e-9, maximum_iterations=iters)
        finally:
            if saved is not None:
                _restore_fd(saved)

    def set_T(self, T: float):
        """Move to temperature T (stepwise; re-solve at current bias)."""
        assert self.T is not None
        while abs(T - self.T) > 1e-9:
            step = max(-40.0, min(40.0, T - self.T))
            self._set_params(self.T + step)
            self._solve()

    def go_to(self, V: float, max_step: float = 0.5):
        """Ramp anode bias adaptively from current bias to V."""
        ds = self.ds
        while abs(V - self.cur_V) > 1e-12:
            step = np.clip(V - self.cur_V, -max_step, max_step)
            target = self.cur_V + step
            ds.set_parameter(device=self.name, name="anode_bias",
                             value=target)
            try:
                self._solve()
                self.cur_V = target
            except Exception:
                # back off: restore last good bias, halve step
                ds.set_parameter(device=self.name, name="anode_bias",
                                 value=self.cur_V)
                max_step *= 0.5
                if max_step < 1e-3:
                    raise RuntimeError(
                        f"bias ramp stuck at {self.cur_V} -> {V}")

    # ------------------------------------------------------------- profile
    def profile(self):
        """Return solved profile + DD contact current (mech-independent)."""
        ds = self.ds
        dev, reg = self.name, self.region
        gn = lambda nm: np.array(ds.get_node_model_values(
            device=dev, region=reg, name=nm))[self._nsort]
        n = gn("Electrons")
        p = gn("Holes")
        usrh = gn("USRH")
        E = np.array(ds.get_edge_model_values(
            device=dev, region=reg, name="ElectricField"))[self._esort]
        Eabs = np.abs(E)
        Enode = np.empty(self.n_nodes)
        Enode[0] = Eabs[0]
        Enode[-1] = Eabs[-1]
        Enode[1:-1] = 0.5 * (Eabs[1:] + Eabs[:-1])
        ia = sum(ds.get_contact_current(device=dev, contact="anode",
                                        equation=e)
                 for e in ("ElectronContinuityEquation",
                           "HoleContinuityEquation"))
        # devsim sign: anode contact current is positive under forward bias
        # I_dd_pure removes the engine-SRH contribution (added back by mech)
        i_dd_raw = ia * AREA_CM2
        i_srh_eq = Q * float(np.sum(usrh * self.vol)) * AREA_CM2
        # impact-ionization integral over the solved field profile
        k_ii = float(np.sum(ALPHA_A * np.exp(-ALPHA_B /
                                             np.maximum(Enode, 1.0)) *
                            self.vol))
        return {"T": self.T, "V": self.cur_V, "ni": self.ni,
                "n": n, "p": p, "E": Enode, "K_ii": k_ii,
                "i_dd_pure": i_dd_raw - i_srh_eq}

    def close(self):
        pass  # devsim has no per-device delete; use unique names / fresh proc


# ======================================================== mechanism menu ==
MENU_DEV = ["srh_mid", "trap_shallow", "surface"]
MENU_SEALED = ["srh_mid", "trap_shallow", "surface", "hurkx"]

_SURF_SIG = 4.0e-6  # cm, width of surface-generation sheet at the junction


def mech_current(prof, mech: str, prm: dict, x: np.ndarray,
                 vol: np.ndarray) -> float:
    """Post-processed G-R terminal current (A) for a mechanism variant.

    prm: tau300, alpha_tau, dEt (trap offset eV), E0/m (hurkx), s0 (surface)
    """
    T, ni = prof["T"], prof["ni"]
    n, p, E = prof["n"], prof["p"], prof["E"]
    kt = KB_EV * T
    tau = prm["tau300"] * (T / 300.0) ** prm.get("alpha_tau", 1.0)
    if mech == "surface":
        w = np.exp(-0.5 * ((x - XJ_CM) / _SURF_SIG) ** 2)
        w = w / np.sum(w * vol)          # normalized sheet, integral = 1
        R = prm["s0"] * w * (n * p - ni * ni) / (n + p + 2.0 * ni)
    else:
        dEt = prm.get("dEt", 0.0)
        n1 = ni * math.exp(dEt / kt)
        p1 = ni * math.exp(-dEt / kt)
        R = (n * p - ni * ni) / (tau * (n + n1) + tau * (p + p1))
        if mech == "hurkx":
            R = R * (1.0 + (E / prm["E0"]) ** prm["m"])
    return Q * float(np.sum(R * vol)) * AREA_CM2


def total_current(prof, mech, prm, x, vol):
    m_av = 1.0 / max(1.0 - min(prof["K_ii"], K_CAP), 1.0 - K_CAP)
    return m_av * (prof["i_dd_pure"] + mech_current(prof, mech, prm, x, vol))


# ------------------------------------------------- vectorized model eval --
def stack_profiles(profs):
    """Stack a list of profile dicts into arrays for fast fitting."""
    return {
        "T": np.array([p["T"] for p in profs]),
        "V": np.array([p["V"] for p in profs]),
        "ni": np.array([p["ni"] for p in profs]),
        "idd": np.array([p["i_dd_pure"] for p in profs]),
        "n": np.stack([p["n"] for p in profs]),
        "p": np.stack([p["p"] for p in profs]),
        "E": np.stack([p["E"] for p in profs]),
        "K": np.array([p["K_ii"] for p in profs]),
    }


def model_currents(st, mech: str, prm: dict, x: np.ndarray,
                   vol: np.ndarray) -> np.ndarray:
    """Vectorized total terminal current (A) for all stacked points."""
    T = st["T"][:, None]
    ni = st["ni"][:, None]
    n, p, E = st["n"], st["p"], st["E"]
    kt = KB_EV * T
    if mech == "surface":
        s = prm["s0"] * (T[:, 0] / 300.0) ** prm.get("alpha_s", 0.0)
        w = np.exp(-0.5 * ((x - XJ_CM) / _SURF_SIG) ** 2)
        w = w / np.sum(w * vol)
        R = s[:, None] * w[None, :] * (n * p - ni * ni) / (n + p + 2.0 * ni)
    else:
        tau = prm["tau300"] * (T / 300.0) ** prm.get("alpha_tau", 1.0)
        dEt = prm.get("dEt", 0.0)
        n1 = ni * np.exp(dEt / kt)
        p1 = ni * np.exp(-dEt / kt)
        R = (n * p - ni * ni) / (tau * (n + n1) + tau * (p + p1))
        if mech == "hurkx":
            R = R * (1.0 + (E / prm["E0"]) ** prm["m"])
    i_mech = Q * np.sum(R * vol[None, :], axis=1) * AREA_CM2
    m_av = 1.0 / np.maximum(1.0 - np.minimum(st["K"], K_CAP), 1.0 - K_CAP)
    return m_av * (st["idd"] + i_mech)


SLOG_I0 = 3.0e-13


def slog(i):
    """Signed log10 transform for currents spanning signs and decades."""
    return np.sign(i) * np.log10(1.0 + np.abs(i) / SLOG_I0)


# ==================================================== measurement protocol =
WINDOW_T = [290.0, 300.0, 312.0, 325.0]
WINDOW_VF = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
WINDOW_VR = [-0.4, -0.8, -1.4, -2.0, -2.8, -3.6]
EXTREME_NEAR = {"T": [390.0, 405.0], "V": [-12.0, -19.0, -26.0]}
EXTREME_FAR = {"T": [415.0, 430.0], "V": [-18.0, -26.0, -34.0]}


def collect_profiles(sim: DiodeSim, temps, v_fwd, v_rev):
    """Solve all (T, V) points; return list of profile dicts.

    For each T: solve V=0, sweep forward ascending, return to 0,
    sweep reverse descending.  Temperatures visited ascending.
    """
    out = []
    for T in sorted(temps):
        sim.go_to(0.0, max_step=0.3)
        sim.set_T(T)
        for V in sorted(v_fwd):
            sim.go_to(V, max_step=0.1)
            out.append(sim.profile())
        sim.go_to(0.0, max_step=0.15)
        for V in sorted(v_rev, reverse=True):
            sim.go_to(V, max_step=2.0)
            out.append(sim.profile())
    return out
