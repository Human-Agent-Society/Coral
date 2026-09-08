#!/usr/bin/env python3
"""Minimal working DEVSIM drift-diffusion model of the lot's architecture.

Run: python3 examples/devsim_diode_demo.py

Builds a 1D p+/n junction diode matching the as-documented architecture of
this device family (see data/practice/DATA_CARD.md): p+ region 5e17 cm^-3
to a junction depth of 1.2 um, uniform n-side donor level (unknown per
device -- 1e16 cm^-3 used here as a placeholder), total length 7 um,
scaling area 1e-3 cm^2. It solves the coupled Poisson + electron/hole
continuity equations, sweeps bias at two temperatures and prints the
terminal current.

This script is a STARTING POINT for building your own forward models with
the DEVSIM Python API: mesh + doping setup, the simple_physics equation
assembly helpers, parameter setting (temperature, mobilities, lifetimes),
robust bias ramping with automatic step back-off, and contact-current
readout. What physics you put in a model, and whether you use DEVSIM at
all, is entirely up to you. API reference: https://devsim.net (the package
also ships docstrings: `python3 -c "import devsim; help(devsim.solve)"`).

Practical convergence tips baked in below:
  - solve potential-only first, then switch on drift-diffusion;
  - ramp bias in small steps (forward bias needs ~0.1 V steps; reverse can
    take ~2 V steps) and halve the step on a failed solve;
  - change temperature in steps of <= 40 K, re-solving at each step;
  - keep unique device/mesh names within one process (DEVSIM has no
    per-device delete); use fresh subprocesses for many devices.
"""
import os

# DEVSIM resolves BLAS/LAPACK at import time via this variable (already set to
# this exact value in this image; assigned unconditionally so the script also
# runs, identically, outside the image).
os.environ["DEVSIM_MATH_LIBS"] = "liblapack.so.3:libblas.so.3"

import devsim as ds
from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim.python_packages.simple_physics import (
    CreateSiliconDriftDiffusion, CreateSiliconDriftDiffusionAtContact,
    CreateSiliconPotentialOnly, CreateSiliconPotentialOnlyContact,
    GetContactBiasName, SetSiliconParameters)

DEVICE, MESH, REGION = "demo", "m_demo", "bulk"
NA_P = 5.0e17     # p+ doping, cm^-3 (documented architecture constant)
ND_N = 1.0e16     # n-side doping, cm^-3 (per-device unknown; placeholder)
XJ = 1.2e-4       # junction depth, cm
LEN = 7.0e-4      # device length, cm
AREA = 1.0e-3     # current scaling area, cm^2


def build():
    ds.create_1d_mesh(mesh=MESH)
    ds.add_1d_mesh_line(mesh=MESH, pos=0.0, ps=2.0e-6, tag="anode_t")
    ds.add_1d_mesh_line(mesh=MESH, pos=XJ, ps=6.0e-7, tag="jct")
    ds.add_1d_mesh_line(mesh=MESH, pos=4.0e-4, ps=6.0e-6, tag="mid")
    ds.add_1d_mesh_line(mesh=MESH, pos=LEN, ps=2.0e-5, tag="cathode_t")
    ds.add_1d_contact(mesh=MESH, name="anode", tag="anode_t", material="metal")
    ds.add_1d_contact(mesh=MESH, name="cathode", tag="cathode_t",
                      material="metal")
    ds.add_1d_region(mesh=MESH, material="Si", region=REGION,
                     tag1="anode_t", tag2="cathode_t")
    ds.finalize_mesh(mesh=MESH)
    ds.create_device(mesh=MESH, device=DEVICE)

    set_temperature(300.0)
    CreateNodeModel(DEVICE, REGION, "Acceptors",
                    "{0:g}*step({1:g}-x)".format(NA_P, XJ))
    CreateNodeModel(DEVICE, REGION, "Donors",
                    "{0:g}*step(x-{1:g})".format(ND_N, XJ))
    CreateNodeModel(DEVICE, REGION, "NetDoping", "Donors-Acceptors")

    # stage 1: electrostatic (potential-only) solution
    CreateSiliconPotentialOnly(DEVICE, REGION)
    for c in ("anode", "cathode"):
        ds.set_parameter(device=DEVICE, name=GetContactBiasName(c), value=0.0)
        CreateSiliconPotentialOnlyContact(DEVICE, REGION, c)
    solve()

    # stage 2: full drift-diffusion, initialized from stage 1
    CreateSolution(DEVICE, REGION, "Electrons")
    CreateSolution(DEVICE, REGION, "Holes")
    ds.set_node_values(device=DEVICE, region=REGION, name="Electrons",
                       init_from="IntrinsicElectrons")
    ds.set_node_values(device=DEVICE, region=REGION, name="Holes",
                       init_from="IntrinsicHoles")
    CreateSiliconDriftDiffusion(DEVICE, REGION)
    for c in ("anode", "cathode"):
        CreateSiliconDriftDiffusionAtContact(DEVICE, REGION, c)
    solve()


def set_temperature(T):
    """SetSiliconParameters fixes T=300 values; set the T-dependent ones."""
    SetSiliconParameters(DEVICE, REGION, T)
    ni = 5.29e19 * (T / 300.0) ** 2.54 * pow(2.718281828, -6726.0 / T)
    for nm, val in (("T", T), ("kT", 1.3806503e-23 * T),
                    ("V_t", 1.3806503e-23 * T / 1.6e-19), ("n_i", ni),
                    ("mu_n", 1300.0 * (T / 300.0) ** -2.2),
                    ("mu_p", 450.0 * (T / 300.0) ** -2.2),
                    ("taun", 1e-6), ("taup", 1e-6),
                    ("n1", ni), ("p1", ni)):
        ds.set_parameter(device=DEVICE, region=REGION, name=nm, value=val)


def solve():
    ds.solve(type="dc", absolute_error=1e10, relative_error=1e-9,
             maximum_iterations=40)


CUR_V = [0.0]


def ramp_to(V, max_step=0.5):
    """Adaptive bias ramp with step back-off on convergence failure."""
    while abs(V - CUR_V[0]) > 1e-12:
        step = max(-max_step, min(max_step, V - CUR_V[0]))
        target = CUR_V[0] + step
        ds.set_parameter(device=DEVICE, name="anode_bias", value=target)
        try:
            solve()
            CUR_V[0] = target
        except Exception:
            ds.set_parameter(device=DEVICE, name="anode_bias", value=CUR_V[0])
            max_step *= 0.5
            if max_step < 1e-3:
                raise


def terminal_current():
    """Anode terminal current in A (positive under forward bias)."""
    ia = sum(ds.get_contact_current(device=DEVICE, contact="anode",
                                    equation=e)
             for e in ("ElectronContinuityEquation",
                       "HoleContinuityEquation"))
    return ia * AREA


def goto_temperature(T_target):
    """Step temperature by <= 40 K, re-solving at each step."""
    T = ds.get_parameter(device=DEVICE, region=REGION, name="T")
    while abs(T_target - T) > 1e-9:
        T += max(-40.0, min(40.0, T_target - T))
        set_temperature(T)
        solve()


def main():
    build()
    for T in (300.0, 350.0):
        ramp_to(0.0, max_step=0.3)
        goto_temperature(T)
        print(f"--- T = {T:.0f} K ---")
        for V in (0.2, 0.4):
            ramp_to(V, max_step=0.1)
            print(f"  V={V:+5.2f} V  I={terminal_current():+.4e} A")
        ramp_to(0.0, max_step=0.15)
        for V in (-1.0, -3.0, -5.0):
            ramp_to(V, max_step=2.0)
            print(f"  V={V:+5.2f} V  I={terminal_current():+.4e} A")


if __name__ == "__main__":
    main()
