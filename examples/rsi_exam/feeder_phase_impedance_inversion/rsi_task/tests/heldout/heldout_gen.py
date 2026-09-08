"""SEALED instance generator for feeder_phase_impedance_inversion_v1.

This module ships ONLY in the verifier image (tests/ build context) and is
deleted from disk by the trusted grader parent before any submitted code
runs. It carries the secret evaluation seeds and the sealed family
configurations (the shifted operating-condition family and the rural
long-feeder topology family).

Families derive from the Stage-C spike buckets
(campaigns/campaign-001/c106/spike/spec.json); "practice" is the visible
fleet's family (used once at authoring time to build
environment/data/practice/instances.json). Because generation drives
OpenDSS timeseries solves (float-context sensitive), the sealed fleet is
BAKED once at authoring time into instances_sealed.json:

    python3 heldout_gen.py     # regenerates instances_sealed.json

grade.py never regenerates; it loads the baked file, keeps truth in memory
and deletes this directory from the image's disk before any submitted code
runs. environment/data/assessment/records.json is the truth-stripped view
of the same baked instances (byte-identity asserted by
evidence_local/gen_data.py).
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

import feeder_core as fc

# ---------------------------------------------------------------- families
FAMILIES = {
    # visible fleet: suburban feeder, mild record error, low PV, dense AMI
    "practice": {
        "trunk": [4, 5], "laterals": [3, 4], "lat_segs": [1, 2],
        "seg_km": [0.4, 1.2], "n_loads": [26, 34], "kw": [18, 60],
        "metered_frac": 0.8, "gis_flip": 0.14, "swaps": [1, 2],
        "scale_mu": 0.0, "scale_common_sig": 0.08, "scale_dev_sig": 0.05,
        "pv_frac": 0.1, "pv_kw": [0.8, 1.5], "idio": [0.25, 0.4],
        "src_sig": 0.0012, "noise_v": 0.0015, "noise_p": 0.01,
        "T": 192, "n_instances": 12,
    },
    # sealed family 1 (L-028 rework): SAME distribution as the practice
    # fleet — identical parameter support, fresh secret seed, more
    # instances. Difficulty comes from the inversion skill, not from a
    # population shift the practice fleet never showed.
    "indist": {
        "trunk": [4, 5], "laterals": [3, 4], "lat_segs": [1, 2],
        "seg_km": [0.4, 1.2], "n_loads": [26, 34], "kw": [18, 60],
        "metered_frac": 0.8, "gis_flip": 0.14, "swaps": [1, 2],
        "scale_mu": 0.0, "scale_common_sig": 0.08, "scale_dev_sig": 0.05,
        "pv_frac": 0.1, "pv_kw": [0.8, 1.5], "idio": [0.25, 0.4],
        "src_sig": 0.0012, "noise_v": 0.0015, "noise_p": 0.01,
        "T": 192, "n_instances": 8,
    },
    # sealed family 2 (L-028 rework): hard end of the SAME practice
    # support — every range is a subset of the practice range (largest
    # networks, longest segments, max record swaps); no scalar leaves the
    # practice value. In-distribution difficulty weighting, not a shift.
    "indist_hard": {
        "trunk": [5, 5], "laterals": [4, 4], "lat_segs": [1, 2],
        "seg_km": [0.8, 1.2], "n_loads": [30, 34], "kw": [18, 60],
        "metered_frac": 0.8, "gis_flip": 0.14, "swaps": [2, 2],
        "scale_mu": 0.0, "scale_common_sig": 0.08, "scale_dev_sig": 0.05,
        "pv_frac": 0.1, "pv_kw": [0.8, 1.5], "idio": [0.25, 0.4],
        "src_sig": 0.0012, "noise_v": 0.0015, "noise_p": 0.01,
        "T": 192, "n_instances": 8,
    },
}

# SECRET evaluation seeds (never leave tests/). The practice seed is public
# in effect (its 12 instances ship in full with truth), the sealed ones are
# not.
SEALED = {
    "practice_seed": 20107,
    "eval_seeds": {"indist": 74093, "indist_hard": 15881},
    "eval_order": ["indist", "indist_hard"],
}

TRUTH_KEYS = ("truth",)


def make_fleet(family: str, seed: int):
    """Deterministic fleet for one family (spike-compatible rng scheme)."""
    cfg = dict(FAMILIES[family])
    n = cfg.pop("n_instances")
    tag_cfg = dict(cfg)
    tag_cfg["n_instances"] = n
    bucket_tag = zlib.crc32(
        json.dumps(tag_cfg, sort_keys=True).encode()) % 100003
    rng = np.random.default_rng(910000 + 7919 * seed + bucket_tag)
    insts = []
    for i in range(n):
        inst = fc.sample_instance(rng, tag_cfg)
        inst["id"] = f"{family}_{i:02d}"
        insts.append(inst)
    return insts


def features_only(inst: dict) -> dict:
    """Strip every truth field; this is all a submitted method may see."""
    return {k: v for k, v in inst.items() if k not in TRUTH_KEYS}


def main() -> None:
    """Bake the sealed fleets (authoring time only)."""
    out = {"families": {}}
    for fam in SEALED["eval_order"]:
        out["families"][fam] = make_fleet(fam, SEALED["eval_seeds"][fam])
        print(f"baked {fam}: {len(out['families'][fam])} instances")
    p = Path(__file__).resolve().parent / "instances_sealed.json"
    p.write_text(json.dumps(out))
    print(f"wrote {p} ({p.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
