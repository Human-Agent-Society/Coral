"""Build a v24 visible-only strongest-captured-agent pressure summary.

The frozen Human anchor builder supplies all measurement, stability, role
order, and reproduction validation.  This wrapper only binds the preregistered
agent hash and relabels the legacy serialized ``sota`` role so it cannot be
mistaken for a Human-SOTA claim.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


BASE = Path(__file__).with_name("build_anchor_manifest.py")
BASE_SHA256 = "6c062994a7f0d8d1871a372f5ad1d679da9b501994a92c7cd936db98012d4d33"
CANDIDATE_SHA256 = "08c107570f1d860114a0fc9e9d6f58a7ecdf431314e6f67c8bc6094860faff5c"
PRESSURE_STATUS = "MEASURED_A6000_STRONGEST_CAPTURED_AGENT_PRESSURE_SUMMARY"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_path_from_argv() -> Path:
    if "--output" not in sys.argv:
        raise RuntimeError("pressure builder requires --output")
    index = sys.argv.index("--output")
    if index + 1 >= len(sys.argv):
        raise RuntimeError("pressure builder --output has no value")
    return Path(sys.argv[index + 1]).resolve()


def main() -> None:
    if sha256(BASE) != BASE_SHA256:
        raise RuntimeError("immutable Human anchor builder hash drift")
    output = output_path_from_argv()
    spec = importlib.util.spec_from_file_location("paged_v24_pressure_builder_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load immutable Human anchor builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EXPECTED_SUBMISSIONS = {
        "baseline_sha256": module.EXPECTED_SUBMISSIONS["baseline_sha256"],
        "sota_sha256": CANDIDATE_SHA256,
    }
    module.CANDIDATE_STATUS = PRESSURE_STATUS
    module.main()

    value = json.loads(output.read_text(encoding="utf-8"))
    if value.get("status") != PRESSURE_STATUS:
        raise RuntimeError("base pressure summary status drift")
    summary = value.pop("global_summary")
    raw_speedup = float(summary.pop("human_sota_geomean_raw_speedup"))
    value["global_summary"] = {
        "strongest_captured_agent_geomean_raw_speedup": raw_speedup,
        "aggregation": (
            "geometric mean of per-case baseline median / "
            "strongest-captured-agent median"
        ),
    }
    for row in value["cases"].values():
        row["strongest_captured_agent"] = row.pop("human_sota")
        medians = row.get("reproduction_medians") or {}
        medians["strongest_captured_agent"] = medians.pop("sota")
    value["candidate_source"] = {
        "kind": "agent-written strongest captured valid visible attempt",
        "sha256": CANDIDATE_SHA256,
        "human_sota_claim": False,
        "serialized_measurement_role": "sota",
        "semantic_measurement_role": "strongest_captured_agent",
    }
    value["runtime_dependency"] = value.pop("upstream")
    value["pressure_only"] = True
    value["heldout_read_or_evaluated"] = False
    value.pop("promotion", None)
    temporary = output.with_suffix(".json.relabel.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
