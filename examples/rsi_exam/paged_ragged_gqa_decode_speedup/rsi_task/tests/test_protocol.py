from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path

import grade
import re
import statistics

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 author workstations
    tomllib = None


HERE = Path(__file__).resolve().parent
TASK = HERE.parent


def load_stability_functions():
    """Load the pure timing helpers without requiring torch on author hosts."""
    protocol_path = HERE / "protocol.py"
    if not protocol_path.is_file():
        protocol_path = Path("/runner/protocol.py")
    tree = ast.parse(protocol_path.read_text())
    selected = []
    names = {"repeat_stability_report", "repeat_stability_ratio", "timing_report"}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            selected.append(node)
    namespace = {
        "math": math,
        "statistics": statistics,
        "DEFAULT_REPEATS": 21,
        "DEFAULT_TRIM_EACH_SIDE": 4,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), "protocol.py", "exec"), namespace)
    return namespace["repeat_stability_report"], namespace["timing_report"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verifier_env(path: Path) -> dict[str, str]:
    text = path.read_text()
    if tomllib is not None:
        return tomllib.loads(text)["verifier"]["env"]
    match = re.search(r"(?ms)^\[verifier\.env\]\s*$\n(.*?)(?=^\[|\Z)", text)
    assert match is not None
    return dict(re.findall(r'^([A-Z0-9_]+)\s*=\s*"([^"]*)"\s*$', match.group(1), re.MULTILINE))


def validate_panel(path: Path, expected_split: str) -> set[tuple]:
    panel = json.loads(path.read_text())
    assert panel["schema_version"] == 1
    assert (
        panel["protocol"]
        == "paged_ragged_gqa_decode_v25_d256_extreme_gqa_streamed_exact_snapshot"
    )
    assert int(panel["timing"]["warmup"]) == 128
    assert int(panel["timing"]["repeats"]) == 21
    assert int(panel["timing"]["timed_inner_calls"]) == 8
    assert float(panel["timing"]["max_min_ratio"]) == 1.20
    assert int(panel["timing"]["symmetric_trim_each_side"]) == 4
    assert float(panel["timing"]["bracket_max_min_ratio"]) == 1.20
    assert panel["timing"]["dispersion_hard_gate"] is True
    assert panel["timing"]["streamed_fresh_input_groups"] is True
    assert panel["timing"]["cpu_exact_mutation_snapshots"] is True
    assert int(panel["timing"]["resident_calls_per_group"]) == 8
    assert int(panel["timing"]["max_peak_allocated_bytes"]) == 36 * 1024**3
    assert int(panel["timing"]["max_peak_reserved_bytes"]) == 40 * 1024**3
    assert panel["timing"]["caller_owned_output_ring"] is True
    assert panel["timing"]["output_ring_preallocated_untimed"] is True
    assert int(panel["timing"]["output_ring_entries"]) == 297
    assert panel["timing"]["solver_output_copy_into_ring_timed"] is True
    assert panel["timing"]["submission_temporary_output_retained"] is False
    identities = set()
    for case in panel["cases"]:
        assert case["id"].startswith(expected_split)
        assert case["dtype"] in {"float16", "bfloat16"}
        assert case["batch"] == len(case["page_counts"]) == len(case["last_page_len"])
        assert case["batch"] >= 12
        assert case["query_heads"] % case["kv_heads"] == 0
        assert case["query_heads"] // case["kv_heads"] in {8, 128}
        assert case["head_dim"] in {64, 128, 256}
        assert case["page_size"] in {1, 8, 16}
        assert 0.0 < float(case["logits_soft_cap"]) <= 50.0
        assert int(case["pos_encoding_mode"]) in {0, 1}
        assert int(case["window_left"]) >= -1
        assert int(case["window_left"]) == -1
        assert float(case["rope_scale"]) > 0.0
        assert float(case["rope_theta"]) > 1.0
        assert all(value > 0 for value in case["page_counts"])
        assert all(1 <= value <= case["page_size"] for value in case["last_page_len"])
        identity = (
            case["seed"], case["dtype"], case["batch"], case["query_heads"],
            case["kv_heads"], case["head_dim"], case["page_size"],
            tuple(case["page_counts"]), tuple(case["last_page_len"]),
            float(case["logits_soft_cap"]), int(case["pos_encoding_mode"]),
            int(case["window_left"]), float(case["rope_scale"]), float(case["rope_theta"]),
        )
        assert identity not in identities
        identities.add(identity)
    return identities


def main() -> None:
    repeat_stability_report, timing_report = load_stability_functions()
    interrupted = [
        0.09770400077104568, 0.0612960010766983, 0.09094399958848953,
        0.06647200137376785, 0.08817599713802338, 0.06077200174331665,
        0.08404800295829773, 0.06070400029420853, 0.09978000074625015,
        0.06154799833893776, 0.09886399656534195, 0.06249599903821945,
        0.10130400210618973, 0.06193599849939346, 0.09628000110387802,
        0.06223199889063835, 0.11129999905824661, 0.06173599883913994,
        0.08711999654769897, 0.062171999365091324, 0.09367600083351135,
    ]
    stability = repeat_stability_report(interrupted, 4)
    assert stability["sample_count"] == 21 and stability["retained_count"] == 13
    assert stability["phase_period"] == 2 and stability["phase_trim_each_side"] == 2
    assert sum(row["retained_count"] for row in stability["phases"].values()) == 13
    assert stability["full_max_min_ratio"] > 1.80
    assert stability["trimmed_max_min_ratio"] <= 1.20
    timing = timing_report(interrupted, 100.0, 1.20, 4)
    assert timing["passed"] is True
    assert timing["median_ms"] == statistics.median(interrupted)
    assert timing["full_max_min_ratio"] == stability["full_max_min_ratio"]
    assert timing["trimmed_max_min_ratio"] == stability["trimmed_max_min_ratio"]
    assert timing["dispersion_hard_gate"] is True
    assert timing["robust_interval"]["definition"].startswith("alternating_phase_balanced")
    five_high_interruptions = [
        (1.0 if index % 2 == 0 else 0.7) * (1.0 + 0.08 * (index // 2))
        for index in range(21)
    ]
    diagnostic_outlier = timing_report(
        five_high_interruptions, 100.0, 1.20, 4
    )
    assert diagnostic_outlier["passed"] is False
    assert diagnostic_outlier["trimmed_max_min_ratio"] > 1.20
    assert diagnostic_outlier["diagnostic_within_limit"] is False
    assert diagnostic_outlier["dispersion_hard_gate"] is True
    extreme_dispersion = timing_report(
        [0.01 * (2 ** index) for index in range(21)],
        1000.0,
        1.20,
        4,
    )
    assert extreme_dispersion["passed"] is False
    assert extreme_dispersion["diagnostic_within_limit"] is False
    assert extreme_dispersion["full_max_min_ratio"] > 1_000_000
    for bad_vector in (
        [1.0] * 20,
        [1.0] * 20 + [0.0],
        [1.0] * 20 + [math.inf],
        [1.0] * 20 + [math.nan],
    ):
        assert timing_report(bad_vector, 100.0, 1.20, 4)["passed"] is False
    impossible_wall = timing_report([1.0] * 21, 5.0, 1.20, 4)
    assert impossible_wall["passed"] is False
    assert impossible_wall["wall_time_possible"] is False
    assert timing_report([1.0] * 21, math.inf, 1.20, 4)["passed"] is False

    visible = HERE / "visible_spec.json"
    repo_visible = TASK / "environment" / "problems" / "visible_spec.json"
    calibration = HERE / "calibration" / "test_spec.json"
    heldout = HERE / "heldout" / "test_spec.json"
    visible_ids = validate_panel(visible, "visible_")
    calibration_ids = validate_panel(calibration, "calibration_")
    heldout_ids = validate_panel(heldout, "heldout_")
    assert visible_ids.isdisjoint(calibration_ids)
    assert visible_ids.isdisjoint(heldout_ids)
    assert calibration_ids.isdisjoint(heldout_ids)
    if repo_visible.is_file():
        assert repo_visible.read_bytes() == visible.read_bytes()

    task_toml = TASK / "task.toml"
    env = verifier_env(task_toml) if task_toml.is_file() else os.environ
    if task_toml.is_file() and tomllib is not None:
        assert not {"BASELINE", "SOTA", "UPPER_BOUND", "ANCHOR_STATUS"}.intersection(env)
    for key, path in (
        ("visible", visible),
        ("calibration", calibration),
        ("heldout", heldout),
    ):
        assert grade.EXPECTED_HASHES[key] == sha256(path)
    production_baseline = HERE / "production_baseline" / "solver.py"
    expected_baseline_hash = grade.EXPECTED_BASELINE_SHA256
    assert sha256(production_baseline) == expected_baseline_hash
    repo_starter = TASK / "environment" / "methods" / "main" / "solver.py"
    if repo_starter.is_file():
        assert sha256(repo_starter) == expected_baseline_hash

    assert not {
        "EXPECTED_CPU_AFFINITY",
        "EXPECTED_GPU_NAME",
        "PRODUCTION_BASELINE_SHA256",
        "HUMAN_SOTA_RAW_SPEEDUP",
        "UPPER_BOUND_RAW_SPEEDUP",
    }.intersection(env)

    for split, path in (("author_calibration", calibration), ("sealed_final", heldout)):
        manifest_dir = "calibration" if split == "author_calibration" else "heldout"
        manifest = json.loads((HERE / manifest_dir / "split_manifest.json").read_text())
        assert manifest["split"] == split
        assert (
            manifest["protocol"]
            == "paged_ragged_gqa_decode_v25_d256_extreme_gqa_streamed_exact_snapshot"
        )
        assert manifest["panel_sha256"] == sha256(path)
    anchors_path = HERE / "heldout" / "anchors.json"
    if anchors_path.is_file():
        anchors = json.loads(anchors_path.read_text())
        assert anchors["status"] == "NONSCORING_TOMBSTONE_NO_ANCHORS"
        assert anchors["used_for_scoring"] is False
        assert anchors["anchor_values_present"] is False
        assert not {
            "baseline",
            "sota",
            "human_reference",
            "upper_bound",
        }.intersection(anchors)
    print("test_protocol: PASS")


if __name__ == "__main__":
    main()
