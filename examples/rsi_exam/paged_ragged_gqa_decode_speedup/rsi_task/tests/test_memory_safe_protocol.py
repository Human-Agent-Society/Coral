from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
TASK = HERE.parent
sys.path.insert(0, str(HERE))
from protocol import immutability_report, snapshot_inputs  # noqa: E402


def input_bytes(case: dict) -> int:
    dtype_bytes = 2
    total_pages = sum(int(value) for value in case["page_counts"])
    physical_pages = total_pages + max(8, total_pages // 7)
    q = (
        int(case["batch"])
        * int(case["query_heads"])
        * int(case["head_dim"])
        * dtype_bytes
    )
    one_cache = (
        physical_pages
        * int(case["page_size"])
        * int(case["kv_heads"])
        * int(case["head_dim"])
        * dtype_bytes
    )
    metadata = (int(case["batch"]) + 1 + total_pages + int(case["batch"])) * 4
    return q + 2 * one_cache + metadata


def main() -> None:
    args = (
        torch.arange(16, dtype=torch.float32).reshape(4, 4),
        torch.arange(8, dtype=torch.int32),
        1.0,
    )
    snapshot = snapshot_inputs(args)
    copies, _ = snapshot
    assert copies[0] is not None and copies[0].device.type == "cpu"
    assert copies[1] is not None and copies[1].device.type == "cpu"
    pristine = immutability_report(args, snapshot)
    assert pristine == {
        "passed": True,
        "changed": [],
        "snapshot_storage": "cpu_exact_copy",
        "cuda_snapshot_bytes": 0,
    }
    args[0].add_(1)
    mutated = immutability_report(args, snapshot)
    assert mutated["passed"] is False and mutated["changed"] == ["q"]

    panel = json.loads((HERE / "calibration/test_spec.json").read_text())
    timing = panel["timing"]
    assert (
        panel["protocol"]
        == "paged_ragged_gqa_decode_v25_d256_extreme_gqa_streamed_exact_snapshot"
    )
    assert timing["resident_calls_per_group"] == timing["timed_inner_calls"] == 8
    assert timing["max_peak_allocated_bytes"] == 36 * 1024**3
    assert timing["max_peak_reserved_bytes"] == 40 * 1024**3
    total_calls = 1 + int(timing["warmup"]) + int(timing["repeats"]) * int(
        timing["timed_inner_calls"]
    )
    assert total_calls == timing["output_ring_entries"] == 297
    assert timing["caller_owned_output_ring"] is True
    assert timing["output_ring_preallocated_untimed"] is True
    assert timing["solver_output_copy_into_ring_timed"] is True
    assert timing["submission_temporary_output_retained"] is False
    largest = max(input_bytes(case) for case in panel["cases"])
    resident_input_bytes = largest * int(timing["resident_calls_per_group"])
    all_call_input_bytes = largest * total_calls
    assert all_call_input_bytes > 30 * resident_input_bytes
    assert resident_input_bytes < 4 * 1024**3
    largest_output_bytes = max(
        int(case["batch"])
        * int(case["query_heads"])
        * int(case["head_dim"])
        * 2
        for case in panel["cases"]
    )
    assert largest_output_bytes * timing["output_ring_entries"] < 2 * 1024**3

    child_path = HERE / "child_eval.py"
    if not child_path.is_file():
        child_path = Path("/runner/child_eval.py")
    assert child_path.is_file()
    child = child_path.read_text()
    assert "prepare_group" in child and "measure_one" in child
    assert '"streamed_fresh_input_groups": True' in child
    assert '"cpu_exact_mutation_snapshots": True' in child
    assert '"cuda_snapshot_bytes": 0' in child
    assert "prepared = {}" not in child
    repo_protocol = TASK / "environment/protocol.py"
    repo_child = TASK / "environment/child_eval.py"
    if repo_protocol.is_file() and repo_child.is_file():
        assert (HERE / "protocol.py").read_bytes() == repo_protocol.read_bytes()
        assert (HERE / "child_eval.py").read_bytes() == repo_child.read_bytes()
    print("test_memory_safe_protocol: PASS")


if __name__ == "__main__":
    main()
