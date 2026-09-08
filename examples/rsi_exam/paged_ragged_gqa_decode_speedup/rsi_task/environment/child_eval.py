from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/runner")
from protocol import (  # noqa: E402
    TrustedCudaTimer,
    case_from_dict,
    immutability_report,
    make_inputs,
    make_metadata,
    phase_seed,
    snapshot_inputs,
    structure_report,
)


_NP_SAVE = np.save
_PATH_WRITE = Path.write_text
_SCHED_GETAFFINITY = os.sched_getaffinity
_CUDA_RESET_PEAK_MEMORY_STATS = torch.cuda.reset_peak_memory_stats
_CUDA_MAX_MEMORY_ALLOCATED = torch.cuda.max_memory_allocated
_CUDA_MAX_MEMORY_RESERVED = torch.cuda.max_memory_reserved
_REQUIRED_CPU_AFFINITY = [4]


def load_solver(root: Path):
    spec = importlib.util.spec_from_file_location("candidate_solver", root / "solver.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load solver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.paged_gqa_decode


def output_ring_storage_report(outputs: list[torch.Tensor]) -> dict:
    pointers = [int(output.untyped_storage().data_ptr()) for output in outputs]
    unique = len(set(pointers))
    return {
        "entry_count": len(outputs),
        "unique_storage_count": unique,
        "all_storages_unique": unique == len(outputs),
    }


def main() -> int:
    submission, challenge_path, output_path = map(Path, sys.argv[1:4])
    challenge = json.loads(challenge_path.read_text())
    case = case_from_dict(challenge["case"])
    timing = challenge["timing"]
    warmup = int(timing["warmup"])
    repeats = int(timing["repeats"])
    trim_each_side = int(timing["symmetric_trim_each_side"])
    output_dir = output_path.parent / "arrays"
    output_dir.mkdir(parents=True, exist_ok=True)
    challenge_path.unlink()

    cpu_affinity_before_submission_import = sorted(_SCHED_GETAFFINITY(0))
    timer = TrustedCudaTimer(repeats)
    measure_one = timer.measure_one
    prime_clock = timer.prime_clock
    metadata = make_metadata(case)
    solver = load_solver(submission)

    timed_inner_calls = int(timing["timed_inner_calls"])
    required_output_ring_entries = 1 + warmup + repeats * timed_inner_calls
    if (
        timing.get("caller_owned_output_ring") is not True
        or timing.get("output_ring_preallocated_untimed") is not True
        or int(timing.get("output_ring_entries", -1))
        != required_output_ring_entries
        or timing.get("solver_output_copy_into_ring_timed") is not True
        or timing.get("submission_temporary_output_retained") is not False
    ):
        raise RuntimeError("caller-owned output-ring timing contract drift")
    max_peak_allocated_bytes = int(timing["max_peak_allocated_bytes"])
    max_peak_reserved_bytes = int(timing["max_peak_reserved_bytes"])
    if warmup % timed_inner_calls:
        raise RuntimeError("warmup must be divisible by timed_inner_calls")
    calls = []
    live_outputs = []
    output_storage_ptrs = set()
    timed_ms = []
    # Allocate every externally observed return storage before correctness,
    # warmup, clock priming, and all CUDA event pairs. The submission result
    # remains a temporary group-local tensor. Each retained event measures the
    # submission call followed by the copy into its next unique ring target.
    output_ring = [
        torch.empty(
            (case.batch, case.query_heads, case.head_dim),
            device="cuda",
            dtype=case.dtype,
        )
        for _ in range(required_output_ring_entries)
    ]
    output_ring_report = output_ring_storage_report(output_ring)
    if not output_ring_report["all_storages_unique"]:
        raise RuntimeError("caller-owned output ring contains aliased storages")
    output_ring_cursor = 0

    def retain_call(
        phase,
        index,
        seed,
        args,
        snapshot,
        output,
        submission_output,
        elapsed_ms,
        wall_ms,
        timed_event_index,
    ) -> None:
        immutable = immutability_report(args, snapshot)
        structure = structure_report(output, args)
        submission_structure = structure_report(submission_output, args)
        if structure["passed"]:
            output_ptr = int(output.untyped_storage().data_ptr())
            if output_ptr in output_storage_ptrs:
                structure["passed"] = False
                structure["failures"].append("output storage was reused across calls")
            else:
                output_storage_ptrs.add(output_ptr)
        array_path = output_dir / f"{phase}_{index}.npy"
        if structure["passed"]:
            # NumPy cannot represent torch.bfloat16. Structure/dtype is checked
            # above inside the trusted CUDA child; serialize values as FP32 for
            # the independent parent-side numerical comparison.
            _NP_SAVE(array_path, output.detach().float().cpu().numpy(), allow_pickle=False)
            array_name = array_path.name
        else:
            array_name = None
        calls.append({
            "phase": phase,
            "index": index,
            "seed": seed,
            "array": array_name,
            "immutability": immutable,
            "structure": structure,
            "submission_structure": submission_structure,
            "elapsed_ms": elapsed_ms,
            "wall_ms": wall_ms,
            "timed_event_index": timed_event_index,
        })
        if isinstance(output, torch.Tensor):
            # Keep every caller-owned ring target live and independently
            # auditable. Submission temporaries remain group-local.
            live_outputs.append(output)

    def prepare_group(phase: str, start: int, count: int):
        rows = []
        for index in range(start, start + count):
            seed = phase_seed(case, phase, index)
            args = make_inputs(case, seed=seed, metadata=metadata)
            rows.append((index, seed, args, snapshot_inputs(args)))
        return rows

    def run_group(group):
        nonlocal output_ring_cursor
        rows = []
        for row in group:
            if output_ring_cursor >= len(output_ring):
                raise RuntimeError("caller-owned output ring exhausted")
            submission_output = solver(*row[2])
            target = output_ring[output_ring_cursor]
            output_ring_cursor += 1
            # The copy is deliberately inside the same timed function as the
            # solver call. Ring allocation and target creation are untimed.
            target.copy_(submission_output)
            rows.append((target, submission_output))
        return tuple(rows)

    def retain_group(phase, group, outputs, elapsed_ms, wall_ms, event_index):
        if len(outputs) != len(group):
            raise RuntimeError(f"{phase} group returned the wrong number of outputs")
        for (index, seed, args, snapshot), output_pair in zip(
            group, outputs, strict=True
        ):
            output, submission_output = output_pair
            retain_call(
                phase,
                index,
                seed,
                args,
                snapshot,
                output,
                submission_output,
                elapsed_ms,
                wall_ms,
                event_index,
            )

    def memory_record() -> dict:
        allocated = int(_CUDA_MAX_MEMORY_ALLOCATED())
        reserved = int(_CUDA_MAX_MEMORY_RESERVED())
        passed = bool(
            allocated <= max_peak_allocated_bytes
            and reserved <= max_peak_reserved_bytes
        )
        record = {
            "passed": passed,
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
            "max_peak_allocated_bytes": max_peak_allocated_bytes,
            "max_peak_reserved_bytes": max_peak_reserved_bytes,
        }
        if not passed:
            raise RuntimeError(f"CUDA peak-memory gate failed: {record}")
        return record

    # V24 keeps every tensor value fresh but bounds residency to one eight-call
    # group. Exact CPU snapshots, input generation, synchronization, mutation
    # checks, and output serialization all occur outside retained event pairs.
    _CUDA_RESET_PEAK_MEMORY_STATS()
    correctness_group = prepare_group("correctness", 0, 1)
    correctness_outputs = run_group(correctness_group)
    retain_group(
        "correctness", correctness_group, correctness_outputs, None, None, None
    )
    memory_record()
    del correctness_group, correctness_outputs

    for group_start in range(0, warmup, timed_inner_calls):
        group = prepare_group("warmup", group_start, timed_inner_calls)
        if group_start == 0:
            # The first group is already resident; the bound primer is therefore
            # immediately before the first candidate warmup call.
            prime_clock()
        outputs = run_group(group)
        retain_group("warmup", group, outputs, None, None, None)
        memory_record()
        del group, outputs

    for event_index in range(repeats):
        group_start = event_index * timed_inner_calls
        group = prepare_group("timed", group_start, timed_inner_calls)
        outputs, elapsed_ms, wall_ms = measure_one(
            event_index,
            lambda current=group: run_group(current),
        )
        per_call_elapsed_ms = elapsed_ms / timed_inner_calls
        per_call_wall_ms = wall_ms / timed_inner_calls
        timed_ms.append(per_call_elapsed_ms)
        retain_group(
            "timed",
            group,
            outputs,
            per_call_elapsed_ms,
            per_call_wall_ms,
            event_index,
        )
        memory_record()
        del group, outputs

    cpu_affinity_after_all_candidate_calls = sorted(_SCHED_GETAFFINITY(0))
    final_memory = memory_record()
    if output_ring_cursor != required_output_ring_entries:
        raise RuntimeError(
            "caller-owned output ring was not consumed exactly once per call"
        )

    payload = {
        "schema_version": 1,
        "case_id": case.case_id,
        "timer": {
            "created_before_submission_import": True,
            "fresh_values_per_call": True,
            "fresh_output_storage_per_call": True,
            "caller_owned_output_ring": True,
            "output_ring_preallocated_untimed": True,
            "output_ring_entries": required_output_ring_entries,
            "output_ring_unique_storage_count": output_ring_report[
                "unique_storage_count"
            ],
            "output_ring_all_storages_unique": output_ring_report[
                "all_storages_unique"
            ],
            "solver_output_copy_into_ring_timed": True,
            "submission_temporary_output_retained": False,
            "timed_region_definition": (
                "submission solver call followed by target.copy_(submission_output) "
                "into the next untimed-preallocated unique caller-owned output"
            ),
            "streamed_fresh_input_groups": True,
            "resident_calls_per_group": timed_inner_calls,
            "cpu_exact_mutation_snapshots": True,
            "cuda_snapshot_bytes": 0,
            "input_setup_inside_timing": False,
            "validation_inside_timing": False,
            "contiguous_calls_inside_each_timed_event": True,
            "synchronize_after_each_timed_event": True,
            "warmup_batch_size": warmup,
            "warmup_group_count": warmup // timed_inner_calls,
            "fixed_steady_state_warmup": warmup == 128,
            "adaptive_warmup_or_posthoc_selection": False,
            "timed_batch_size": repeats,
            "timed_event_count": repeats,
            "timed_calls_per_event": timed_inner_calls,
            "timed_call_count": repeats * timed_inner_calls,
            "event_latency_divided_by_timed_calls": True,
            "clock_primer_rounds": timer._clock_primer_rounds,
            "clock_primer_immediately_before_warmup": True,
            "symmetric_trim_each_side": trim_each_side,
            "stability_retained_count": repeats - 2 * trim_each_side,
            "dispersion_hard_gate": True,
            "required_cpu_affinity": _REQUIRED_CPU_AFFINITY,
            "cpu_affinity_before_submission_import": cpu_affinity_before_submission_import,
            "cpu_affinity_after_all_candidate_calls": cpu_affinity_after_all_candidate_calls,
            "sched_affinity_verified": bool(
                cpu_affinity_before_submission_import == _REQUIRED_CPU_AFFINITY
                and cpu_affinity_after_all_candidate_calls == _REQUIRED_CPU_AFFINITY
            ),
            "stable_page_table_per_case": True,
            "sealed_challenge_unlinked_before_submission_import": True,
        },
        "memory": final_memory,
        "calls": calls,
        "candidate_repeat_ms": timed_ms,
        "candidate_median_ms": statistics.median(timed_ms),
    }
    _PATH_WRITE(output_path, json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
