from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Callable

import torch


DEFAULT_WARMUP = 128
DEFAULT_REPEATS = 21
DEFAULT_MAX_MIN_RATIO = 1.20
DEFAULT_TRIM_EACH_SIDE = 4
CLOCK_PRIMER_ROUNDS = 64

_CUDA_EVENT = torch.cuda.Event
_CUDA_SYNC = torch.cuda.synchronize
_PERF_COUNTER = __import__("time").perf_counter
_TORCH_EQUAL = torch.equal
_TORCH_MM = torch.mm

INPUT_NAMES = (
    "q", "k_cache", "v_cache", "page_indptr", "page_indices",
    "last_page_len", "sm_scale", "logits_soft_cap", "pos_encoding_mode",
    "window_left", "rope_scale", "rope_theta",
)


@dataclass(frozen=True)
class Case:
    case_id: str
    seed: int
    dtype_name: str
    batch: int
    query_heads: int
    kv_heads: int
    head_dim: int
    page_size: int
    page_counts: tuple[int, ...]
    last_page_len: tuple[int, ...]
    logits_soft_cap: float
    pos_encoding_mode: int
    window_left: int
    rope_scale: float
    rope_theta: float

    @property
    def dtype(self) -> torch.dtype:
        return {"float16": torch.float16, "bfloat16": torch.bfloat16}[self.dtype_name]

    @property
    def total_pages(self) -> int:
        return sum(self.page_counts)


def case_from_dict(raw: dict) -> Case:
    case = Case(
        case_id=str(raw["id"]),
        seed=int(raw["seed"]),
        dtype_name=str(raw["dtype"]),
        batch=int(raw["batch"]),
        query_heads=int(raw["query_heads"]),
        kv_heads=int(raw["kv_heads"]),
        head_dim=int(raw["head_dim"]),
        page_size=int(raw["page_size"]),
        page_counts=tuple(int(x) for x in raw["page_counts"]),
        last_page_len=tuple(int(x) for x in raw["last_page_len"]),
        logits_soft_cap=float(raw["logits_soft_cap"]),
        pos_encoding_mode=int(raw["pos_encoding_mode"]),
        window_left=int(raw["window_left"]),
        rope_scale=float(raw["rope_scale"]),
        rope_theta=float(raw["rope_theta"]),
    )
    if case.dtype_name not in {"float16", "bfloat16"}:
        raise ValueError(f"{case.case_id}: unsupported dtype")
    if case.batch <= 0 or len(case.page_counts) != case.batch:
        raise ValueError(f"{case.case_id}: page_counts must match batch")
    if len(case.last_page_len) != case.batch:
        raise ValueError(f"{case.case_id}: last_page_len must match batch")
    if case.query_heads <= 0 or case.kv_heads <= 0 or case.query_heads % case.kv_heads:
        raise ValueError(f"{case.case_id}: invalid GQA head ratio")
    if case.head_dim not in {64, 128, 256} or case.page_size not in {1, 8, 16, 32}:
        raise ValueError(f"{case.case_id}: unsupported head_dim or page_size")
    if any(count <= 0 for count in case.page_counts):
        raise ValueError(f"{case.case_id}: every request needs a page")
    if any(length <= 0 or length > case.page_size for length in case.last_page_len):
        raise ValueError(f"{case.case_id}: invalid last_page_len")
    if not math.isfinite(case.logits_soft_cap) or not 0.0 < case.logits_soft_cap <= 50.0:
        raise ValueError(f"{case.case_id}: invalid logits_soft_cap")
    if case.pos_encoding_mode not in {0, 1}:
        raise ValueError(f"{case.case_id}: pos_encoding_mode must be 0 (NONE) or 1 (ROPE_LLAMA)")
    if case.window_left < -1:
        raise ValueError(f"{case.case_id}: window_left must be -1 or non-negative")
    if (
        not math.isfinite(case.rope_scale)
        or case.rope_scale <= 0.0
        or not math.isfinite(case.rope_theta)
        or case.rope_theta <= 1.0
    ):
        raise ValueError(f"{case.case_id}: invalid RoPE parameters")
    return case


def phase_seed(case: Case, phase: str, index: int = 0) -> int:
    offsets = {"correctness": 10_000, "warmup": 100_000, "timed": 200_000}
    if phase not in offsets:
        raise ValueError(f"unknown phase {phase!r}")
    return case.seed + offsets[phase] + 1009 * int(index)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def make_metadata(case: Case, device: str | torch.device = "cuda") -> tuple[torch.Tensor, ...]:
    device = torch.device(device)
    indptr = [0]
    for count in case.page_counts:
        indptr.append(indptr[-1] + count)
    slack = max(8, case.total_pages // 7)
    physical_pages = case.total_pages + slack
    permutation = torch.randperm(
        physical_pages,
        generator=_generator(device, case.seed + 701),
        dtype=torch.int64,
        device=device,
    )
    page_indices = permutation[: case.total_pages].to(torch.int32)
    return (
        torch.tensor(indptr, dtype=torch.int32, device=device),
        page_indices.contiguous(),
        torch.tensor(case.last_page_len, dtype=torch.int32, device=device),
    )


def make_inputs(
    case: Case,
    *,
    seed: int,
    metadata: tuple[torch.Tensor, ...] | None = None,
    device: str | torch.device = "cuda",
) -> tuple[object, ...]:
    device = torch.device(device)
    page_indptr, page_indices, last_page_len = (
        make_metadata(case, device) if metadata is None else metadata
    )
    physical_pages = int(page_indices.max().item()) + 1
    physical_pages = max(physical_pages, case.total_pages + max(8, case.total_pages // 7))
    generator = _generator(device, seed)
    q = torch.randn(
        (case.batch, case.query_heads, case.head_dim),
        generator=generator,
        device=device,
        dtype=case.dtype,
    )
    cache_shape = (physical_pages, case.page_size, case.kv_heads, case.head_dim)
    k_cache = torch.randn(cache_shape, generator=generator, device=device, dtype=case.dtype)
    v_cache = torch.randn(cache_shape, generator=generator, device=device, dtype=case.dtype)
    sm_scale = 1.0 / math.sqrt(case.head_dim)
    return (
        q,
        k_cache,
        v_cache,
        page_indptr,
        page_indices,
        last_page_len,
        sm_scale,
        case.logits_soft_cap,
        case.pos_encoding_mode,
        case.window_left,
        case.rope_scale,
        case.rope_theta,
    )


def _rope_llama_half_split(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    rope_scale: float,
    rope_theta: float,
) -> torch.Tensor:
    """Required ROPE_LLAMA semantics: half-split rotation in float32."""
    head_dim = tensor.shape[-1]
    inv_freq = 1.0 / (
        float(rope_theta)
        ** (
            torch.arange(0, head_dim, 2, device=tensor.device, dtype=torch.float32)
            / head_dim
        )
    )
    inv_freq = inv_freq / float(rope_scale)
    angles = positions.float().unsqueeze(-1) * inv_freq
    while angles.ndim < tensor.ndim:
        angles = angles.unsqueeze(-2)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    source = tensor.float()
    first, second = source[..., : head_dim // 2], source[..., head_dim // 2 :]
    return torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine),
        dim=-1,
    )


def reference_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_indptr: torch.Tensor,
    page_indices: torch.Tensor,
    last_page_len: torch.Tensor,
    sm_scale: float,
    logits_soft_cap: float,
    pos_encoding_mode: int,
    window_left: int,
    rope_scale: float,
    rope_theta: float,
) -> torch.Tensor:
    batch, query_heads, head_dim = q.shape
    kv_heads = k_cache.shape[2]
    group_size = query_heads // kv_heads
    page_size = k_cache.shape[1]
    outputs = []
    for request in range(batch):
        start = int(page_indptr[request].item())
        end = int(page_indptr[request + 1].item())
        ids = page_indices[start:end].to(torch.int64)
        valid = (end - start - 1) * page_size + int(last_page_len[request].item())
        keys = k_cache.index_select(0, ids).reshape(-1, kv_heads, head_dim)[:valid]
        values = v_cache.index_select(0, ids).reshape(-1, kv_heads, head_dim)[:valid]
        first_token = 0 if int(window_left) < 0 else max(0, valid - int(window_left) - 1)
        keys = keys[first_token:valid]
        values = values[first_token:valid]
        if int(pos_encoding_mode) == 1:
            query = _rope_llama_half_split(
                q[request],
                torch.tensor(valid - 1, device=q.device),
                rope_scale,
                rope_theta,
            )
            keys = _rope_llama_half_split(
                keys,
                torch.arange(first_token, valid, device=q.device),
                rope_scale,
                rope_theta,
            )
        else:
            query = q[request].float()
            keys = keys.float()
        keys = keys.repeat_interleave(group_size, dim=1)
        values = values.float().repeat_interleave(group_size, dim=1)
        scores = torch.einsum("hd,lhd->hl", query, keys) * float(sm_scale)
        scores = float(logits_soft_cap) * torch.tanh(scores / float(logits_soft_cap))
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hl,lhd->hd", probabilities, values))
    return torch.stack(outputs, dim=0).to(q.dtype)


_BASELINE_PLAN = None


def _vectorized_page_plan(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_indptr: torch.Tensor,
    page_indices: torch.Tensor,
    last_page_len: torch.Tensor,
    window_left: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global _BASELINE_PLAN
    key = (
        q.device,
        tuple(q.shape),
        tuple(k_cache.shape),
        page_indptr.data_ptr(),
        page_indices.data_ptr(),
        last_page_len.data_ptr(),
        int(window_left),
    )
    if _BASELINE_PLAN is not None and _BASELINE_PLAN[0] == key:
        return (
            _BASELINE_PLAN[4],
            _BASELINE_PLAN[5],
            _BASELINE_PLAN[6],
            _BASELINE_PLAN[7],
        )
    indptr = [int(value) for value in page_indptr.detach().cpu().tolist()]
    page_ids = [int(value) for value in page_indices.detach().cpu().tolist()]
    tails = [int(value) for value in last_page_len.detach().cpu().tolist()]
    page_size = int(k_cache.shape[1])
    token_rows = []
    position_rows = []
    query_positions = []
    for request in range(int(q.shape[0])):
        start, end = indptr[request], indptr[request + 1]
        valid_tokens = (end - start - 1) * page_size + tails[request]
        first_token = (
            0
            if int(window_left) < 0
            else max(0, valid_tokens - int(window_left) - 1)
        )
        physical_tokens = [
            page_id * page_size + offset
            for page_id in page_ids[start:end]
            for offset in range(page_size)
        ][:valid_tokens]
        token_rows.append(physical_tokens[first_token:])
        position_rows.append(list(range(first_token, valid_tokens)))
        query_positions.append(valid_tokens - 1)
    width = max(len(row) for row in token_rows)
    gather = torch.tensor(
        [row + [0] * (width - len(row)) for row in token_rows],
        dtype=torch.int64,
        device=q.device,
    )
    valid = torch.tensor(
        [
            [True] * len(row) + [False] * (width - len(row))
            for row in token_rows
        ],
        dtype=torch.bool,
        device=q.device,
    )
    positions = torch.tensor(
        [row + [0] * (width - len(row)) for row in position_rows],
        dtype=torch.float32,
        device=q.device,
    )
    query_position = torch.tensor(
        query_positions, dtype=torch.float32, device=q.device
    )
    _BASELINE_PLAN = (
        key,
        page_indptr,
        page_indices,
        last_page_len,
        gather,
        valid,
        positions,
        query_position,
    )
    return gather, valid, positions, query_position


def vectorized_baseline_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_indptr: torch.Tensor,
    page_indices: torch.Tensor,
    last_page_len: torch.Tensor,
    sm_scale: float,
    logits_soft_cap: float,
    pos_encoding_mode: int,
    window_left: int,
    rope_scale: float,
    rope_theta: float,
) -> torch.Tensor:
    """Batched PyTorch baseline; the loop reference remains correctness-only."""
    batch, query_heads, head_dim = q.shape
    kv_heads = int(k_cache.shape[2])
    group_size = query_heads // kv_heads
    gather, valid, positions, query_position = _vectorized_page_plan(
        q,
        k_cache,
        page_indptr,
        page_indices,
        last_page_len,
        window_left,
    )
    width = int(gather.shape[1])
    keys = k_cache.reshape(-1, kv_heads, head_dim).index_select(
        0, gather.reshape(-1)
    ).reshape(batch, width, kv_heads, head_dim).float()
    values = v_cache.reshape(-1, kv_heads, head_dim).index_select(
        0, gather.reshape(-1)
    ).reshape(batch, width, kv_heads, head_dim).float()
    query = q.reshape(batch, kv_heads, group_size, head_dim).float()
    if int(pos_encoding_mode) == 1:
        query = _rope_llama_half_split(
            query,
            query_position[:, None, None],
            rope_scale,
            rope_theta,
        )
        keys = _rope_llama_half_split(
            keys,
            positions[:, :, None],
            rope_scale,
            rope_theta,
        )
    scores = torch.einsum("bkgd,blkd->bkgl", query, keys) * float(sm_scale)
    scores = float(logits_soft_cap) * torch.tanh(
        scores / float(logits_soft_cap)
    )
    scores.masked_fill_(~valid[:, None, None, :], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("bkgl,blkd->bkgd", probabilities, values)
    return output.reshape(batch, query_heads, head_dim).to(q.dtype)


baseline_decode = vectorized_baseline_decode


def snapshot_inputs(args: tuple[object, ...]) -> tuple[tuple[torch.Tensor | None, ...], tuple[int | None, ...]]:
    """Take one exact, non-CUDA mutation snapshot.

    A full CUDA clone doubles the already-large paged KV working set and made
    the superseded v23 attempt exceed a 48 GiB A6000 before the first call. A device-to-host copy is
    exact, lives outside every retained event, and cannot alias candidate
    inputs.  The tensor version remains an independent mutation signal.
    """
    tensors = tuple(
        value.detach().to(device="cpu", copy=True)
        if isinstance(value, torch.Tensor)
        else None
        for value in args
    )
    versions = tuple(int(value._version) if isinstance(value, torch.Tensor) else None for value in args)
    return tensors, versions


def immutability_report(args: tuple[object, ...], snapshot) -> dict:
    copies, versions = snapshot
    changed = []
    for name, value, copy, version in zip(INPUT_NAMES, args, copies, versions):
        if not isinstance(value, torch.Tensor):
            continue
        current = value.detach().to(device="cpu", copy=True)
        if int(value._version) != version or not _TORCH_EQUAL(current, copy):
            changed.append(name)
    return {
        "passed": not changed,
        "changed": changed,
        "snapshot_storage": "cpu_exact_copy",
        "cuda_snapshot_bytes": 0,
    }


def structure_report(output: object, args: tuple[object, ...]) -> dict:
    q = args[0]
    failures = []
    if not isinstance(output, torch.Tensor):
        failures.append("output is not a tensor")
    else:
        if not output.is_cuda or output.device != q.device:
            failures.append("output device mismatch")
        if output.dtype != q.dtype:
            failures.append("output dtype mismatch")
        if tuple(output.shape) != tuple(q.shape):
            failures.append("output shape mismatch")
        if not bool(torch.isfinite(output).all()):
            failures.append("output contains non-finite values")
        input_ptrs = {x.untyped_storage().data_ptr() for x in args if isinstance(x, torch.Tensor)}
        if output.untyped_storage().data_ptr() in input_ptrs:
            failures.append("output aliases input storage")
    return {"passed": not failures, "failures": failures}


def tolerance(case: Case) -> tuple[float, float]:
    return (2.0e-2, 2.0e-2) if case.dtype_name == "float16" else (3.0e-2, 3.0e-2)


def correctness_report(output: torch.Tensor, expected: torch.Tensor, case: Case) -> dict:
    rtol, atol = tolerance(case)
    diff = (output.float() - expected.float()).abs()
    envelope = atol + rtol * expected.float().abs()
    normalized = float((diff / envelope.clamp_min(torch.finfo(torch.float32).tiny)).max().item())
    relative_l2 = float(torch.linalg.vector_norm(diff).item()) / max(
        float(torch.linalg.vector_norm(expected.float()).item()), 1.0e-20
    )
    return {
        "passed": bool(normalized <= 1.0),
        "max_abs_error": float(diff.max().item()),
        "relative_l2_error": relative_l2,
        "normalized_error": normalized,
        "rtol": rtol,
        "atol": atol,
    }


class TrustedCudaTimer:
    def __init__(self, repeats: int):
        self._synchronize = _CUDA_SYNC
        self._clock = _PERF_COUNTER
        self._slots = []
        for _ in range(repeats):
            start, end = _CUDA_EVENT(enable_timing=True), _CUDA_EVENT(enable_timing=True)
            self._slots.append((start.record, end.record, start.elapsed_time, end))
        # Prime the device clock immediately before the retained batch. Without
        # this, an A6000 can enter the batch at an idle clock for sub-0.1 ms
        # kernels. These tensors and the bound torch.mm primitive are created
        # before submission import and the primer is outside every event pair.
        lhs = torch.ones((1024, 1024), device="cuda", dtype=torch.float16)
        rhs = torch.ones((1024, 1024), device="cuda", dtype=torch.float16)
        out = torch.empty_like(lhs)
        self._clock_primer = (lhs, rhs, out)
        self._clock_primer_rounds = CLOCK_PRIMER_ROUNDS
        self.prime_clock()

    def prime_clock(self) -> None:
        lhs, rhs, out = self._clock_primer
        for _ in range(self._clock_primer_rounds):
            _TORCH_MM(lhs, rhs, out=out)
        self.synchronize()

    def synchronize(self) -> None:
        self._synchronize()

    def measure_batch(
        self,
        functions: list[Callable[[], torch.Tensor]],
    ) -> list[tuple[torch.Tensor, float, float]]:
        if len(functions) != len(self._slots):
            raise ValueError("timed function count does not match precreated event slots")
        outputs = []
        dispatch_ms = []
        for fn, (start_record, end_record, _, _) in zip(functions, self._slots):
            wall_start = self._clock()
            start_record()
            outputs.append(fn())
            end_record()
            dispatch_ms.append((self._clock() - wall_start) * 1.0e3)
        self.synchronize()
        results = []
        for output, wall_ms, (_, _, elapsed_time, end) in zip(outputs, dispatch_ms, self._slots):
            event_ms = float(elapsed_time(end))
            # Event pairs and their bound methods are captured before submission
            # import. An invalid event fails closed instead of substituting host
            # dispatch time for a sub-0.1 ms GPU measurement.
            elapsed_ms = event_ms if event_ms > 0.0 else math.inf
            results.append((output, elapsed_ms, max(wall_ms, 1.0e-12)))
        return results

    def measure_one(
        self,
        index: int,
        function: Callable[[], torch.Tensor],
    ) -> tuple[torch.Tensor, float, float]:
        """Measure one pre-created event slot and synchronize before reuse.

        V24 streams one fresh eight-call input group at a time. Input creation,
        CPU mutation snapshots, synchronization, and validation all stay
        outside this event pair; only the eight contiguous candidate calls are
        retained. Per-group synchronization bounds memory without changing the
        CUDA-event definition.
        """
        if not 0 <= index < len(self._slots):
            raise IndexError("timed event slot is outside the precreated range")
        start_record, end_record, elapsed_time, end = self._slots[index]
        wall_start = self._clock()
        start_record()
        output = function()
        end_record()
        wall_ms = (self._clock() - wall_start) * 1.0e3
        self.synchronize()
        event_ms = float(elapsed_time(end))
        elapsed_ms = event_ms if event_ms > 0.0 else math.inf
        return output, elapsed_ms, max(wall_ms, 1.0e-12)


def repeat_stability_report(repeat_ms: list[float], trim_each_side: int) -> dict:
    """Audit a fixed two-phase robust range without changing the score vector.

    The fresh-allocation path can have a deterministic period-two
    latency caused by alternating allocator/address phases.  Pooling both
    phases makes a perfectly repeatable two-mode process look unstable.  The
    gate therefore keeps the schedule fixed, splits samples by event parity,
    applies half of the original trim to each phase, and uses the worse of the
    two within-phase ranges.  The ordinary median still scores all 21 samples.
    """
    finite = bool(repeat_ms) and all(math.isfinite(x) and x > 0.0 for x in repeat_ms)
    phase_period = 2
    phase_trim_each_side = trim_each_side // phase_period
    valid_trim = (
        trim_each_side >= 0
        and trim_each_side % phase_period == 0
        and all(
            len(repeat_ms[phase::phase_period]) > 2 * phase_trim_each_side
            for phase in range(phase_period)
        )
    )
    report = {
        "estimator": "alternating_phase_balanced_symmetric_sorted_trim",
        "sample_count": len(repeat_ms),
        "trim_each_side": int(trim_each_side),
        "phase_period": phase_period,
        "phase_trim_each_side": phase_trim_each_side,
        "retained_count": max(len(repeat_ms) - 2 * trim_each_side, 0),
        "worst_phase": None,
        "phases": {},
        "trimmed_low_indices": [],
        "trimmed_high_indices": [],
        "retained_indices": [],
        "full_min_index": None,
        "full_min_cuda_ms": None,
        "full_max_index": None,
        "full_max_cuda_ms": None,
        "retained_min_index": None,
        "retained_min_cuda_ms": None,
        "retained_max_index": None,
        "retained_max_cuda_ms": None,
        "full_max_min_ratio": math.inf,
        "trimmed_max_min_ratio": math.inf,
    }
    if not finite or not valid_trim:
        return report

    ordered = sorted(enumerate(repeat_ms), key=lambda item: (item[1], item[0]))
    full_min, full_max = ordered[0], ordered[-1]
    retained_all = []
    trimmed_low = []
    trimmed_high = []
    phase_reports = {}
    for phase in range(phase_period):
        phase_ordered = sorted(
            ((index, repeat_ms[index]) for index in range(phase, len(repeat_ms), phase_period)),
            key=lambda item: (item[1], item[0]),
        )
        phase_retained = phase_ordered[
            phase_trim_each_side : len(phase_ordered) - phase_trim_each_side or None
        ]
        phase_min, phase_max = phase_retained[0], phase_retained[-1]
        low = [index for index, _ in phase_ordered[:phase_trim_each_side]]
        high = (
            [index for index, _ in phase_ordered[len(phase_ordered) - phase_trim_each_side :]]
            if phase_trim_each_side
            else []
        )
        phase_reports[str(phase)] = {
            "sample_count": len(phase_ordered),
            "trim_each_side": phase_trim_each_side,
            "retained_count": len(phase_retained),
            "trimmed_low_indices": low,
            "trimmed_high_indices": high,
            "retained_indices": [index for index, _ in phase_retained],
            "retained_min_index": phase_min[0],
            "retained_min_cuda_ms": phase_min[1],
            "retained_max_index": phase_max[0],
            "retained_max_cuda_ms": phase_max[1],
            "trimmed_max_min_ratio": phase_max[1] / phase_min[1],
        }
        retained_all.extend(phase_retained)
        trimmed_low.extend(low)
        trimmed_high.extend(high)
    worst_phase = max(
        phase_reports,
        key=lambda phase: (phase_reports[phase]["trimmed_max_min_ratio"], int(phase)),
    )
    worst = phase_reports[worst_phase]
    report.update({
        "worst_phase": int(worst_phase),
        "phases": phase_reports,
        "trimmed_low_indices": sorted(trimmed_low),
        "trimmed_high_indices": sorted(trimmed_high),
        "retained_indices": sorted(index for index, _ in retained_all),
        "full_min_index": full_min[0],
        "full_min_cuda_ms": full_min[1],
        "full_max_index": full_max[0],
        "full_max_cuda_ms": full_max[1],
        "retained_min_index": worst["retained_min_index"],
        "retained_min_cuda_ms": worst["retained_min_cuda_ms"],
        "retained_max_index": worst["retained_max_index"],
        "retained_max_cuda_ms": worst["retained_max_cuda_ms"],
        "full_max_min_ratio": full_max[1] / full_min[1],
        "trimmed_max_min_ratio": worst["trimmed_max_min_ratio"],
    })
    return report


def repeat_stability_ratio(
    repeat_ms: list[float], trim_each_side: int = DEFAULT_TRIM_EACH_SIDE
) -> float:
    """Return the center-trimmed robust-range ratio used by the hard gate."""
    return float(repeat_stability_report(repeat_ms, trim_each_side)["trimmed_max_min_ratio"])


def timing_report(
    repeat_ms: list[float],
    parent_wall_ms: float,
    diagnostic_max_min_ratio: float,
    trim_each_side: int,
) -> dict:
    finite_positive = bool(repeat_ms) and all(
        math.isfinite(x) and x > 0.0 for x in repeat_ms
    )
    stability = repeat_stability_report(repeat_ms, trim_each_side)
    ratio = float(stability["trimmed_max_min_ratio"])
    median_ms = statistics.median(repeat_ms) if finite_positive else math.inf
    gpu_sum_ms = sum(repeat_ms) if finite_positive else math.inf
    wall_finite_positive = math.isfinite(parent_wall_ms) and parent_wall_ms > 0.0
    process_ratio = (
        parent_wall_ms / gpu_sum_ms
        if wall_finite_positive and finite_positive and gpu_sum_ms > 0.0
        else 0.0
    )
    wall_time_possible = bool(wall_finite_positive and process_ratio >= 0.5)
    vector_shape_ok = bool(
        len(repeat_ms) == DEFAULT_REPEATS
        and trim_each_side == DEFAULT_TRIM_EACH_SIDE
        and stability["retained_count"]
        == DEFAULT_REPEATS - 2 * DEFAULT_TRIM_EACH_SIDE
    )
    hard_gate_reasons = []
    if not finite_positive:
        hard_gate_reasons.append("repeat vector contains a non-finite or non-positive value")
    if not vector_shape_ok:
        hard_gate_reasons.append("repeat vector or diagnostic trim shape does not match protocol")
    if not wall_time_possible:
        hard_gate_reasons.append("parent wall time is non-finite, non-positive, or impossibly short")
    diagnostic_within_limit = bool(
        finite_positive
        and math.isfinite(diagnostic_max_min_ratio)
        and diagnostic_max_min_ratio > 0.0
        and ratio <= diagnostic_max_min_ratio
    )
    if not diagnostic_within_limit:
        hard_gate_reasons.append(
            "worst alternating-phase retained interval exceeds the configured dispersion ceiling"
        )
    return {
        "passed": not hard_gate_reasons,
        "median_ms": median_ms,
        "max_min_ratio": ratio,
        "full_max_min_ratio": stability["full_max_min_ratio"],
        "trimmed_max_min_ratio": ratio,
        "diagnostic_max_min_ratio": float(diagnostic_max_min_ratio),
        "diagnostic_within_limit": diagnostic_within_limit,
        "dispersion_hard_gate": True,
        "scoring_estimator": "ordinary_median_all_21_retained_samples",
        "robust_interval": {
            "definition": "alternating_phase_balanced_center_13_after_two_low_and_two_high_per_phase",
            "phase_period": 2,
            "worst_phase": stability["worst_phase"],
            "low_cuda_ms": stability["retained_min_cuda_ms"],
            "high_cuda_ms": stability["retained_max_cuda_ms"],
            "max_min_ratio": ratio,
            "ceiling": float(diagnostic_max_min_ratio),
        },
        "outlier_rule": (
            "within each fixed event-parity phase, two lowest and two highest "
            "samples are excluded only from dispersion; all 21 samples remain "
            "in the scoring median"
        ),
        "parent_wall_to_gpu_sum_ratio": process_ratio,
        "wall_time_possible": wall_time_possible,
        "hard_gate_reasons": hard_gate_reasons,
        "stability": stability,
    }
