from __future__ import annotations

import hashlib
import math

import numpy as np
import torch


ENTRYPOINT = "attention_varlen_feature_full_vjp"
WARMUPS = 5
REPEATS = 21
DISPERSION_CENTER_COUNT = 13
MAX_DISPERSION = 1.20
SIGNATURE_ABS_SUM_RTOL = 0.08
SIGNATURE_SQ_SUM_RTOL = 0.12
SIGNATURE_SIGNED_SUM_TO_L1 = 0.02
SIGNATURE_SAMPLE_RTOL = 0.18
SIGNATURE_SAMPLE_ATOL = 0.20


def phase_seed(case, phase, index):
    payload = f"{case['id']}|{case['seed']}|{phase}|{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF


def make_inputs(case, seed):
    torch.manual_seed(seed)
    dtype = getattr(torch, case["dtype"])
    total_q = sum(case["q_lengths"])
    total_k = sum(case["k_lengths"])

    def packed(total, heads):
        dim = int(case["head_dim"])
        padding = (
            int(case["head_padding"])
            if case.get("layout", "contiguous") == "padded"
            else 0
        )
        token_step = int(case.get("token_step", 1))
        base = torch.randn(
            total * token_step, heads, dim + padding,
            device="cuda", dtype=dtype,
        )
        return (
            base[::token_step, :, :dim]
            .detach()
            .requires_grad_()
        )

    q = packed(total_q, case["q_heads"])
    k = packed(total_k, case["kv_heads"])
    v = packed(total_k, case["kv_heads"])
    cu_q = torch.tensor(
        [0] + list(torch.tensor(case["q_lengths"]).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )
    cu_k = torch.tensor(
        [0] + list(torch.tensor(case["k_lengths"]).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )
    alibi_shape = (
        (case["q_heads"],)
        if case.get("alibi_mode", "per_batch") == "shared"
        else (len(case["q_lengths"]), case["q_heads"])
    )
    alibi_slopes = 0.00001 + 0.00039 * torch.rand(
        *alibi_shape, device="cuda", dtype=torch.float32
    )
    dout = packed(total_q, case["q_heads"]).detach()
    return q, k, v, cu_q, cu_k, alibi_slopes, dout


def _local_mask(length_q, length_k, left, right, device):
    row = torch.arange(length_q, device=device, dtype=torch.long)[:, None]
    col = torch.arange(length_k, device=device, dtype=torch.long)[None, :]
    shifted = row + length_k - length_q
    if left < 0 and right < 0:
        return torch.zeros(
            (length_q, length_k), device=device, dtype=torch.bool
        )
    if left < 0:
        return col > shifted + right
    return torch.logical_or(
        col > torch.minimum(
            shifted + right,
            torch.full_like(shifted, length_k),
        ),
        col < shifted - left,
    )


def _reference(
    q,
    k,
    v,
    cu_q,
    cu_k,
    alibi_slopes,
    dout,
    causal,
    window_left,
    window_right,
    softcap,
    scale_multiplier,
):
    outputs = []
    group = q.shape[1] // k.shape[1]
    scale = float(scale_multiplier) / math.sqrt(q.shape[-1])
    right = 0 if causal else int(window_right)
    for index in range(cu_q.numel() - 1):
        q_lo, q_hi = int(cu_q[index]), int(cu_q[index + 1])
        k_lo, k_hi = int(cu_k[index]), int(cu_k[index + 1])
        qi = q[q_lo:q_hi].transpose(0, 1)
        ki = k[k_lo:k_hi].repeat_interleave(group, dim=1).transpose(0, 1)
        vi = v[k_lo:k_hi].repeat_interleave(group, dim=1).transpose(0, 1)
        scores = torch.matmul(qi.float(), ki.float().transpose(-1, -2)) * scale
        if softcap > 0:
            scores = float(softcap) * torch.tanh(scores / float(softcap))

        length_q, length_k = q_hi - q_lo, k_hi - k_lo
        mask = _local_mask(
            length_q, length_k, int(window_left), right, q.device
        )
        scores = scores.masked_fill(mask[None, :, :], -torch.inf)
        q_pos = torch.arange(length_q, device=q.device)[:, None]
        k_pos = torch.arange(length_k, device=q.device)[None, :]
        distance = (q_pos + length_k - length_q - k_pos).abs().float()
        slopes = alibi_slopes if alibi_slopes.dim() == 1 else alibi_slopes[index]
        scores = scores - slopes[:, None, None] * distance

        all_masked = mask.all(dim=-1)
        safe_scores = scores.masked_fill(all_masked[None, :, None], 0.0)
        probs = torch.softmax(safe_scores, dim=-1).to(v.dtype)
        probs = probs.masked_fill(all_masked[None, :, None], 0.0)
        outputs.append(torch.matmul(probs, vi).transpose(0, 1))
    out = torch.cat(outputs, dim=0)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    return out, dq, dk, dv


def run_reference(args, case):
    return _reference(
        *args,
        bool(case["causal"]),
        int(case["window_left"]),
        int(case["window_right"]),
        float(case["softcap"]),
        float(case["scale_multiplier"]),
    )


def run_candidate(fn, args, case):
    return tuple(
        fn(
            *args,
            bool(case["causal"]),
            int(case["window_left"]),
            int(case["window_right"]),
            float(case["softcap"]),
            float(case["scale_multiplier"]),
        )
    )


def snapshot_inputs(args):
    return tuple(value.detach().clone() for value in args)


def immutability(args, snapshots):
    changed = [
        index
        for index, (actual, saved) in enumerate(zip(args, snapshots))
        if not torch.equal(actual.detach(), saved)
    ]
    return {"passed": not changed, "changed_indices": changed}


def structure(values, args, case):
    q, k, v, _, _, _, _ = args
    shapes = [q.shape, q.shape, k.shape, v.shape]
    dtypes = [q.dtype, q.dtype, k.dtype, v.dtype]
    passed = len(values) == 4 and all(
        isinstance(value, torch.Tensor)
        and value.is_cuda
        and value.shape == shape
        and value.dtype == dtype
        and torch.isfinite(value).all().item()
        for value, shape, dtype in zip(values, shapes, dtypes)
    )
    return {"passed": bool(passed), "count": len(values)}


def compare(candidate, truth, case):
    bf16 = case["dtype"] == "bfloat16"
    wide = int(case["head_dim"]) > 128
    if bf16 and wide:
        rtol, atol, l2_limit = 0.08, 0.10, 0.020
    elif bf16:
        rtol, atol, l2_limit = 0.12, 0.14, 0.045
    elif wide:
        rtol, atol, l2_limit = 0.050, 0.040, 0.015
    else:
        rtol, atol, l2_limit = 0.065, 0.045, 0.022
    rows = []
    for index, (got, ref) in enumerate(zip(candidate, truth)):
        got64 = np.asarray(got, dtype=np.float64)
        ref64 = np.asarray(ref, dtype=np.float64)
        diff = got64 - ref64
        max_abs = float(np.max(np.abs(diff)))
        scale = float(np.max(np.abs(ref64)))
        rel_l2 = float(
            np.linalg.norm(diff.ravel())
            / max(np.linalg.norm(ref64.ravel()), 1e-8)
        )
        rows.append({
            "index": index,
            "passed": max_abs <= atol + rtol * scale and rel_l2 <= l2_limit,
            "max_abs": max_abs,
            "rel_l2": rel_l2,
        })
    if len(candidate) != len(truth):
        rows.append(
            {"index": -1, "passed": False, "reason": "wrong component count"}
        )
    return rows


def signature(values, seed):
    result = []
    for index, value in enumerate(values):
        flat = value.detach().float().reshape(-1)
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed + 997 * index)
        ids = torch.randint(
            0,
            flat.numel(),
            (min(32, flat.numel()),),
            device=value.device,
            generator=generator,
        )
        result.append({
            "sum": float(flat.sum()),
            "abs_sum": float(flat.abs().sum()),
            "sq_sum": float((flat * flat).sum()),
            "samples": [float(item) for item in flat[ids].cpu()],
        })
    return result


def signature_close(got, ref):
    if len(got) != len(ref):
        return False
    for actual, expected in zip(got, ref):
        actual_sum = float(actual["sum"])
        expected_sum = float(expected["sum"])
        actual_abs_sum = float(actual["abs_sum"])
        expected_abs_sum = float(expected["abs_sum"])
        actual_sq_sum = float(actual["sq_sum"])
        expected_sq_sum = float(expected["sq_sum"])
        scalars = (
            actual_sum,
            expected_sum,
            actual_abs_sum,
            expected_abs_sum,
            actual_sq_sum,
            expected_sq_sum,
        )
        if not all(math.isfinite(value) for value in scalars):
            return False

        # A signed sum is a cancellation-prone statistic.  Relative error
        # against a near-zero signed reference rejected numerically correct
        # FlashAttention gradients in v20 even when full-array relative L2 was
        # below 0.5%.  Normalize its error by the reference L1 norm instead.
        signed_tolerance = (
            1.0
            + SIGNATURE_SIGNED_SUM_TO_L1 * max(expected_abs_sum, 1.0)
        )
        if abs(actual_sum - expected_sum) > signed_tolerance:
            return False
        if not math.isclose(
            actual_abs_sum,
            expected_abs_sum,
            rel_tol=SIGNATURE_ABS_SUM_RTOL,
            abs_tol=1.0,
        ):
            return False
        if not math.isclose(
            actual_sq_sum,
            expected_sq_sum,
            rel_tol=SIGNATURE_SQ_SUM_RTOL,
            abs_tol=1.0,
        ):
            return False
        if len(actual["samples"]) != len(expected["samples"]):
            return False
        for actual_value, expected_value in zip(
            actual["samples"], expected["samples"]
        ):
            if not (
                math.isfinite(float(actual_value))
                and math.isfinite(float(expected_value))
            ):
                return False
            if not math.isclose(
                float(actual_value),
                float(expected_value),
                rel_tol=SIGNATURE_SAMPLE_RTOL,
                abs_tol=SIGNATURE_SAMPLE_ATOL,
            ):
                return False
    return True
