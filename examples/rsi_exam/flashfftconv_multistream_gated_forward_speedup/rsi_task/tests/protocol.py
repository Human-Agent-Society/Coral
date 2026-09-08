from __future__ import annotations

import hashlib
import math

import numpy as np
import torch


ENTRYPOINT = "flashfftconv_multistream_gated_forward"
WARMUPS = 4
REPEATS = 7
MAX_DISPERSION = 2.25


def phase_seed(case, phase, index):
    payload = f"{case['id']}|{case['seed']}|{phase}|{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF


def make_inputs(case, seed):
    torch.manual_seed(seed)
    dtype = getattr(torch, case["dtype"])
    shape = (
        case["batch"], case["streams"], case["heads"], case["length"]
    )
    x = torch.randn(shape, device="cuda", dtype=dtype)
    in_gate = torch.sigmoid(
        torch.randn(shape, device="cuda", dtype=torch.float32)
        + float(case["gate_bias"])
    ).to(dtype)
    out_gate = torch.sigmoid(
        torch.randn(shape, device="cuda", dtype=torch.float32)
        - 0.5 * float(case["gate_bias"])
    ).to(dtype)
    k = (
        torch.randn(
            case["streams"], case["heads"], case["length"],
            device="cuda", dtype=torch.float32,
        )
        * (float(case["kernel_scale"]) / math.sqrt(case["length"]))
    )
    return x.contiguous(), k.contiguous(), in_gate.contiguous(), out_gate.contiguous()


def _reference(x, k, in_gate, out_gate, fft_size):
    batch, streams, heads, length = x.shape
    flat_x = x.reshape(batch, streams * heads, length)
    flat_in = in_gate.reshape_as(flat_x)
    flat_out = out_gate.reshape_as(flat_x)
    flat_k = k.reshape(streams * heads, length)
    x_f = torch.fft.rfft(
        flat_x.float() * flat_in.float(), n=fft_size, dim=-1
    )
    k_f = torch.fft.rfft(flat_k, n=fft_size, dim=-1)
    conv = torch.fft.irfft(
        x_f * k_f.unsqueeze(0), n=fft_size, dim=-1
    )[..., :length]
    return (
        (conv * flat_out.float())
        .to(x.dtype)
        .reshape(batch, streams, heads, length)
        .contiguous()
    )


def run_reference(args, case):
    return (_reference(*args, int(case["fft_size"])),)


def run_candidate(fn, args, case):
    return (fn(*args, int(case["fft_size"])),)


def snapshot_inputs(args):
    return tuple(value.detach().clone() for value in args)


def immutability(args, snapshots):
    changed = [
        index
        for index, (actual, saved) in enumerate(zip(args, snapshots))
        if not torch.equal(actual, saved)
    ]
    return {"passed": not changed, "changed_indices": changed}


def structure(values, args, case):
    x = args[0]
    passed = (
        len(values) == 1
        and isinstance(values[0], torch.Tensor)
        and values[0].is_cuda
        and values[0].shape == x.shape
        and values[0].dtype == x.dtype
        and values[0].is_contiguous()
        and torch.isfinite(values[0]).all().item()
    )
    return {"passed": bool(passed), "count": len(values)}


def compare(candidate, truth, case):
    if case["dtype"] == "bfloat16":
        rtol, atol, l2_limit = 0.13, 0.18, 0.035
    else:
        rtol, atol, l2_limit = 0.065, 0.060, 0.015
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
        rows.append({"index": -1, "passed": False})
    return rows


def signature(values, seed):
    rows = []
    for index, value in enumerate(values):
        flat = value.detach().float().reshape(-1)
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed + 997 * index)
        ids = torch.randint(
            0, flat.numel(), (32,), device=value.device, generator=generator
        )
        rows.append({
            "sum": float(flat.sum()),
            "abs_sum": float(flat.abs().sum()),
            "sq_sum": float((flat * flat).sum()),
            "samples": [float(item) for item in flat[ids].cpu()],
        })
    return rows


def signature_close(got, ref):
    if len(got) != len(ref):
        return False
    for actual, expected in zip(got, ref):
        for key in ("sum", "abs_sum", "sq_sum"):
            if not math.isclose(
                float(actual[key]), float(expected[key]),
                rel_tol=0.14, abs_tol=2.0,
            ):
                return False
        for actual_value, expected_value in zip(
            actual["samples"], expected["samples"]
        ):
            if not math.isclose(
                float(actual_value), float(expected_value),
                rel_tol=0.16, abs_tol=0.20,
            ):
                return False
    return True
