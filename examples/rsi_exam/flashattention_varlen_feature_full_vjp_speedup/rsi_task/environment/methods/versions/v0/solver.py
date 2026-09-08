import math

import torch


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
            shifted + right, torch.full_like(shifted, length_k)
        ),
        col < shifted - left,
    )


def attention_varlen_feature_full_vjp(
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
    right = 0 if causal else window_right
    for index in range(cu_q.numel() - 1):
        q_lo, q_hi = int(cu_q[index]), int(cu_q[index + 1])
        k_lo, k_hi = int(cu_k[index]), int(cu_k[index + 1])
        qi = q[q_lo:q_hi].transpose(0, 1)
        ki = k[k_lo:k_hi].repeat_interleave(group, dim=1).transpose(0, 1)
        vi = v[k_lo:k_hi].repeat_interleave(group, dim=1).transpose(0, 1)
        scores = torch.matmul(qi.float(), ki.float().transpose(-1, -2)) * scale
        if softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)
        length_q, length_k = q_hi - q_lo, k_hi - k_lo
        mask = _local_mask(
            length_q, length_k, window_left, right, q.device
        )
        scores = scores.masked_fill(mask[None], -torch.inf)
        q_pos = torch.arange(length_q, device=q.device)[:, None]
        k_pos = torch.arange(length_k, device=q.device)[None]
        distance = (q_pos + length_k - length_q - k_pos).abs().float()
        scores = scores - alibi_slopes[index, :, None, None] * distance
        all_masked = mask.all(dim=-1)
        scores = scores.masked_fill(all_masked[None, :, None], 0.0)
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        probs = probs.masked_fill(all_masked[None, :, None], 0.0)
        outputs.append(torch.matmul(probs, vi).transpose(0, 1))
    out = torch.cat(outputs, dim=0)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    return out, dq, dk, dv
