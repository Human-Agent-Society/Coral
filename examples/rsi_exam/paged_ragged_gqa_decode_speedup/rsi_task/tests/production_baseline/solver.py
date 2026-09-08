from __future__ import annotations

import torch


_PLAN = None


def _page_plan(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_indptr: torch.Tensor,
    page_indices: torch.Tensor,
    last_page_len: torch.Tensor,
    window_left: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one padded token-gather plan for the stable paged metadata."""
    global _PLAN
    key = (
        q.device,
        tuple(q.shape),
        tuple(k_cache.shape),
        page_indptr.data_ptr(),
        page_indices.data_ptr(),
        last_page_len.data_ptr(),
        int(window_left),
    )
    if _PLAN is not None and _PLAN[0] == key:
        return _PLAN[4], _PLAN[5], _PLAN[6], _PLAN[7]

    indptr = [int(value) for value in page_indptr.detach().cpu().tolist()]
    page_ids = [int(value) for value in page_indices.detach().cpu().tolist()]
    tails = [int(value) for value in last_page_len.detach().cpu().tolist()]
    page_size = int(k_cache.shape[1])
    token_rows: list[list[int]] = []
    position_rows: list[list[int]] = []
    query_positions: list[int] = []
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
    gather_rows = [row + [0] * (width - len(row)) for row in token_rows]
    position_rows = [
        row + [0] * (width - len(row)) for row in position_rows
    ]
    valid_rows = [
        [True] * len(row) + [False] * (width - len(row))
        for row in token_rows
    ]
    gather = torch.tensor(gather_rows, dtype=torch.int64, device=q.device)
    valid = torch.tensor(valid_rows, dtype=torch.bool, device=q.device)
    positions = torch.tensor(position_rows, dtype=torch.float32, device=q.device)
    query_position = torch.tensor(
        query_positions, dtype=torch.float32, device=q.device
    )
    # Keep the metadata tensors live. Besides avoiding a stale data-pointer
    # collision when moving between cases, this documents that planning is
    # valid only while the immutable page table remains the same.
    _PLAN = (
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


def _rope_half_split(
    tensor: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    half = tensor.shape[-1] // 2
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    first, second = tensor[..., :half], tensor[..., half:]
    return torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine),
        dim=-1,
    )


def paged_gqa_decode(
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
    """Vectorized PyTorch starter covering the complete paged-decode ABI."""
    batch, query_heads, head_dim = q.shape
    kv_heads = int(k_cache.shape[2])
    group_size = query_heads // kv_heads
    gather, valid, positions, query_position = _page_plan(
        q,
        k_cache,
        page_indptr,
        page_indices,
        last_page_len,
        window_left,
    )
    width = int(gather.shape[1])
    flat_k = k_cache.reshape(-1, kv_heads, head_dim)
    flat_v = v_cache.reshape(-1, kv_heads, head_dim)
    keys = flat_k.index_select(0, gather.reshape(-1)).reshape(
        batch, width, kv_heads, head_dim
    ).float()
    values = flat_v.index_select(0, gather.reshape(-1)).reshape(
        batch, width, kv_heads, head_dim
    ).float()
    query = q.reshape(batch, kv_heads, group_size, head_dim).float()

    if int(pos_encoding_mode) == 1:
        inv_freq = 1.0 / (
            float(rope_theta)
            ** (
                torch.arange(
                    0, head_dim, 2, device=q.device, dtype=torch.float32
                )
                / head_dim
            )
        )
        inv_freq = inv_freq / float(rope_scale)
        query = _rope_half_split(
            query,
            query_position[:, None, None, None] * inv_freq[None, None, None, :],
        )
        keys = _rope_half_split(
            keys,
            positions[:, :, None, None] * inv_freq[None, None, None, :],
        )

    scores = torch.einsum("bkgd,blkd->bkgl", query, keys) * float(sm_scale)
    scores = float(logits_soft_cap) * torch.tanh(
        scores / float(logits_soft_cap)
    )
    scores.masked_fill_(~valid[:, None, None, :], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("bkgl,blkd->bkgd", probabilities, values)
    return output.reshape(batch, query_heads, head_dim).to(q.dtype)
