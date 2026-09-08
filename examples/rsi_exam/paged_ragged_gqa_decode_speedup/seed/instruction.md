# Paged Ragged GQA Decode Attention

You inherit a correct batch-vectorized PyTorch implementation of grouped-query decode attention over a paged, variable-length KV cache. Make it as fast as possible while it keeps computing the same thing within the accuracy the verifier enforces.

## Hard Constraints

- Keep the callable signature in `methods/main/solver.py` unchanged. Modify implementation files only under `methods/main`.
- Inputs are CUDA tensors. `q` has shape `(batch, query_heads, head_dim)`. `k_cache` and `v_cache` have NHD layout `(physical_pages, page_size, kv_heads, head_dim)`.
- `page_indptr`, `page_indices`, and `last_page_len` are CUDA `int32` tensors. They encode a CSR page table with at least one page per request; the final valid page length is in `[1, page_size]`.
- Query and cache tensors share dtype, either `float16` or `bfloat16`. `query_heads` is divisible by `kv_heads`; each contiguous query-head group maps to one KV head.
- Return one fresh CUDA tensor with shape `(batch, query_heads, head_dim)` and the input floating dtype. Do not mutate or alias any input.
- Implement the documented attention for every request: gather only pages named by its page-table row, trim the final page, optionally apply the specified Llama half-split RoPE to query position `kv_len - 1` and key positions `0..kv_len-1`, retain only the last `window_left + 1` tokens when `window_left >= 0`, form `scores = q @ k.T * sm_scale`, apply `scores = logits_soft_cap * tanh(scores / logits_soft_cap)`, apply softmax in each GQA head, and multiply by values. RoPE frequencies are `theta^(-arange(0, head_dim, 2)/head_dim) / rope_scale`; mode `0` means no positional encoding and mode `1` means `ROPE_LLAMA`.
- The sealed panel uses page tables, request lengths, shapes, dtypes and tensor values you have not seen. Your implementation must stay correct and fast on them. Do not enumerate visible cases, fixed seeds, or any other property you can only observe in the public panel.
- Caching an output across calls, mutating metadata, or replacing the timing primitives is detected and scored as incorrect.
- Submission code may import `torch`, `triton`, standard typing/math helpers, and sibling modules. It may not access files, processes, clocks, environment secrets, network services, verifier paths, or runtime introspection.

## What You Have

- `methods/main/solver.py`: the correct batch-vectorized PyTorch starting implementation covering the full callable ABI.
- `problems/visible_spec.json`: a public development panel with real shuffled physical page tables and ragged request lengths.
- `protocol.py`: the public data contract, the reference semantics, and the exact measurement protocol the verifier uses.
- `selfcheck.py`: an isolated visible evaluator. It generates fresh values, checks every output against an independent float32 formulation, and reports `visible_geomean_speedup` against the inherited starter. Higher is better.
- `last_page_len` gives each request's valid tail length; slots past it are unwritten cache memory and must not be attended to.
- `/app/methods/experiment_log.md`: the append-only experiment ledger. Record each attempt's visible latency, its `visible_geomean_speedup`, and the keep/revert decision, and save the matching source under `/app/methods/versions/vN/`.

## What You Submit

The contents of `/app/methods`, with your final implementation at `/app/methods/main/solver.py`. The evaluator imports:

```python
from solver import paged_gqa_decode
```

Module-level compilation caches and reusable workspaces are allowed. They must be keyed by general runtime properties and stay correct when a fresh process presents unseen cases. Planning and compilation happen during untimed warmup; timed calls still receive fresh `q`, `k_cache`, and `v_cache` tensors.

## How It Is Judged

Each sealed case runs the verifier's own hash-bound copy of the inherited starter immediately before your candidate and immediately after it. All three runs use fresh isolated processes, identical inputs and seeds, the same correctness checks, and the same timer. The case score is the bracketed baseline latency divided by your candidate latency, and the cases are combined by geometric mean. The inherited starter therefore scores 1.0x.

Before a case scores at all it must pass output structure, finite values, numerical accuracy, input immutability, no-alias, fresh-call, timing-integrity, dispersion, and baseline-drift checks. **Any failed case zeroes the entire score.** Timing measurement is defined in `protocol.py`; input generation, validation and output transfer are all outside the timed window.

Run the visible loop with:

```bash
python /app/selfcheck.py
```
