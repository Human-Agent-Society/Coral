# Make masked grouped-query attention fast on a TPU v6e chip

## Background

Attention is the dominant cost in serving and training large language models. Production
configurations rarely run the textbook dense form: queries share key/value heads in groups
(GQA/MQA), long contexts use sliding windows, some models cap attention logits with a tanh,
and prefill mixes long KV caches with short query blocks. Hardware vendors ship hand-tuned
kernels for exactly these shapes because the naive formulation materializes a quadratic
score matrix in memory.

On TPU the gap between a straightforward XLA implementation and a well-designed kernel is
large but shape-dependent: the win comes from never materializing the masked-out region,
streaming KV through fast memory, and keeping the matrix units busy across a grid of
blocks. Doing this across many different mask/grouping/shape combinations with one
implementation is the hard part.

## Instruction

You are given a working attention implementation in `/app/methods/main/attention.py` and a
set of visible benchmark cases in `/app/data/visible_strata.json`. Make `make_attention`
return the fastest callable you can for each configuration, on the single TPU v6e chip
visible inside this container. The shipped implementation is the reward zero point: matching
it scores 0. You may delete and rewrite everything under `/app/methods/main/`; only the
entry signature and output contract are fixed. Grading runs your code on held-out
configurations drawn from the same distribution as the visible ones (different shapes and
seeds), each case gated on numerical correctness, and rewards the per-case speedup over the
same baseline implementation measured on the same chip.

### Hard Constraints

- Submit an algorithm, not answers: no branching on case `name` or `seed`; your code must
  handle any configuration from the documented family.
- Entry signature, verbatim: `make_attention(cfg)` in `/app/methods/main/attention.py`,
  returning a callable `f(q, k, v)`.
- Output contract: bfloat16, same shape as `q`; a case counts only if it matches the fp32
  reference within `rtol=2e-2, atol=2e-2`.
- Deterministic: same inputs, same output.
- JAX only (any of jax/jnp/pallas as installed in the image); no additional packages.

### What You Have

- `/app/methods/main/attention.py` — the baseline (plain XLA, fp32 softmax). This is what
  your speedup is measured against.
- `/app/data/visible_strata.json` — visible cases: decode/prefill/encoder shapes, GQA
  ratios 1:1 to 16:1, optional causal mask, optional sliding window, optional logit
  softcap, head_dim 128, bf16 inputs.
- `/app/eval/attn_eval.py` — the exact correctness check and timing harness grading uses.
- `python /app/selfcheck.py` — free, unlimited: correctness + speedup on every visible
  case against a frozen copy of the baseline.
- One TPU v6e chip. `jax.devices()` shows exactly it.

### What You Submit

```python
# /app/methods/main/attention.py
def make_attention(cfg: dict):
    """cfg: batch, q_len, kv_len, num_q_heads, num_kv_heads, head_dim,
    causal, window, softcap, seed, name."""
    def f(q, k, v):  # bf16 in, bf16 out, q.shape
        ...
    return f
```

```text
output: bfloat16 array of shape (batch, num_q_heads, q_len, head_dim)
```

### How It Is Judged

Per held-out case: run `make_attention(cfg)`, check correctness against the fp32 reference
(fail -> that case scores 0), then time it (median over repeated rounds after warmup) and
compute `speedup = baseline_time / your_time` where `baseline_time` is the frozen
measurement of the shipped implementation on the same chip. Each case's speedup maps onto
[0, 1] through the task's anchor band (log-linear within segments); the final reward is the
arithmetic mean over all held-out cases. Compilation happens once per case before timing
and is not counted.
