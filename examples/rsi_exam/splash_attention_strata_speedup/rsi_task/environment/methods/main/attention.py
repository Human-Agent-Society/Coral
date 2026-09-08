"""Starting point: plain XLA attention. Replace freely; keep the entry signature."""
import functools

import jax
import jax.numpy as jnp


def make_attention(cfg):
    """Return a callable f(q, k, v) -> bf16 output of q's shape.

    cfg keys: batch, q_len, kv_len, num_q_heads, num_kv_heads, head_dim,
    causal (bool), window (int or None), softcap (float or None), seed, name.
    """
    ql, kl = cfg["q_len"], cfg["kv_len"]
    offset = kl - ql
    qi = jnp.arange(ql)[:, None]
    kj = jnp.arange(kl)[None, :]
    mask = jnp.ones((ql, kl), bool)
    if cfg["causal"]:
        mask &= kj <= qi + offset
    if cfg.get("window"):
        mask &= (qi + offset - kj) < cfg["window"]
    g = cfg["num_q_heads"] // cfg["num_kv_heads"]
    scale = cfg["head_dim"] ** -0.5
    softcap = cfg.get("softcap")
    mask_value = -0.7 * 3.4e38  # finite: -inf NaNs with softcap under XLA fusion

    @jax.jit
    def f(q, k, v):
        k = jnp.repeat(k, g, axis=1)
        v = jnp.repeat(v, g, axis=1)
        s = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32),
                       k.astype(jnp.float32)) * scale
        if softcap:
            s = softcap * jnp.tanh(s / softcap)
        s = jnp.where(mask[None, None], s, mask_value)
        p = jax.nn.softmax(s, axis=-1)
        o = jnp.einsum("bhqk,bhkd->bhqd", p, v.astype(jnp.float32))
        return o.astype(jnp.bfloat16)

    return f
