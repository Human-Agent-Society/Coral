"""Shared eval core: configs, fp32 reference, correctness gate, timing."""
import json
import time
import functools

import jax
import jax.numpy as jnp

RTOL = 2e-2
ATOL = 2e-2
# finite fill: -inf + tanh softcap NaNs under XLA fusion on TPU
MASK_VALUE = -0.7 * 3.4e38
WARMUP = 3
ROUNDS = 5
ITERS = 10


def load_strata(path):
    with open(path) as f:
        return json.load(f)["cases"]


def make_inputs(cfg):
    key = jax.random.PRNGKey(cfg["seed"])
    kq, kk, kv = jax.random.split(key, 3)
    b, ql, kl = cfg["batch"], cfg["q_len"], cfg["kv_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    q = jax.random.normal(kq, (b, hq, ql, d), jnp.bfloat16)
    k = jax.random.normal(kk, (b, hkv, kl, d), jnp.bfloat16)
    v = jax.random.normal(kv, (b, hkv, kl, d), jnp.bfloat16)
    return q, k, v


def _mask(cfg, ql, kl):
    # position i attends j iff j <= i+offset (causal) and i+offset-j < window
    offset = kl - ql
    qi = jnp.arange(ql)[:, None]
    kj = jnp.arange(kl)[None, :]
    m = jnp.ones((ql, kl), bool)
    if cfg["causal"]:
        m &= kj <= qi + offset
    if cfg.get("window"):
        m &= (qi + offset - kj) < cfg["window"]
    return m


def reference_attention(q, k, v, cfg):
    """fp32 ground truth; also defines the exact semantics submissions must match."""
    qf, kf, vf = (x.astype(jnp.float32) for x in (q, k, v))
    g = cfg["num_q_heads"] // cfg["num_kv_heads"]
    kf = jnp.repeat(kf, g, axis=1)
    vf = jnp.repeat(vf, g, axis=1)
    scale = cfg["head_dim"] ** -0.5
    s = jnp.einsum("bhqd,bhkd->bhqk", qf, kf) * scale
    if cfg.get("softcap"):
        c = cfg["softcap"]
        s = c * jnp.tanh(s / c)
    m = _mask(cfg, cfg["q_len"], cfg["kv_len"])
    s = jnp.where(m[None, None], s, MASK_VALUE)
    p = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", p, vf)


def check_correct(out, q, k, v, cfg):
    if out.shape != q.shape:
        return False, f"shape {out.shape} != {q.shape}"
    if out.dtype != jnp.bfloat16:
        return False, f"dtype {out.dtype} != bfloat16"
    ref = reference_attention(q, k, v, cfg)
    ok = jnp.allclose(out.astype(jnp.float32), ref, rtol=RTOL, atol=ATOL)
    return bool(ok), "" if ok else "value mismatch beyond tolerance"


def time_fn(fn, *args):
    """Median-of-rounds wall time per call, seconds."""
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            r = fn(*args)
        jax.block_until_ready(r)
        ts.append((time.perf_counter() - t0) / ITERS)
    ts.sort()
    return ts[len(ts) // 2]


def evaluate_case(make_attention, cfg, baseline_t=None):
    """Returns dict with ok/err/t/speedup for one case."""
    q, k, v = make_inputs(cfg)
    try:
        fn = make_attention(dict(cfg))
        out = jax.block_until_ready(fn(q, k, v))
    except Exception as e:  # noqa: BLE001 - submission failures must not kill the run
        return {"name": cfg["name"], "ok": False, "err": f"{type(e).__name__}: {e}"}
    ok, why = check_correct(out, q, k, v, cfg)
    if not ok:
        return {"name": cfg["name"], "ok": False, "err": why}
    t = time_fn(fn, q, k, v)
    r = {"name": cfg["name"], "ok": True, "t": t}
    if baseline_t:
        r["speedup"] = baseline_t / t
    return r
