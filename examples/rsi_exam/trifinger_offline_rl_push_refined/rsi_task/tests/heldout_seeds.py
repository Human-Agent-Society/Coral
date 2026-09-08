"""Sealed held-out episode seeds. NEVER copied into the agent image — lives only under tests/.

Each seed drives `trifinger_score.seed_episode(seed_k)` before an independent `env.reset()`, so every
held-out episode is a fully independent, deterministically-reproducible job start (no soft-reset
chaining between episodes, unlike the vendor `Evaluation.evaluate()` loop: it
sidesteps having to also prove the physical cube-reset trajectory replay is bit-identical, and reuses
only the primitive that was actually verified: seed_episode(n) + env.reset() -> bit-identical rollout).

the sealed batch used to be `list(range(70001, 70033))` — 32 CONSECUTIVE integers.
That is enumerable from a single guess: an agent that writes `range(70001, 70033)` can evaluate and
overfit directly on the graded episodes, and no network policy stops it (the simulator ships inside
the agent image, so the attack needs nothing from outside). The sealed batch is now drawn from the
SAME generator family as the visible pool but in a DISJOINT domain — a different salt prefix, so the
two sides provably cannot collide:

    visible (environment/selfcheck.py): k -> sha256("trifinger-visible-v2-{k}") mod (2**31-1), k=0..99
    sealed  (this file)               : k -> sha256("trifinger-sealed-v3-{k}")  mod (2**31-1), k=0..31

Verified at authoring time: 32 distinct values, zero intersection with the 100 visible seeds.
This CHANGES THE SEALED SPLIT — anchor values measured before 2026-07-27 are NOT comparable; see
tests/anchors.json.
"""
import hashlib

_SEALED_SALT = "trifinger-sealed-v3-"   # deliberately != the visible salt "trifinger-visible-v2-"
_N_SEALED = 32


def _derive(prefix: str, n: int) -> list:
    return [int(hashlib.sha256(f"{prefix}{k}".encode()).hexdigest(), 16) % (2 ** 31 - 1)
            for k in range(n)]


HELDOUT_SEEDS = _derive(_SEALED_SALT, _N_SEALED)  # 32 episodes
