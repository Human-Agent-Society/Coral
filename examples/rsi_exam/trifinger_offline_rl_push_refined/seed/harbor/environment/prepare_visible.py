"""Download the visible offline-RL training dataset into /app/data/trifinger_dataset at image-build
time. The verifier never downloads anything — sim assets ship inside the trifinger_simulation /
robot_properties_fingers wheels, so a sealed rollout needs no dataset and no network.

2026-07-19: the upstream dataset host (robots.real-robot-challenge.com) is serving an EXPIRED TLS
certificate, which fails the default verified download. Fallback: retry with an unverified TLS
context, then verify the downloaded LMDB archive against a sha256 PINNED from a certificate-valid
copy (2026-07-04) — integrity comes from the pin, not from TLS. A fresh unverified download was
byte-identical to the pinned copy when this fallback was added.
"""
from __future__ import annotations

import hashlib
import ssl
import sys
from pathlib import Path

import gymnasium as gym
import trifinger_rl_datasets  # noqa: F401

DATA_DIR = Path("/app/data/trifinger_dataset")
# sha256 of trifinger-cube-push-sim-mixed-v0.zarr/data.mdb (850000000 bytes), recorded from a
# certificate-valid download (2026-07-04).
DATA_MDB_SHA256 = "36645a7dfa1d4fcbe5a296e7f43ee65d9d4e5f88b55ccd4e3e4851182dc2e260"


def _download() -> None:
    env = gym.make("trifinger-cube-push-sim-mixed-v0", disable_env_checker=True, data_dir=str(DATA_DIR))
    stats = env.unwrapped.get_dataset_stats()
    print(f"downloaded trifinger-cube-push-sim-mixed-v0 -> {DATA_DIR}: {stats}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _download()
    except Exception as exc:  # noqa: BLE001 — expired-cert path, see module docstring
        print(f"[prepare_visible] verified-TLS download failed ({type(exc).__name__}: {exc}); "
              "retrying with unverified TLS + pinned sha256", flush=True)
        import urllib.request
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request._opener = None  # drop the cached opener built with the verified context
        _download()

    mdb = DATA_DIR / "trifinger-cube-push-sim-mixed-v0.zarr" / "data.mdb"
    h = hashlib.sha256()
    with open(mdb, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != DATA_MDB_SHA256:
        sys.exit(f"dataset integrity check FAILED: sha256 {h.hexdigest()} != pinned {DATA_MDB_SHA256}")
    print(f"[prepare_visible] dataset integrity verified (sha256 {DATA_MDB_SHA256[:16]}...)")


if __name__ == "__main__":
    main()
