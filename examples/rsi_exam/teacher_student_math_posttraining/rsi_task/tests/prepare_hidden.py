#!/usr/bin/env python3
"""Build sealed evaluator assets from immutable revisions; never copied into the agent image."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

SETS = (
    ("MathArena/aime_2025", "c94da77eb22bbd6439e62a323bec18493a421302"),
    ("MathArena/aime_2026", "d2de22f3c656b4f56cf8981212186377d1e23bc3"),
)
STUDENT = ("Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1")
EXPECTED_SHA256 = "d9f359e4a7568138d4bda1b7dcf7af15adfb04b523e18f7b53d63a288bb92097"


def main() -> None:
    scratch = Path(tempfile.mkdtemp(prefix="sealed_build_"))
    rows = []
    try:
        for index, (repo, revision) in enumerate(SETS):
            root = Path(snapshot_download(repo_id=repo, revision=revision, repo_type="dataset", local_dir=scratch / str(index)))
            for parquet in sorted(root.rglob("*.parquet")):
                for row in pq.read_table(parquet).to_pylist():
                    rows.append({"problem": str(row["problem"]).strip(), "answer": int(str(row["answer"]).strip())})
        if len(rows) != 60:
            raise RuntimeError("sealed source count changed")
        random.Random(251026).shuffle(rows)
        payload = [{"id": f"sealed_{i:03d}", **row} for i, row in enumerate(rows)]
        encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        digest = hashlib.sha256(encoded).hexdigest()
        if digest != EXPECTED_SHA256:
            raise RuntimeError(f"sealed dataset commitment changed: {digest}")
        heldout = Path("/tests/heldout")
        heldout.mkdir(parents=True, exist_ok=True)
        (heldout / "sealed.json").write_bytes(encoded)

        snapshot_download(
            repo_id=STUDENT[0], revision=STUDENT[1], local_dir="/tests/trusted-tokenizer",
            allow_patterns=["config.json", "generation_config.json", "tokenizer*", "vocab.json", "merges.txt", "*.jinja"],
        )
        for cache in Path("/tests/trusted-tokenizer").rglob(".cache"):
            shutil.rmtree(cache, ignore_errors=True)
        Path("/tests/ASSET_MANIFEST.json").write_text(json.dumps({
            "schema": "sealed_integer_math_v1", "cases": 60,
            "sealed_sha256": digest, "student_revision": STUDENT[1],
        }, sort_keys=True) + "\n")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
