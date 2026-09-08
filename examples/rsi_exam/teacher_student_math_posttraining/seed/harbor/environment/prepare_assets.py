#!/usr/bin/env python3
"""Download immutable assets at image-build time, then emit only sanitized runtime files."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

STUDENT = ("Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1")
TEACHER = ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")
TRAIN = ("agentica-org/DeepScaleR-Preview-Dataset", "b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29")
DEV = (
    ("MathArena/aime_2024_I", "ea5b061c3e8039dc9858defaafc407d04b995e9f"),
    ("MathArena/aime_2024_II", "29d5d31e9b46e215fc24d9b2a3047506823dd101"),
)
TRAIN_ROWS = 40315
TRAIN_JSONL_SHA256 = "5997e7c324093988a3fc96ade61426ff932d4e8707dba7d8be6e3e8af956eaf8"


def download(repo: str, revision: str, destination: Path, *, repo_type: str | None = None) -> Path:
    return Path(snapshot_download(repo_id=repo, revision=revision, repo_type=repo_type, local_dir=destination))


def parquet_rows(snapshot: Path) -> list[dict[str, object]]:
    files = sorted(snapshot.rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"no parquet files in immutable snapshot {snapshot}")
    rows: list[dict[str, object]] = []
    for path in files:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            handle.write(payload)
            digest.update(payload)
    return digest.hexdigest()


def sanitized_dev(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, (repo, revision) in enumerate(DEV):
        snap = download(repo, revision, root / f"dev_{index}", repo_type="dataset")
        for row in parquet_rows(snap):
            rows.append({"problem": str(row["problem"]).strip(), "answer": str(row["answer"]).strip()})
    if len(rows) != 30 or any(not row["answer"].lstrip("-").isdigit() for row in rows):
        raise RuntimeError("development snapshot schema/count changed")
    return [{"id": f"dev_{i:03d}", **row} for i, row in enumerate(rows)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("environment", "visible"), required=True)
    args = parser.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix="fixed_assets_"))
    try:
        if args.role == "environment":
            models = Path("/app/models")
            models.mkdir(parents=True, exist_ok=True)
            download(*STUDENT, models / "student")
            download(*TEACHER, models / "teacher")
            for cache in models.rglob(".cache"):
                shutil.rmtree(cache, ignore_errors=True)

            snap = download(*TRAIN, scratch / "train", repo_type="dataset")
            source_files = sorted(path for path in snap.rglob("*.json") if ".cache" not in path.parts)
            if len(source_files) != 1:
                raise RuntimeError("training snapshot payload changed")
            source_rows = json.loads(source_files[0].read_text())
            clean = [{"problem": str(row["problem"]), "answer": str(row["answer"]).strip()} for row in source_rows]
            if len(clean) != TRAIN_ROWS or any(set(row) != {"problem", "answer"} for row in clean):
                raise RuntimeError("training snapshot schema/count changed")
            data = Path("/app/data")
            data.mkdir(parents=True, exist_ok=True)
            digest = write_jsonl(data / "train.jsonl", clean)
            if digest != TRAIN_JSONL_SHA256:
                raise RuntimeError(f"sanitized training commitment changed: {digest}")
            dev = sanitized_dev(scratch / "dev")
            write_jsonl(data / "dev.jsonl", [{"id": row["id"], "problem": row["problem"]} for row in dev])
            (data / "MANIFEST.json").write_text(json.dumps({
                "schema": "anonymous_math_problem_answer_v1",
                "train_rows": TRAIN_ROWS,
                "train_jsonl_sha256": digest,
                "dev_rows": 30,
            }, sort_keys=True) + "\n")
        else:
            rows = sanitized_dev(scratch / "dev")
            output = Path("/opt/visible")
            output.mkdir(parents=True, exist_ok=True)
            (output / "answers.json").write_text(json.dumps(
                {row["id"]: int(row["answer"]) for row in rows}, sort_keys=True
            ) + "\n")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
