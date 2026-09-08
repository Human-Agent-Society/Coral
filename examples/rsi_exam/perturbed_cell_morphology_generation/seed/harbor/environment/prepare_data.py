#!/usr/bin/env python3
"""Download and prepare the official 100-condition RxRx1 evaluation subset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_URL = "https://zenodo.org/api/records/8307629/files/IMPA_reproducibility.zip/content"
SOURCE_MEMBER = "IMPA_reproducibility/datasets/rxrx1.tar"
SOURCE_MEMBER_SHA256 = "e037f7356cc58614e8d775053eb1847d9e67119a0e8e2252a38b23f785e0eeb1"
INDEX_URL = "https://huggingface.co/suyc21/CellFlux/resolve/main/data/rxrx1/rxrx1_df_subset.csv"
INDEX_SHA256 = "aed8785f197431b2e78f00fecf3d3dea1916e5893edf56a2584d5a8cc06d0ead"
EMBEDDING_URL = "https://raw.githubusercontent.com/theislab/IMPA/13eb82f582268b43ffe5173e75c3ba8b3f42c8d8/embeddings/csv/rxrx1_gene2vec_embeddings.csv"
EMBEDDING_SHA256 = "ea80e88865d448063079a863043d36b73de06eede7b7b8941971833224132b53"
INCEPTION_URL = "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth"
INCEPTION_SHA256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
EXPECTED_SEALED_MANIFEST_SHA256 = "a1b5d5197333f18fbd38fca9b6870a3963379400e5da8bf0181f9a98447e4b9b"
VISIBLE_PER_CONDITION = 10


class RangeReader(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            self.length = int(response.headers["Content-Length"])
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self.position + offset
        elif whence == os.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def readinto(self, buffer: bytearray) -> int:
        if self.position >= self.length:
            return 0
        count = min(len(buffer), self.length - self.position)
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self.position}-{self.position + count - 1}"},
        )
        # A multi-GB range read takes thousands of requests; one transient 5xx
        # must not lose the whole download.
        for attempt in range(8):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read(count)
                break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(60, 2 ** attempt))
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination) == expected_sha256:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    actual = sha256_file(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {url}: {actual}")
    temporary.replace(destination)


def download_zip_member(destination: Path) -> None:
    if destination.exists() and sha256_file(destination) == SOURCE_MEMBER_SHA256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    reader = io.BufferedReader(RangeReader(SOURCE_URL), buffer_size=32 * 1024 * 1024)
    with zipfile.ZipFile(reader) as outer:
        with outer.open(SOURCE_MEMBER) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    actual = sha256_file(temporary)
    if actual != SOURCE_MEMBER_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"dataset member checksum mismatch: {actual}")
    temporary.replace(destination)


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe tar member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"links are not accepted in dataset tar: {member.name}")
        archive.extractall(destination)


def locate_raw_root(extraction_root: Path) -> Path:
    matches = list(extraction_root.rglob("metadata/rxrx1_df.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one metadata file, found {len(matches)}")
    return matches[0].parent.parent


def seed_for(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def image_path(root: Path, sample_key: str) -> Path:
    pieces = sample_key.split("-")
    if len(pieces) < 2:
        raise ValueError(f"invalid sample key: {sample_key}")
    fields = pieces[1].split("_")
    return root / "_".join(fields[:2]) / fields[2] / ("_".join(fields[3:]) + ".npy")


def load_images(root: Path, keys: list[str]) -> np.ndarray:
    result = np.stack([np.load(image_path(root, key), allow_pickle=False) for key in keys])
    if result.dtype != np.uint8 or result.shape[1:] != (96, 96, 6):
        raise ValueError(f"unexpected image tensor: {result.shape} {result.dtype}")
    return result


def deterministic_rows(group: pd.DataFrame, salt: str) -> pd.DataFrame:
    name = str(group["CPD_NAME"].iloc[0])
    order = np.random.default_rng(seed_for(salt, name)).permutation(len(group))
    return group.iloc[order].reset_index(drop=True)


def pair_controls(controls: pd.DataFrame, treated: pd.DataFrame, salt: str) -> pd.DataFrame:
    pools = {str(batch): rows.reset_index(drop=True) for batch, rows in controls.groupby("BATCH")}
    result = treated.copy()
    result["control_key"] = [
        str(pools[str(row.BATCH)].iloc[seed_for(salt, row.SAMPLE_KEY) % len(pools[str(row.BATCH)])]["SAMPLE_KEY"])
        for row in result.itertuples()
    ]
    return result


def hash_rows(rows: pd.DataFrame) -> str:
    payload = rows[["SAMPLE_KEY", "control_key", "CPD_NAME", "BATCH"]].to_csv(
        index=False, lineterminator="\n"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def evaluation_arrays(
    root: Path,
    rows: pd.DataFrame,
    embeddings: pd.DataFrame,
    condition_to_id: dict[str, int],
    batch_to_id: dict[str, int],
) -> dict[str, np.ndarray]:
    names = rows["CPD_NAME"].astype(str).tolist()
    return {
        "control": load_images(root, rows["control_key"].astype(str).tolist()),
        "target": load_images(root, rows["SAMPLE_KEY"].astype(str).tolist()),
        "embedding": embeddings.loc[names].to_numpy(np.float32),
        "condition": np.asarray([condition_to_id[name] for name in names], dtype=np.int64),
        "batch": np.asarray([batch_to_id[str(name)] for name in rows["BATCH"]], dtype=np.int64),
        "sample_id": np.arange(len(rows), dtype=np.int64),
    }


def save_inputs(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        path,
        control=arrays["control"],
        embedding=arrays["embedding"],
        condition=arrays["condition"],
        batch=arrays["batch"],
        sample_id=arrays["sample_id"],
    )


def prepare(
    raw_root: Path,
    index_path: Path,
    embedding_path: Path,
    output_root: Path,
    role: str,
) -> dict:
    index = pd.read_csv(index_path)
    required = {"SAMPLE_KEY", "BATCH", "CPD_NAME", "ANNOT", "SPLIT"}
    if not required.issubset(index.columns):
        raise ValueError(f"index missing columns: {sorted(required - set(index.columns))}")
    missing = [key for key in index["SAMPLE_KEY"].astype(str) if not image_path(raw_root, key).is_file()]
    if missing:
        raise FileNotFoundError(f"official subset has {len(missing)} missing images; first: {missing[0]}")
    embeddings = pd.read_csv(embedding_path, index_col=0)
    conditions = sorted(index.loc[index["ANNOT"] == "treated", "CPD_NAME"].astype(str).unique())
    if len(conditions) != 100:
        raise ValueError(f"expected 100 treated conditions, found {len(conditions)}")
    absent_embeddings = sorted(set(conditions) - set(embeddings.index.astype(str)))
    if absent_embeddings:
        raise ValueError(f"missing embeddings: {absent_embeddings[:3]}")
    condition_to_id = {name: position for position, name in enumerate(conditions)}
    batches = sorted(index["BATCH"].astype(str).unique())
    batch_to_id = {name: position for position, name in enumerate(batches)}
    output_root.mkdir(parents=True, exist_ok=True)

    if role == "agent":
        source_train = index.loc[index["SPLIT"] == "train"].copy()
        train_treated_all = source_train.loc[source_train["ANNOT"] == "treated"].copy()
        ordered = pd.concat(
            [deterministic_rows(group, "rxrx1-official100-visible-v1") for _, group in train_treated_all.groupby("CPD_NAME")],
            ignore_index=True,
        )
        group_sizes = ordered.groupby("CPD_NAME").size()
        if int(group_sizes.min()) < VISIBLE_PER_CONDITION:
            raise ValueError(
                f"every condition needs at least {VISIBLE_PER_CONDITION} train-treated images; "
                f"smallest group has {int(group_sizes.min())}"
            )
        visible = (
            ordered.groupby("CPD_NAME", sort=True, group_keys=False)
            .head(VISIBLE_PER_CONDITION)
            .reset_index(drop=True)
        )
        train_treated = train_treated_all.loc[
            ~train_treated_all["SAMPLE_KEY"].isin(set(visible["SAMPLE_KEY"]))
        ].reset_index(drop=True)
        train_controls = source_train.loc[source_train["ANNOT"] == "negative_control"].reset_index(drop=True)
        visible = pair_controls(train_controls, visible, "rxrx1-official100-visible-control-v1")
        arrays = evaluation_arrays(raw_root, visible, embeddings, condition_to_id, batch_to_id)
        np.savez_compressed(
            output_root / "train.npz",
            treated=load_images(raw_root, train_treated["SAMPLE_KEY"].astype(str).tolist()),
            treated_condition=np.asarray([condition_to_id[str(x)] for x in train_treated["CPD_NAME"]], dtype=np.int64),
            treated_batch=np.asarray([batch_to_id[str(x)] for x in train_treated["BATCH"]], dtype=np.int64),
            treated_sample_id=np.arange(len(train_treated), dtype=np.int64),
            control_bank=load_images(raw_root, train_controls["SAMPLE_KEY"].astype(str).tolist()),
            control_batch=np.asarray([batch_to_id[str(x)] for x in train_controls["BATCH"]], dtype=np.int64),
            condition_embedding=embeddings.loc[conditions].to_numpy(np.float32),
        )
        save_inputs(output_root / "validation_inputs.npz", arrays)
        np.savez_compressed(output_root / "validation_targets.npz", target=arrays["target"])
        manifest = {
            "format_version": 1,
            "source": "official CellFlux HF 100-condition subset",
            "channels": 6,
            "image_size": 96,
            "embedding_dimension": 200,
            "conditions": len(conditions),
            "train_treated_samples": int(len(train_treated)),
            "train_control_samples": int(len(train_controls)),
            "visible_samples": int(len(visible)),
            "visible_manifest_sha256": hash_rows(visible),
        }
    else:
        rows = index.loc[
            (index["SPLIT"] == "test") & (index["ANNOT"] == "treated")
        ].copy().reset_index(drop=True)
        test_controls = index.loc[
            (index["SPLIT"] == "test") & (index["ANNOT"] == "negative_control")
        ].copy()
        rows = pair_controls(test_controls, rows, "rxrx1-official100-sealed-control-v1")
        manifest_hash = hash_rows(rows)
        if manifest_hash != EXPECTED_SEALED_MANIFEST_SHA256:
            raise ValueError(f"sealed manifest drift: {manifest_hash}")
        arrays = evaluation_arrays(raw_root, rows, embeddings, condition_to_id, batch_to_id)
        save_inputs(output_root / "inputs.npz", arrays)
        np.savez_compressed(output_root / "targets.npz", target=arrays["target"])
        manifest = {
            "format_version": 1,
            "source": "official CellFlux HF 100-condition subset",
            "channels": 6,
            "image_size": 96,
            "embedding_dimension": 200,
            "samples": int(len(rows)),
            "conditions": int(rows["CPD_NAME"].nunique()),
            "manifest_sha256": manifest_hash,
        }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("agent", "verifier"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--embedding-path", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/cellflux-data"))
    parser.add_argument(
        "--inception-path",
        type=Path,
        default=Path("/opt/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"),
    )
    parser.add_argument("--skip-inception", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    raw_root = args.raw_root
    downloaded = raw_root is None
    if downloaded:
        dataset_tar = args.cache_dir / "rxrx1.tar"
        extraction_root = args.cache_dir / "extracted"
        download_zip_member(dataset_tar)
        if not extraction_root.exists():
            safe_extract_tar(dataset_tar, extraction_root)
        raw_root = locate_raw_root(extraction_root)
    index_path = args.index_path or (args.cache_dir / "rxrx1_df_subset.csv")
    embedding_path = args.embedding_path or (args.cache_dir / "rxrx1_gene2vec_embeddings.csv")
    download_file(INDEX_URL, index_path, INDEX_SHA256)
    download_file(EMBEDDING_URL, embedding_path, EMBEDDING_SHA256)
    if not args.skip_inception:
        download_file(INCEPTION_URL, args.inception_path, INCEPTION_SHA256)
    manifest = prepare(
        raw_root.resolve(),
        index_path.resolve(),
        embedding_path.resolve(),
        args.output_root.resolve(),
        args.role,
    )
    print(json.dumps(manifest, indent=2))
    if downloaded and not args.keep_cache:
        shutil.rmtree(args.cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
