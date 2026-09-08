#!/usr/bin/env python3
"""Build-time integrity/schema gate for materialized verifier inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import anndata as ad
import pandas as pd

ROOT = Path("/tests")
INPUT = ROOT / "inputs"
SEALED = ROOT / "sealed"
EXPECTED = {
    INPUT / "labeled.h5ad": (1_138_579, "e009455de016df7dfbd16e1c5201f9f73da182f2c0b257c5c94e25e579929f5c"),
    INPUT / "unlabeled.h5ad": (97_708_447, "6db99bc3eb5ec31ab1a1e10e757275a6d5fbec9a66b3f83d39043d4131c67b27"),
    INPUT / "query.h5ad": (4_967_777, "724dd36e19ba8692a1237fa508831522bb012d991b8b0e0a603dc915f691f8bc"),
    INPUT / "classes.txt": (218, "0dfb8f66ce2207f047887089a42ceb1ccb057df6c2ddaf159328578e7300b53d"),
    INPUT / "reference_model.pkl": (774_940, "ca34a816b693613eed7591efc1ed1c5581d10a310942ed5cf25d7e98884fae66"),
    SEALED / "truth.csv": (28_661, "4af62b3a9302583dea5a562401dc1f0c3eb8eb6488d5d966fe7374e1d74a666b"),
    SEALED / "config.json": (93, "962e08fa3b30b7adb8c49db7194c9261441c45266c06f12287079820a8381054"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_unlabeled(value: ad.AnnData, name: str) -> None:
    forbidden = [column for column in value.obs.columns if "label" in str(column).lower() or str(column) == "ann_finest_level"]
    if forbidden:
        raise RuntimeError(f"{name} contains label-like observation columns")


def main() -> None:
    for path, (size, digest) in EXPECTED.items():
        if path.stat().st_size != size or sha256(path) != digest:
            raise RuntimeError(f"materialized verifier asset failed integrity: {path.name}")
    classes = [line.strip() for line in (INPUT / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(classes) != 15 or len(set(classes)) != 15:
        raise RuntimeError("class vocabulary must contain 15 unique labels")
    labeled = ad.read_h5ad(INPUT / "labeled.h5ad", backed="r")
    unlabeled = ad.read_h5ad(INPUT / "unlabeled.h5ad", backed="r")
    query = ad.read_h5ad(INPUT / "query.h5ad", backed="r")
    try:
        if labeled.shape != (75, 27_402) or unlabeled.shape != (22_740, 27_402) or query.shape != (1_050, 27_402):
            raise RuntimeError("verifier H5AD shape mismatch")
        if "ann_finest_level" not in labeled.obs:
            raise RuntimeError("labeled input lacks the public label column")
        counts = labeled.obs["ann_finest_level"].astype(str).value_counts()
        if set(counts.index) != set(classes) or not (counts == 5).all():
            raise RuntimeError("public labeled input violates the five-shot contract")
        assert_unlabeled(unlabeled, "unlabeled input")
        assert_unlabeled(query, "sealed query")
        if not labeled.var_names.equals(unlabeled.var_names) or not labeled.var_names.equals(query.var_names):
            raise RuntimeError("verifier inputs do not share the exact gene axis")
        query_ids = pd.Index(query.obs_names.astype(str), name="cell_id")
    finally:
        for value in (labeled, unlabeled, query):
            value.file.close()
    truth = pd.read_csv(SEALED / "truth.csv", dtype=str)
    if list(truth.columns) != ["cell_id", "label"] or truth["cell_id"].duplicated().any():
        raise RuntimeError("sealed truth schema is invalid")
    if len(truth) != len(query_ids) or set(truth["cell_id"]) != set(query_ids):
        raise RuntimeError("sealed truth IDs do not match the query")
    if set(truth["label"]) - set(classes):
        raise RuntimeError("sealed truth contains an unknown class")


if __name__ == "__main__":
    main()
