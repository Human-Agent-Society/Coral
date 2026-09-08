#!/usr/bin/env python3
"""Fail-fast integrity and schema gate for materialized visible-image assets."""

from __future__ import annotations

import hashlib
import stat
import sys
from pathlib import Path


LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
AGENT_ROOT = Path("/app/data")
EVALUATOR_ROOT = Path("/opt/cell-visible")

AGENT_EXPECTED = {
    AGENT_ROOT / "visible_labeled.h5ad": (
        1_138_579,
        "e009455de016df7dfbd16e1c5201f9f73da182f2c0b257c5c94e25e579929f5c",
    ),
    AGENT_ROOT / "visible_unlabeled.h5ad": (
        97_708_447,
        "6db99bc3eb5ec31ab1a1e10e757275a6d5fbec9a66b3f83d39043d4131c67b27",
    ),
    AGENT_ROOT / "visible_query.h5ad": (
        5_671_262,
        "dc19fbdd164bfb43e4ea790a89514617d146795c41508604651788e3342450ab",
    ),
    AGENT_ROOT / "classes.txt": (
        218,
        "0dfb8f66ce2207f047887089a42ceb1ccb057df6c2ddaf159328578e7300b53d",
    ),
    AGENT_ROOT / "reference_model.pkl": (
        774_940,
        "ca34a816b693613eed7591efc1ed1c5581d10a310942ed5cf25d7e98884fae66",
    ),
}

EVALUATOR_EXPECTED = {
    EVALUATOR_ROOT / "visible_labels.csv": (
        34_827,
        "c2b4ce39f176f5def7d3a0c0583f565399ee5a1ee0e8582ce0420052c80a9aed",
    ),
    EVALUATOR_ROOT / "classes.txt": (
        218,
        "0dfb8f66ce2207f047887089a42ceb1ccb057df6c2ddaf159328578e7300b53d",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_materialized(expected: dict[Path, tuple[int, str]]) -> None:
    for path, (expected_size, expected_digest) in expected.items():
        try:
            info = path.lstat()
        except OSError as exc:
            raise RuntimeError(f"visible-image asset is missing: {path.name}") from exc
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"visible-image asset is not one regular file: {path.name}")
        with path.open("rb") as handle:
            prefix = handle.read(len(LFS_HEADER))
        if prefix == LFS_HEADER:
            raise RuntimeError(f"unsmudged Git-LFS pointer reached image build: {path.name}")
        if info.st_size != expected_size or sha256(path) != expected_digest:
            raise RuntimeError(f"visible-image asset failed integrity: {path.name}")


def read_classes(path: Path) -> list[str]:
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(classes) != 15 or len(set(classes)) != 15:
        raise RuntimeError("class vocabulary must contain 15 unique labels")
    return classes


def assert_unlabeled(value: object, name: str) -> None:
    columns = getattr(value, "obs").columns
    forbidden = [column for column in columns if "label" in str(column).lower() or str(column) == "ann_finest_level"]
    if forbidden:
        raise RuntimeError(f"{name} contains label-like observation columns")


def validate_agent() -> None:
    validate_materialized(AGENT_EXPECTED)
    # Imports happen only after the byte gate so an unsmudged checkout fails
    # immediately and cannot masquerade as a dependency/schema error.
    import anndata as ad

    classes = read_classes(AGENT_ROOT / "classes.txt")
    labeled = ad.read_h5ad(AGENT_ROOT / "visible_labeled.h5ad", backed="r")
    unlabeled = ad.read_h5ad(AGENT_ROOT / "visible_unlabeled.h5ad", backed="r")
    query = ad.read_h5ad(AGENT_ROOT / "visible_query.h5ad", backed="r")
    try:
        if labeled.shape != (75, 27_402) or unlabeled.shape != (22_740, 27_402) or query.shape != (1_200, 27_402):
            raise RuntimeError("agent H5AD shape mismatch")
        if "ann_finest_level" not in labeled.obs:
            raise RuntimeError("labeled input lacks its public label column")
        counts = labeled.obs["ann_finest_level"].astype(str).value_counts()
        if set(counts.index) != set(classes) or not (counts == 5).all():
            raise RuntimeError("public labeled input violates the five-shot contract")
        assert_unlabeled(unlabeled, "unlabeled input")
        assert_unlabeled(query, "visible query")
        if not labeled.var_names.equals(unlabeled.var_names) or not labeled.var_names.equals(query.var_names):
            raise RuntimeError("agent inputs do not share the exact gene axis")
        if not labeled.obs_names.is_unique or not unlabeled.obs_names.is_unique or not query.obs_names.is_unique:
            raise RuntimeError("agent input cell IDs are not unique")
    finally:
        for value in (labeled, unlabeled, query):
            value.file.close()


def validate_evaluator() -> None:
    validate_materialized(EVALUATOR_EXPECTED)
    import pandas as pd

    classes = read_classes(EVALUATOR_ROOT / "classes.txt")
    frame = pd.read_csv(EVALUATOR_ROOT / "visible_labels.csv", dtype=str)
    if list(frame.columns) != ["cell_id", "label"] or frame.isna().any().any() or frame["cell_id"].duplicated().any():
        raise RuntimeError("visible evaluator labels have invalid schema")
    if len(frame) != 1_200 or set(frame["label"]) != set(classes):
        raise RuntimeError("visible evaluator labels violate row/class coverage")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"agent", "evaluator"}:
        raise SystemExit("usage: build_validate_assets.py {agent|evaluator}")
    if sys.argv[1] == "agent":
        validate_agent()
    else:
        validate_evaluator()


if __name__ == "__main__":
    main()
