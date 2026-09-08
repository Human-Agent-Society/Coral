"""Inherited weak baseline for five-shot cell-type annotation."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


LABEL_COLUMN = "ann_finest_level"


def read_classes(path: Path) -> list[str]:
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(classes) != 15 or len(classes) != len(set(classes)):
        raise ValueError("classes file must contain exactly 15 unique labels")
    return classes


def log_normalize(value: ad.AnnData) -> ad.AnnData:
    result = value.copy()
    sc.pp.normalize_total(result, target_sum=1e4)
    sc.pp.log1p(result)
    return result


def dense_features(value: ad.AnnData, genes: list[str]) -> np.ndarray:
    result = log_normalize(value)[:, genes].X
    if sp.issparse(result):
        result = result.toarray()
    return np.asarray(result, dtype=np.float32)


def predict(
    labeled_path: Path,
    unlabeled_path: Path,
    query_path: Path,
    classes_path: Path,
) -> pd.DataFrame:
    classes = read_classes(classes_path)
    labeled = ad.read_h5ad(labeled_path)
    unlabeled = ad.read_h5ad(unlabeled_path)
    query = ad.read_h5ad(query_path)
    if LABEL_COLUMN not in labeled.obs:
        raise ValueError(f"labeled input is missing {LABEL_COLUMN}")
    if not labeled.var_names.equals(unlabeled.var_names) or not labeled.var_names.equals(query.var_names):
        raise ValueError("all inputs must use the same genes in the same order")

    feature_source = log_normalize(ad.concat([labeled, unlabeled, query], axis=0, join="inner", merge="same"))
    sc.pp.highly_variable_genes(feature_source, n_top_genes=2000, flavor="seurat")
    genes = feature_source.var_names[feature_source.var["highly_variable"]].astype(str).tolist()

    encoder = LabelEncoder().fit(classes)
    train_labels = labeled.obs[LABEL_COLUMN].astype(str).to_numpy()
    if set(train_labels) != set(classes):
        raise ValueError("labeled input does not cover the class vocabulary")
    classifier = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=0)
    classifier.fit(dense_features(labeled, genes), encoder.transform(train_labels))
    predictions = encoder.inverse_transform(classifier.predict(dense_features(query, genes)))
    return pd.DataFrame({"cell_id": query.obs_names.astype(str), "pred_label": predictions})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", type=Path, required=True)
    parser.add_argument("--unlabeled", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = predict(args.labeled, args.unlabeled, args.query, args.classes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"wrote {len(output)} predictions to {args.output}")


if __name__ == "__main__":
    main()
