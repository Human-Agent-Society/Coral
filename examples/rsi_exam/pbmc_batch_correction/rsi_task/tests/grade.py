"""Harbor verifier for pbmc_batch_correction.

Runs AFTER the agent finishes. Answer-key-isolated grading:

  1. load the test ATAC (binary peaks), its cell-type labels (`annot`) and batch ids into memory;
  2. SEAL: delete the test file (pbmc_atlas_atac_test.h5ad) from disk — its `annot` is the answer;
  3. write a FEATURE-ONLY boundary (ATAC + batch ids, NO cell types) and run the submission's
     embed() in an isolated subprocess (tests/solve.py) -> cell embedding. The submission gets
     numpy/scipy arrays only and the test file is gone, so it cannot read the cell types;
  4. score cell-type NMI of the embedding (best-over-resolution Leiden, scIB protocol; KMeans
     fallback if scanpy/igraph are unavailable) and map to reward.

Reward (RL/eval scalar), linear on the higher-is-better NMI axis, anchored at
baseline=0 and upper_bound(perfect=1.0)=1. Any failure -> 0.
"""

from __future__ import annotations

import gc
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import sparse

# Anchors live in a sealed root-owned file, never in [verifier.env]:
# anything injected there is readable from /proc/1/environ by submitted code.
ANCHOR_PATH = Path(__file__).resolve().with_name("anchors.json")
with ANCHOR_PATH.open(encoding="utf-8") as _f:
    _ANCHORS = json.load(_f)
BASELINE = float(_ANCHORS["BASELINE"])     # PCA, measured -> 0
UPPER_BOUND = float(_ANCHORS["UPPER_BOUND"])  # perfect cluster match -> 1

SUBMISSION_DIR = Path("/app/methods/main")
DATA = Path("/tests/heldout")
REWARD_DIR = Path("/logs/verifier")
SOLVE = Path("/tests/solve.py")

SEED = 2002
RESOLUTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0]


def _reward(v: float) -> float:
    """baseline->0, upper->1, linear in between."""
    b, u = BASELINE, UPPER_BOUND   # b < u
    if v <= b:
        return 0.0
    if v <= u:
        return float((v - b) / max(1e-12, u - b))
    return 1.0  # v > upper is mathematically impossible (NMI <= 1); reward is capped


def _load_atac(path):
    import anndata as ad
    a = ad.read_h5ad(path)
    X = a.X
    X = (sparse.csr_matrix(X) if not sparse.issparse(X) else X.tocsr()).astype(np.float32)
    X.data[:] = 1.0
    names = list(a.var["features"]) if "features" in a.var.columns else list(a.var_names)
    return (X, names, a.obs["annot"].astype(str).tolist(),
            a.obs["batch"].astype(str).tolist())


def _nmi_scanpy(Z, celltype):
    """Best-over-resolution Leiden NMI (scIB protocol), matching the paper."""
    import anndata as ad
    import scanpy as sc
    from sklearn.metrics import normalized_mutual_info_score

    a = ad.AnnData(X=Z.astype(np.float32))
    a.obs["celltype"] = celltype
    sc.pp.neighbors(a, use_rep="X", n_neighbors=15, random_state=SEED)
    best = 0.0
    for res in RESOLUTIONS:
        sc.tl.leiden(a, resolution=res, key_added="cl", random_state=SEED,
                     flavor="igraph", n_iterations=2, directed=False)
        best = max(best, normalized_mutual_info_score(celltype, a.obs["cl"].tolist()))
    return best, "leiden"


def _nmi_kmeans(Z, celltype):
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    n_ct = len(set(celltype))
    best = 0.0
    for k in sorted({max(2, n_ct - 2), n_ct, n_ct + 2, n_ct + 5}):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Z)
        best = max(best, normalized_mutual_info_score(celltype, km.labels_))
    return best, "kmeans-fallback"


def main() -> None:
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    out = {"metric": None, "reward": 0.0, "correctness": False, "errors": []}
    try:
        X, peaks, celltype, batch = _load_atac(DATA / "pbmc_atlas_atac_test.h5ad")
        bset = sorted(set(batch))
        bidx = {b: i for i, b in enumerate(bset)}
        batch_ids = np.array([bidx[b] for b in batch], dtype=np.int64)

        # Answer-key seal: the cell types (celltype) are in memory; remove the file so the embed
        # subprocess cannot read its `annot`.
        f = DATA / "pbmc_atlas_atac_test.h5ad"
        if f.exists():
            f.unlink()
        ANCHOR_PATH.unlink(missing_ok=True)

        bnd = Path(tempfile.mkdtemp(prefix="batchcorr_bnd_"))
        sparse.save_npz(bnd / "atac.npz", X)
        np.savez(bnd / "aux.npz", batch_ids=batch_ids)
        json.dump({"peak_names": peaks}, open(bnd / "names.json", "w"))
        shutil.rmtree(DATA)
        # free the big sparse matrix before the solver subprocess runs (scoring below only needs
        # Z + celltype + n_cells); avoids parent+child both holding the peak matrix -> OOM.
        n_cells = X.shape[0]
        del X
        gc.collect()
        z_path = bnd / "embed.npy"
        proc = subprocess.run(
            [sys.executable, str(SOLVE), "--boundary", str(bnd),
             "--submission", str(SUBMISSION_DIR), "--out", str(z_path)],
            capture_output=True, text=True, timeout=2100,
        )
        if not z_path.exists():
            tail = (proc.stderr or proc.stdout or "<no output>")[-2000:]
            raise RuntimeError(
                f"submission produced no embedding (rc={proc.returncode}); "
                f"solve output tail: {tail}")
        Z = np.load(z_path).astype(np.float64)
        if Z.ndim != 2 or Z.shape[0] != n_cells:
            raise ValueError(f"embedding shape {Z.shape} invalid for {n_cells} cells")
        if not np.all(np.isfinite(Z)):
            raise ValueError("embedding contains non-finite values")
        if Z.shape[1] < 2 or Z.shape[1] > 512:
            raise ValueError(f"embedding dim {Z.shape[1]} out of allowed range [2, 512]")

        errs = []
        try:
            nmi, method = _nmi_scanpy(Z, celltype)
        except Exception as e:  # noqa: BLE001
            errs.append(f"scanpy NMI unavailable, using KMeans fallback: {e}")
            nmi, method = _nmi_kmeans(Z, celltype)

        out = {"metric": round(float(nmi), 6), "reward": round(_reward(float(nmi)), 6),
               "correctness": True, "errors": errs, "nmi_method": method,
               "embedding_dim": int(Z.shape[1]), "n_test_cells": int(n_cells),
               "n_celltypes": len(set(celltype)), "n_batches": len(bset)}
    except Exception as exc:  # noqa: BLE001 — any failure -> reward 0, never crash
        out = {"metric": None, "reward": 0.0, "correctness": False,
               "errors": [f"{type(exc).__name__}: {exc}"]}

    # reward.json keeps only the reward; nmi_celltype is in score_details.json.
    rewards = {"reward": float(out["reward"])}
    (REWARD_DIR / "reward.txt").write_text(f"{out['reward']}\n", encoding="utf-8")
    (REWARD_DIR / "reward.json").write_text(json.dumps(rewards), encoding="utf-8")
    details = {**out, "anchors": _ANCHORS}
    (REWARD_DIR / "score_details.json").write_text(json.dumps(details), encoding="utf-8")
    (REWARD_DIR / "grade_debug.json").write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
