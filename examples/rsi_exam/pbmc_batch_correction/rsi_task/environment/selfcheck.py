"""Local self-test on the VISIBLE split only.

Runs the embedding in methods/main on the visible cells and computes cell-type
NMI (best-over-resolution Leiden if scanpy is available, else KMeans fallback).

    python environment/selfcheck.py [--data <root>]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
from scipy import sparse

SEED = 2002
HERE = os.path.dirname(os.path.abspath(__file__))
RESOLUTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0]


def _resolve_root(cli):
    for cand in (cli, os.environ.get("PBMC_ATLAS_DIR"), os.path.join(HERE, "data")):
        if cand and os.path.exists(os.path.join(cand, "pbmc_atlas_atac.h5ad")):
            return cand
    raise FileNotFoundError("Set --data or PBMC_ATLAS_DIR.")


def _load(path):
    import anndata as ad
    a = ad.read_h5ad(path)
    X = a.X
    X = (sparse.csr_matrix(X) if not sparse.issparse(X) else X.tocsr()).astype(np.float32)
    X.data[:] = 1.0
    names = list(a.var["features"]) if "features" in a.var.columns else list(a.var_names)
    return X, names, a.obs["annot"].astype(str).tolist(), a.obs["batch"].astype(str).tolist()


def _nmi(Z, celltype):
    from sklearn.metrics import normalized_mutual_info_score
    try:
        import anndata as ad
        import scanpy as sc
        a = ad.AnnData(X=Z.astype(np.float32)); a.obs["ct"] = celltype
        sc.pp.neighbors(a, use_rep="X", n_neighbors=15, random_state=SEED)
        best = 0.0
        for res in RESOLUTIONS:
            sc.tl.leiden(a, resolution=res, key_added="cl", random_state=SEED,
                         flavor="igraph", n_iterations=2, directed=False)
            best = max(best, normalized_mutual_info_score(celltype, a.obs["cl"].tolist()))
        return best, "leiden"
    except Exception:
        from sklearn.cluster import KMeans
        n_ct = len(set(celltype)); best = 0.0
        for k in sorted({max(2, n_ct - 2), n_ct, n_ct + 2, n_ct + 5}):
            km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Z)
            best = max(best, normalized_mutual_info_score(celltype, km.labels_))
        return best, "kmeans-fallback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    root = _resolve_root(args.data)
    X, peaks, celltype, batch = _load(os.path.join(root, "pbmc_atlas_atac.h5ad"))
    bset = sorted(set(batch)); bidx = {b: i for i, b in enumerate(bset)}
    batch_ids = np.array([bidx[b] for b in batch])

    spec = importlib.util.spec_from_file_location(
        "main_solver", os.path.join(HERE, "methods", "main", "solver.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.join(HERE, "methods", "main"))
    spec.loader.exec_module(mod)

    Z = np.asarray(mod.embed(X, batch_ids, peaks), dtype=np.float64)
    nmi, method = _nmi(Z, celltype)
    print(f"[selfcheck] cells={X.shape[0]} dim={Z.shape[1]} celltypes={len(set(celltype))} batches={len(bset)}")
    print(f"[selfcheck] NMI ({method})={nmi:.4f}")
    print("[selfcheck] visible proxy only; the sealed held-out score will differ.")
    import os as _os, subprocess as _sp, sys as _sys
    if _os.path.exists("/app/budget.py"):
        _sys.stdout.flush()
        _sp.run([_sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()
