"""Public helpers for the filtered vector-search task: raw-file readers, the tag predicate,
and a plain k-means IVF index. Free to use, edit, or replace entirely -- Solver.build /
Solver.search in solver.py are the only entry points the grader calls.

Data conventions handed to Solver.build / Solver.search:
  base_vectors : uint8 memmap [N, 192]      -- library vectors, raw Big-ANN yfcc bytes
  tag_indptr   : int64  [N + 1]             -- CSR row pointers into tag_indices
  tag_indices  : int32  [nnz]               -- tag ids of row i are tag_indices[indptr[i]:indptr[i+1]]
  query_vector : uint8  [192]               -- one query vector
  query_tags   : int32  [1] or [2]          -- a library row is a HIT iff its tag set is a
                                               SUPERSET of these tags

N = 10,000,000 rows, 200,386 distinct tags, 10.82 tags per row on average. Tag frequency spans six
orders of magnitude: the rarest tag has 1 row, the median 54, the most common 3,386,745.
Distance is squared L2 on the raw uint8 values (cast to a wider dtype before subtracting).
"""
from __future__ import annotations

import numpy as np


# ── raw Big-ANN file readers ──────────────────────────────────────────────────
def read_u8bin(path):
    """[n, d] uint8 memmap over a Big-ANN .u8bin (int32 n, int32 d, then n*d bytes)."""
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype="int32", count=2)
    return np.memmap(path, dtype="uint8", mode="r", offset=8, shape=(int(n), int(d)))


def read_spmat(path):
    """(indptr int64 [nrow+1], indices int32 [nnz], ncol) from a Big-ANN .spmat
    (int64 nrow, ncol, nnz; then indptr; then indices)."""
    with open(path, "rb") as f:
        nrow, ncol, nnz = np.fromfile(f, dtype="int64", count=3)
    nrow, ncol, nnz = int(nrow), int(ncol), int(nnz)
    indptr = np.memmap(path, dtype="int64", mode="r", offset=24, shape=(nrow + 1,))
    indices = np.memmap(path, dtype="int32", mode="r", offset=24 + 8 * (nrow + 1), shape=(nnz,))
    return indptr, indices, ncol


# ── the predicate and the distance ────────────────────────────────────────────
def l2_sq(q, X):
    """Squared L2 from one uint8 query [D] to uint8 rows X [n, D] -> float32 [n]."""
    Xf = np.asarray(X, dtype=np.float32)
    qf = np.asarray(q, dtype=np.float32)
    return ((Xf - qf) ** 2).sum(axis=1)


def matches(query_tags, row, tag_indptr, tag_indices):
    """True iff row's tag set contains every tag in query_tags."""
    row_tags = tag_indices[tag_indptr[row]:tag_indptr[row + 1]]
    return bool(np.isin(query_tags, row_tags).all())


def build_inverted_index(tag_indptr, tag_indices, n_tags):
    """tag -> sorted array of row ids, as one flat array plus offsets.

    Returns (rows int32 [nnz], starts int64 [n_tags + 1]) so the rows carrying tag t are
    rows[starts[t]:starts[t + 1]]. This is the transpose of the CSR you are handed.
    """
    ix = np.asarray(tag_indices)
    n_rows = len(tag_indptr) - 1
    row_of = np.repeat(np.arange(n_rows, dtype=np.int32), np.diff(np.asarray(tag_indptr)))
    order = np.argsort(ix, kind="stable")
    rows = row_of[order]
    counts = np.bincount(ix, minlength=n_tags)
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return rows, starts


class IVFIndex:
    """k-means-style IVF over the library: assigns every row to one of `nlist` cells, and at
    search time returns the rows in the `nprobe` cells nearest the query."""

    def __init__(self, X, nlist=1024, seed=0, iters=8, sample=200_000, chunk=500_000):
        rng = np.random.default_rng(seed)
        n = len(X)
        samp = np.asarray(X[rng.choice(n, min(sample, n), replace=False)], dtype=np.float32)
        C = samp[rng.choice(len(samp), nlist, replace=False)].copy()
        for _ in range(iters):
            a = (-2.0 * samp @ C.T + (C * C).sum(1)[None, :]).argmin(1)
            for j in range(nlist):
                m = a == j
                if m.any():
                    C[j] = samp[m].mean(0)
        self.centroids = C
        self.nlist = nlist
        assign = np.empty(n, dtype=np.int16)
        Cn = (C * C).sum(1)[None, :]
        for s in range(0, n, chunk):
            Xf = np.asarray(X[s:s + chunk], dtype=np.float32)
            assign[s:s + chunk] = (Cn - 2.0 * Xf @ C.T).argmin(1)
        self.assign = assign

    def nearest_cells(self, q, nprobe):
        d = ((self.centroids - np.asarray(q, dtype=np.float32)) ** 2).sum(1)
        return np.argpartition(d, min(nprobe, len(d) - 1))[:nprobe]
