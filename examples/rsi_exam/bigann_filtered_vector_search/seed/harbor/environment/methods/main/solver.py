"""Starter filtered-ANN solver -- the inherited baseline you improve IN PLACE.

Contract (the grader re-runs exactly this):

    class Solver:
        def build(self, base_vectors, tag_indptr, tag_indices):        # once, untimed, all cores
        def search_batch(self, qv, q_indptr, q_indices, k=10) -> [n, k] row indices   # timed, 1 core

`base_vectors` is a uint8 memmap [10000000, 192]; the tag sets arrive as CSR, so the tags of row i
are `tag_indices[tag_indptr[i]:tag_indptr[i+1]]`, and the queries' predicates use the same layout.
A library row is a HIT for a query iff its tag set is a SUPERSET of that query's tags. Distance is
squared L2 on the raw uint8 values.

This starter is a competent textbook attempt: resolve the predicate through an inverted index,
prune with one global IVF index, score what survives. It clears the recall gate and it is correct.

It is also the ZERO of the scoring band: submitted unchanged it scores 0. Everything you are
measured on is what you add to it.
"""
from __future__ import annotations

import numpy as np

from ann_utils import IVFIndex, build_inverted_index

N_TAGS = 200386
NLIST = 1024
NPROBE = 32


class Solver:
    def build(self, base_vectors, tag_indptr, tag_indices):
        self.X = base_vectors
        # tag -> rows carrying it, so a predicate can be resolved without scanning the library
        self.rows, self.starts = build_inverted_index(tag_indptr, tag_indices, N_TAGS)
        self.ivf = IVFIndex(base_vectors, nlist=NLIST, seed=0)

    def _candidates(self, query_tags):
        s, r = self.starts, self.rows
        cand = r[s[query_tags[0]]:s[query_tags[0] + 1]]
        for t in query_tags[1:]:
            cand = np.intersect1d(cand, r[s[t]:s[t + 1]], assume_unique=True)
        return cand

    def _search_one(self, query_vector, query_tags, k):
        cand = self._candidates(query_tags)
        if len(cand) == 0:
            return np.zeros(k, dtype=np.int64)
        q = np.asarray(query_vector, dtype=np.float32)
        near = self.ivf.nearest_cells(q, NPROBE)
        keep = cand[np.isin(self.ivf.assign[cand], near)]
        if len(keep) < k:
            keep = cand
        d = ((np.asarray(self.X[keep], dtype=np.float32) - q) ** 2).sum(1)
        order = np.argsort(d)[:k] if len(d) <= k else np.argpartition(d, k - 1)[:k]
        out = keep[order]
        if len(out) < k:                      # pad so every row of the answer has k entries
            out = np.concatenate([out, np.zeros(k - len(out), dtype=out.dtype)])
        return out.astype(np.int64)

    def search_batch(self, query_vectors, query_tag_indptr, query_tag_indices, k=10):
        n = len(query_vectors)
        out = np.zeros((n, k), dtype=np.int64)
        for i in range(n):
            tags = query_tag_indices[query_tag_indptr[i]:query_tag_indptr[i + 1]]
            out[i] = self._search_one(query_vectors[i], tags, k)
        return out
