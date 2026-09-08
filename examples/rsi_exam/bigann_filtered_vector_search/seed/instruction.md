# Filtered nearest-neighbour search at 10M scale

A photo library holds **10,000,000 images**. Each is a 192-dimensional `uint8` embedding plus a set
of metadata tags (uploader, camera, place, year, and so on). A query arrives as an embedding plus a
**tag predicate**: one or two tags. An image is a valid answer only if its tag set **contains every
tag in the predicate**. Among the valid images, return the **10 nearest** by squared L2 distance.
Your job is to make that fast.

## Hard Constraints

- Edit `/app/methods/main/solver.py` in place. `/app/methods/main/` is what gets graded.
- Keep the exact `Solver` class contract below. `build()` is called once, then `search_batch()`
  is called **once** with the entire workload; the row order of what `search_batch()` returns must
  match the query order.
- Answers must satisfy the predicate: an image counts only if its tag set contains every tag in
  the query's predicate.
- `build()` may use every core. Before the timed call every thread of the process is pinned to
  **one CPU** (`sched_setaffinity`, applied to each existing thread and inherited by any created
  later), so search throughput cannot be bought with parallelism.
- There is no network at run time: whatever you use has to be already installed or written here.
  `pip list` and the image's package manifest are the authoritative record of what is available.

## What You Have

`/app/data/` holds the library in its published binary form:

| file | contents |
|---|---|
| `base.10M.u8bin` | `int32 n`, `int32 d`, then `n*d` raw bytes -- the `[10000000, 192]` uint8 matrix |
| `base.metadata.10M.spmat` | `int64 nrow, ncol, nnz`, then `int64 indptr[nrow+1]`, then `int32 indices[nnz]` -- the tag sets as CSR |
| `query_vectors.npy` | 500 visible queries, uint8 `[500, 192]` |
| `query_tag_indptr.npy`, `query_tag_indices.npy` | the visible queries' predicates, same CSR convention |
| `ground_truth.npy` | the correct top-10 row ids for each visible query, `int32 [500, 10]` |

`ann_utils.py` (next to your solver) has readers for both binary formats, the tag predicate, a
squared-L2 helper, an inverted-index builder and a plain k-means IVF index. All of it is yours to
use, rewrite or delete.

`base_vectors` is memory-mapped. The container has 32 GB of RAM.

Run `python /app/selfcheck.py` to get your own recall@10 and QPS on the visible queries under the
same protocol as the grader, including the one-CPU pinning. It reports the raw metrics only.

## What You Submit

Leave your best implementation in `/app/methods/main/solver.py`, exposing exactly:

```python
class Solver:
    def build(self, base_vectors, tag_indptr, tag_indices):
        """Called once, before search_batch(). Untimed, and free to use every core.
        base_vectors : uint8 memmap [10000000, 192]
        tag_indptr   : int64 [10000001]   CSR row pointers
        tag_indices  : int32 [108210476]  row i carries tags tag_indices[indptr[i]:indptr[i+1]]
        """

    def search_batch(self, query_vectors, query_tag_indptr, query_tag_indices, k=10):
        """Called ONCE with the entire workload. This call is what gets timed.
        query_vectors    : uint8 [n, 192]
        query_tag_indptr : int64 [n + 1]    same CSR convention as the library's tags
        query_tag_indices: int32 [nnz]      query i has tags indices[indptr[i]:indptr[i+1]]
        returns          : [n, k] library row indices; row order must match the queries,
                           order within a row does not matter
        """
```

There is no submit step and no per-attempt feedback. Work and self-check for as long as your run
window allows, then leave your best `solver.py` in place.

## How It Is Judged

Two numbers, both computed on a **sealed query workload you never see** — a different, larger
sample from the same source as your visible queries:

- **recall@10** against the official ground truth, averaged over the hidden queries. This is a
  **gate**: below **0.90** the submission scores zero no matter how fast it is.
- **QPS** = hidden query count divided by the wall-clock seconds of the single `search_batch()`
  call. `build()` is not counted and has its own, generous, budget.

Your score rises monotonically with QPS once the gate is met, and is not capped at the top.
The starter as shipped is the zero of that scale: submitted unchanged it scores 0.
