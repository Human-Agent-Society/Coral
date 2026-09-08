# Batch-integrated cell embedding from scATAC-seq peaks (PBMC)

Single-cell data is collected in separate batches (samples, labs, runs), and technical batch
effects can dominate and obscure real biology. You inherit a no-correction **PCA** embedding (which
keeps batch variation) and must produce a low-dimensional embedding that **removes batch variation
while preserving cell-type structure**, using only each cell's binary peak set and its batch label.
The embedding is re-computed by a sealed verifier on a hidden split and scored by how well cell
types stay clustered, so only a method that genuinely *integrates* across batches counts.

## Hard Constraints

- Submit an **algorithm** (`embed`), not a precomputed embedding — the verifier re-runs your code on
  hidden cells.
- Keep the exact signature
  `embed(atac, batch_labels, peak_names) -> np.ndarray (n_cells, d)`.
- Input peaks are **binary**; you also receive each cell's integer `batch` id. You do **not** receive
  cell-type labels — the embedding must be unsupervised w.r.t. cell type (the judge uses cell types
  only for scoring).
- Output a real-valued embedding with a modest `d` in **[2, 512]** (e.g. 10–128); finite values, one
  row per input cell, or the submission scores 0.
- There is no network at grade time. Do not hardcode or look up answers.

## What You Have

- The workspace `/app/`:
  - Visible data (`/app/data/`): `pbmc_atlas_atac.h5ad` — binary peak-cell matrix
    (`var_names="chr-start-end"`) with `obs['batch']` and (for local scoring only) `obs['annot']`.
  - The editable baseline `/app/methods/main/` — **this directory is what gets graded**: the
    no-correction PCA embedding. Improve it in place or
    rewrite it entirely.
- **Your self-check surface** (free): `python /app/selfcheck.py` runs your embedding on the visible
  cells and reports cell-type NMI (best-over-resolution Leiden, or a KMeans fallback). This is a
  **proxy** — the sealed split differs, so a good self-check is necessary, not sufficient.

## What You Submit

Edit `/app/methods/main/solver.py` to expose:

```python
def embed(atac, batch_labels, peak_names):
    # atac         : scipy.sparse CSR (n_cells, n_peaks), binary {0,1}
    # batch_labels : np.ndarray (n_cells,) integer batch ids
    # peak_names   : list[str] length n_peaks, "chr-start-end"
    # returns      : np.ndarray float (n_cells, d) cell embedding, d in [2, 512]
```

You may add helper modules next to `solver.py`. The runtime provides `numpy / scipy /
scikit-learn / torch` (and `anndata` / `scanpy` for the self-check). **There is no submit step and
no per-attempt feedback** — self-check for as long as your run window allows, then leave your best
`solver.py` in place; it is graded once at the end on the hidden cells.

## How It Is Judged

After your run, the verifier runs your `embed` on the hidden split (your code runs in an isolated
subprocess that receives only the peaks + batch ids, never the cell types). It clusters the
embedding with Leiden at a grid of resolutions and scores the best **normalized mutual information
(NMI)** between the clustering and the hidden cell-type labels:

```
metric = max over resolutions of  NMI(Leiden(embedding, res), cell_types)
```

The metric is **cell-type NMI (higher is better)**. A no-correction embedding keeps batch
variation; strong methods mix batches while keeping cell types separable.
