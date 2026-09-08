# Visible data — batch correction

This directory contains the visible development split. Evaluation uses a
separate split that is not present in the agent environment.

Files:
- `pbmc_atlas_atac.h5ad` — binary peak-cell matrix; `var_names="chr-start-end"`,
  `obs['batch']` = batch id, `obs['annot']` = cell type (for LOCAL scoring only;
  your `embed` must not use cell type).

Only the ATAC file is needed for this scenario.
