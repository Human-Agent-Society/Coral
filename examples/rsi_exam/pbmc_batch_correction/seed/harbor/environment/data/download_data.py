"""Download the PBMC_ATLAS files used by all three scenarios.

Usage:
    python download_data.py --out /path/to/data_root

Requires `huggingface_hub`. After download, point evaluate.py / selfcheck.py at
the chosen data root via --data or the PBMC_ATLAS_DIR env var.
"""
import argparse
import os

REPO = "Chtholly17/PBMC_ATLAS"
FILES = [
    "pbmc_atlas_atac.h5ad",
    "pbmc_atlas_atac_test.h5ad",
    "pbmc_atlas_rna_binning_10.h5ad",
    "pbmc_atlas_rna_binning_10_test.h5ad",
    "pbmc_atlas_variable_genes.csv",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Target data root directory.")
    ap.add_argument("--files", nargs="*", default=FILES, help="Subset of files.")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    os.makedirs(args.out, exist_ok=True)
    for fn in args.files:
        print(f"Downloading {fn} ...")
        hf_hub_download(
            repo_id=REPO,
            filename=fn,
            repo_type="dataset",
            local_dir=args.out,
            local_dir_use_symlinks=False,
        )
    print(f"Done. Data root: {args.out}")


if __name__ == "__main__":
    main()
