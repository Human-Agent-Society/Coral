"""Editable deterministic single-sample co-folding baseline.

Fixed entry point (the judge's evaluate.py imports it by this name and calls it on
the hidden data):
    predict_complex(item: dict) -> {"protein_pdb": str, "ligand_sdf": str}

You may add sibling helper modules inside this directory (main/) and import them.
Do not return pre-computed coordinates.
"""
from cofold_utils import run_cofolding, split_complex


def predict_complex(item: dict) -> dict:
    cif_text = run_cofolding(
        item,
        num_samples=1,
        recycling=3,
        use_msa=True,
        relax=False,          # generic relaxation is not silently applied
    )
    protein_pdb, ligand_sdf = split_complex(cif_text, item["ligand_smiles"])
    return {"protein_pdb": protein_pdb, "ligand_sdf": ligand_sdf}
