def predict_complex(item: dict) -> dict:
    """Fast invalid fixture used only to exercise fail-closed aggregation."""
    return {"protein_pdb": "not a PDB", "ligand_sdf": "not an SDF"}
