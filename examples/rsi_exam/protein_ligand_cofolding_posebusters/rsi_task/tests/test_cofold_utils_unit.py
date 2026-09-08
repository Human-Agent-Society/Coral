import importlib.util
import inspect
import math
from pathlib import Path

import pytest
import yaml


TASK_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = TASK_ROOT / "environment" / "methods" / "main" / "cofold_utils.py"


@pytest.fixture(scope="module")
def cofold_utils():
    spec = importlib.util.spec_from_file_location("cofold_utils_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_boltz_yaml_is_safe_and_identical_sequences_share_msa(tmp_path, cofold_utils):
    msa_dir = tmp_path / "source_msa"
    msa_dir.mkdir()
    (msa_dir / "A.a3m").write_bytes(b">query\nAAAA\n\x00>hit\nAAAA\n")
    (msa_dir / "B.a3m").write_text(">query\nSHOULD_NOT_BE_USED\n")
    (msa_dir / "C.a3m").write_text(">query\nCCCC\n")
    work = tmp_path / "work"
    work.mkdir()
    smiles = "C/C=C\\C'special:#"
    item = {
        "protein_chains": [
            {"chain_id": "A", "sequence": "AAAA"},
            {"chain_id": "B", "sequence": "AAAA"},
            {"chain_id": "C", "sequence": "CCCC"},
        ],
        "ligand_smiles": smiles,
        "msa_dir": str(msa_dir),
    }

    input_path = cofold_utils._write_boltz_input(item, str(work), use_msa=True)
    parsed = yaml.safe_load(Path(input_path).read_text())
    proteins = [entry["protein"] for entry in parsed["sequences"][:-1]]

    assert parsed["sequences"][-1]["ligand"]["smiles"] == smiles
    assert proteins[0]["msa"] == proteins[1]["msa"]
    assert proteins[0]["msa"] != proteins[2]["msa"]
    assert Path(proteins[0]["msa"]).read_bytes() == b">query\nAAAA\n>hit\nAAAA\n"
    assert b"SHOULD_NOT_BE_USED" not in Path(proteins[0]["msa"]).read_bytes()


def test_no_msa_uses_documented_empty_value_without_touching_msa_dir(
    tmp_path, cofold_utils
):
    item = {
        "protein_chains": [
            {"chain_id": "A", "sequence": "AAAA"},
            {"chain_id": "B", "sequence": "BBBB"},
        ],
        "ligand_smiles": "CCO",
    }
    input_path = cofold_utils._write_boltz_input(
        item, str(tmp_path), use_msa=False
    )
    parsed = yaml.safe_load(Path(input_path).read_text())

    assert [x["protein"]["msa"] for x in parsed["sequences"][:-1]] == [
        "empty",
        "empty",
    ]


def test_boltz_command_is_reproducible_offline_and_tempdir_is_cleaned(
    tmp_path, monkeypatch, cofold_utils
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        work = Path(command[command.index("--out_dir") + 1])
        output = work / "boltz_results_input" / "predictions" / "input"
        output.mkdir(parents=True)
        (output / "input_model_0.cif").write_text("CIF_ZERO")

    cache = tmp_path / "boltz cache"
    monkeypatch.setenv("BOLTZ_CACHE", str(cache))
    monkeypatch.setattr(cofold_utils.subprocess, "run", fake_run)
    result = cofold_utils.run_cofolding(
        {
            "protein_chains": [{"chain_id": "A", "sequence": "AAAA"}],
            "ligand_smiles": "CCO",
        },
        use_msa=False,
    )

    assert result == "CIF_ZERO"
    command, kwargs = calls[0]
    assert command[command.index("--cache") + 1] == str(cache.resolve())
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--num_workers") + 1] == "0"
    assert command[command.index("--output_format") + 1] == "mmcif"
    assert "--use_msa_server" not in command
    assert kwargs == {"check": True, "capture_output": True, "text": True}
    assert not Path(command[command.index("--out_dir") + 1]).exists()


def test_default_output_selects_exact_rank_zero_not_rank_ten(tmp_path, cofold_utils):
    out = tmp_path / "predictions"
    out.mkdir()
    model_0 = out / "target_model_0.cif"
    model_10 = out / "target_model_10.cif"
    model_0.write_text("ZERO")
    model_10.write_text("TEN")

    assert cofold_utils._pick_default_cif(str(tmp_path)) == str(model_0)
    assert cofold_utils._sample_rank(str(model_0)) == 0
    assert cofold_utils._sample_rank(str(model_10)) == 10


def test_starter_does_not_prepackage_hidden_portfolio_recipe(cofold_utils):
    parameters = inspect.signature(cofold_utils.run_cofolding).parameters
    assert "model" not in parameters
    assert "ensemble" not in parameters
    assert not hasattr(cofold_utils, "_run_chai")
    assert not hasattr(cofold_utils, "_pick_by_confidence")
    assert not hasattr(cofold_utils, "_pick_best_cif")


def test_unimplemented_relaxation_fails_explicitly(cofold_utils):
    with pytest.raises(NotImplementedError, match="no validated generic"):
        cofold_utils._relax_complex("CIF")


def test_ligand_output_uses_clean_template_and_explicit_hydrogens(cofold_utils):
    pytest.importorskip("rdkit")
    from rdkit import Chem

    smiles = "CS(=O)(=O)N"
    template = Chem.MolFromSmiles(smiles)
    lig_atoms = [
        (atom.GetSymbol(), float(index) * 4.0, 0.0, 0.0)
        for index, atom in enumerate(template.GetAtoms())
    ]
    block = cofold_utils._atoms_to_sdf(lig_atoms, smiles)
    result = Chem.MolFromMolBlock(block, removeHs=False, sanitize=True)
    expected = Chem.AddHs(Chem.MolFromSmiles(smiles))

    assert result is not None
    assert result.GetNumAtoms() == expected.GetNumAtoms()
    assert sum(a.GetAtomicNum() == 1 for a in result.GetAtoms()) > 0
    assert sum(a.GetNumRadicalElectrons() for a in result.GetAtoms()) == 0
    conformer = result.GetConformer()
    assert all(
        math.isfinite(value)
        for atom_index in range(result.GetNumAtoms())
        for value in tuple(conformer.GetAtomPosition(atom_index))
    )
