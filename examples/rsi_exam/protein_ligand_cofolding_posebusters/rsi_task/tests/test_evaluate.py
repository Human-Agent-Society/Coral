from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).with_name("evaluate.py")
SPEC = importlib.util.spec_from_file_location("protein_cofold_metric", MODULE_PATH)
metric = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = metric
assert SPEC.loader is not None
SPEC.loader.exec_module(metric)


def _flat_report(value=True):
    return pd.DataFrame(
        [[np.bool_(value) for _ in metric.POSEBUSTERS_SUCCESS_COLUMNS]],
        columns=metric.POSEBUSTERS_SUCCESS_COLUMNS,
    )


def _raw_multiindex_report():
    columns = pd.MultiIndex.from_tuples(list(metric._RAW_POSEBUSTERS_COLUMNS))
    return pd.DataFrame([[np.bool_(True) for _ in columns]], columns=columns)


def _chain(chain_id, sequence, coordinates):
    residues = tuple(
        metric.ProteinResidue(code, None if coordinate is None else tuple(coordinate))
        for code, coordinate in zip(sequence, coordinates, strict=True)
    )
    return metric.ProteinChain(chain_id, residues)


def _coordinates(count, offset=(0.0, 0.0, 0.0)):
    return [
        (
            float(index) + offset[0],
            float((index * index + index) % 5) + offset[1],
            float((index * 3 + index * index) % 7) + offset[2],
        )
        for index in range(count)
    ]


def _transform(coordinates):
    rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    translation = np.asarray((13.0, -7.0, 4.5))
    return [tuple(rotation @ np.asarray(point) + translation) for point in coordinates]


def test_posebusters_report_accepts_numpy_bool_and_full_flat_schema():
    pb_valid, rmsd_ok, checks = metric.evaluate_posebusters_report(_flat_report())
    assert pb_valid is True
    assert rmsd_ok is True
    assert tuple(checks) == metric.POSEBUSTERS_SUCCESS_COLUMNS
    assert all(checks.values())


def test_posebusters_report_accepts_raw_multiindex_schema():
    pb_valid, rmsd_ok, checks = metric.evaluate_posebusters_report(_raw_multiindex_report())
    assert pb_valid is True
    assert rmsd_ok is True
    assert len(checks) == 26


def test_posebusters_report_separates_validity_from_rmsd_failure():
    report = _flat_report()
    report.loc[0, metric.RMSD_SUCCESS_COLUMN] = np.bool_(False)
    pb_valid, rmsd_ok, checks = metric.evaluate_posebusters_report(report)
    assert pb_valid is True
    assert rmsd_ok is False
    assert checks[metric.RMSD_SUCCESS_COLUMN] is False


@pytest.mark.parametrize("bad_value", [1, "True", None, pd.NA, np.nan])
def test_posebusters_report_rejects_non_boolean_values(bad_value):
    report = _flat_report().astype(object)
    report.loc[0, "sanitization"] = bad_value
    with pytest.raises(metric.MetricError, match="must contain a boolean"):
        metric.evaluate_posebusters_report(report)


def test_posebusters_report_rejects_missing_empty_and_multiple_rows():
    with pytest.raises(metric.MetricError, match="missing required columns"):
        metric.evaluate_posebusters_report(_flat_report().drop(columns=["sanitization"]))
    with pytest.raises(metric.MetricError, match="exactly one row"):
        metric.evaluate_posebusters_report(_flat_report().iloc[:0])
    with pytest.raises(metric.MetricError, match="exactly one row"):
        metric.evaluate_posebusters_report(pd.concat([_flat_report(), _flat_report()]))


def test_prediction_json_rejects_duplicate_keys_extra_keys_and_symlink(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"protein_pdb":"A","protein_pdb":"B","ligand_sdf":"C"}', encoding="utf-8"
    )
    with pytest.raises(metric.MetricError, match="duplicate JSON key"):
        metric.load_prediction_json(duplicate)

    extra = tmp_path / "extra.json"
    extra.write_text(
        json.dumps({"protein_pdb": "A", "ligand_sdf": "B", "score": 1}), encoding="utf-8"
    )
    with pytest.raises(metric.MetricError, match="keys mismatch"):
        metric.load_prediction_json(extra)

    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"protein_pdb": "A", "ligand_sdf": "B"}), encoding="utf-8")
    link = tmp_path / "prediction-link.json"
    link.symlink_to(valid)
    with pytest.raises(metric.MetricError, match="non-symlink"):
        metric.load_prediction_json(link)


def test_sequence_alignment_matches_positions_across_internal_deletion():
    alignment = metric._align_sequence("ACEFG", "ACDEFG")
    assert alignment.matches == 5
    assert alignment.identity == 1.0
    assert alignment.expected_to_observed == {0: 0, 1: 1, 3: 2, 4: 3, 5: 4}


def test_superposition_uses_sequence_positions_not_flat_truncation():
    expected_sequence = "ACDEFGHIKLMNPQRSTVWY"
    predicted_full = _coordinates(len(expected_sequence))
    missing_index = 3
    predicted_sequence = expected_sequence[:missing_index] + expected_sequence[missing_index + 1 :]
    predicted_coordinates = predicted_full[:missing_index] + predicted_full[missing_index + 1 :]
    crystal_coordinates = _transform(predicted_full)

    transform = metric._superpose_proteins(
        (_chain("prediction", predicted_sequence, predicted_coordinates),),
        (_chain("crystal", expected_sequence, crystal_coordinates),),
        ({"chain_id": "A", "sequence": expected_sequence},),
    )
    assert transform.matched_ca_count == len(expected_sequence) - 1
    assert transform.alignment_rmsd < 1e-10


def test_superposition_resolves_homomer_chain_permutation_with_protein_geometry():
    sequence = "ACDEFGHIK"
    first = _coordinates(len(sequence), offset=(0.0, 0.0, 0.0))
    second = [
        (x + 21.0, y * 1.7 - 3.0, z * 0.6 + 8.0)
        for x, y, z in _coordinates(len(sequence), offset=(2.0, 1.0, 0.0))
    ]
    crystal_first = _transform(first)
    crystal_second = _transform(second)

    transform = metric._superpose_proteins(
        (_chain("X", sequence, first), _chain("Y", sequence, second)),
        (_chain("A", sequence, crystal_second), _chain("B", sequence, crystal_first)),
        (
            {"chain_id": "A", "sequence": sequence},
            {"chain_id": "B", "sequence": sequence},
        ),
    )
    assert transform.matched_ca_count == 2 * len(sequence)
    assert transform.alignment_rmsd < 1e-10


def test_superposition_selects_expected_crystal_chain_but_rejects_extra_prediction_chain():
    expected_sequence = "ACDEFGHIK"
    extra_sequence = "LMNPQRSTV"
    expected_coordinates = _coordinates(len(expected_sequence))
    crystal_expected = _transform(expected_coordinates)
    crystal_extra = _transform(
        _coordinates(len(extra_sequence), offset=(30.0, -5.0, 2.0))
    )
    expected = ({"chain_id": "A", "sequence": expected_sequence},)

    transform = metric._superpose_proteins(
        (_chain("prediction", expected_sequence, expected_coordinates),),
        (
            _chain("crystal-extra", extra_sequence, crystal_extra),
            _chain("crystal-target", expected_sequence, crystal_expected),
        ),
        expected,
    )
    assert transform.matched_ca_count == len(expected_sequence)
    assert transform.alignment_rmsd < 1e-10

    with pytest.raises(metric.MetricError, match="chain count mismatch"):
        metric._superpose_proteins(
            (
                _chain("prediction", expected_sequence, expected_coordinates),
                _chain("prediction-extra", extra_sequence, crystal_extra),
            ),
            (_chain("crystal-target", expected_sequence, crystal_expected),),
            expected,
        )


def test_superposition_fails_closed_on_missing_chain_or_low_ca_coverage():
    sequence = "ACDEFGHIK"
    full = _coordinates(len(sequence))
    expected = (
        {"chain_id": "A", "sequence": sequence},
        {"chain_id": "B", "sequence": sequence},
    )
    with pytest.raises(metric.MetricError, match="chain count mismatch"):
        metric._superpose_proteins(
            (_chain("A", sequence, full),),
            (_chain("A", sequence, full), _chain("B", sequence, full)),
            expected,
        )

    sparse = [point if index < 3 else None for index, point in enumerate(full)]
    with pytest.raises(metric.MetricError, match="sufficient sequence-matched CA coverage"):
        metric._superpose_proteins(
            (_chain("A", sequence, sparse),),
            (_chain("A", sequence, full),),
            ({"chain_id": "A", "sequence": sequence},),
        )


def test_superposition_rejects_partial_domain_alignment_shortcut():
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    full = _coordinates(len(sequence))
    partial_sequence = sequence[: len(sequence) // 2]
    partial_coordinates = full[: len(partial_sequence)]

    with pytest.raises(metric.MetricError, match="sequence identity/coverage"):
        metric._superpose_proteins(
            (_chain("A", partial_sequence, partial_coordinates),),
            (_chain("A", sequence, full),),
            ({"chain_id": "A", "sequence": sequence},),
        )


def test_success_rate_is_unweighted_and_rejects_empty_input():
    def score(passed):
        return metric.PoseScore(passed, passed, passed, {}, 3, 0.0)

    assert metric.success_rate([score(True), score(False), score(True)]) == pytest.approx(2 / 3)
    with pytest.raises(metric.MetricError, match="empty"):
        metric.success_rate([])


def test_score_pose_runs_with_real_parsers_and_pinned_config(tmp_path):
    pytest.importorskip("gemmi")
    pytest.importorskip("rdkit")
    posebusters_module = pytest.importorskip("posebusters")

    sequence = "ACDEFG"
    coordinates = _coordinates(len(sequence))
    residue_names = ("ALA", "CYS", "ASP", "GLU", "PHE", "GLY")
    pdb_lines = []
    for serial, (residue_name, point) in enumerate(zip(residue_names, coordinates, strict=True), 1):
        x, y, z = point
        pdb_lines.append(
            f"ATOM  {serial:5d}  CA  {residue_name:>3s} A{serial:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        )
    pdb_text = "\n".join(pdb_lines + ["TER", "END", ""])
    sdf_text = """one-carbon ligand
  metric-test

  1  0  0  0  0  0  0  0  0  0  1 V2000
    1.0000    2.0000    3.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""
    crystal_protein = tmp_path / "crystal.pdb"
    crystal_ligand = tmp_path / "crystal.sdf"
    metric_work = tmp_path / "metric-work"
    crystal_protein.write_text(pdb_text, encoding="utf-8")
    crystal_ligand.write_text(sdf_text, encoding="utf-8")
    metric_work.mkdir()

    class FakePoseBusters:
        def __init__(self):
            self.config = posebusters_module.PoseBusters(config="redock").config

        def bust(self, **_kwargs):
            return _flat_report()

    result = metric.score_pose(
        {"protein_pdb": pdb_text, "ligand_sdf": sdf_text},
        crystal_ligand_path=crystal_ligand,
        crystal_protein_path=crystal_protein,
        expected_chains=({"chain_id": "A", "sequence": sequence},),
        expected_ligand_smiles="C",
        work_dir=metric_work,
        posebusters_factory=FakePoseBusters,
    )
    assert result.passed is True
    assert result.matched_ca_count == len(sequence)
    assert result.protein_alignment_rmsd < 1e-10


def test_ligand_identity_rejects_a_different_connectivity_graph():
    pytest.importorskip("rdkit")
    molecule = metric._parse_ligand_sdf(
        """one-carbon ligand
  metric-test

  1  0  0  0  0  0  0  0  0  0  1 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""
    )
    metric._validate_ligand_identity(molecule, "C")
    with pytest.raises(metric.MetricError, match="does not match"):
        metric._validate_ligand_identity(molecule, "CC")
