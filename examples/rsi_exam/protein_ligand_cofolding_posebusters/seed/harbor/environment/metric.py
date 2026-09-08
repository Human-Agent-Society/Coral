#!/usr/bin/env python3
"""Trusted scientific metric for protein-ligand co-folding.

This module never imports or executes a submission.  The verifier's isolated
child produces one bounded JSON prediction for the current anonymized case;
the trusted parent loads it with :func:`load_prediction_json` and calls
:func:`score_pose` while the crystal structures remain parent-only.

A case succeeds exactly when the predicted ligand, after sequence-aware
protein-frame alignment, passes every binary check in PoseBusters 0.3.1's
``redock`` configuration, including its heavy-atom ``RMSD <= 2 A`` check.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import stat
import unicodedata
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


POSEBUSTERS_VERSION = "0.3.1"
RMSD_THRESHOLD_ANGSTROM = 2.0

MAX_PREDICTION_JSON_BYTES = 24 * 1024 * 1024
MAX_PROTEIN_PDB_BYTES = 16 * 1024 * 1024
MAX_LIGAND_SDF_BYTES = 4 * 1024 * 1024
MAX_PROTEIN_CHAINS = 8
MAX_LIGAND_ATOMS = 4096
MAX_PREDICTED_LIGAND_ATOMS = 192
MAX_STRUCTURE_COORDINATE_ABS = 100_000.0
MIN_SEQUENCE_IDENTITY = 0.80
# Protein coordinates define the frame in which ligand RMSD is measured.  A
# partial-domain prediction must not be usable as an alignment shortcut around
# full-complex co-folding, while allowing a small number of unresolved termini.
MIN_SEQUENCE_COVERAGE = 0.95
MIN_CA_COVERAGE = 0.95

# PoseBusters==0.3.1, config/redock.yml, in declared order.  Keeping the
# complete schema explicit prevents missing/renamed columns from turning into
# a vacuous all([]) pass.
POSEBUSTERS_SUCCESS_COLUMNS = (
    "mol_pred_loaded",
    "mol_true_loaded",
    "mol_cond_loaded",
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "molecular_formula",
    "molecular_bonds",
    "double_bond_stereochemistry",
    "tetrahedral_chirality",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "double_bond_flatness",
    "internal_energy",
    "protein-ligand_maximum_distance",
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
    "rmsd_≤_2å",
)
RMSD_SUCCESS_COLUMN = "rmsd_≤_2å"
POSEBUSTERS_VALIDITY_COLUMNS = tuple(
    column for column in POSEBUSTERS_SUCCESS_COLUMNS if column != RMSD_SUCCESS_COLUMN
)

# ``PoseBusters.bust`` returns flattened display labels in 0.3.1.  These
# aliases also accept the library's raw two-level (module, output) form, which
# is useful for defensive parsing and focused tests.
_RAW_POSEBUSTERS_COLUMNS = {
    ("loading", "mol_pred_loaded"): "mol_pred_loaded",
    ("loading", "mol_true_loaded"): "mol_true_loaded",
    ("loading", "mol_cond_loaded"): "mol_cond_loaded",
    ("chemistry", "passes_rdkit_sanity_checks"): "sanitization",
    ("chemistry", "inchi_convertible"): "inchi_convertible",
    ("chemistry", "all_atoms_connected"): "all_atoms_connected",
    ("chemistry", "formula"): "molecular_formula",
    ("chemistry", "connections"): "molecular_bonds",
    ("chemistry", "stereo_dbond"): "double_bond_stereochemistry",
    ("chemistry", "stereo_tetrahedral"): "tetrahedral_chirality",
    ("geometry", "bond_lengths_within_bounds"): "bond_lengths",
    ("geometry", "bond_angles_within_bounds"): "bond_angles",
    ("geometry", "no_internal_clash"): "internal_steric_clash",
    ("ring_flatness", "flatness_passes"): "aromatic_ring_flatness",
    ("double_bond_flatness", "flatness_passes"): "double_bond_flatness",
    ("energy_ratio", "energy_ratio_passes"): "internal_energy",
    ("distance_to_protein", "not_too_far_away"): "protein-ligand_maximum_distance",
    ("distance_to_protein", "no_clashes"): "minimum_distance_to_protein",
    ("distance_to_organic_cofactors", "no_clashes"): "minimum_distance_to_organic_cofactors",
    ("distance_to_inorganic_cofactors", "no_clashes"): "minimum_distance_to_inorganic_cofactors",
    ("distance_to_waters", "no_clashes"): "minimum_distance_to_waters",
    ("volume_overlap_with_protein", "no_volume_clash"): "volume_overlap_with_protein",
    ("volume_overlap_with_organic_cofactors", "no_volume_clash"): "volume_overlap_with_organic_cofactors",
    ("volume_overlap_with_inorganic_cofactors", "no_volume_clash"): "volume_overlap_with_inorganic_cofactors",
    ("volume_overlap_with_waters", "no_volume_clash"): "volume_overlap_with_waters",
    ("rmsd", "rmsd_within_threshold"): RMSD_SUCCESS_COLUMN,
}


class MetricError(ValueError):
    """A malformed prediction, incompatible metric environment, or invalid report."""


@dataclass(frozen=True)
class ProteinResidue:
    one_letter_code: str
    ca: tuple[float, float, float] | None


@dataclass(frozen=True)
class ProteinChain:
    chain_id: str
    residues: tuple[ProteinResidue, ...]

    @property
    def sequence(self) -> str:
        return "".join(residue.one_letter_code for residue in self.residues)


@dataclass(frozen=True)
class SequenceAlignment:
    expected_to_observed: Mapping[int, int]
    matches: int
    aligned: int
    identity: float
    expected_coverage: float


@dataclass(frozen=True)
class ProteinTransform:
    rotation: Any
    translation: Any
    matched_ca_count: int
    alignment_rmsd: float


@dataclass(frozen=True)
class PoseScore:
    passed: bool
    pb_valid: bool
    rmsd_within_2a: bool
    checks: Mapping[str, bool]
    matched_ca_count: int
    protein_alignment_rmsd: float


def _reject_json_constant(token: str) -> None:
    raise MetricError(f"non-finite JSON constant is forbidden: {token}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MetricError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_prediction(prediction: Mapping[str, Any]) -> dict[str, str]:
    """Validate the typed child output and return an immutable-by-copy payload."""
    if not isinstance(prediction, Mapping):
        raise MetricError("prediction must be a JSON object")

    expected = {"protein_pdb", "ligand_sdf"}
    actual_keys = list(prediction)
    if any(not isinstance(key, str) for key in actual_keys):
        raise MetricError("prediction keys must be strings")
    actual = set(actual_keys)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise MetricError(f"prediction keys mismatch (missing={missing}, extra={extra})")

    limits = {
        "protein_pdb": MAX_PROTEIN_PDB_BYTES,
        "ligand_sdf": MAX_LIGAND_SDF_BYTES,
    }
    clean: dict[str, str] = {}
    for key in sorted(expected):
        value = prediction[key]
        if not isinstance(value, str):
            raise MetricError(f"{key} must be a string")
        if not value.strip():
            raise MetricError(f"{key} must not be empty")
        if "\x00" in value:
            raise MetricError(f"{key} contains a NUL byte")
        size = len(value.encode("utf-8"))
        if size > limits[key]:
            raise MetricError(f"{key} is too large ({size} > {limits[key]} bytes)")
        clean[key] = value
    return clean

def _finite_bounded_coordinate(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise MetricError("structure contains an invalid coordinate") from exc
    if not math.isfinite(value) or abs(value) > MAX_STRUCTURE_COORDINATE_ABS:
        raise MetricError("structure contains an invalid coordinate")
    return value


def _sdf_v2000_counts(text: str) -> tuple[int, int, list[str]]:
    lines = text.splitlines()
    if len(lines) < 4 or any(len(line) > 4096 for line in lines):
        raise MetricError("SDF has an invalid bounded text layout")
    counts = lines[3]
    if len(counts) < 6 or "V2000" not in counts:
        raise MetricError("SDF must use the bounded V2000 layout")
    try:
        atoms = int(counts[0:3])
        bonds = int(counts[3:6])
    except ValueError as exc:
        raise MetricError("SDF counts line is malformed") from exc
    if atoms < 1 or bonds < 0 or len(lines) < 4 + atoms + bonds:
        raise MetricError("SDF counts are invalid")
    return atoms, bonds, lines


def _trusted_ligand_atom_count(path: str | os.PathLike[str]) -> int:
    source = Path(path)
    try:
        info = source.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise MetricError("crystal ligand must be a regular file")
        if info.st_size <= 0 or info.st_size > MAX_LIGAND_SDF_BYTES:
            raise MetricError("crystal ligand size is invalid")
        text = source.read_text(encoding="utf-8")
    except MetricError:
        raise
    except (OSError, UnicodeError) as exc:
        raise MetricError("crystal ligand cannot be read") from exc
    atoms, _, _ = _sdf_v2000_counts(text)
    return atoms


def validate_prediction_work_bounds(
    prediction: Mapping[str, Any],
    *,
    expected_chains: Sequence[Mapping[str, Any]],
    crystal_ligand_path: str | os.PathLike[str],
) -> dict[str, str]:
    """Apply the same task-relative parser-work bounds on both splits."""
    clean = validate_prediction(prediction)
    expected_lengths = [len(str(chain["sequence"])) for chain in expected_chains]
    expected_total = sum(expected_lengths)
    if not expected_lengths or expected_total <= 0:
        raise MetricError("expected protein sequence is empty")

    pdb_lines = clean["protein_pdb"].splitlines()
    atom_limit = max(4096, expected_total * 40 + 1024)
    line_limit = atom_limit * 4 + 2048
    if len(pdb_lines) > line_limit or any(len(line) > 4096 for line in pdb_lines):
        raise MetricError("protein PDB exceeds task-relative work bounds")

    atom_count = 0
    residue_runs = 0
    residues_per_chain: dict[str, int] = {}
    chain_ids: set[str] = set()
    previous_residue: tuple[str, str, str, str] | None = None
    for line in pdb_lines:
        if line.startswith(("TER", "MODEL", "ENDMDL")):
            previous_residue = None
            continue
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_count += 1
        if atom_count > atom_limit or len(line) < 54:
            raise MetricError("protein PDB exceeds task-relative atom bounds")
        _finite_bounded_coordinate(line[30:38])
        _finite_bounded_coordinate(line[38:46])
        _finite_bounded_coordinate(line[46:54])
        chain_id = line[21:22]
        residue_key = (chain_id, line[22:26], line[26:27], line[17:20])
        chain_ids.add(chain_id)
        if residue_key != previous_residue:
            residue_runs += 1
            residues_per_chain[chain_id] = residues_per_chain.get(chain_id, 0) + 1
            previous_residue = residue_key

    if atom_count < 1 or residue_runs < 1:
        raise MetricError("protein PDB contains no bounded protein structure")
    residue_limit = expected_total * 2 + 32
    per_chain_limit = max(expected_lengths) * 2 + 32
    if (
        residue_runs > residue_limit
        or len(chain_ids) > len(expected_lengths)
        or any(value > per_chain_limit for value in residues_per_chain.values())
    ):
        raise MetricError("protein PDB exceeds task-relative residue bounds")

    ligand_atoms, ligand_bonds, ligand_lines = _sdf_v2000_counts(clean["ligand_sdf"])
    crystal_atoms = _trusted_ligand_atom_count(crystal_ligand_path)
    ligand_limit = min(
        MAX_PREDICTED_LIGAND_ATOMS,
        max(64, crystal_atoms * 5 + 16),
    )
    if ligand_atoms > ligand_limit or ligand_bonds > max(256, ligand_atoms * 8):
        raise MetricError("ligand SDF exceeds task-relative graph bounds")
    if len(ligand_lines) > ligand_atoms + ligand_bonds + 512:
        raise MetricError("ligand SDF exceeds task-relative line bounds")
    for line in ligand_lines[4 : 4 + ligand_atoms]:
        if len(line) < 30:
            raise MetricError("ligand SDF atom line is malformed")
        _finite_bounded_coordinate(line[0:10])
        _finite_bounded_coordinate(line[10:20])
        _finite_bounded_coordinate(line[20:30])
    return clean


def load_prediction_json(path: str | os.PathLike[str]) -> dict[str, str]:
    """Load one regular, non-symlinked, size-bounded child result file."""
    result_path = Path(path)
    try:
        info = result_path.lstat()
    except OSError as exc:
        raise MetricError(f"cannot stat prediction JSON: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise MetricError("prediction JSON must be a regular non-symlink file")
    if info.st_size <= 0 or info.st_size > MAX_PREDICTION_JSON_BYTES:
        raise MetricError(
            f"prediction JSON size is invalid ({info.st_size} bytes; "
            f"limit {MAX_PREDICTION_JSON_BYTES})"
        )
    try:
        text = result_path.read_text(encoding="utf-8")
        parsed = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MetricError(f"invalid prediction JSON: {exc}") from exc
    return validate_prediction(parsed)


def _normalise_label(value: Any) -> str:
    label = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return "_".join(label.split())


def _normalise_posebusters_column(column: Any) -> str:
    if isinstance(column, tuple):
        parts = tuple(_normalise_label(part) for part in column if str(part).strip())
        for part in reversed(parts):
            if part in POSEBUSTERS_SUCCESS_COLUMNS:
                return part
        if len(parts) >= 2:
            raw_key = (parts[-2], parts[-1])
            if raw_key in _RAW_POSEBUSTERS_COLUMNS:
                return _RAW_POSEBUSTERS_COLUMNS[raw_key]
        return parts[-1] if parts else ""
    return _normalise_label(column)


def evaluate_posebusters_report(report: Any) -> tuple[bool, bool, dict[str, bool]]:
    """Parse exactly one PoseBusters row with strict schema and bool handling.

    Returns ``(pb_valid, rmsd_within_2a, checks)``.  Missing/duplicate columns,
    empty or multi-row reports, ``NA``, integers, and strings all fail closed.
    Python ``bool`` and NumPy ``bool_`` are accepted.
    """
    try:
        row_count = int(report.shape[0])
        columns = list(report.columns)
    except Exception as exc:
        raise MetricError("PoseBusters report is not a DataFrame-like table") from exc
    if row_count != 1:
        raise MetricError(f"PoseBusters must return exactly one row, got {row_count}")

    normalised: dict[str, Any] = {}
    row = report.iloc[0]
    for original in columns:
        name = _normalise_posebusters_column(original)
        if name in normalised:
            raise MetricError(f"duplicate PoseBusters column after normalisation: {name}")
        normalised[name] = row[original]

    missing = [name for name in POSEBUSTERS_SUCCESS_COLUMNS if name not in normalised]
    if missing:
        raise MetricError(f"PoseBusters report is missing required columns: {missing}")

    import numpy as np

    checks: dict[str, bool] = {}
    for name in POSEBUSTERS_SUCCESS_COLUMNS:
        value = normalised[name]
        if not isinstance(value, (bool, np.bool_)):
            raise MetricError(
                f"PoseBusters column {name!r} must contain a boolean, got {type(value).__name__}"
            )
        checks[name] = bool(value)

    pb_valid = all(checks[name] for name in POSEBUSTERS_VALIDITY_COLUMNS)
    rmsd_within_2a = checks[RMSD_SUCCESS_COLUMN]
    return pb_valid, rmsd_within_2a, checks


def _validate_posebusters_config(config: Mapping[str, Any]) -> None:
    modules = config.get("modules")
    if not isinstance(modules, list):
        raise MetricError("PoseBusters redock configuration has no modules list")

    declared: list[str] = []
    rmsd_threshold: float | None = None
    for module in modules:
        if not isinstance(module, Mapping):
            raise MetricError("PoseBusters redock module is malformed")
        module_name = _normalise_label(module.get("name", ""))
        rename = module.get("rename_outputs", {})
        suffix = module.get("rename_suffix", "")
        for raw_name in module.get("chosen_binary_test_output", []):
            display_name = rename.get(raw_name, f"{raw_name}{suffix}")
            declared.append(_normalise_label(display_name))
        if module.get("function") == "rmsd":
            value = module.get("parameters", {}).get("rmsd_threshold")
            try:
                rmsd_threshold = float(value)
            except (TypeError, ValueError) as exc:
                raise MetricError("PoseBusters RMSD threshold is malformed") from exc
        if not module_name:
            raise MetricError("PoseBusters redock module has no name")

    if tuple(declared) != POSEBUSTERS_SUCCESS_COLUMNS:
        raise MetricError("PoseBusters redock binary schema differs from pinned 0.3.1 schema")
    if rmsd_threshold != RMSD_THRESHOLD_ANGSTROM:
        raise MetricError(
            f"PoseBusters RMSD threshold mismatch: {rmsd_threshold} != {RMSD_THRESHOLD_ANGSTROM}"
        )


def _new_posebusters() -> Any:
    try:
        installed = metadata.version("posebusters")
    except metadata.PackageNotFoundError as exc:
        raise MetricError("PoseBusters is not installed") from exc
    if installed != POSEBUSTERS_VERSION:
        raise MetricError(
            f"PoseBusters version mismatch: installed {installed}, required {POSEBUSTERS_VERSION}"
        )
    from posebusters import PoseBusters

    instance = PoseBusters(config="redock")
    _validate_posebusters_config(instance.config)
    return instance


def _one_letter_code(residue_info: Any) -> str:
    code = getattr(residue_info, "one_letter_code", "X")
    code = str(code).strip().upper()
    return code if len(code) == 1 else "X"


def _read_protein_chains(pdb_path: str | os.PathLike[str]) -> tuple[ProteinChain, ...]:
    import gemmi

    try:
        structure = gemmi.read_structure(str(pdb_path))
    except Exception as exc:
        raise MetricError(f"protein PDB cannot be parsed: {exc}") from exc
    if len(structure) != 1:
        raise MetricError(f"protein PDB must contain exactly one model, got {len(structure)}")

    chains: list[ProteinChain] = []
    seen_chain_ids: set[str] = set()
    for chain in structure[0]:
        residues: list[ProteinResidue] = []
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if not (info and info.is_amino_acid()):
                continue
            atom = residue.find_atom("CA", "*")
            ca: tuple[float, float, float] | None = None
            if atom is not None:
                candidate = (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))
                if not all(math.isfinite(value) for value in candidate):
                    raise MetricError(f"protein PDB chain {chain.name!r} contains non-finite CA coordinates")
                ca = candidate
            residues.append(ProteinResidue(_one_letter_code(info), ca))
        if not residues:
            continue
        chain_id = str(chain.name)
        if chain_id in seen_chain_ids:
            raise MetricError(f"protein PDB has duplicate amino-acid chain ID: {chain_id!r}")
        seen_chain_ids.add(chain_id)
        chains.append(ProteinChain(chain_id, tuple(residues)))

    if not chains:
        raise MetricError("protein PDB contains no amino-acid chains")
    if len(chains) > MAX_PROTEIN_CHAINS:
        raise MetricError(f"protein PDB contains too many chains ({len(chains)} > {MAX_PROTEIN_CHAINS})")
    return tuple(chains)


def _validate_expected_chains(expected_chains: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    if not isinstance(expected_chains, Sequence) or isinstance(expected_chains, (str, bytes)):
        raise MetricError("expected protein_chains must be a sequence")
    if not expected_chains or len(expected_chains) > MAX_PROTEIN_CHAINS:
        raise MetricError("expected protein_chains count is invalid")

    result: list[tuple[str, str]] = []
    ids: set[str] = set()
    for chain in expected_chains:
        if not isinstance(chain, Mapping):
            raise MetricError("expected protein chain must be an object")
        if set(chain) != {"chain_id", "sequence"}:
            raise MetricError("expected protein chain must contain only chain_id and sequence")
        chain_id = chain["chain_id"]
        sequence = chain["sequence"]
        if not isinstance(chain_id, str) or not chain_id or chain_id in ids:
            raise MetricError(f"expected protein chain ID is invalid or duplicated: {chain_id!r}")
        if not isinstance(sequence, str) or not sequence:
            raise MetricError(f"expected protein chain {chain_id!r} has no sequence")
        sequence = "".join(sequence.split()).upper()
        if not sequence.isalpha():
            raise MetricError(f"expected protein chain {chain_id!r} has an invalid sequence")
        ids.add(chain_id)
        result.append((chain_id, sequence))
    return tuple(result)


def _align_sequence(observed: str, expected: str) -> SequenceAlignment:
    """Needleman-Wunsch mapping from expected positions to observed residues."""
    import numpy as np

    if not observed or not expected:
        raise MetricError("cannot align an empty protein sequence")
    n, m = len(observed), len(expected)
    score = np.empty((n + 1, m + 1), dtype=np.int32)
    trace = np.empty((n + 1, m + 1), dtype=np.uint8)
    score[:, 0] = np.arange(n + 1, dtype=np.int32) * -2
    score[0, :] = np.arange(m + 1, dtype=np.int32) * -2
    trace[:, 0] = 1
    trace[0, :] = 2
    trace[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = int(score[i - 1, j - 1]) + (2 if observed[i - 1] == expected[j - 1] else -1)
            up = int(score[i - 1, j]) - 2
            left = int(score[i, j - 1]) - 2
            best = max(diagonal, up, left)
            score[i, j] = best
            trace[i, j] = 0 if diagonal == best else (1 if up == best else 2)

    mapping: dict[int, int] = {}
    matches = aligned = 0
    i, j = n, m
    while i or j:
        direction = int(trace[i, j])
        if i and j and direction == 0:
            i -= 1
            j -= 1
            mapping[j] = i
            aligned += 1
            matches += int(observed[i] == expected[j])
        elif i and (not j or direction == 1):
            i -= 1
        elif j:
            j -= 1
        else:  # defensive guard against a malformed traceback table
            raise MetricError("protein sequence alignment traceback failed")

    identity = matches / aligned if aligned else 0.0
    coverage = aligned / m
    return SequenceAlignment(mapping, matches, aligned, identity, coverage)


def _candidate_assignments(
    observed: Sequence[ProteinChain],
    expected: Sequence[tuple[str, str]],
    *,
    allow_extra_observed: bool = False,
) -> tuple[tuple[int, ...], ...]:
    if len(observed) < len(expected) or (
        not allow_extra_observed and len(observed) != len(expected)
    ):
        raise MetricError(
            f"protein chain count mismatch (observed={len(observed)}, expected={len(expected)})"
        )

    alignments = {
        (observed_index, expected_index): _align_sequence(chain.sequence, expected_sequence)
        for observed_index, chain in enumerate(observed)
        for expected_index, (_, expected_sequence) in enumerate(expected)
    }
    candidates: list[tuple[tuple[int, int], tuple[int, ...]]] = []
    for assignment in itertools.permutations(
        range(len(observed)), len(expected)
    ):
        selected = [alignments[(observed_index, expected_index)] for expected_index, observed_index in enumerate(assignment)]
        if any(
            alignment.identity < MIN_SEQUENCE_IDENTITY
            or alignment.expected_coverage < MIN_SEQUENCE_COVERAGE
            for alignment in selected
        ):
            continue
        quality = (sum(a.matches for a in selected), sum(a.aligned for a in selected))
        candidates.append((quality, assignment))
    if not candidates:
        raise MetricError("no one-to-one protein chain assignment passes sequence identity/coverage")
    best_quality = max(quality for quality, _ in candidates)
    return tuple(assignment for quality, assignment in candidates if quality == best_quality)


def _coordinates_by_expected_position(
    chain: ProteinChain, expected_sequence: str
) -> tuple[dict[int, tuple[float, float, float]], SequenceAlignment]:
    alignment = _align_sequence(chain.sequence, expected_sequence)
    coordinates = {
        expected_index: chain.residues[observed_index].ca
        for expected_index, observed_index in alignment.expected_to_observed.items()
        if chain.residues[observed_index].ca is not None
    }
    return coordinates, alignment


def _kabsch_transform(predicted: Any, crystal: Any) -> tuple[Any, Any, float]:
    import numpy as np

    predicted = np.asarray(predicted, dtype=float)
    crystal = np.asarray(crystal, dtype=float)
    if predicted.shape != crystal.shape or predicted.ndim != 2 or predicted.shape[1:] != (3,):
        raise MetricError("matched CA coordinate arrays have incompatible shapes")
    if len(predicted) < 3:
        raise MetricError(f"at least three matched CA atoms are required, got {len(predicted)}")
    if not np.isfinite(predicted).all() or not np.isfinite(crystal).all():
        raise MetricError("matched CA coordinates contain non-finite values")

    predicted_center = predicted.mean(axis=0)
    crystal_center = crystal.mean(axis=0)
    p_centered = predicted - predicted_center
    q_centered = crystal - crystal_center
    if np.linalg.matrix_rank(p_centered) < 2 or np.linalg.matrix_rank(q_centered) < 2:
        raise MetricError("matched CA atoms are degenerate and cannot define a stable rigid transform")

    u, _, vt = np.linalg.svd(p_centered.T @ q_centered)
    handedness = -1.0 if np.linalg.det(vt.T @ u.T) < 0 else 1.0
    rotation = vt.T @ np.diag([1.0, 1.0, handedness]) @ u.T
    translation = crystal_center - rotation @ predicted_center
    transformed = predicted @ rotation.T + translation
    residual = float(np.sqrt(np.mean(np.sum((transformed - crystal) ** 2, axis=1))))
    if not math.isfinite(residual):
        raise MetricError("protein alignment produced a non-finite residual")
    return rotation, translation, residual


def _superpose_proteins(
    predicted_chains: Sequence[ProteinChain],
    crystal_chains: Sequence[ProteinChain],
    expected_chains: Sequence[Mapping[str, Any]],
) -> ProteinTransform:
    """Match chains/residues by sequence and find the best protein-only frame.

    Sequence quality is optimized before geometry.  Geometry only resolves
    sequence-equivalent assignments (for example homomer chain permutations),
    so crystal ligand coordinates never influence chain selection.
    """
    import numpy as np

    expected = _validate_expected_chains(expected_chains)
    predicted_assignments = _candidate_assignments(predicted_chains, expected)
    # Deposited crystal structures can contain additional biological-assembly
    # chains that are not part of the supplied prediction target.  Select the
    # expected chains by sequence on the trusted crystal side only.  Submitted
    # predictions still have to contain exactly the declared chain count.
    crystal_assignments = _candidate_assignments(
        crystal_chains, expected, allow_extra_observed=True
    )

    best: ProteinTransform | None = None
    for predicted_assignment in predicted_assignments:
        for crystal_assignment in crystal_assignments:
            predicted_coordinates: list[tuple[float, float, float]] = []
            crystal_coordinates: list[tuple[float, float, float]] = []
            valid = True
            for expected_index, (_, expected_sequence) in enumerate(expected):
                pred_map, _ = _coordinates_by_expected_position(
                    predicted_chains[predicted_assignment[expected_index]], expected_sequence
                )
                crystal_map, _ = _coordinates_by_expected_position(
                    crystal_chains[crystal_assignment[expected_index]], expected_sequence
                )
                if len(pred_map) / len(expected_sequence) < MIN_CA_COVERAGE:
                    valid = False
                    break
                if len(crystal_map) / len(expected_sequence) < MIN_CA_COVERAGE:
                    valid = False
                    break
                common = sorted(set(pred_map) & set(crystal_map))
                if len(common) < 3 or len(common) / len(expected_sequence) < MIN_CA_COVERAGE:
                    valid = False
                    break
                predicted_coordinates.extend(pred_map[index] for index in common)
                crystal_coordinates.extend(crystal_map[index] for index in common)
            if not valid:
                continue
            try:
                rotation, translation, residual = _kabsch_transform(
                    np.asarray(predicted_coordinates), np.asarray(crystal_coordinates)
                )
            except MetricError:
                continue
            candidate = ProteinTransform(
                rotation=rotation,
                translation=translation,
                matched_ca_count=len(predicted_coordinates),
                alignment_rmsd=residual,
            )
            if best is None or candidate.alignment_rmsd < best.alignment_rmsd:
                best = candidate
    if best is None:
        raise MetricError("protein structures do not have sufficient sequence-matched CA coverage")
    return best


def _parse_ligand_sdf(sdf_text: str) -> Any:
    from rdkit import Chem
    import numpy as np

    records = sdf_text.split("$$$$")
    nonempty = [record for record in records if record.strip()]
    if len(nonempty) != 1:
        raise MetricError(f"ligand SDF must contain exactly one molecule, got {len(nonempty)}")
    try:
        molecule = Chem.MolFromMolBlock(
            nonempty[0], sanitize=False, removeHs=False, strictParsing=True
        )
    except Exception as exc:
        raise MetricError(f"ligand SDF cannot be parsed: {exc}") from exc
    if molecule is None:
        raise MetricError("ligand SDF cannot be parsed")
    if molecule.GetNumAtoms() <= 0 or molecule.GetNumAtoms() > MAX_LIGAND_ATOMS:
        raise MetricError(f"ligand atom count is invalid: {molecule.GetNumAtoms()}")
    if molecule.GetNumConformers() != 1:
        raise MetricError(f"ligand SDF must contain one conformer, got {molecule.GetNumConformers()}")
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    if coordinates.shape != (molecule.GetNumAtoms(), 3) or not np.isfinite(coordinates).all():
        raise MetricError("ligand SDF contains invalid or non-finite coordinates")
    return molecule


def _validate_ligand_identity(molecule: Any, expected_smiles: str) -> None:
    """Require the predicted coordinates to describe the supplied ligand graph."""
    from rdkit import Chem

    if not isinstance(expected_smiles, str) or not expected_smiles.strip():
        raise MetricError("expected ligand SMILES is invalid")
    try:
        expected = Chem.MolFromSmiles(expected_smiles, sanitize=True)
        observed = Chem.Mol(molecule)
        Chem.SanitizeMol(observed)
        if expected is None:
            raise ValueError("SMILES parser returned no molecule")
        expected_graph = Chem.MolToSmiles(
            Chem.RemoveHs(expected), canonical=True, isomericSmiles=False
        )
        observed_graph = Chem.MolToSmiles(
            Chem.RemoveHs(observed), canonical=True, isomericSmiles=False
        )
    except Exception as exc:
        raise MetricError("ligand graph cannot be normalized") from exc
    if observed_graph != expected_graph:
        raise MetricError("predicted ligand graph does not match the supplied SMILES")


def _write_aligned_ligand(molecule: Any, transform: ProteinTransform, output_path: Path) -> None:
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    import numpy as np

    aligned = Chem.Mol(molecule)
    conformer = aligned.GetConformer()
    for atom_index in range(aligned.GetNumAtoms()):
        point = conformer.GetAtomPosition(atom_index)
        coordinate = transform.rotation @ np.asarray([point.x, point.y, point.z]) + transform.translation
        if not np.isfinite(coordinate).all():
            raise MetricError("ligand alignment produced non-finite coordinates")
        conformer.SetAtomPosition(
            atom_index,
            Point3D(float(coordinate[0]), float(coordinate[1]), float(coordinate[2])),
        )
    try:
        Chem.MolToMolFile(aligned, str(output_path))
    except Exception as exc:
        raise MetricError(f"cannot write aligned ligand SDF: {exc}") from exc
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise MetricError("aligned ligand SDF was not written")


def score_pose(
    prediction: Mapping[str, Any],
    *,
    crystal_ligand_path: str | os.PathLike[str],
    crystal_protein_path: str | os.PathLike[str],
    expected_chains: Sequence[Mapping[str, Any]],
    expected_ligand_smiles: str,
    work_dir: str | os.PathLike[str],
    posebusters_factory: Callable[[], Any] | None = None,
) -> PoseScore:
    """Score one already-isolated prediction against trusted crystal files."""
    clean = validate_prediction_work_bounds(
        prediction,
        expected_chains=expected_chains,
        crystal_ligand_path=crystal_ligand_path,
    )
    work = Path(work_dir)
    if not work.is_dir():
        raise MetricError("metric work_dir must already exist")
    predicted_protein_path = work / "predicted_protein.pdb"
    aligned_ligand_path = work / "predicted_ligand_aligned.sdf"
    predicted_protein_path.write_text(clean["protein_pdb"], encoding="utf-8")

    predicted_chains = _read_protein_chains(predicted_protein_path)
    crystal_chains = _read_protein_chains(crystal_protein_path)
    transform = _superpose_proteins(predicted_chains, crystal_chains, expected_chains)
    ligand = _parse_ligand_sdf(clean["ligand_sdf"])
    _validate_ligand_identity(ligand, expected_ligand_smiles)
    _write_aligned_ligand(ligand, transform, aligned_ligand_path)

    posebusters = posebusters_factory() if posebusters_factory is not None else _new_posebusters()
    if not hasattr(posebusters, "config"):
        raise MetricError("PoseBusters instance has no configuration")
    _validate_posebusters_config(posebusters.config)
    try:
        report = posebusters.bust(
            mol_pred=str(aligned_ligand_path),
            mol_true=str(crystal_ligand_path),
            mol_cond=str(crystal_protein_path),
        )
    except Exception as exc:
        raise MetricError(f"PoseBusters evaluation failed: {exc}") from exc
    pb_valid, rmsd_within_2a, checks = evaluate_posebusters_report(report)
    return PoseScore(
        passed=pb_valid and rmsd_within_2a,
        pb_valid=pb_valid,
        rmsd_within_2a=rmsd_within_2a,
        checks=checks,
        matched_ca_count=transform.matched_ca_count,
        protein_alignment_rmsd=transform.alignment_rmsd,
    )


def success_rate(scores: Sequence[PoseScore]) -> float:
    """Return the unweighted fraction of successful independent complexes."""
    if not scores:
        raise MetricError("cannot aggregate an empty score list")
    value = sum(int(score.passed) for score in scores) / len(scores)
    if not math.isfinite(value):
        raise MetricError("aggregate success rate is non-finite")
    return value
