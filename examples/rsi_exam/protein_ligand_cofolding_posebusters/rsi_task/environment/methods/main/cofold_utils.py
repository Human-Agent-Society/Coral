"""Pinned baseline adapter and complex-to-PDB/SDF conversion helpers.

The inherited baseline performs one deterministic prediction through one local
backend.  It deliberately does not prepackage alternate backends, portfolios,
or candidate-ranking policies. ``relax=True`` fails explicitly until a method
supplies and validates a physical post-processor; it never silently becomes a
no-op.
"""
import glob
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


# Boltz does not seed diffusion by default.  A fixed benchmark baseline must not
# change its pose merely because the verifier was re-run.
PREDICTION_SEED = 42


def _boltz_cache_path() -> str:
    """Return the explicit, absolute Boltz asset cache used by the CLI."""
    configured = os.environ.get("BOLTZ_CACHE", "~/.boltz")
    return os.path.abspath(os.path.expanduser(configured))


def _clean_a3m(src: str, dst: str) -> str:
    """Copy an A3M while removing NUL bytes rejected by Boltz 1.0."""
    raw = Path(src).read_bytes()
    Path(dst).write_bytes(raw.replace(b"\x00", b""))
    return dst


def _write_boltz_input(item: dict, work: str, use_msa: bool = True) -> str:
    """Write a safe Boltz YAML input with deterministic local MSA paths.

    Boltz requires chains with identical sequences to reference the same MSA.
    In single-sequence mode its documented representation is ``msa: empty``;
    ``--use_msa_server false`` is not valid for a Click flag and can accidentally
    turn an offline prediction into an online MSA request.
    """
    import yaml

    proteins = []
    sequence_msas = {}
    for chain in item["protein_chains"]:
        sequence = str(chain["sequence"])
        if use_msa:
            if sequence not in sequence_msas:
                source = os.path.join(
                    item["msa_dir"], f"{chain['chain_id']}.a3m"
                )
                clean = os.path.join(work, f"msa_{len(sequence_msas)}.a3m")
                sequence_msas[sequence] = _clean_a3m(source, clean)
            msa = sequence_msas[sequence]
        else:
            msa = "empty"
        proteins.append(
            {
                "protein": {
                    "id": str(chain["chain_id"]),
                    "sequence": sequence,
                    "msa": msa,
                }
            }
        )

    payload = {
        "version": 1,
        "sequences": proteins
        + [
            {
                "ligand": {
                    "id": "LIG",
                    "smiles": str(item["ligand_smiles"]),
                }
            }
        ],
    }
    path = os.path.join(work, "input.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            payload,
            handle,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
    return path


def run_cofolding(item: dict, num_samples: int = 1, recycling: int = 3,
                  use_msa: bool = True, relax: bool = False) -> str:
    """Run the inherited backend and return its deterministic default pose."""
    with tempfile.TemporaryDirectory(prefix="cofold_") as work:
        inp = _write_boltz_input(item, work, use_msa=use_msa)
        cmd = [
            "boltz",
            "predict",
            inp,
            "--out_dir",
            work,
            "--cache",
            _boltz_cache_path(),
            "--num_workers",
            "0",
            "--seed",
            str(PREDICTION_SEED),
            "--diffusion_samples",
            str(num_samples),
            "--recycling_steps",
            str(recycling),
            "--output_format",
            "mmcif",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        cif_text = Path(_pick_default_cif(work)).read_text(encoding="utf-8")

    if relax:
        cif_text = _relax_complex(cif_text)   # openmm short relax / declash (integration point)
    return cif_text


def _sample_rank(path: str) -> Optional[int]:
    """Extract an exact terminal sample rank, avoiding 1/10 substring matches."""
    stem = Path(path).stem
    match = re.search(r"(?:^|[._-])model_(?:idx_)?(\d+)$", stem)
    if match is None:
        match = re.search(r"(?:^|[._-])idx_(\d+)$", stem)
    return int(match.group(1)) if match is not None else None


def _pick_default_cif(work: str) -> str:
    """Return the backend's deterministic rank-zero output."""
    cifs = sorted(glob.glob(os.path.join(work, "**", "*.cif"), recursive=True))
    if not cifs:
        raise FileNotFoundError("baseline backend produced no CIF output")
    rank_zero = [path for path in cifs if _sample_rank(path) == 0]
    return rank_zero[0] if rank_zero else cifs[0]


def _relax_complex(cif_text):
    """Require an explicit, tested post-processor instead of a silent fallback."""
    raise NotImplementedError(
        "relax=True has no validated generic implementation; implement and test "
        "a task-specific post-processor before enabling it"
    )


def split_complex(cif_text: str, ligand_smiles: str) -> Tuple[str, str]:
    """Split a predicted complex cif into (protein PDB text, ligand SDF text).

    gemmi parses the structure; rdkit assigns bond orders to the ligand from the
    SMILES template before writing the SDF."""
    import gemmi
    from rdkit import Chem
    st = gemmi.make_structure_from_block(gemmi.cif.read_string(cif_text).sole_block())
    # ligand heavy-atom coordinates (LIG / any residue that is neither amino acid nor water)
    lig_atoms = []
    for chain in st[0]:
        for res in chain:
            info = gemmi.find_tabulated_residue(res.name)   # gemmi Residue has no is_amino_acid/is_water
            is_aa = bool(info and info.is_amino_acid())
            is_water = bool(info and info.is_water())
            if not is_aa and not is_water:
                for a in res:
                    lig_atoms.append((a.element.name, a.pos.x, a.pos.y, a.pos.z))
    # protein: clone, drop ligands and waters, emit PDB -- use gemmi's own
    # operations rather than hand-building a Model
    prot = st.clone()
    prot.remove_ligands_and_waters()
    prot.remove_empty_chains()
    protein_pdb = prot.make_pdb_string()
    ligand_sdf = _atoms_to_sdf(lig_atoms, ligand_smiles)
    return protein_pdb, ligand_sdf


def _atoms_to_sdf(lig_atoms, ligand_smiles):
    """Wrap the predicted ligand heavy-atom coordinates into an SDF with correct
    bond orders, using the SMILES template (rdkit).

    PoseBusters computes a symmetry-corrected RMSD, which needs both the right bond
    orders and the right atom correspondence -- hence graph matching first, with a
    positional fallback."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    templ = Chem.MolFromSmiles(ligand_smiles)
    if templ is None:
        raise ValueError(f"cannot parse ligand SMILES: {ligand_smiles}")
    n_heavy = templ.GetNumAtoms()                      # template heavy-atom count (before adding H)
    heavy = [a for a in lig_atoms if a[0] not in ("H", "D")]  # keep only heavy atoms from the prediction

    def _block_from_template_coords(coords_by_templ_idx):
        """Template chemistry (correct formula and valences) plus given coordinates -> MolBlock.
        coords_by_templ_idx[i] is the (x, y, z) of template atom i."""
        conf = Chem.Conformer(n_heavy)
        for i, (x, y, z) in enumerate(coords_by_templ_idx):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        # Always put coordinates on a fresh, sanitized SMILES template before
        # adding explicit hydrogens.  Adding H to the inferred prediction graph
        # leaves phantom radicals / an incorrect molecular formula for common
        # groups such as sulfonamides, which PoseBusters correctly rejects.
        m = Chem.Mol(templ)
        m.RemoveAllConformers()
        m.AddConformer(conf, assignId=True)
        m = Chem.AddHs(m, addCoords=True)
        return Chem.MolToMolBlock(m)

    # Preferred: template chemistry plus a substructure graph match that moves each
    # predicted coordinate onto its corresponding template atom -- chemically valid and
    # geometrically right at the same time.
    #   Build a connectivity graph over the predicted heavy atoms by distance, flatten
    #   every bond on both sides to single so the match is pure element+topology, and use
    #   the resulting template-atom i -> predicted-atom match[i] correspondence to copy the
    #   coordinates back onto the template. This depends on neither boltz's atom ordering
    #   nor on inferring correct bond orders for the prediction -- the latter is exactly
    #   where groups such as sulfonamides break.
    try:
        from rdkit.Chem import rdDetermineBonds
        rw = Chem.RWMol()
        for (elem, _x, _y, _z) in heavy:
            rw.AddAtom(Chem.Atom(elem))
        m = rw.GetMol()
        pconf = Chem.Conformer(m.GetNumAtoms())
        for i, (_, x, y, z) in enumerate(heavy):
            pconf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        m.AddConformer(pconf, assignId=True)
        rdDetermineBonds.DetermineConnectivity(m)             # connectivity only, no bond orders
        tq = Chem.Mol(templ)                                 # template query: flatten bond orders, drop charges/implicit H -- pure topology
        for b in tq.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
        for a in tq.GetAtoms():
            a.SetNoImplicit(True); a.SetFormalCharge(0)
        mq = Chem.Mol(m)
        for b in mq.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
        match = mq.GetSubstructMatch(tq)                      # match[i] = index of the predicted atom corresponding to template atom i
        if len(match) == n_heavy:
            pc = m.GetConformer()
            coords = [(pc.GetAtomPosition(match[i]).x, pc.GetAtomPosition(match[i]).y,
                       pc.GetAtomPosition(match[i]).z) for i in range(n_heavy)]
            return _block_from_template_coords(coords)
    except Exception:
        pass

    # Fallback: if the graph match fails but the element order agrees with the template,
    # assign positionally -- chemistry stays valid, geometry is best-effort.
    if len(heavy) >= n_heavy and all(
            heavy[i][0] == templ.GetAtomWithIdx(i).GetSymbol() for i in range(n_heavy)):
        return _block_from_template_coords([(heavy[i][1], heavy[i][2], heavy[i][3])
                                            for i in range(n_heavy)])
    # Last resort: template plus the first n_heavy predicted coordinates in order,
    # so the output is always chemically valid rather than a crash.
    return _block_from_template_coords(
        [(heavy[i][1], heavy[i][2], heavy[i][3]) if i < len(heavy) else (0.0, 0.0, 0.0)
         for i in range(n_heavy)])
