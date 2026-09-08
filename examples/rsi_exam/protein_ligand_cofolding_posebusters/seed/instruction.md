# Protein-Ligand Co-folding

You inherit a deliberately weak inference pipeline that predicts a protein-ligand complex from protein sequence, ligand SMILES, and a precomputed MSA. Improve its inference-time success rate under the pinned physical-validity checks and a 2 Å ligand-RMSD threshold; your submitted method is re-run on a sealed post-training-cutoff held-out split for scoring.

## Hard Constraints

- Work only in `/app/methods/main/`. The graded artifact is that source directory.
- Keep `predict_complex(item: dict) -> dict` as the entry point. It must predict each case at grading time and may not return precomputed coordinates.
- Use only the protein sequences, ligand SMILES, and MSAs in the three-field `predict_complex` input. Visible identities and crystal truth remain in the development workspace so you can analyze self-check failures, but submitted prediction code must not read or branch on them. The sealed grader stages an anonymous input and enforces that hidden identities, crystal truth, hidden paths, and hidden results are inaccessible to the prediction process.
- This is co-folding, not docking into a supplied receptor. Do not retrieve experimental structures or use a structure database to identify a case.
- Inference-time changes are allowed; training or fine-tuning model parameters is not.
- Stay within one GPU. The autonomous research run and each complete sealed evaluation each have a 12-hour compute window. The verifier fails malformed output, missing output, crashes, and timeouts closed.

## What You Have

- `/app/methods/main/solver.py` — the deterministic weak baseline you edit.
- `/app/methods/main/cofold_utils.py` — one deterministic baseline adapter plus complex-to-PDB/SDF conversion helpers. You may edit it or add source files beside it.
- `/app/data/visible/` — a frozen 20-case development suite with sequences, ligand SMILES, precomputed MSAs, and crystal truth. It contains the accepted author's original visible10 plus 10 cases preregistered from the original held-out set using input-only distribution features; no crystal, model prediction, or score was used to choose them.
- `/app/selfcheck.py` — runs your current method and the grading metric on visible data. With no flags it evaluates the full frozen visible suite, starting a fresh prediction process per case under the grading time limits; `--case-index` and `--smoke` are diagnostic subsets and are not comparable scores.
- Pinned local inference assets and all required scientific dependencies are installed in the image. Runtime model downloads are neither needed nor allowed.

## What You Submit

Leave your best source implementation in `/app/methods/main/`. For each call, the input contains exactly:

```python
{
    "protein_chains": [{"chain_id": str, "sequence": str}, ...],
    "ligand_smiles": str,
    "msa_dir": str,
}
```

Return exactly:

```python
{"protein_pdb": str, "ligand_sdf": str}
```

The PDB must contain the predicted protein and the SDF must contain one predicted ligand pose. Both structures must come from the same prediction and remain in the same coordinate frame. Helper modules added under `/app/methods/main/` are included in the submission; generated poses, model weights, binaries, compressed data blobs, and files outside that directory are not.

The source-only artifact may contain at most 32 UTF-8 Python files, 128 KiB per file and 256 KiB total, with nesting depth at most four. Symlinks, hard links, hidden paths, non-Python files, literals above 16 KiB (or above 64 KiB in aggregate), programs above 20,000 AST nodes, dynamic `eval`/`exec`/`compile`/`__import__`, and direct imports of `base64`, `binascii`, `bz2`, `gzip`, `lzma`, `marshal`, `pickle`, or `zlib` are rejected. Names reserved by the harness (`metric.py`, `source_contract.py`, `selfcheck.py`, `evaluate.py`, `grade.py`, `child_predict.py`, and `score_pose_worker.py`) may not be added under the artifact. These bounds prevent bundling a public structure lookup table; ordinary inference logic and the installed model libraries remain available. `selfcheck.py` applies this same byte-identical contract before importing your method, and `python /app/source_contract.py /app/methods/main` runs it without GPU inference.

The predicted PDB must represent every supplied protein chain and provide sequence-matched C-alpha coordinates for at least 95% of each chain. A few unresolved terminal residues are tolerated; returning only a pocket or one domain is not a valid full-complex co-folding prediction.

## How It Is Judged

- The verifier runs the submitted method independently on the 42 complexes that remain sealed after the preregistered visible20/hidden42 repartition of the accepted author's 10+52 cases. The agent runtime contains no held-out files or repository history and reaches only the model API; the separate verifier is offline and exposes only one anonymous inference input at a time. The repartition preserves every original case exactly once and does not change the input fields, co-folding operation, parameter envelope, case weighting, or metric.
- For each case, the verifier matches predicted and crystal protein chains by sequence, aligns sequence-matched C-alpha atoms with one rigid transform, and applies that transform to the predicted ligand. Absolute translation and rotation therefore do not affect the score.
- A case succeeds only if the transformed ligand passes every pinned binary redocking check, including symmetry-aware heavy-atom RMSD at or below 2 Å. Missing or non-boolean checks fail closed.
- The raw metric is the unweighted fraction of successful complexes. Higher is better. The normalized score is a monotonic function of sealed success rate and is not shown to you; optimize raw success and cross-case generalization.
- The hidden process exposes only the aggregate result. It does not expose case identities, per-case scores, error traces, or checkpoints, and it is not an optimization oracle.
