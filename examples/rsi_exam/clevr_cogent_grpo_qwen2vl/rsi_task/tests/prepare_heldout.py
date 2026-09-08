#!/usr/bin/env python
"""Build-time prep for the VERIFIER image (has network during build, sealed at run time).

Bakes in the two SEALED held-out test sets, each saved as a HF dataset with columns
image (PIL) / problem (str) / solution (gold str), capped at 5000 counting examples:
    - CoGenT-B  : CLEVR-CoGenT v1.0 valB  -> /tests/heldout/cogent_b_val
    - SuperCLEVR: Super-CLEVR test         -> /tests/heldout/superclevr_test

These sets live ONLY in the verifier image; they never enter the agent build context.

The base model is intentionally NOT downloaded here: grade.py loads the *submitted* model from
/app/submission (which Harbor uploads from the agent box), so the verifier never needs the base.

⚠️ The exact on-disk layout of the public archives can drift. The filtering below keeps only
*counting* questions (integer answers), matching the R1-V keyword-matching eval. If a field name
or archive path differs when you build, fix it here — then sanity-check the printed counts.
URLs (from the task spec):
  CoGenT  : https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0.zip
  SuperCLEVR: https://github.com/Lizw14/Super-CLEVR  (images + question json)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from pathlib import Path

CAP = 5000
OOD_VAL_N = 1000   # MUST match environment/prepare_data.py: the first OOD_VAL_N SuperCLEVR counting
                   # questions are the agent's VISIBLE OOD val; the sealed test skips them (disjoint).
CLEVR_COGENT_ZIP = "https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0.zip"  # ~24.7 GB
# SuperCLEVR is hosted on HF (RyanWW/Super-CLEVR); the old JHU URLs are dead (404). Verified
# reachable 2026-06-25. Counting questions are CLEVR-style with integer answers.
SUPERCLEVR_IMAGES = "https://huggingface.co/datasets/RyanWW/Super-CLEVR/resolve/main/images.zip?download=true"
SUPERCLEVR_QUESTIONS = "https://huggingface.co/datasets/RyanWW/Super-CLEVR/resolve/main/superCLEVR_questions_30k.json?download=true"


def _download(url: str, dest: Path) -> Path:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"[heldout] downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def _is_counting(answer: str) -> bool:
    return str(answer).strip().isdigit()


def build_cogent_b(work: Path, out: Path) -> None:
    from datasets import Dataset, Features, Image, Value

    zip_path = _download(CLEVR_COGENT_ZIP, work / "cogent.zip")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # Robust to the archive's exact prefix (inside the FAIR zip the files are
        # CLEVR_CoGenT_v1.0/questions/CLEVR_valB_questions.json and images/valB/CLEVR_valB_*.png).
        q_name = next(n for n in names if n.endswith("valB_questions.json"))
        questions = json.loads(zf.read(q_name))["questions"]
        img_members = {Path(n).name: n for n in names
                       if "/images/valB/" in n and n.lower().endswith(".png")}
        rows = {"image": [], "problem": [], "solution": []}
        for q in questions:
            ans = q.get("answer", "")
            if not _is_counting(ans):
                continue
            member = img_members.get(Path(q["image_filename"]).name)
            if member is None:
                continue
            rows["image"].append({"bytes": zf.read(member), "path": None})
            rows["problem"].append(q["question"])
            rows["solution"].append(str(ans))
            if len(rows["problem"]) >= CAP:
                break
    feats = Features({"image": Image(), "problem": Value("string"), "solution": Value("string")})
    Dataset.from_dict(rows, features=feats).save_to_disk(str(out))
    print(f"[heldout] CoGenT-B counting={len(rows['problem'])} -> {out}")


def build_superclevr(work: Path, out: Path) -> None:
    from datasets import Dataset, Features, Image, Value

    q_path = _download(SUPERCLEVR_QUESTIONS, work / "superclevr_q.json")
    img_zip = _download(SUPERCLEVR_IMAGES, work / "superclevr_images.zip")
    raw = json.loads(q_path.read_text())
    questions = raw["questions"] if isinstance(raw, dict) else raw  # CLEVR-style {"questions":[...]}
    rows = {"image": [], "problem": [], "solution": []}
    kept = 0  # count of (counting + image-present) questions, in order
    with zipfile.ZipFile(img_zip) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for q in questions:
            ans = q.get("answer", "")
            if not _is_counting(ans):
                continue
            fn = q["image_filename"]
            member = names.get(Path(fn).name)
            if member is None:
                continue
            kept += 1
            # The FIRST OOD_VAL_N such questions are the agent's VISIBLE OOD val
            # (environment/prepare_data.py build_ood_val) — exclude them from the sealed test so the
            # two never overlap. Same source + order guarantees a clean, deterministic partition.
            if kept <= OOD_VAL_N:
                continue
            rows["image"].append({"bytes": zf.read(member), "path": None})
            rows["problem"].append(q["question"])
            rows["solution"].append(str(ans))
            if len(rows["problem"]) >= CAP:
                break
    feats = Features({"image": Image(), "problem": Value("string"), "solution": Value("string")})
    Dataset.from_dict(rows, features=feats).save_to_disk(str(out))
    print(f"[heldout] SuperCLEVR counting={len(rows['problem'])} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-dir", default="/tests/heldout")
    args = ap.parse_args()

    work = Path("/tmp/heldout_build")
    work.mkdir(parents=True, exist_ok=True)
    heldout = Path(args.heldout_dir)
    build_cogent_b(work, heldout / "cogent_b_val")
    build_superclevr(work, heldout / "superclevr_test")
    # free the raw archives so they don't bloat the image layer
    os.system(f"rm -rf {work}")


if __name__ == "__main__":
    main()
