#!/usr/bin/env python
"""Build-time data prep for the AGENT image (runs inside the Dockerfile, has network).

Bakes, all offline-loadable at run time (task.toml sets HF_HUB_OFFLINE=1 / HF_DATASETS_OFFLINE=1):
  1. the base model at /app/models/Qwen2-VL-2B-Instruct (R1-V's --model_name_or_path);
  2. a fixed **train / val split of CoGenT-A** (same distribution):
       - /app/data/train : the training split the trainer sees (val is EXCLUDED);
       - /app/data/val   : a held-out, IN-distribution validation set (training-health signal);
  3. an **OOD validation set** /app/data/ood_val : the first OOD_VAL_N SuperCLEVR counting questions.
       This is out-of-distribution vs CoGenT-A and VISIBLE to the agent, so it can iterate on real
       generalization. It is kept DISJOINT from the sealed SuperCLEVR test (tests/prepare_heldout.py
       skips exactly these first OOD_VAL_N questions).
The splits are FIXED by the task (seeded shuffle / fixed source order) so the agent cannot train on
val, and cannot see the sealed test.

The sealed OOD TEST (CoGenT-B + the rest of SuperCLEVR) is NOT here — it lives only in the verifier.
"""
from __future__ import annotations

import argparse
from pathlib import Path

BASE_MODEL_REPO = "Qwen/Qwen2-VL-2B-Instruct"
TRAIN_REPO = "leonardPKU/clevr_cogen_a_train"   # ~70k: columns image (PIL) / problem / solution
VAL_N = 2000        # in-distribution validation examples, held out from training
SPLIT_SEED = 0      # fixed so the train/val partition is deterministic and non-gameable

# OOD validation set: a slice of SuperCLEVR (out-of-distribution vs CoGenT-A). VISIBLE to the agent
# so it can iterate on real generalization. MUST stay disjoint from the sealed SuperCLEVR TEST slice
# in tests/prepare_heldout.py, which skips these first OOD_VAL_N counting questions (same source +
# order, so [:OOD_VAL_N] here and [OOD_VAL_N:] there never overlap).
OOD_VAL_N = 1000
SUPERCLEVR_QUESTIONS = "https://huggingface.co/datasets/RyanWW/Super-CLEVR/resolve/main/superCLEVR_questions_30k.json?download=true"
SUPERCLEVR_IMAGES = "https://huggingface.co/datasets/RyanWW/Super-CLEVR/resolve/main/images.zip?download=true"


def fetch_base_model(dest: Path) -> None:
    from huggingface_hub import snapshot_download
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=BASE_MODEL_REPO, local_dir=str(dest), local_dir_use_symlinks=False)
    print(f"[prepare] base model -> {dest}")


def build_train_val_split(data_root: Path) -> None:
    from datasets import load_dataset

    ds = load_dataset(TRAIN_REPO, split="train")
    print(f"[prepare] CoGenT-A total: {len(ds)} examples")

    # Deterministic shuffle, then carve a held-out VAL set that training never sees.
    ds = ds.shuffle(seed=SPLIT_SEED)
    val = ds.select(range(VAL_N))
    train = ds.select(range(VAL_N, len(ds)))

    data_root.mkdir(parents=True, exist_ok=True)
    train.save_to_disk(str(data_root / "train"))
    val.save_to_disk(str(data_root / "val"))
    print(f"[prepare] train={len(train)} -> {data_root/'train'}  (val EXCLUDED)")
    print(f"[prepare] val={len(val)} -> {data_root/'val'}  (in-distribution, held out from train)")

    # Remove the raw full-70k cache so ONLY the task's train split is loadable offline at run time
    # (the agent cannot silently reload the original dataset — which still contains val — and train
    # on it). The saved train/ and val/ dirs are self-contained copies, unaffected by this.
    import shutil
    from datasets import config as ds_config
    for d in Path(ds_config.HF_DATASETS_CACHE).glob("*clevr_cogen_a_train*"):
        shutil.rmtree(d, ignore_errors=True)
        print(f"[prepare] removed raw dataset cache: {d}")


def _download(url: str, dest: Path) -> Path:
    import urllib.request
    if not dest.exists():
        print(f"[prepare] downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def build_ood_val(data_root: Path) -> None:
    """OOD validation = the FIRST OOD_VAL_N SuperCLEVR counting questions (disjoint from the sealed
    SuperCLEVR test, which skips these). Same source/order as tests/prepare_heldout.py."""
    import json, zipfile
    from datasets import Dataset, Features, Image, Value

    work = data_root / "_ood_src"; work.mkdir(parents=True, exist_ok=True)
    q_path = _download(SUPERCLEVR_QUESTIONS, work / "superclevr_q.json")
    img_zip = _download(SUPERCLEVR_IMAGES, work / "superclevr_images.zip")
    raw = json.loads(q_path.read_text())
    questions = raw["questions"] if isinstance(raw, dict) else raw
    rows = {"image": [], "problem": [], "solution": []}
    with zipfile.ZipFile(img_zip) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for q in questions:
            ans = q.get("answer", "")
            if not str(ans).strip().isdigit():          # counting questions only
                continue
            member = names.get(Path(q["image_filename"]).name)
            if member is None:
                continue
            rows["image"].append({"bytes": zf.read(member), "path": None})
            rows["problem"].append(q["question"]); rows["solution"].append(str(ans))
            if len(rows["problem"]) >= OOD_VAL_N:
                break
    feats = Features({"image": Image(), "problem": Value("string"), "solution": Value("string")})
    Dataset.from_dict(rows, features=feats).save_to_disk(str(data_root / "ood_val"))
    import shutil; shutil.rmtree(work, ignore_errors=True)
    print(f"[prepare] ood_val (SuperCLEVR, OOD)={len(rows['problem'])} -> {data_root/'ood_val'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["environment"], default="environment")
    ap.add_argument("--model-dir", default="/app/models/Qwen2-VL-2B-Instruct")
    ap.add_argument("--data-root", default="/app/data")
    args = ap.parse_args()

    fetch_base_model(Path(args.model_dir))
    build_train_val_split(Path(args.data_root))
    build_ood_val(Path(args.data_root))


if __name__ == "__main__":
    main()
