"""Build-time data preparation for bigann_filtered_vector_search.

Downloads the real Big-ANN NeurIPS'23 filtered-track yfcc-10M files from the public Meta CDN and
lays out exactly the slice this image is allowed to hold. Runs during `docker build` (host network,
outside the run-time egress lock) and must leave NO raw file behind -- run it in the same RUN layer
as the COPY of the split ids, or the deleted originals stay in the layer history.

  --split visible   -> the agent image: library + the 500 visible queries and their ground truth
  --split hidden    -> the verifier image: library + the 1500 sealed queries and their truth

The library (base vectors + tag CSR) is identical in both and is kept in its native Big-ANN binary
form: a .u8bin is an 8-byte header followed by the raw uint8 matrix and a .spmat is a 24-byte
header followed by int64 indptr and int32 indices, so both memory-map directly with an offset. Not
converting saves 1.9 GB of duplicated bytes per image and keeps the files byte-identical to what
the benchmark publishes.

The FULL public query set and the FULL public ground truth are downloaded (they are needed to cut
either slice) and then deleted. Neither image may keep them: the agent image would otherwise hold
the answers to the sealed queries, and holding all 100k queries would hand any solver an oracle
answer table keyed by query bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

CDN = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M"

# name -> (bytes, keep?) ; sizes are asserted so a truncated download fails the build loudly
FILES = {
    "base.10M.u8bin": (1_920_000_008, True),
    "base.metadata.10M.spmat": (945_683_840, True),
    "query.public.100K.u8bin": (19_200_008, False),
    "query.metadata.public.100K.spmat": (1_907_024, False),
    "GT.public.ibin": (8_000_008, False),
}


def fetch(name: str, dest: Path) -> Path:
    out = dest / name
    if out.exists() and out.stat().st_size == FILES[name][0]:
        return out
    subprocess.run(["curl", "-fsSL", "--retry", "5", "--retry-delay", "5",
                    f"{CDN}/{name}", "-o", str(out)], check=True)
    got = out.stat().st_size
    want = FILES[name][0]
    if got != want:
        raise SystemExit(f"{name}: expected {want} bytes, got {got} -- truncated download")
    return out


def read_u8bin(path: Path):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype="int32", count=2)
    return np.memmap(path, dtype="uint8", mode="r", offset=8, shape=(int(n), int(d)))


def read_spmat(path: Path):
    with open(path, "rb") as f:
        nrow, ncol, nnz = np.fromfile(f, dtype="int64", count=3)
    nrow, ncol, nnz = int(nrow), int(ncol), int(nnz)
    indptr = np.memmap(path, dtype="int64", mode="r", offset=24, shape=(nrow + 1,))
    indices = np.memmap(path, dtype="int32", mode="r", offset=24 + 8 * (nrow + 1), shape=(nnz,))
    return indptr, indices, ncol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["visible", "hidden"])
    ap.add_argument("--ids", required=True, help=".npy of query row ids for this split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/tmp/yfcc_raw")
    args = ap.parse_args()

    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        print(f"[prep] fetching {name}", flush=True)
        fetch(name, work)

    # library: keep the raw files as-is
    for name, (_, keep) in FILES.items():
        if keep:
            shutil.move(str(work / name), str(out / name))
    print("[prep] library in place", flush=True)

    ids = np.sort(np.load(args.ids).astype(np.int64))
    Q = read_u8bin(work / "query.public.100K.u8bin")
    qip, qix, n_tags = read_spmat(work / "query.metadata.public.100K.spmat")
    with open(work / "GT.public.ibin", "rb") as f:
        nq, k = np.fromfile(f, dtype="int32", count=2)
        GT = np.fromfile(f, dtype="int32", count=int(nq) * int(k)).reshape(int(nq), int(k))

    if ids.min() < 0 or ids.max() >= len(Q):
        raise SystemExit(f"split ids out of range for {len(Q)} queries")

    np.save(out / "query_vectors.npy", np.asarray(Q[ids]))
    # query tags stay CSR, same convention as the library's
    lens = (qip[ids + 1] - qip[ids]).astype(np.int64)
    np.save(out / "query_tag_indptr.npy", np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))
    np.save(out / "query_tag_indices.npy",
            np.concatenate([np.asarray(qix[qip[i]:qip[i + 1]]) for i in ids]).astype(np.int32))
    np.save(out / "ground_truth.npy", GT[ids].astype(np.int32))
    print(f"[prep] {args.split}: {len(ids)} queries, {int(lens.sum())} tag entries", flush=True)

    # the raw query/GT files must not survive into the image
    for name, (_, keep) in FILES.items():
        if not keep:
            (work / name).unlink(missing_ok=True)
    leftovers = sorted(p.name for p in work.iterdir())
    if leftovers:
        raise SystemExit(f"refusing to finish: raw files still present in {work}: {leftovers}")
    work.rmdir()

    # a fingerprint of what this image holds, so the two images can be shown to share a library
    h = hashlib.sha256()
    for name in ("base.10M.u8bin", "base.metadata.10M.spmat"):
        with open(out / name, "rb") as f:
            h.update(f.read(1 << 20))            # header + first MB is enough to catch a mismatch
            f.seek(-(1 << 20), os.SEEK_END); h.update(f.read())
    (out / "LIBRARY_FINGERPRINT").write_text(h.hexdigest() + "\n")
    print(f"[prep] library fingerprint {h.hexdigest()[:16]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
