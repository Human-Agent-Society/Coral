#!/usr/bin/env bash
# Reconstruct the NON-committed sealed grading assets from OFFICIAL sources:
#   * checker_src/          -- official ROADEF 2018 checker source (checker.zip)
#   * heldout/in/           -- the 10 SEALED challenge A-instances (+ global_param)
# The frozen per-case the strong reference reference wastes (heldout/ref_lengths.json) and the
# anchor metric values ARE committed; this only regenerates the instances + checker
# source and hash-verifies against heldout_manifest.json. Requires curl + unzip.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
UA="Mozilla/5.0"
CHECKER_URL="https://roadef.org/challenge/2018/files/checker_18052018.zip"
DATASET_URL="https://roadef.org/challenge/2018/files/dataset_A.zip"
SEALED="A20 A6 A19 A7 A3 A2 A10 A5 A13 A15"

tmp="$(mktemp -d)"
echo "fetching official checker + dataset A ..."
curl -fsSL -A "$UA" -o "$tmp/checker.zip" "$CHECKER_URL"
curl -fsSL -A "$UA" -o "$tmp/dataset_A.zip" "$DATASET_URL"
unzip -q -o "$tmp/checker.zip" -d "$tmp/checker"
unzip -q -o "$tmp/dataset_A.zip" -d "$tmp/ds"

rm -rf "$HERE/checker_src"; mkdir -p "$HERE/checker_src"
cp -r "$tmp/checker/src" "$tmp/checker/include" "$HERE/checker_src/"

ds="$(dirname "$(find "$tmp/ds" -name 'A1_batch.csv' | head -1)")"
gp="$(find "$tmp/checker" -name global_param.csv | head -1)"   # ships in the checker zip, not the dataset zip
mkdir -p "$HERE/heldout/in"
cp "$gp" "$HERE/heldout/"
for I in $SEALED; do
  cp "$ds/${I}_batch.csv" "$ds/${I}_defects.csv" "$HERE/heldout/in/"
done

echo "verifying sealed-input hash against heldout_manifest.json ..."
python3 - "$HERE" "$SEALED" <<'PY'
import hashlib, json, os, sys
here, sealed = sys.argv[1], sys.argv[2].split()
man = json.load(open(os.path.join(here, "heldout_manifest.json")))
d = os.path.join(here, "heldout", "in")
h = hashlib.sha256()
for I in sorted(sealed):
    for k in ("batch", "defects"):
        h.update(open(os.path.join(d, f"{I}_{k}.csv"), "rb").read())
got = h.hexdigest(); exp = man["sealed_inputs_concat_sha256"]
print("expected", exp); print("got     ", got)
assert got == exp, "SEALED INPUT HASH MISMATCH -- dataset changed!"
print("OK: 10 sealed instances match the frozen hash.")
PY
echo "done."
