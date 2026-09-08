#!/usr/bin/env bash
# Reconstruct the NON-committed agent-workbench assets from OFFICIAL sources:
#   * checker_src/            -- official ROADEF 2018 checker source (checker.zip)
#   * tools/instances/        -- the 10 VISIBLE challenge A-instances + global_param
# Everything is hash-verified against assets_manifest.json so a rebuild is
# byte-identical to what the anchors were measured on. Requires curl + unzip.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
UA="Mozilla/5.0"
CHECKER_URL="https://roadef.org/challenge/2018/files/checker_18052018.zip"
DATASET_URL="https://roadef.org/challenge/2018/files/dataset_A.zip"
VISIBLE="A1 A17 A16 A12 A9 A4 A18 A11 A8 A14"

tmp="$(mktemp -d)"
echo "fetching official checker + dataset A ..."
curl -fsSL -A "$UA" -o "$tmp/checker.zip" "$CHECKER_URL"
curl -fsSL -A "$UA" -o "$tmp/dataset_A.zip" "$DATASET_URL"
unzip -q -o "$tmp/checker.zip" -d "$tmp/checker"
unzip -q -o "$tmp/dataset_A.zip" -d "$tmp/ds"

# checker source
rm -rf "$HERE/checker_src"; mkdir -p "$HERE/checker_src"
cp -r "$tmp/checker/src" "$tmp/checker/include" "$HERE/checker_src/"

# visible instances
ds="$(dirname "$(find "$tmp/ds" -name 'A1_batch.csv' | head -1)")"
gp="$(find "$tmp/checker" -name global_param.csv | head -1)"   # ships in the checker zip, not the dataset zip
mkdir -p "$HERE/tools/instances"
cp "$gp" "$HERE/tools/instances/"
for I in $VISIBLE; do
  cp "$ds/${I}_batch.csv" "$ds/${I}_defects.csv" "$HERE/tools/instances/"
done

echo "verifying hashes against assets_manifest.json ..."
python3 - "$HERE" "$VISIBLE" <<'PY'
import hashlib, json, os, sys
here, visible = sys.argv[1], sys.argv[2].split()
man = json.load(open(os.path.join(here, "assets_manifest.json")))
def h(concat):
    d = hashlib.sha256()
    for f in concat: d.update(open(f, "rb").read())
    return d.hexdigest()
insts = os.path.join(here, "tools", "instances")
files = [os.path.join(insts, f"{I}_{k}.csv") for I in sorted(visible) for k in ("batch", "defects")]
got = h(files); exp = man["visible_inputs_concat_sha256"]
print("visible inputs:", "OK" if got == exp else f"MISMATCH got={got} exp={exp}")
assert got == exp, "VISIBLE INPUT HASH MISMATCH"
PY
echo "done."
