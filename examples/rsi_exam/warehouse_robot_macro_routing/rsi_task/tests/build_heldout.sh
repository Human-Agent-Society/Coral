#!/usr/bin/env bash
# Reconstruct the sealed grading assets (NOT committed): the 200 sealed
# instances and the official visualiser binary. The frozen rank-1 per-case
# reference lengths (heldout/rank1_lengths.json) and the anchor metric values
# ARE committed; this script only regenerates the large/binary inputs and
# verifies their hashes against heldout_manifest.json.
#
# Requires: the rebuilt official tools (run ../environment/tools/build_tools.sh
# first, or point $TOOLS at a directory that has gen + vis).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TOOLS="${TOOLS:-$HERE/../environment/tools}"

mkdir -p "$HERE/heldout/in"
echo "generating 200 sealed instances from heldout_seeds.txt ..."
"$TOOLS/gen" "$HERE/heldout_seeds.txt" --dir="$HERE/heldout/in"
cp "$TOOLS/vis" "$HERE/heldout/vis"

echo "verifying sealed-input hash against heldout_manifest.json ..."
python3 - "$HERE" <<'PY'
import hashlib, json, os, sys
here = sys.argv[1]
man = json.load(open(os.path.join(here, "heldout_manifest.json")))
d = os.path.join(here, "heldout", "in")
h = hashlib.sha256()
for f in sorted(os.listdir(d)):
    h.update(open(os.path.join(d, f), "rb").read())
got = h.hexdigest()
exp = man["sealed_inputs_concat_sha256"]
print("expected", exp)
print("got     ", got)
assert got == exp, "SEALED INPUT HASH MISMATCH -- generator or seeds changed!"
print("OK: 200 sealed instances match the frozen hash.")
PY
echo "done."
