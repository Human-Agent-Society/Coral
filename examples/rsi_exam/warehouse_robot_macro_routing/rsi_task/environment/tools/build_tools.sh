#!/usr/bin/env bash
# Rebuild the official AtCoder AHC066 generator + visualiser and the 100 visible
# instances. The tool source and binaries are NOT committed (AtCoder tooling);
# this script reconstructs them.
#
# Requires: rust toolchain (cargo), curl/unzip.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TOOLS_URL="https://img.atcoder.jp/ahc066/O25rQjiK.zip"   # official local tools

if [ ! -f "$HERE/src/lib.rs" ]; then
  echo "fetching official tools from $TOOLS_URL ..."
  tmp="$(mktemp -d)"
  curl -fsSL -A "Mozilla/5.0" "$TOOLS_URL" -o "$tmp/tools.zip"
  unzip -q "$tmp/tools.zip" -d "$tmp"
  src="$(dirname "$(find "$tmp" -name lib.rs | head -1)")/.."
  cp -r "$src/src" "$src/Cargo.toml" "$HERE/"
  [ -f "$src/Cargo.lock" ] && cp "$src/Cargo.lock" "$HERE/" || true
fi

echo "cargo build -r --bins ..."
( cd "$HERE" && cargo build -r --bins )
cp "$HERE/target/release/gen" "$HERE/gen"
cp "$HERE/target/release/vis" "$HERE/vis"

echo "generating 100 visible instances (seeds 0-99) ..."
seq 0 99 > "$HERE/visible_seeds.txt"
"$HERE/gen" "$HERE/visible_seeds.txt" --dir="$HERE/in"
echo "done: $(ls "$HERE/in" | wc -l) instances, gen+vis rebuilt."
