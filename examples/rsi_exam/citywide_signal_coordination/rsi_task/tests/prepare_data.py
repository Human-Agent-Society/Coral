"""Fetch the public TAPASCologne 0.17.0 scenario (net + demand) into --out.

TAPASCologne is an official public SUMO scenario (DLR TAPAS synthetic demand over the
OSM Cologne network). Docs: https://sumo.dlr.de/docs/Data/Scenarios/TAPASCologne.html

This runs at image BUILD time (network available then; run time can be offline). Only
the two files the evaluator needs are kept, renamed to the de-identified
names the evaluator uses: net.xml and demand.rou.xml.

If the primary URL rotates, override with SCENARIO_URL. The scenario is ~250 MB.

NOTE: the bundle is distributed as a ZIP. An earlier revision pointed at a
`TAPASCologne-0.17.0.tar.gz` that does not exist on SourceForge (that directory only ever
held .zip), and then tried to open it with `tarfile` -- so the image build failed twice
over. Both are fixed here; keep them in step if the URL is ever changed again.

INTEGRITY: `BUNDLE_SHA256` pins the downloaded archive. It was EMPTY until 2026-08-11 --
the author's machine had no SUMO and never downloaded the bundle, and inventing a hash is
worse than having none. It is now FILLED IN from an actual download on the review machine,
so a mismatch is a hard failure from here on. (While empty, this script printed the
measured hash and continued, so the build still worked.) This closes step 5 of the
re-anchor runbook `_dev/reanchor_2026-08/RE_ANCHOR.md`. The value is mirrored in
`solution/provenance.md`; the two copies of this script (`tests/prepare_data.py`,
`environment/prepare_data.py`) must stay byte-identical -- verify with `md5sum` after any
edit.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

# Primary public mirror (SourceForge SUMO project, traffic_data/scenarios).
DEFAULT_URL = os.environ.get(
    "SCENARIO_URL",
    "https://sourceforge.net/projects/sumo/files/traffic_data/scenarios/"
    "TAPASCologne/TAPASCologne-0.17.0.zip/download",
)

# SHA256 of the TAPASCologne-0.17.0 bundle.  MEASURED 2026-08-11 on the review machine
# (x86_64 Amazon Linux 2023, eclipse-sumo/libsumo 1.19.0 under python 3.11) by running this
# very script against the DEFAULT_URL below; the download was 252 MB and the extracted
# net.xml / demand.rou.xml are 65 MB / 193 MB.  Non-empty => strict check from now on.
# This closes RE_ANCHOR.md step 5, which had been open since the task was written because
# the authoring machine had no SUMO and never downloaded the bundle.
BUNDLE_SHA256 = "e9344cda6443a5d239897fde64fa205c53f2c51f1d91243921a2a81cd34a703c"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_bundle(path):
    """Pin the archive contents. Empty pin => report the measured hash and continue."""
    got = _sha256(path)
    if not BUNDLE_SHA256:
        print(f"[prepare] WARNING: BUNDLE_SHA256 is unset (never measured on the "
              f"authoring machine). Measured sha256 = {got}\n"
              f"[prepare] WARNING: record this in prepare_data.py (both mirrors) and "
              f"solution/provenance.md — RE_ANCHOR.md step 5.", flush=True)
        return got
    if got != BUNDLE_SHA256:
        sys.exit(f"[prepare] ERROR: bundle sha256 mismatch\n"
                 f"  expected {BUNDLE_SHA256}\n  got      {got}\n"
                 f"The upstream mirror changed the artefact: do NOT build on it, and do "
                 f"NOT re-pin without re-anchoring (the scenario defines every number).")
    print(f"[prepare] sha256 ok: {got}", flush=True)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bundle = os.path.join(a.out, "tapas.zip")

    print(f"[prepare] downloading TAPASCologne bundle -> {bundle}", flush=True)
    req = urllib.request.Request(DEFAULT_URL, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req) as r, open(bundle, "wb") as f:
        f.write(r.read())

    check_bundle(bundle)

    # Extract into a scratch directory that is deleted below.  The upstream bundle's own
    # directory name identifies the source scenario, and extracting straight into --out
    # used to leave that name sitting in the agent's container.
    print("[prepare] extracting net + demand", flush=True)
    scratch = os.path.join(a.out, "_extract")
    with zipfile.ZipFile(bundle) as z:
        z.extractall(scratch)

    # locate the two required files anywhere under the extracted tree
    net = rou = None
    for root, _dirs, files in os.walk(scratch):
        for fn in files:
            if fn == "cologne.net.xml":
                net = os.path.join(root, fn)
            elif fn == "cologne.rou.xml":
                rou = os.path.join(root, fn)
    if not (net and rou):
        sys.exit("[prepare] ERROR: cologne.net.xml / cologne.rou.xml not found in bundle")

    dest = os.path.join(a.out, "scenario")
    os.makedirs(dest, exist_ok=True)
    for src, name in [(net, "net.xml"), (rou, "demand.rou.xml")]:
        if os.path.abspath(src) != os.path.abspath(os.path.join(dest, name)):
            subprocess.run(["cp", src, os.path.join(dest, name)], check=True)
    shutil.rmtree(scratch, ignore_errors=True)
    os.remove(bundle)
    print(f"[prepare] ready: {dest}", flush=True)


if __name__ == "__main__":
    main()
