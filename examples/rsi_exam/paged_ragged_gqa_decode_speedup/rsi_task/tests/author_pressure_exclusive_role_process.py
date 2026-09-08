"""Visible-only v24 pressure harness using disposable GPU role workers.

This invokes the reviewed v24 author calibration unchanged except for binding
its second role to the preregistered strongest captured agent hash.  The role
is still serialized as ``sota`` by the reviewed record schema; downstream
pressure aggregation must relabel it ``strongest_captured_agent`` and must not
interpret it as Human SOTA.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


BASE = Path("/calibration/author_calibrate_base.py")
WORKER_FLAG = "--exclusive-pressure-role-worker"
PRESSURE_SHA256 = "08c107570f1d860114a0fc9e9d6f58a7ecdf431314e6f67c8bc6094860faff5c"


def load_base():
    spec = importlib.util.spec_from_file_location("paged_v24_author_pressure_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load reviewed v24 author calibration")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EXPECTED_SOTA_SHA256 = PRESSURE_SHA256
    return module


def worker_main(arguments: list[str]) -> None:
    if len(arguments) != 3:
        raise RuntimeError("pressure role worker requires payload, output, and submission")
    payload_path, output_path, submission = map(Path, arguments)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    module = load_base()
    result = module.evaluate_role(submission, payload["raw_case"], payload["timing"])
    output_path.write_text(
        json.dumps(module.json_safe(result), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def outer_main() -> None:
    module = load_base()

    def isolated_evaluate_role(submission: Path, raw_case: dict, timing: dict) -> dict:
        with tempfile.TemporaryDirectory(prefix="paged-v24-pressure-role-") as directory:
            root = Path(directory)
            payload = root / "payload.json"
            output = root / "result.json"
            payload.write_text(
                json.dumps({"raw_case": raw_case, "timing": timing}),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    WORKER_FLAG,
                    str(payload),
                    str(output),
                    str(submission),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3600,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "exclusive pressure role worker failed:\n"
                    + completed.stdout
                    + "\n"
                    + completed.stderr
                )
            return json.loads(output.read_text(encoding="utf-8"))

    module.evaluate_role = isolated_evaluate_role
    module.main()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == WORKER_FLAG:
        worker_main(sys.argv[2:])
    else:
        outer_main()
