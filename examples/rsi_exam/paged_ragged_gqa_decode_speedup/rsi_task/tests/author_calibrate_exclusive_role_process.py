"""Run each reviewed v25 calibration role in a disposable GPU process.

Exclusive_Process permits exactly one CUDA context.  The reviewed calibration
first measures a role in ``child_eval.py`` and then reconstructs the trusted
GPU reference in its parent.  Keeping that parent alive would retain its
reference context and block the next role.  This control wrapper moves the
unchanged ``evaluate_role`` call into one disposable worker per role:

    measurement child -> reviewed GPU reference/correctness -> worker exit

The measured child, call schedule, CUDA events, inputs, output checks, and
returned role record are unchanged.  Only process lifetime is shortened so the
reference context is released before the next role begins.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


BASE = Path("/calibration/author_calibrate_base.py")
WORKER_FLAG = "--exclusive-role-worker"


def load_base():
    spec = importlib.util.spec_from_file_location("paged_v25_author_calibrate_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load reviewed v25 author calibration")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def worker_main(arguments: list[str]) -> None:
    if len(arguments) != 3:
        raise RuntimeError("exclusive role worker requires payload, output, and submission")
    payload_path, output_path, submission = map(Path, arguments)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    module = load_base()
    result = module.evaluate_role(
        submission,
        payload["raw_case"],
        payload["timing"],
    )
    output_path.write_text(
        json.dumps(module.json_safe(result), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def outer_main() -> None:
    module = load_base()

    def isolated_evaluate_role(submission: Path, raw_case: dict, timing: dict) -> dict:
        with tempfile.TemporaryDirectory(prefix="paged-v25-exclusive-role-") as directory:
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
                    "exclusive role worker failed:\n"
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
