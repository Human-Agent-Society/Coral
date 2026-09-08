#!/opt/venvs/train/bin/python
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: eval_visible.py MODEL_DIRECTORY")
    pidfile = Path("/app/run/teacher.pid")
    if pidfile.exists():
        try:
            if Path(f"/proc/{int(pidfile.read_text())}").exists():
                raise SystemExit("stop the teacher with teacher_control.sh stop before development evaluation")
        except ValueError:
            pass
    predictions = Path("/app/results/dev_predictions.json")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run([
        "/opt/venvs/vllm/bin/python", "/app/methods/main/infer_visible.py",
        sys.argv[1], "--output", str(predictions),
    ], check=True, env=env)
    output = subprocess.check_output([
        "/opt/venvs/train/bin/python", "/app/visible_client.py", str(predictions)
    ], text=True)
    response = json.loads(output)
    # The only evaluator-originated fields are aggregate score and budget counters.
    print(json.dumps(response, sort_keys=True))


if __name__ == "__main__":
    main()
