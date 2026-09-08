"""Operator setup before agents enter their sandbox; exports public data only."""

import shutil
import subprocess
import uuid
from pathlib import Path

from qlib_docker import PLATFORM, ensure_image

ROOT = Path(__file__).resolve().parent
PUBLIC_FILES = ("train_panel.parquet", "valid_panel.parquet", "feature_catalog.csv")


def main():
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    if not all((data / name).is_file() for name in PUBLIC_FILES):
        logs = ROOT / ".rsi_runs"
        logs.mkdir(exist_ok=True)
        with (logs / "visible-setup.log").open("w") as log:
            image = ensure_image(ROOT / "harbor/environment", log)
            container = "coral-visible-" + uuid.uuid4().hex
            try:
                subprocess.run(
                    [
                        "docker",
                        "create",
                        "--platform",
                        PLATFORM,
                        "--network",
                        "none",
                        "--name",
                        container,
                        image,
                        "true",
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=60,
                )
                for name in PUBLIC_FILES:
                    subprocess.run(
                        ["docker", "cp", f"{container}:/app/data/{name}", str(data / name)],
                        check=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=120,
                    )
            finally:
                subprocess.run(
                    ["docker", "rm", "-f", container],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                )
    target = ROOT / "selfcheck.py"
    if not target.exists():
        shutil.copy2(ROOT / "harbor/environment/selfcheck.py", target)


if __name__ == "__main__":
    main()
