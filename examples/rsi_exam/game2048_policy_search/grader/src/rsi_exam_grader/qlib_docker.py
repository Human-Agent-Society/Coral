"""Docker image preparation for the qlib task (operator-side only)."""

import hashlib
import subprocess
from pathlib import Path

PLATFORM = "linux/amd64"  # pyqlib 0.9.7 has no Linux ARM wheel.


def image_tag(context: Path) -> str:
    digest = hashlib.sha256(PLATFORM.encode())
    for path in sorted(context.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(context).as_posix().encode() + b"\0")
            digest.update(path.read_bytes())
    return "coral-rsi-qlib:" + digest.hexdigest()[:24]


def ensure_image(context: Path, log, timeout: float = 3600) -> str:
    tag = image_tag(context)
    probe = subprocess.run(
        ["docker", "image", "inspect", tag],
        stdout=subprocess.DEVNULL,
        stderr=log,
        timeout=30,
    )
    if probe.returncode:
        subprocess.run(
            ["docker", "build", "--platform", PLATFORM, "-t", tag, str(context)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )
    return tag
