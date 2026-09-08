"""CORAL-managed grader virtual environment.

Creates and bootstraps `.coral/private/grader_venv/` so that grader code
referenced by `grader.entrypoint` can be imported by a worker subprocess
without polluting CORAL's own venv.

Design:
  - venv lives inside `.coral/private/`, which is already covered by the
    Read deny-rule applied to agent worktrees (worktree.py).
  - The grader venv must be able to `import coral` so user grader packages
    can declare `coral` as a dependency. We replicate whatever install
    method produced the running CORAL by reading PEP 610 `direct_url.json`
    out of coral's dist-info:
      * editable install (dev `uv sync` path) -> `uv pip install -e <path>`
      * git VCS install (the README's `install.sh` -> `uv tool install
        git+...` path) -> `uv pip install git+<url>@<commit_id>`
    Both flavors point the grader venv at the exact same code the host is
    running, so there's no version drift between host and grader.
  - User's `grader.setup` shell commands then run with VIRTUAL_ENV pointed
    at the grader venv, so plain `uv pip install ...` lands in the right
    place.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from urllib.parse import urlparse

from coral.config import GraderConfig
from coral.venv_paths import venv_python
from coral.workspace.repo import _clean_env, run_setup_commands

logger = logging.getLogger(__name__)

DEFAULT_UV_HTTP_TIMEOUT = "180"


def _coral_install_origin() -> dict:
    """Return the PEP 610 install origin recorded in coral's dist-info.

    `direct_url.json` is written by every modern installer (pip >= 19, uv)
    and records what URL and (for VCS) what commit the package was installed
    from. Reading it lets us replicate the running install into the grader
    venv without guessing.

    Raises RuntimeError if the file is missing — that means coral was either
    installed by an installer that doesn't write PEP 610 metadata, or
    something went wrong with the install. In either case the caller can't
    safely replicate it.
    """
    host_site = Path(sysconfig.get_paths()["purelib"])
    dist_infos = list(host_site.glob("coral-*.dist-info"))
    if not dist_infos:
        raise RuntimeError(
            f"No coral-*.dist-info found in {host_site}; cannot determine how CORAL was installed."
        )
    direct_url = dist_infos[0] / "direct_url.json"
    if not direct_url.exists():
        raise RuntimeError(
            f"{direct_url} not found; CORAL was installed by an installer that "
            "does not write PEP 610 metadata. Reinstall with `uv tool install "
            "git+https://github.com/Human-Agent-Society/CORAL.git` or "
            "`git clone ... && uv sync`."
        )
    return json.loads(direct_url.read_text(encoding="utf-8"))


def _coral_install_command(origin: dict) -> str:
    """Build the `uv pip install` command that replicates the host coral install."""
    url = origin.get("url")
    if not url:
        raise RuntimeError(f"direct_url.json missing 'url': {origin!r}")

    if origin.get("dir_info", {}).get("editable"):
        # `file:///abs/path` -> `/abs/path`
        local_path = urlparse(url).path
        return f"uv pip install -q -e {local_path}"

    if "vcs_info" in origin:
        vcs = origin["vcs_info"].get("vcs")
        commit = origin["vcs_info"].get("commit_id")
        if vcs != "git" or not commit:
            raise RuntimeError(
                f"Only git VCS installs are supported; got vcs_info={origin['vcs_info']!r}"
            )
        return f"uv pip install -q git+{url}@{commit}"

    raise RuntimeError(
        f"Unsupported coral install origin (not editable, not VCS): {origin!r}. "
        "Reinstall via `uv tool install git+...` or `uv sync`."
    )


def grader_venv_path(coral_dir: Path) -> Path:
    """Path to the grader venv for a given .coral dir."""
    return coral_dir / "private" / "grader_venv"


def grader_python_path(coral_dir: Path) -> Path:
    """Path to the Python interpreter inside the grader venv."""
    return venv_python(grader_venv_path(coral_dir))


def setup_grader_env(
    coral_dir: Path,
    grader_config: GraderConfig,
    config_dir: Path,
    *,
    rebuild: bool = False,
) -> Path:
    """Create the grader venv and run `grader_config.setup` commands in it.

    Steps:
      1. (Optionally) wipe an existing venv if `rebuild=True`.
      2. Run `uv venv .coral/private/grader_venv/` against `sys.executable`
         so the venv matches CORAL's interpreter.
      3. Replicate the host coral install into the venv by reading PEP 610
         `direct_url.json` and running the corresponding install command.
      4. Run each command in `grader_config.setup` with VIRTUAL_ENV /
         PATH pointed at the new venv. `cwd` is `config_dir` so paths
         in setup commands resolve relative to the task directory.

    Returns the path to the venv's Python interpreter.
    Raises RuntimeError on any failure with stdout/stderr in the message.
    """
    venv_dir = grader_venv_path(coral_dir)
    python_path = grader_python_path(coral_dir)

    if rebuild and venv_dir.exists():
        shutil.rmtree(venv_dir)

    venv_dir.parent.mkdir(parents=True, exist_ok=True)

    if not python_path.exists():
        if shutil.which("uv") is None:
            raise RuntimeError(
                "`uv` was not found on PATH; it is required to create the grader venv. "
                "Install it from https://docs.astral.sh/uv/getting-started/installation/ "
                "and re-run."
            )
        logger.info(f"Creating grader venv at {venv_dir}")
        venv_cmd = ["uv", "venv", "--python", sys.executable, str(venv_dir)]
        result = subprocess.run(
            venv_cmd,
            capture_output=True,
            text=True,
            env=_clean_env(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"`{' '.join(venv_cmd)}` failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

    if not python_path.exists():
        raise RuntimeError(
            f"Expected Python interpreter at {python_path} after `uv venv`, but it does not exist"
        )

    extra_env = {
        "VIRTUAL_ENV": str(venv_dir),
        "PATH": f"{venv_dir / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", DEFAULT_UV_HTTP_TIMEOUT),
    }

    coral_install_cmd = _coral_install_command(_coral_install_origin())
    run_setup_commands([coral_install_cmd], cwd=config_dir, extra_env=extra_env)

    if grader_config.setup:
        run_setup_commands(grader_config.setup, cwd=config_dir, extra_env=extra_env)

    return python_path
