"""DeepSeek Harness (``dsh``) CLI subprocess lifecycle."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from coral.agent.exit_classifier import classify_by_uptime
from coral.agent.process import open_agent_stderr_for_log_dir
from coral.agent.runtime import (
    AgentHandle,
    apply_run_as_user,
    apply_sandbox,
    apply_sandbox_env,
    write_coral_log_entry,
)
from coral.sandbox.protocol import AgentSandboxSpec
from coral.venv_paths import venv_bin_dir
from coral.workspace.repo import _clean_env

logger = logging.getLogger(__name__)

_DEEPSEEK_HARNESS_RUNTIME_OPTION_KEYS = {
    "command",
    "profile",
    "patch",
    "provider",
    "permission_mode",
    "tools_mode",
}


class DeepSeekHarnessRuntime:
    """Spawn and manage the official DeepSeek Harness headless CLI."""

    @property
    def instruction_filename(self) -> str:
        return "AGENTS.md"

    @property
    def shared_dir_name(self) -> str:
        return ".dsh"

    def extract_session_id(self, log_path: Path) -> str | None:
        # The shipped headless profile creates a fresh persisted Agent, but its
        # stdout contract exposes only the final assistant text (not a session
        # id). Keep this explicit rather than guessing from user-facing output.
        return None

    def classify_exit(
        self,
        log_path: Path,
        exit_code: int | None,
        uptime_seconds: float | None,
        min_clean_runtime_seconds: int = 60,
    ) -> str:
        return classify_by_uptime(exit_code, uptime_seconds, min_clean_runtime_seconds)

    def start(
        self,
        worktree_path: Path,
        coral_md_path: Path,
        model: str = "deepseek-v4-flash",
        runtime_options: dict[str, Any] | None = None,
        max_turns: int = 0,
        log_dir: Path | None = None,
        verbose: bool = False,
        resume_session_id: str | None = None,
        prompt: str | None = None,
        prompt_source: str | None = None,
        task_name: str | None = None,
        task_description: str | None = None,
        gateway_url: str | None = None,
        gateway_api_key: str | None = None,
        run_as_user: dict[str, Any] | None = None,
        sandbox: AgentSandboxSpec | None = None,
    ) -> AgentHandle:
        """Start ``dsh --profile headless`` in the given worktree.

        DeepSeek Harness currently exposes no session-resume flag from its
        headless profile. A CORAL restart therefore starts a new dsh session
        and supplies the normal continuation prompt.
        """
        agent_id_file = worktree_path / ".coral_agent_id"
        agent_id = agent_id_file.read_text().strip() if agent_id_file.exists() else "unknown"

        if log_dir is None:
            log_dir = worktree_path / ".dsh" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_idx = len(list(log_dir.glob(f"{agent_id}*.log")))
        log_path = log_dir / f"{agent_id}.{log_idx}.log"

        if prompt is None:
            if resume_session_id:
                prompt = "Session restarted. Inspect the workspace and continue the task where it left off."
            else:
                prompt = "Begin working on your task and iterating on the seed solution."

        opts = runtime_options or {}
        for key in opts:
            if key not in _DEEPSEEK_HARNESS_RUNTIME_OPTION_KEYS:
                logger.warning(f"Ignoring unsupported deepseek_harness runtime option: {key}")

        command = str(opts.get("command") or "dsh")
        profile = str(opts.get("profile") or "headless")
        cmd = [command, "--profile", profile]
        patches = opts.get("patch")
        if isinstance(patches, (str, Path)):
            patches = [patches]
        if isinstance(patches, list):
            for patch in patches:
                cmd.extend(["--patch", str(patch)])

        # The headless CLI has no --model flag. A final Cordis overlay keeps
        # CORAL's agents.model contract authoritative over profile defaults and
        # any user-supplied patches.
        dsh_home = worktree_path / ".dsh"
        dsh_home.mkdir(parents=True, exist_ok=True)
        model_patch_path = dsh_home / "coral-model.patch.yml"
        provider = str(opts.get("provider") or "deepseek-official")
        model_patch_path.write_text(
            "- id: agent-default-model\n"
            "  config:\n"
            f"    provider: {json.dumps(provider)}\n"
            f"    model: {json.dumps(model)}\n"
        )
        cmd.extend(["--patch", str(model_patch_path)])
        cmd.append(prompt)
        cmd = apply_sandbox(cmd, sandbox)

        logger.info(f"Starting DeepSeek Harness agent {agent_id} in {worktree_path}")
        logger.info(f"Command: {' '.join(cmd)}")

        agent_env = _clean_env()
        worktree_venv = str(worktree_path / ".venv")
        agent_env["UV_PROJECT_ENVIRONMENT"] = worktree_venv
        agent_env["VIRTUAL_ENV"] = worktree_venv
        # Prepend the venv executable dir (bin or Scripts) to PATH, matching
        # the platform-aware resolution used by the other builtin runtimes.
        venv_bin = str(venv_bin_dir(worktree_path / ".venv"))
        agent_env["PATH"] = venv_bin + os.pathsep + agent_env.get("PATH", "")
        agent_env["DSH_HOME"] = str(dsh_home)

        permission_mode = opts.get("permission_mode")
        if permission_mode:
            agent_env["DSH_PERMISSION_MODE"] = str(permission_mode)
        tools_mode = opts.get("tools_mode")
        if tools_mode:
            agent_env["DSH_TOOLS_MODE"] = str(tools_mode)
        if gateway_url:
            agent_env["DEEPSEEK_BASE_URL"] = gateway_url
        if gateway_api_key:
            agent_env["DEEPSEEK_API_KEY"] = gateway_api_key

        apply_sandbox_env(agent_env, sandbox)
        user_kwargs = apply_run_as_user(agent_env, run_as_user)

        log_file = open(log_path, "w", buffering=1, encoding="utf-8", errors="replace")
        err_path: Path | None = None
        err_file: Any = None
        stderr_target: Any = subprocess.STDOUT
        opened = open_agent_stderr_for_log_dir(log_dir, agent_id)
        if opened is not None:
            err_path, err_file = opened
            stderr_target = err_file

        write_coral_log_entry(
            log_file,
            prompt=prompt,
            source=prompt_source or ("restart" if resume_session_id else "start"),
            agent_id=agent_id,
            session_id=None,
            task_name=task_name,
            task_description=task_description,
        )

        if verbose:
            process = subprocess.Popen(
                cmd,
                cwd=str(worktree_path),
                stdout=subprocess.PIPE,
                stderr=stderr_target,
                start_new_session=True,
                env=agent_env,
                **user_kwargs,
            )

            def _tee_output(proc: subprocess.Popen, log_f: Any, agent: str) -> None:
                try:
                    if proc.stdout is None:
                        return
                    for line in iter(proc.stdout.readline, b""):
                        decoded = line.decode("utf-8", errors="replace")
                        sys.stdout.write(f"[{agent}] {decoded}")
                        sys.stdout.flush()
                        log_f.write(decoded)
                        log_f.flush()
                finally:
                    log_f.close()

            threading.Thread(
                target=_tee_output, args=(process, log_file, agent_id), daemon=True
            ).start()
            log_file_ref = None
        else:
            process = subprocess.Popen(
                cmd,
                cwd=str(worktree_path),
                stdout=log_file,
                stderr=stderr_target,
                start_new_session=True,
                env=agent_env,
                **user_kwargs,
            )
            log_file_ref = log_file

        return AgentHandle(
            agent_id=agent_id,
            process=process,
            worktree_path=worktree_path,
            log_path=log_path,
            _log_file=log_file_ref,
            err_file=err_file,
            err_path=err_path,
        )
