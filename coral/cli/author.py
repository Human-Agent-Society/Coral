"""Commands: init, validate (formerly test-eval)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coral.task.validation import ValidationProgressEvent


def _module_identifier(name: str) -> str:
    """Sanitize a task directory name into a valid Python module identifier.

    'my-task'      -> 'my_task_grader'
    'My.Task!'     -> 'my_task_grader'
    '123-foo'      -> 'task_123_foo_grader'  (avoid leading digit)
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name).lower().strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        cleaned = "task"
    if cleaned[0].isdigit():
        cleaned = f"task_{cleaned}"
    return f"{cleaned}_grader"


def _distribution_name(name: str) -> str:
    """Sanitize a task directory name into a PEP 503 distribution name.

    'my-task'  -> 'my-task-grader'
    'My.Task!' -> 'my-task-grader'
    """
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", name).lower().strip("-")
    if not cleaned:
        cleaned = "task"
    return f"{cleaned}-grader"


def cmd_init(args: argparse.Namespace) -> None:
    """Create a new task directory with a packaged grader.

    Examples:
      coral init my-task            Scaffold at ./my-task/
      coral init my-task --name "My Task"
    """
    task_path = Path(args.path).resolve()
    task_name = args.name or task_path.name
    module_name = _module_identifier(task_path.name)
    dist_name = _distribution_name(task_path.name)

    if task_path.exists() and any(task_path.iterdir()):
        print(f"Error: {task_path} already exists and is not empty.", file=sys.stderr)
        sys.exit(1)

    task_path.mkdir(parents=True, exist_ok=True)
    (task_path / "seed").mkdir()
    grader_pkg_dir = task_path / "grader" / "src" / module_name
    grader_pkg_dir.mkdir(parents=True)

    (task_path / "task.yaml").write_text(
        f"task:\n"
        f'  name: "{task_name}"\n'
        f"  description: |\n"
        f"    Describe your task here. Agents read this verbatim from CORAL.md.\n"
        f"    Reference the program file by name (solution.py) and describe what\n"
        f'    it must do — e.g. "solution.py must print a single float to stdout".\n'
        f"\n"
        f"grader:\n"
        f'  entrypoint: "{module_name}.grader:Grader"\n'
        f"  setup:\n"
        f'    - "uv pip install -e ./grader"\n'
        f"  timeout: 300\n"
        f"  direction: maximize          # or 'minimize'\n"
        f"  args:\n"
        f'    program_file: "solution.py"\n'
        f"\n"
        f"agents:\n"
        f"  count: 1\n"
        f"  runtime: claude_code         # claude_code | codex | cursor | kiro | opencode | 'pkg.module:Cls' for a custom runtime\n"
        f"\n"
        f"workspace:\n"
        f'  repo_path: "./seed"          # relative to where you run `coral start`\n',
        encoding="utf-8",
    )

    (task_path / "seed" / "solution.py").write_text(
        f'"""Baseline solution for the {task_name} task.\n'
        "\n"
        "The grader runs this file and parses a single floating-point number\n"
        "from stdout as the score. Replace with your real implementation.\n"
        '"""\n'
        "\n"
        "print(0.0)\n",
        encoding="utf-8",
    )

    (task_path / "grader" / "pyproject.toml").write_text(
        f"[project]\n"
        f'name = "{dist_name}"\n'
        f'version = "0.1.0"\n'
        f'description = "CORAL grader for the {task_name} task."\n'
        f'requires-python = ">=3.11"\n'
        f"dependencies = [\n"
        f'    "coral",\n'
        f"]\n"
        f"\n"
        f"[build-system]\n"
        f'requires = ["hatchling"]\n'
        f'build-backend = "hatchling.build"\n'
        f"\n"
        f"[tool.hatch.build.targets.wheel]\n"
        f'packages = ["src/{module_name}"]\n',
        encoding="utf-8",
    )

    (grader_pkg_dir / "__init__.py").write_text(
        f'"""{task_name} grader (entrypoint: {module_name}.grader:Grader)."""\n'
        f"\n"
        f"from .grader import Grader\n"
        f"\n"
        f'__all__ = ["Grader"]\n',
        encoding="utf-8",
    )

    (grader_pkg_dir / "grader.py").write_text(
        f'"""{task_name} grader."""\n'
        f"\n"
        f"from coral.grader import TaskGrader\n"
        f"\n"
        f"\n"
        f"class Grader(TaskGrader):\n"
        f'    """Evaluate agent submissions for the {task_name} task."""\n'
        f"\n"
        f"    def evaluate(self) -> float:\n"
        f"        # self.codebase_path  — agent's worktree (read-only; writes are discarded)\n"
        f"        # self.private_dir    — .coral/private/ (hidden answer keys, fixtures)\n"
        f"        # self.args           — dict from task.yaml -> grader.args\n"
        f"        # self.timeout        — eval timeout in seconds\n"
        f"        #\n"
        f"        # Return a float, or use self.score(value, explanation=...)\n"
        f"        # or self.fail(reason) to record a failure with feedback.\n"
        f'        program_file = self.args.get("program_file", "solution.py")\n'
        f"        result = self.run_program(program_file)\n"
        f"\n"
        f"        if result.returncode != 0:\n"
        f'            return self.fail(f"{{program_file}} crashed: {{result.stderr[:200]}}")\n'
        f"\n"
        f"        try:\n"
        f"            return float(result.stdout.strip())\n"
        f"        except ValueError:\n"
        f'            return self.fail(f"Expected a single float on stdout, got: {{result.stdout[:80]!r}}")\n',
        encoding="utf-8",
    )

    print(f"Created task at {task_path}/")
    print("  task.yaml                 — task config + grader entrypoint")
    print("  seed/solution.py          — baseline the agent will iterate on")
    print(f"  grader/                   — packaged grader ({dist_name})")
    print(f"  grader/src/{module_name}/grader.py")
    print("\nNext:")
    print(f"  cd {task_path.name}")
    print("  coral validate .          # bootstraps grader venv + runs grader on seed/")
    print("  coral start -c task.yaml  # launch agents")


def _render_validation_progress(event: ValidationProgressEvent) -> None:
    """Render shared validation progress with the legacy CLI wording."""
    if event.stage == "structure" and event.status == "completed":
        print("Validation: OK")
    elif event.stage == "workspace" and event.status == "completed":
        print(event.message)
    elif event.stage == "grader_environment" and event.status == "started":
        print(event.message)
    elif event.stage == "baseline" and event.status == "started":
        print(f"\n{event.message}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Test your grader against seed code.

    Examples:
      coral validate my-task        Dry-run the grader in my-task/
    """
    from coral.task.validation import run_validation

    task_dir = Path(args.path).resolve()
    json_output = getattr(args, "json", False)
    on_event = None if json_output else _render_validation_progress
    result = run_validation(task_dir, on_event=on_event)

    if json_output:
        print(json.dumps(result.to_dict()))
        if not result.successful:
            sys.exit(1)
        return

    if result.failure is not None:
        if result.failure.stage == "structure" and result.report.error_messages:
            print("Validation errors:", file=sys.stderr)
            for error in result.report.error_messages:
                print(f"  - {error}", file=sys.stderr)
        elif result.failure.stage == "baseline":
            print(f"\n{result.failure.message}", file=sys.stderr)
        else:
            print(result.failure.message, file=sys.stderr)
        sys.exit(1)

    if result.baseline is None:
        print("Validation failed: no baseline result", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 50}")
    print(f"Score: {result.baseline.aggregated}")
    for name, score in result.baseline.scores.items():
        if score.explanation:
            print(f"  {name}: {score.explanation}")
    print(f"{'=' * 50}")
