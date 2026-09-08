"""CORAL CLI — clean, grouped command-line interface."""

from __future__ import annotations

import argparse
import difflib
import sys


class _GroupedHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter that suppresses the default subcommand list.

    We print our own grouped help in the epilog instead.
    """

    def _format_usage(self, usage, actions, groups, prefix):
        # Show clean usage without the giant {cmd1,cmd2,...} list
        return "usage: coral <command> [options]\n"

    def _format_action(self, action: argparse.Action) -> str:
        # Hide the auto-generated subcommand choices list
        if isinstance(action, argparse._SubParsersAction):
            return ""
        return super()._format_action(action)


class _CommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Formatter for individual commands — preserves docstring examples."""

    pass


class _HelpOnErrorParser(argparse.ArgumentParser):
    """ArgumentParser that prints help alongside error messages."""

    def error(self, message: str) -> None:
        sys.stderr.write(f"\nerror: {message}\n\n")
        self.print_help(sys.stderr)
        sys.exit(2)


# All visible commands for "did you mean?" suggestions
_VISIBLE_COMMANDS = [
    "init",
    "validate",
    "start",
    "resume",
    "stop",
    "status",
    "log",
    "show",
    "notes",
    "skills",
    "runs",
    "ui",
    "eval",
    "wait",
    "diff",
    "revert",
    "checkout",
    "export",
    "heartbeat",
    "setup",
    "agents",
]


class _MainParser(_HelpOnErrorParser):
    """Top-level parser with 'did you mean?' suggestions for unknown commands."""

    def error(self, message: str) -> None:
        # Check for unknown command and suggest closest match
        if "invalid choice:" in message:
            # Extract the bad command from the error message
            try:
                bad_cmd = message.split("'")[1]
            except IndexError:
                bad_cmd = None
            if bad_cmd:
                matches = difflib.get_close_matches(bad_cmd, _VISIBLE_COMMANDS, n=3, cutoff=0.5)
                sys.stderr.write(f"\nerror: unknown command '{bad_cmd}'\n")
                if matches:
                    sys.stderr.write("\nDid you mean?\n")
                    for m in matches:
                        sys.stderr.write(f"  coral {m}\n")
                sys.stderr.write("\n")
                self.print_help(sys.stderr)
                sys.exit(2)
        super().error(message)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Add the common --task and --run flags."""
    parser.add_argument("--task", help="Task name (auto-detected if omitted)")
    parser.add_argument("--run", help="Run ID (defaults to latest)")


def _non_empty_status(value: str) -> str:
    """Validate and normalize a status supplied on the command line."""
    normalized = value.strip()
    if not normalized:
        raise argparse.ArgumentTypeError("status must be a non-empty string")
    return normalized


def main() -> None:
    from coral import __version__

    epilog = """\
Getting Started:
  init            Create a new task directory
  validate        Test your grader against seed code

Running Agents:
  start           Launch agents on a task
  resume          Resume a previous run
  stop            Shut down running agents
  status          Show agent health and leaderboard

Inspecting Results:
  log             List and search attempts (leaderboard)
  show            Show details of a specific attempt
  notes           Browse shared notes
  skills          Browse shared skills
  runs            List runs (active only; --all for stopped)

User Setup:
  setup           Detect installed agent runtimes + configure bindings
  agents          List, inspect, and validate agent bindings

Dashboard:
  ui              Launch the web dashboard

Agent Internals:
  eval            Stage, commit, and evaluate changes
  wait            Wait for a submitted eval's score
  diff            Show uncommitted changes
  revert          Undo the last commit
  checkout        Reset to a previous attempt
  export          Export an attempt as a git branch
  heartbeat       View/modify per-agent heartbeat actions

Run 'coral <command> --help' for details on any command."""

    parser = _MainParser(
        prog="coral",
        description=f"CORAL v{__version__} \u2014 Autonomous agent orchestration",
        epilog=epilog,
        formatter_class=_GroupedHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"coral {__version__}")
    sub = parser.add_subparsers(dest="command", prog="coral")

    # --- Getting Started ---

    p_init = sub.add_parser(
        "init",
        help="Create a new task directory",
        description="Create a new task directory with scaffolded config and grader.",
        epilog="Examples:\n  coral init my-task\n  coral init my-task --name 'My Task'",
        formatter_class=_CommandHelpFormatter,
    )
    p_init.add_argument("path", help="Path for the new task directory")
    p_init.add_argument("--name", help="Task name (default: directory name)")

    p_validate = sub.add_parser(
        "validate",
        help="Test your grader against seed code",
        description="Validate task structure and dry-run the grader against seed code.",
        epilog="Examples:\n  coral validate my-task\n  coral validate my-task --json",
        formatter_class=_CommandHelpFormatter,
    )
    p_validate.add_argument("path", help="Path to the task directory")
    p_validate.add_argument(
        "--json",
        action="store_true",
        help="Output one machine-readable validation result",
    )
    # Hidden alias: test-eval -> validate (mirror validate's arguments so the
    # alias can actually dispatch; cmd_validate reads args.path)
    p_test_eval = sub.add_parser("test-eval", help=argparse.SUPPRESS)
    p_test_eval.add_argument("path", help=argparse.SUPPRESS)
    p_test_eval.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    # --- Running Agents ---

    p_start = sub.add_parser(
        "start",
        help="Launch agents on a task",
        description="Launch autonomous agents on a task. Auto-wraps in tmux by default.",
        epilog=(
            "Examples:\n"
            "  coral start -c task.yaml\n"
            "  coral start -c task.yaml agents.count=4 agents.model=opus\n"
            "  coral start -c task.yaml run.verbose=true run.ui=true run.session=local"
        ),
        formatter_class=_CommandHelpFormatter,
    )
    p_start.add_argument("--config", "-c", required=True, help="Path to task config YAML")
    # Internal: the tmux/docker wrapper sets this so the inner (run.session=local)
    # process knows which session mode to restore into the saved config. Not for
    # direct use — inferring from in_tmux()/in_docker() would misfire when a user
    # runs run.session=local from their own tmux session or container.
    p_start.add_argument(
        "--wrapped-session",
        dest="wrapped_session",
        choices=["tmux", "docker"],
        help=argparse.SUPPRESS,
    )
    p_start.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help="Config overrides as key=value (e.g. agents.count=4 run.verbose=true run.session=local)",
    )

    p_resume = sub.add_parser(
        "resume",
        help="Resume a previous run",
        description="Resume agents from a previous run, restoring their sessions.",
        epilog="Examples:\n  coral resume\n  coral resume --task my-task agents.model=opus",
        formatter_class=_CommandHelpFormatter,
    )
    _add_run_args(p_resume)
    p_resume.add_argument(
        "--instruction",
        "-i",
        type=str,
        default=None,
        help="Additional instruction to inject into agents at resume time",
    )
    p_resume.add_argument(
        "--from",
        dest="resume_from",
        type=str,
        default=None,
        help="Attempt hash to reset an agent worktree to before resuming",
    )
    p_resume.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help="Config overrides as key=value (e.g. agents.model=opus run.verbose=true)",
    )

    p_stop = sub.add_parser(
        "stop",
        help="Shut down running agents",
        description="Gracefully stop the CORAL manager and all agents.",
        formatter_class=_CommandHelpFormatter,
    )
    p_stop.add_argument("--all", action="store_true", help="Stop all active runs")
    _add_run_args(p_stop)

    p_status = sub.add_parser(
        "status",
        help="Show agent health and leaderboard",
        description="Show manager/agent status and top leaderboard entries.",
        formatter_class=_CommandHelpFormatter,
    )
    p_status.add_argument(
        "--all",
        action="store_true",
        help="Include tune-mode and grader-error attempts in the body summary "
        "and leaderboard (hidden by default, matching `coral log`)",
    )
    _add_run_args(p_status)

    # --- Inspecting Results ---

    p_log = sub.add_parser(
        "log",
        help="List and search attempts (leaderboard)",
        description="List and search attempts. Default: top 20 sorted by score.",
        epilog=(
            "Examples:\n"
            "  coral log                     Top 20 by score\n"
            "  coral log -n 5                Top 5\n"
            "  coral log --recent            Sort by time instead of score\n"
            "  coral log --agent agent-1     Filter by agent\n"
            "  coral log --search 'kernel'   Full-text search\n"
            "  coral log --all               Include tune + grader_error attempts\n"
            "  coral log --class tune        Show only tune-mode attempts"
        ),
        formatter_class=_CommandHelpFormatter,
    )
    p_log.add_argument(
        "-n", "--count", type=int, default=20, help="Number of results (default: 20)"
    )
    p_log.add_argument("--recent", action="store_true", help="Sort by time instead of score")
    p_log.add_argument("--agent", help="Filter by agent ID")
    p_log.add_argument("--search", help="Full-text search")
    g_class = p_log.add_mutually_exclusive_group()
    g_class.add_argument(
        "--all",
        action="store_true",
        help="Include tune-mode and grader-error attempts (hidden by default)",
    )
    g_class.add_argument(
        "--class",
        dest="budget_class",
        choices=("real", "tune", "grader_error"),
        help="Show only attempts of this budget class (mutually exclusive with --all)",
    )
    _add_run_args(p_log)
    # Hidden alias: attempts -> log
    p_attempts_alias = sub.add_parser("attempts", help=argparse.SUPPRESS)
    p_attempts_alias.add_argument("--top", type=int, help=argparse.SUPPRESS)
    p_attempts_alias.add_argument("--recent", type=int, help=argparse.SUPPRESS)
    p_attempts_alias.add_argument("--agent", help=argparse.SUPPRESS)
    p_attempts_alias.add_argument("--search", help=argparse.SUPPRESS)
    _add_run_args(p_attempts_alias)

    p_show = sub.add_parser(
        "show",
        help="Show details of a specific attempt",
        description="Show full details and diff for a specific attempt.",
        epilog="Examples:\n  coral show abc123\n  coral show <full-commit-hash>",
        formatter_class=_CommandHelpFormatter,
    )
    p_show.add_argument("hash", help="Commit hash or prefix")
    p_show.add_argument(
        "--diff", action="store_true", default=False, help="Show full code diff instead of summary"
    )
    _add_run_args(p_show)
    # Hidden alias: attempt -> show
    p_attempt_alias = sub.add_parser("attempt", help=argparse.SUPPRESS)
    p_attempt_alias.add_argument("hash", help=argparse.SUPPRESS)
    p_attempt_alias.add_argument(
        "--diff", action="store_true", default=False, help=argparse.SUPPRESS
    )
    _add_run_args(p_attempt_alias)

    p_notes = sub.add_parser(
        "notes",
        help="Browse shared notes",
        description="List, search, or read agent notes.",
        epilog=(
            "Examples:\n"
            "  coral notes                   List all notes\n"
            "  coral notes -n 5              Last 5 notes\n"
            "  coral notes --search 'idea'   Search notes\n"
            "  coral notes --status confirmed  Filter by note status\n"
            "  coral notes --read 3          Read note #3\n"
            "  coral notes --audit           Check recorded experiment evidence"
        ),
        formatter_class=_CommandHelpFormatter,
    )
    p_notes.add_argument("--search", "-s", help="Search notes by keyword")
    p_notes.add_argument(
        "--status",
        metavar="STATUS",
        type=_non_empty_status,
        help="Filter by frontmatter status (e.g. confirmed, refuted, untested)",
    )
    p_notes.add_argument("-n", "--recent", type=int, help="Show N most recent")
    p_notes.add_argument("--read", "-r", help="Read a specific note by number or name")
    p_notes.add_argument(
        "--history", action="store_true", help="Show shared state checkpoint history"
    )
    p_notes.add_argument("--diff", metavar="HASH", help="Show diff for a checkpoint commit")
    p_notes.add_argument(
        "--audit",
        action="store_true",
        help="Read-only evidence audit (supports --status, --search and -n)",
    )
    _add_run_args(p_notes)

    p_skills = sub.add_parser(
        "skills",
        help="Browse shared skills",
        description="List skills or show details of a specific skill.",
        epilog="Examples:\n  coral skills\n  coral skills --read optimizer",
        formatter_class=_CommandHelpFormatter,
    )
    p_skills.add_argument("--read", "-r", help="Show details of a skill (name or prefix)")
    _add_run_args(p_skills)

    p_runs = sub.add_parser(
        "runs",
        help="List all runs across tasks",
        description="List all CORAL runs. Default: active runs only, most recent first.",
        epilog=(
            "Examples:\n"
            "  coral runs                    Active runs only\n"
            "  coral runs --all              Include stopped runs\n"
            "  coral runs --task my-task     Filter by task\n"
            "  coral runs -n 5              Show at most 5 runs"
        ),
        formatter_class=_CommandHelpFormatter,
    )
    p_runs.add_argument("--all", "-a", action="store_true", help="Include stopped runs")
    p_runs.add_argument("--task", "-t", help="Filter by task name")
    p_runs.add_argument(
        "-n", "--count", type=int, default=20, help="Number of results (default: 20)"
    )
    p_runs.add_argument("--verbose", "-v", action="store_true", help="Show full paths")

    # --- Dashboard ---

    p_ui = sub.add_parser(
        "ui",
        help="Launch the web dashboard",
        description="Start the CORAL web dashboard for monitoring runs.",
        epilog="Examples:\n  coral ui\n  coral ui --port 9000",
        formatter_class=_CommandHelpFormatter,
    )
    p_ui.add_argument("--port", type=int, default=None, help="Port (default: 8420)")
    p_ui.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    _add_run_args(p_ui)
    p_ui.add_argument("--no-open", action="store_true", help="Don't auto-open browser")

    # --- Agent Internals ---

    p_eval = sub.add_parser(
        "eval",
        help="Stage, commit, and evaluate changes",
        description=(
            "Stage all changes, commit with a message, and submit for grading.\n"
            "By default blocks until the grader daemon returns a score.\n"
            "Use --no-wait to return immediately with a pending status and\n"
            "poll later via `coral wait <hash>`."
        ),
        epilog=(
            "Examples:\n"
            '  coral eval -m "Optimized inner loop"\n'
            '  coral eval -m "Try variant A" --no-wait\n'
            '  coral eval -m "Heavy benchmark" --timeout 1800\n'
            '  coral eval --tune -m "Sweep lr=1e-3 vs 3e-4"\n'
            '  coral eval -m "Rewrite prompt" --operator prompt --mutation rewrite'
        ),
        formatter_class=_CommandHelpFormatter,
    )
    p_eval.add_argument(
        "-m", "--message", required=True, help="Description of what you changed and why"
    )
    p_eval.add_argument("--agent", help="Agent ID (default: read from .coral_agent_id)")
    p_eval.add_argument("--workdir", help="Working directory (default: cwd)")
    p_eval.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for grader to return a score (default). Use --no-wait to return immediately.",
    )
    p_eval.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Seconds to wait for grader (default: derived from grader.timeout).",
    )
    p_eval.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help=(
            "Submit as a tune-mode attempt: scored and recorded normally, but "
            "excluded from the agent's plateau / heartbeat budget. Use for "
            "hyperparameter sweeps and config exploration that shouldn't "
            "trigger pivot prompts."
        ),
    )
    p_eval.add_argument(
        "--operator",
        help="Meta-evolve operator for this eval (requires --mutation)",
    )
    p_eval.add_argument(
        "--mutation",
        help="Meta-evolve mutation for this eval (requires --operator)",
    )

    p_wait = sub.add_parser(
        "wait",
        help="Wait for a submitted eval's score",
        description=(
            "Block until the grader daemon finalizes a previously submitted\n"
            "attempt (e.g. one submitted with `coral eval --no-wait`)."
        ),
        epilog="Examples:\n  coral wait abc123\n  coral wait abc123 --timeout 600",
        formatter_class=_CommandHelpFormatter,
    )
    p_wait.add_argument("hash", help="Commit hash or prefix of the attempt to wait on")
    p_wait.add_argument("--workdir", help="Working directory (default: cwd)")
    p_wait.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Seconds to wait (default: derived from grader.timeout).",
    )

    p_diff = sub.add_parser(
        "diff",
        help="Show uncommitted changes",
        description="Show staged and unstaged changes in the working tree.",
        formatter_class=_CommandHelpFormatter,
    )
    p_diff.add_argument("--workdir", help="Working directory (default: cwd)")

    p_revert = sub.add_parser(
        "revert",
        help="Undo the last commit",
        description="Reset to HEAD~1, discarding the last commit and its changes.",
        formatter_class=_CommandHelpFormatter,
    )
    p_revert.add_argument("--workdir", help="Working directory (default: cwd)")

    p_checkout = sub.add_parser(
        "checkout",
        help="Reset to a previous attempt",
        description="Reset the working tree to a previous attempt's commit.",
        epilog="Examples:\n  coral checkout abc123",
        formatter_class=_CommandHelpFormatter,
    )
    p_checkout.add_argument("hash", help="Commit hash or prefix")
    p_checkout.add_argument("--workdir", help="Working directory (default: cwd)")
    _add_run_args(p_checkout)

    p_export = sub.add_parser(
        "export",
        help="Export an attempt as a git branch",
        description=(
            "Create a normal git branch pointing at an attempt's commit in the "
            "run's source repo, so it can be checked out with a standard git workflow."
        ),
        epilog="Examples:\n  coral export abc123 --branch coral/better-scheduler",
        formatter_class=_CommandHelpFormatter,
    )
    p_export.add_argument("hash", help="Commit hash or prefix")
    p_export.add_argument("-b", "--branch", required=True, help="Name of the branch to create")
    p_export.add_argument(
        "-f", "--force", action="store_true", help="Overwrite the branch if it exists"
    )
    _add_run_args(p_export)

    p_heartbeat = sub.add_parser(
        "heartbeat",
        help="View/modify per-agent heartbeat actions",
        description="Show or modify per-agent heartbeat configuration.",
        epilog=(
            "Examples:\n"
            "  coral heartbeat                              Show current config\n"
            "  coral heartbeat set reflect --every 3        Reflect every 3 evals\n"
            '  coral heartbeat set review --every 5 --prompt "..."  Custom action\n'
            "  coral heartbeat remove consolidate           Remove action\n"
            "  coral heartbeat reset                        Reset to task YAML defaults"
        ),
        formatter_class=_CommandHelpFormatter,
    )
    _add_run_args(p_heartbeat)
    hb_sub = p_heartbeat.add_subparsers(dest="heartbeat_command")

    hb_set = hb_sub.add_parser("set", help="Add or update a heartbeat action")
    hb_set.add_argument("name", help="Action name (e.g. reflect, consolidate)")
    hb_set.add_argument(
        "--every",
        type=int,
        required=True,
        help="Trigger every N evals (or stall threshold for plateau)",
    )
    hb_set.add_argument("--prompt", help="Prompt text (required for custom actions)")
    hb_set.add_argument(
        "--trigger",
        choices=["interval", "plateau"],
        default=None,
        help="Trigger type: 'interval' (every N evals) or 'plateau' (after N non-improving evals)",
    )
    hb_set.add_argument(
        "--global",
        dest="is_global",
        action="store_true",
        default=None,
        help="Use global eval counter (shared across all agents)",
    )
    hb_set.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help=(
            "Minimum score delta over the prior plateau-anchor that counts as "
            "improvement (only meaningful for --trigger plateau). Default 0 = "
            "any nudge resets the streak. Set to your task's noise floor "
            "(e.g. 0.001) so tiny inch-ups don't keep blocking pivot."
        ),
    )
    _add_run_args(hb_set)

    hb_remove = hb_sub.add_parser("remove", help="Remove a heartbeat action")
    hb_remove.add_argument("name", help="Action name to remove")
    _add_run_args(hb_remove)

    hb_reset = hb_sub.add_parser("reset", help="Reset to task YAML defaults")
    _add_run_args(hb_reset)

    # --- User Setup: agent bindings ---

    p_setup = sub.add_parser(
        "setup",
        help="Detect agent runtimes / configure user-level agent bindings",
        description=(
            "With no subcommand, scan PATH for installed agent runtime CLIs and "
            "offer to create a binding for each one (interactive). With "
            "`coral setup agent`, create or update a named binding non-interactively. "
            "A binding bundles a runtime, CLI command, model, runtime options, and "
            "optional role seed so tasks can reference it by name."
        ),
        epilog=(
            "Examples:\n"
            "  coral setup                               Detect runtimes + wizard\n"
            "  coral setup --non-interactive             Just print the detection report\n"
            "  coral setup agent                         Interactive single-binding setup\n"
            "  coral setup agent --name claude-opus --runtime claude_code --model opus\n"
            "  coral setup agent --name codex-high --runtime codex --option model_reasoning_effort=high"
        ),
        formatter_class=_CommandHelpFormatter,
    )
    p_setup.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt; just print the detection report and exit",
    )
    p_setup.add_argument("--config", help="Path to the user bindings file (advanced)")
    setup_sub = p_setup.add_subparsers(dest="setup_command")
    sp_agent = setup_sub.add_parser(
        "agent",
        help="Create or update a named agent binding",
        formatter_class=_CommandHelpFormatter,
    )
    sp_agent.add_argument("--name", help="Binding name (e.g. claude-opus)")
    sp_agent.add_argument("--runtime", help="Runtime (claude_code, codex, opencode, ...)")
    sp_agent.add_argument(
        "--command",
        dest="command_path",
        help="CLI binary (defaults to the runtime's command)",
    )
    sp_agent.add_argument("--model", help="Default model for this binding")
    sp_agent.add_argument("--role-file", dest="role_file", help="Path to a role seed .md file")
    sp_agent.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Runtime option (repeatable)",
    )
    sp_agent.add_argument("--default", action="store_true", help="Make this the default binding")
    sp_agent.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt; require values via flags",
    )
    sp_agent.add_argument("--config", help="Path to the user bindings file (advanced)")

    p_agents = sub.add_parser(
        "agents",
        help="List, inspect, and validate agent bindings",
        description="Manage user-level agent bindings stored in ~/.config/coral/agents.yaml.",
        epilog=(
            "Examples:\n"
            "  coral agents list                 List all bindings\n"
            "  coral agents show claude-opus     Inspect one binding\n"
            "  coral agents doctor               Validate all bindings\n"
            "  coral agents remove claude-opus   Delete a binding"
        ),
        formatter_class=_CommandHelpFormatter,
    )
    ag_sub = p_agents.add_subparsers(dest="agents_command")
    ag_list = ag_sub.add_parser("list", help="List all bindings")
    ag_list.add_argument("--config", help="Path to the user bindings file (advanced)")
    ag_show = ag_sub.add_parser("show", help="Show one binding")
    ag_show.add_argument("name", help="Binding name")
    ag_show.add_argument("--config", help="Path to the user bindings file (advanced)")
    ag_remove = ag_sub.add_parser(
        "remove",
        help="Delete one or more bindings (interactive when no names given)",
        description=(
            "Delete bindings. Pass one or more names to remove them directly, or "
            "pass no name to launch an interactive numbered-selection wizard."
        ),
        epilog=(
            "Examples:\n"
            "  coral agents remove                        Interactive selection\n"
            "  coral agents remove claude-opus            Remove one binding\n"
            "  coral agents remove claude-opus codex-high Remove several at once"
        ),
        formatter_class=_CommandHelpFormatter,
    )
    ag_remove.add_argument(
        "names",
        nargs="*",
        help="Binding name(s); omit for interactive selection",
    )
    ag_remove.add_argument("--config", help="Path to the user bindings file (advanced)")
    ag_doctor = ag_sub.add_parser(
        "doctor",
        help="Validate bindings (includes a live hello-ping by default)",
        description=(
            "Validate bindings. By default this includes a live hello-ping that "
            "spawns the runtime CLI, sends a tiny prompt, and waits for output — "
            "useful for catching auth failures and stale installs, but every ping "
            "is one LLM round-trip per binding. Pass --no-live for the cheap, "
            "metadata-only checks (resolves to AgentSpec / CLI present / "
            "--version works / role_file exists)."
        ),
        epilog=(
            "Examples:\n"
            "  coral agents doctor                 Validate all bindings (with live ping)\n"
            "  coral agents doctor claude-opus     Validate one binding\n"
            "  coral agents doctor --no-live       Skip the LLM round-trip (CI / quick check)\n"
            "  coral agents doctor --timeout 60    Allow up to 60s per ping"
        ),
        formatter_class=_CommandHelpFormatter,
    )
    ag_doctor.add_argument("name", nargs="?", help="Binding to check (default: all)")
    ag_doctor.add_argument("--config", help="Path to the user bindings file (advanced)")
    ag_doctor.add_argument(
        "--no-live",
        dest="no_live",
        action="store_true",
        help="Skip the live hello-ping (only run cheap metadata checks)",
    )
    ag_doctor.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-binding ping timeout in seconds (default: 30)",
    )

    # --- Parse and dispatch ---

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Lazy imports for fast startup
    from coral.cli.agents import cmd_agents, cmd_setup
    from coral.cli.author import cmd_init, cmd_validate
    from coral.cli.eval import (
        cmd_checkout,
        cmd_diff,
        cmd_eval,
        cmd_export,
        cmd_revert,
        cmd_wait,
    )
    from coral.cli.heartbeat import cmd_heartbeat
    from coral.cli.query import cmd_log, cmd_notes, cmd_runs, cmd_show, cmd_skills
    from coral.cli.start import cmd_resume, cmd_start, cmd_status, cmd_stop
    from coral.cli.ui import cmd_ui

    commands = {
        "start": cmd_start,
        "resume": cmd_resume,
        "stop": cmd_stop,
        "status": cmd_status,
        "eval": cmd_eval,
        "wait": cmd_wait,
        "revert": cmd_revert,
        "checkout": cmd_checkout,
        "export": cmd_export,
        "diff": cmd_diff,
        "heartbeat": cmd_heartbeat,
        "log": cmd_log,
        "show": cmd_show,
        "notes": cmd_notes,
        "skills": cmd_skills,
        "runs": cmd_runs,
        "init": cmd_init,
        "validate": cmd_validate,
        "setup": cmd_setup,
        "agents": cmd_agents,
        "ui": cmd_ui,
        # Hidden aliases for backward compatibility
        "attempts": _cmd_attempts_compat,
        "attempt": cmd_show,
        "test-eval": cmd_validate,
    }
    commands[args.command](args)


def _cmd_attempts_compat(args: argparse.Namespace) -> None:
    """Backward-compatible wrapper: translates old attempts flags to new log flags."""
    from coral.cli.query import cmd_log

    # Map old --top N and --recent N to new --count N and --recent (bool)
    if hasattr(args, "top") and args.top:
        args.count = args.top
    elif not hasattr(args, "count") or args.count is None:
        args.count = 20

    if hasattr(args, "recent") and isinstance(args.recent, int) and args.recent:
        args.count = args.recent
        args.recent = True
    elif not hasattr(args, "recent") or args.recent is None:
        args.recent = False

    cmd_log(args)


if __name__ == "__main__":
    main()
