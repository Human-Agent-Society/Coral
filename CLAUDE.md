# CORAL

`CLAUDE.md` is the shared project and contribution guide for all coding agents.
`AGENTS.md` is a relative symlink to this file; edit this file to update either entrypoint.

An orchestration system for **autonomous coding agents** — agents follow a CORAL.md guide, run experiments, share knowledge, and loop forever.

## Project Overview

Core pattern: **Spawn agents → agents read CORAL.md → commit changes → grader daemon scores them → repeat**

Key concepts:
- **Agents are the optimizers** — Claude Code (or Codex / Cursor / Kiro / OpenCode) subprocesses, each in its own git worktree.
- **Shared state via `.coral/`** — split into `public/` (visible to agents through a runtime-specific symlink like `.claude/`, `.codex/`, `.opencode/`) and `private/` (grader venv, hidden inputs — denied to agents). The grader's own *source* is surfaced read-only to agents as `<shared_dir>/grader/` (a symlink to the real `grader/` package) so they can read how they're scored.
- **Async eval loop** — `coral eval -m "..."` stages+commits and writes a *pending* attempt; a long-running grader daemon picks it up, grades it inside a detached worktree, and writes the final score back. Default behavior blocks until the score lands; `--no-wait` returns immediately.
- **Optional meta-evolve loop** — configured operator/mutation pairs are recorded on real attempts, rebuilt into direction-aware parent lift statistics, and recommended only in `pivot` heartbeats. It is disabled by default.
- **CLI orchestration** — 18 commands (see Commands below), grouped under `coral start / status / eval / log / ...`.

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `coral/types.py` | Core types: `Task`, `Score`, `ScoreBundle`, `Attempt` |
| `coral/config.py` | OmegaConf-backed YAML configuration (`CoralConfig`, `GraderConfig`, `AgentConfig`, `GatewayConfig`, `WarmStartConfig`, `HeartbeatActionConfig`, ...) |
| `coral/task/` | Frontend-independent task validation with structured reports, progress events, and baseline grading |
| `coral/agent/` | Agent lifecycle: `manager.py` (multi-agent supervisor), `meta_evolve.py` (attempt attribution, lift statistics, and recommendation), `runtime.py` (abstract), `state.py`, `heartbeat.py`, `exit_classifier.py`, `warmstart.py`, `process.py`, `registry.py` |
| `coral/sandbox/` | Pluggable agent sandboxing (`agents.sandbox`): `protocol.py` (`SandboxProvider` + spec/context types), `registry.py` (name or `module:Class` entrypoint resolution), `srt.py` (built-in srt provider: OS-level FS/network enforcement + allow-all proxy) |
| `coral/agent/builtin/` | Concrete runtimes: `claude_code`, `codex`, `cursor_agent`, `kiro`, `opencode` |
| `coral/grader/` | Grader stack: `protocol.py`, `base.py`, `task_grader.py`, `loader.py`, `subprocess_grader.py`, `daemon.py` (long-running grader), `builtin/function_grader.py` |
| `coral/hub/` | Shared state: `attempts.py`, `notes.py`, `skills.py`, `checkpoint.py` (git-tracked snapshots of `.coral/public/`), `heartbeat.py`, `prompts/` (built-in heartbeat prompts) |
| `coral/hooks/` | `post_commit.py` — implements `submit_eval` (called by `coral eval`) |
| `coral/workspace/` | Run layout: `project.py` (run dir setup), `worktree.py` (per-agent git worktrees + symlinks), `repo.py` (clone/init), `grader_env.py` (`.coral/private/grader_venv/`) |
| `coral/template/` | `coral_md.py` + `coral.md.template` / `coral_single.md.template`; bundled `agents/` (deep-researcher, librarian) and `skills/` (deep-research, organize-files, skill-creator) seeded into every run |
| `coral/gateway/` | Optional LiteLLM gateway (`server.py`, `middleware.py`, `config.py`) for intercepting agent model traffic |
| `coral/web/` | Starlette web dashboard (`app.py`, `api.py`, `events.py`, `logs.py`, `static/`) |
| `coral/cli/` | CLI package: `start.py`, `query.py`, `eval.py`, `heartbeat.py`, `agents.py` (user-level bindings), `ui.py`, `author.py`, `validation.py`, `_helpers.py` |
| `coral/user_agents.py` | User-level agent bindings: load/save `~/.config/coral/agents.yaml`, expanded into concrete agent fields by `config._expand_bindings` |
| `examples/` | Task configs (circle_packing, swebench-verified, kernel_engineering, mnist, ...) — each is a `task.yaml` + `seed/` + packaged grader (`grader/` referenced by `grader.entrypoint`); hidden data is declared in `grader.private` (copied into `.coral/private/`) and must live **outside** the agent-visible `grader/` package |
| `plugin/` | Skills-first, multi-harness plugin for driving `coral` from another harness (Superpowers-style: one shared `skills/`, per-harness `.claude-plugin/` + `.codex-plugin/` manifests, per-harness `hooks/` with a SessionStart install check). Per-harness marketplace manifests at the repo root — `.claude-plugin/marketplace.json` (Claude) and `.agents/plugins/marketplace.json` (Codex git-backed, `git-subdir` → `./plugin`) — for `owner/repo` discovery. No MCP. See `plugin/README.md`. |
| `tests/` | Pytest suite (config, grader, hooks, hub, manager reliability, daemon, workspace, ...) |

## How It Works

```
coral start --config task.yaml
  → results/<task-slug>/<timestamp>/        ← run_dir
    ├── .coral/
    │   ├── public/        symlinked into each worktree as .claude/ (or .codex/.opencode/...)
    │   │   ├── attempts/  pending + final ScoreBundle JSONs (one per commit hash)
    │   │   ├── notes/     agent-written markdown
    │   │   ├── skills/    agent-built reusable tools (seeded with deep-research, ...)
    │   │   ├── agents/    subagent definitions (seeded with deep-researcher, librarian)
    │   │   ├── logs/, eval_logs/, heartbeat/, eval_count
    │   │   └── grader_daemon.pid, grader_daemon_heartbeat
    │   ├── private/
    │   │   ├── grader_venv/   isolated uv venv where the grader entrypoint runs
    │   │   └── ...        anything listed in grader.private (hidden from agents)
    │   ├── config.yaml, config_dir
    │   └── .git/          checkpoint repo for shared-state versioning
    ├── repo/              cloned source repo (each run is independent)
    └── agents/<agent_id>/ git worktree on branch coral/<agent_id>; .claude/ → .coral/public/

  → Bootstraps .coral/private/grader_venv/ via `uv venv` and runs
    grader.setup commands inside it (grader.entrypoint is required).
  → Spawns the chosen runtime per agent (claude_code default).
  → Starts the grader daemon as a sibling process.

Each agent loop:
  → Reads CORAL.md (generated by coral/template/coral_md.py)
  → Edits files, then runs `coral eval -m "description"`
    - submit_eval (coral/hooks/post_commit.py) does git add -A + commit
    - when agents.meta_evolve is enabled, a configured --operator/--mutation pair is validated before git mutation and stored in Attempt.metadata
    - writes a pending Attempt JSON to .coral/public/attempts/<hash>.json
    - by default, polls until the daemon finalizes the score (use --no-wait to return immediately + `coral wait <hash>` later)
  → Grader daemon (coral/grader/daemon.py):
    - watches .coral/public/attempts/ for new pending entries
    - dispatches through a thread pool of size grader.parallel.max_workers (default 1)
    - each worker grades inside `git worktree add --detach <commit>` so agent commits during grading don't perturb the grader's view
    - reuses one TaskGrader instance per worker (no per-eval cold start)
    - writes the final ScoreBundle back atomically (tmp + rename)
  → Heartbeat actions (reflect / consolidate / pivot / lint_wiki) fire on
    interval or plateau triggers and inject prompts into the agent.
    - an enabled meta-evolve recommendation is appended only when pivot fires
```

## Tech Stack

- **Python 3.11+**, Hatchling build, **uv** for environment management.
- **Core deps**: `pyyaml`, `omegaconf`, `starlette` (dashboard).
- **Optional extras**: `swebench`, `datasets`, `docker`, `harbor` (heavyweight task graders).
- **Runtimes** are external CLIs invoked as subprocesses — Claude Code (`claude`), Codex (`codex`), Cursor Agent (`cursor-agent`), Kiro (`kiro`), OpenCode (`opencode`).

## Commands

```bash
# Install
uv sync                    # Basic
uv sync --extra dev        # With pytest, ruff
uv sync --all-extras       # Everything

# Authoring
coral init my-task                                # Scaffold task.yaml + grader/ package + seed/
coral validate my-task                            # Type-check task structure and dry-run grader against seed/
coral validate my-task --json                     # Emit one structured validation result for tools/frontends

# User-level agent bindings (~/.config/coral/agents.yaml)
coral setup                                       # Scan PATH + numbered wizard (one runtime can yield N bindings)
coral setup --non-interactive                     # Just print the detection report (no prompts)
coral setup agent --name claude-opus --runtime claude_code --model opus   # Create/update a single binding
coral agents list                                 # List bindings (numbered, default marked)
coral agents show <name>                          # Inspect a binding
coral agents doctor [name] [--no-live] [--timeout 30]  # Validate bindings (incl. live hello-ping unless --no-live)
coral agents remove                               # Interactive numbered-selection wizard
coral agents remove <name> [<name>...]            # Delete one or more by name
# In task.yaml: `agents.binding: claude-opus` (or `agents.assignments[].binding`) expands into runtime/model/runtime_options

# Running agents
coral start -c task.yaml                          # Launch agents (auto-tmux)
coral start -c task.yaml agents.count=4 agents.model=opus       # Dotlist overrides
coral start -c task.yaml run.verbose=true run.ui=true           # Verbose + dashboard
coral start -c task.yaml run.stop.max_real_attempts=30          # Stop after 30 finalized real attempts
coral start -c task.yaml agents.sandbox.enabled=true            # Wrap agents in srt OS-level sandbox (or `preset: sandbox`)
coral start -c task.yaml run.session=local                      # No tmux session
coral resume                                      # Resume latest run (sessions restored)
coral resume -i "Try greedy approaches"           # Inject an instruction at resume
coral resume --from <hash> -i "Continue this fork" # Reset an agent to an attempt (later attempts are archived = soft-deleted), then inject instruction
coral stop [--all]                                # Stop one or all active runs
coral status                                      # Agent health + leaderboard

# Inspecting results
coral log                                         # Top 20 by score
coral log -n 5 --recent                           # Sort by time
coral log --search "kernel" --agent agent-1       # Full-text + filter
coral show <hash> [--diff]                        # Attempt details (file summary or full diff)
coral notes [--status STATUS] [--search KW] [-n N] [--read N] [--history]  # Browse / filter / read / show checkpoint history
coral notes --audit [--status STATUS] [--search KW] [-n N]  # Read-only evidence checks; author status is not verification
coral skills [--read NAME]                        # List or read a shared skill
coral runs [--all] [--task NAME]                  # Active runs (or all)

# Dashboard
coral ui [--port 8420]                            # Web dashboard

# Agent-side commands (run inside an agent worktree)
coral eval -m "what changed and why"              # Stage + commit + grade (blocking)
coral eval -m "..." --no-wait                     # Submit and return immediately
coral eval -m "rewrite prompt" --operator prompt --mutation rewrite  # Attribute an enabled meta-evolve arm
coral wait <hash> [--timeout 600]                 # Block until daemon finalizes a prior submission
coral diff                                        # Show uncommitted changes
coral revert                                      # Undo last commit
coral checkout <hash>                             # Reset working tree to a previous attempt
coral export <hash> -b <branch> [-f]              # Export an attempt's commit as a normal git branch in the run repo
coral heartbeat [set|remove|reset]                # Inspect or rewrite per-agent heartbeat actions

# Tests + lint
uv run pytest tests/ -v
uv run ruff check .
uv run ruff format .
```

## Code Patterns

1. **`GraderInterface`** protocol (`@runtime_checkable`):
   ```python
   class GraderInterface(Protocol):
       async def grade(self, codebase_path: str, tasks: list[Task], **kwargs) -> ScoreBundle: ...
   ```

2. **`BaseGrader`** with helpers `_make_score()`, `_make_bundle()`, `grade_sync()`. **`TaskGrader`** (`coral/grader/task_grader.py`) is the recommended base for task-specific graders — implement `evaluate()` and use `self.codebase_path`, `self.private_dir`, `self.args`, `self.score(...)`, `self.fail(...)`.

3. **Wiring a grader**: `grader.entrypoint = "module.path:ClassName"` (required) plus `grader.setup: ["uv pip install -e ./grader"]`. The daemon resolves the entrypoint inside `.coral/private/grader_venv/` via `coral.grader.subprocess_grader.SubprocessGrader`. The legacy `eval/grader.py` auto-discovery has been removed. Hidden data (answer keys, test fixtures) must be declared under `grader.private` (copied into `.coral/private/`, read via `self.private_dir`) — `.coral/private/` is the only path agent runtimes are denied. The rest of the grader package is the opposite of hidden: **everything inside `grader/` is visible to agents** — the whole source is surfaced read-only at `<shared_dir>/grader/` (a symlink to the real package) so they can read how they're scored. So a `grader.private` path must live **outside** `grader/` (conventionally a sibling `taskdata/`, declared as `taskdata`) — `coral validate` errors if one is inside the package, since it would be both copied to `.coral/private/` *and* leaked via the surfaced source. Non-secret bundled data read via `Path(__file__).parent` may sit inside `grader/`, but it's visible — never put a secret there. `FunctionGrader` exists for wrapping plain callables but is no longer wired through `task.yaml` — ship a thin `TaskGrader` subclass instead.

4. **Eval is async by default**: `coral eval` writes a pending `Attempt` to `.coral/public/attempts/<hash>.json` and the daemon writes the final `ScoreBundle` back. `grader.max_pending_per_agent` (default 1) caps in-flight submissions per agent; `grader.parallel.max_workers` (default 1) controls daemon concurrency — bump only when the grader is concurrency-safe.

5. **Hub modules** are pure I/O over `.coral/public/`:
   - `attempts.py` — JSON CRUD, leaderboard, search, eval counters
   - `notes.py` — markdown + YAML frontmatter
   - `skills.py` — `SKILL.md` discovery
   - `checkpoint.py` — `git init` + lock-protected commits inside `.coral/public/` so agents can browse the history of shared state
   - `heartbeat.py` — per-agent action storage
   - `steering.py` — stopped-run dashboard steering queue drained by `coral resume`

6. **Heartbeat actions** (`coral/agent/heartbeat.py`): each agent has a list of `HeartbeatAction`s with `trigger ∈ {"interval", "plateau"}`. Defaults: `reflect` every 1 eval, `consolidate` every 10 (global), `pivot` after 5 plateau evals, `lint_wiki` every 10 (global). Edit at runtime with `coral heartbeat set/remove/reset`.

7. **Meta-evolve** (`coral/agent/meta_evolve.py`): disabled by default. Enabled real attempts require a configured `--operator`/`--mutation` pair before `git add`; tune and grader-error attempts never count. Lift is `current - parent` for maximize and `parent - current` for minimize. Selection explores unobserved arms in config order, then uses deterministic UCB. The manager rebuilds evidence from Attempt JSON and appends the advisory result only to `pivot` prompts.

8. **Multi-runtime**: `coral.agent.registry` maps `agents.runtime` to a runtime class. `claude_code` is the default. Each runtime knows its native shared-state directory (`.claude`, `.codex`, `.opencode`, ...) — `generate_coral_md(..., shared_dir=...)` renders the right paths into CORAL.md.

## Key Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package config, dependencies, `coral` console entrypoint |
| `coral/types.py` | `Task`, `Score`, `ScoreBundle`, `Attempt` |
| `coral/config.py` | YAML config dataclasses + OmegaConf merge + dotlist overrides |
| `coral/grader/protocol.py` | `GraderInterface` protocol |
| `coral/grader/base.py` | `BaseGrader` base class |
| `coral/grader/task_grader.py` | `TaskGrader` — recommended base for task graders |
| `coral/grader/loader.py` | Resolve grader from `grader.entrypoint` (subprocess in grader venv) |
| `coral/grader/subprocess_grader.py` | Worker-subprocess grader runtime used by entrypoint path |
| `coral/grader/daemon.py` | Long-running grader daemon (one per run) |
| `coral/grader/builtin/function_grader.py` | Wrap functions as graders |
| `coral/workspace/project.py` | `setup_run_dir()` — builds `.coral/{public,private}/`, clones repo, seeds bundled skills/agents |
| `coral/workspace/worktree.py` | Per-agent git worktree creation + `.claude/` symlink + permissions |
| `coral/workspace/repo.py` | Source-repo clone/init helpers + `run_setup_commands` |
| `coral/workspace/grader_env.py` | `setup_grader_env()` — `uv venv .coral/private/grader_venv/` + grader.setup |
| `coral/hub/attempts.py` | Attempt CRUD + leaderboard + per-agent pending caps |
| `coral/hub/checkpoint.py` | Git-tracked checkpoints of `.coral/public/` |
| `coral/agent/manager.py` | Multi-agent lifecycle, supervises grader daemon, restart-burst circuit breaker |
| `coral/agent/meta_evolve.py` | Opt-in attempt attribution, directional lift aggregation, deterministic arm recommendation, and prompt rendering |
| `coral/agent/runtime.py` | `AgentRuntime` abstract base |
| `coral/agent/builtin/*.py` | Concrete runtimes (claude_code, codex, cursor_agent, kiro, opencode) |
| `coral/agent/heartbeat.py` | `HeartbeatAction` / `HeartbeatRunner` (interval + plateau triggers) |
| `coral/hooks/post_commit.py` | `submit_eval()` — git add/commit + write pending attempt + optional poll |
| `coral/template/coral_md.py` | Renders the CORAL.md each agent reads |
| `coral/cli/__init__.py` | Top-level argparse + dispatch (incl. `setup` / `agents` binding commands) |

## Developer Workflows

Project-local skills live under `.claude/skills/`. Claude Code loads them on demand by description match — describe the task and the matching skill triggers automatically.

All ordinary PRs target `dev`. A release PR promotes `dev` to `main` with a
merge commit; the promotion workflow verifies identical trees and puts the
version tag on the promoted `dev` parent, so release ancestry remains shared
without merging `main` back into `dev`. Do not make main-only release changes.

| Skill | Use when |
|---|---|
| [coral-debug](.claude/skills/coral-debug/SKILL.md) | Editing existing code under `coral/` or chasing a bug — reproduce loops, where-to-look pointers, run inspection, lint/test |
| [coral-new-task](.claude/skills/coral-new-task/SKILL.md) | Adding a new `examples/<task>/` — seed + task.yaml + grader package, validation loop, common pitfalls |
| [coral-extend](.claude/skills/coral-extend/SKILL.md) | Extending the framework itself — new runtime, CLI command, bundled skill, hook, or config field |

## AI-assisted contributions

CORAL is itself infrastructure for running autonomous coding agents, so we
fully expect contributors to use Claude Code, Codex, Cursor, Kiro, OpenCode,
or other agents while developing changes. This document sets the ground rules.

These rules apply to **any** PR where an AI coding agent meaningfully wrote,
refactored, or generated code, tests, or docs. If you are not sure whether
your workflow counts, it does — read on.

> **TL;DR**
> - All PRs target the `dev` branch, never `main` (see CONTRIBUTING.md).
> - Release tags point to the promoted `dev` commit, not the `main` merge commit;
>   do not add main-only release changes or routine main-to-dev sync PRs.
> - A human author must read every changed line and be able to defend the design end-to-end.
> - No drive-by mechanical PRs (lone typo fixes, single-line style nits, batch reformat across the repo).
> - Don't open duplicate PRs against the same issue. Check first.
> - Edit CLAUDE.md for shared guidance; keep AGENTS.md as its relative symlink.
> - Don't dump raw agent transcripts into the PR description. Summarize.

### 1. Human accountability

- **Pure agent-authored PRs are not accepted.** A human must understand the
  change, review every line, and respond to review comments themselves —
  not by piping reviewer questions back into an agent and posting the reply.
- The submitting human is responsible for:
  - Correctness of the diff,
  - Running tests + lint locally before pushing,
  - Following up on review feedback,
  - Any regressions introduced by the change.
- "An agent wrote it" is not an explanation for design choices in review.

### 2. Scope discipline

- **Bundle agent output with substantive work.** Don't open standalone PRs
  for things like:
  - a single typo fix,
  - one isolated `ruff`-style nit,
  - reformatting a file the agent happened to touch,
  - changing a comment without changing behavior.
  Roll these into a larger PR that actually does something, or batch many
  of them into one explicit cleanup PR with a clear scope statement.
- **One concern per PR.** If your agent generated a refactor and a bug fix
  in the same run, split them.
- **Don't reformat the world.** Repo-wide formatting / import-sorting /
  rename-the-variable PRs are a "no" unless coordinated with maintainers
  first (and then they go on `.git-blame-ignore-revs`).

### 3. Duplicate-work checks

Before opening a PR, verify nobody is already on it:

```bash
gh issue view <issue_number> --repo Human-Agent-Society/CORAL --comments
gh pr list --repo Human-Agent-Society/CORAL --state open --search "<issue_number> in:body"
gh pr list --repo Human-Agent-Society/CORAL --state open --search "<short keywords>"
```

If an open PR is already in flight:

- If it looks stalled, **comment there** to check in. Don't open a parallel PR.
- If your approach is materially different, file an issue (or comment on the
  existing one) explaining the difference before submitting.

### 4. Shared agent guidance

`CLAUDE.md` is the single source for project guidance and contribution rules.
The repository-root `AGENTS.md` is a relative symlink to `CLAUDE.md`, so all
agents read the same instructions. Edit `CLAUDE.md` and preserve the symlink;
do not replace `AGENTS.md` with a separate copy.

### 5. Disclosure in PR descriptions

If an agent did the bulk of the writing, say so plainly. We don't penalize
agent-assisted PRs — we just want the review framed correctly.

Use `.github/PULL_REQUEST_TEMPLATE.md` as-is. Do not replace it with a custom
short summary. In particular:

- Keep the `Motivation`, `Changes`, `Test plan`, `Affected areas`, and
  `Checklist` sections.
- Tick every affected area that applies.
- Tick checklist items only when they are true. If the full-suite template
  commands were not run, leave those boxes unchecked and list the narrower
  commands you did run in `Test plan`.
- Reference issues in the body (`Closes #123`) when applicable.

A minimal disclosure looks like:

> *Authored with assistance from \<tool\>. I read every changed line, ran the
> tests below locally, and verified the behavior described in the test plan.*

What **not** to do:

- Don't paste raw agent transcripts, multi-thousand-line reasoning traces,
  or hidden "system prompt" blocks. Summarize the design instead.
- Don't fabricate test results. If you didn't run it, don't claim you did.
- Don't link to private agent sessions. Reviewers can't follow those.

### 6. CORAL-specific gotchas for agents

These tend to bite AI-generated PRs in this repo specifically:

- **`.coral/private/` is hidden from agents at runtime.** Don't write code
  that assumes agents can read answer keys or hidden test inputs. Grader
  source is surfaced read-only through the shared `grader/` link; keep
  hidden data outside that package. See `coral/workspace/project.py`.
- **The grader runs in its own venv** (`.coral/private/grader_venv/`).
  Adding a dep to `pyproject.toml` does not put it on the grader's path —
  use `grader.setup` in `task.yaml` instead.
- **Grader output must not write to `codebase_path`.** The daemon force-
  removes that directory after grading. Anything the grader wants to
  persist goes through `ScoreBundle` or `self.private_dir`.
- **Score direction matters.** `grader.direction` controls which way is
  better — agents often flip this. Double-check against existing examples.
- **Don't commit hidden answers under `seed/`.** Anything in `seed/` ends
  up in the agent's worktree, visible to the running agent.
- **CORAL.md is generated** by `coral/template/coral_md.py` — don't hand-
  edit the rendered file in a worktree, edit the template instead.
- **Audit note evidence with `coral notes --audit`** (optionally `--status STATUS`,
  `--search KW`, or `-n N`). This read-only command checks recorded observations separately
  from author status. A score comparison needs `evidence.baseline`; missing
  conditions remain unchecked, and matching arithmetic does not verify causality.

If you're adding a new task, the [`coral-new-task` skill](.claude/skills/coral-new-task/SKILL.md)
walks through these in detail. For framework-level changes, see
[`coral-extend`](.claude/skills/coral-extend/SKILL.md); for debugging,
[`coral-debug`](.claude/skills/coral-debug/SKILL.md).

### 7. Reviewer guidance

Reviewers should:

- Flag PRs that look like raw agent output without human review (telltale
  signs: unrelated drive-by changes, invented APIs, broken imports, fake
  citations, "test plan" sections that don't match the diff).
- Ask the human author — not the agent — to explain non-obvious choices.
- Close low-value mechanical PRs with a pointer to this document.

---

By submitting a PR you confirm you've read CONTRIBUTING.md and, if
applicable, this document.
