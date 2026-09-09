"""YAML-based project configuration for CORAL, powered by OmegaConf."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from omegaconf import MISSING, OmegaConf

from coral.harbor_task import (
    HARBOR_ADAPTER_MARKER,
    HARBOR_GRADER_ENTRYPOINT,
    HARBOR_PRIVATE_TASK_DIR,
    HARBOR_RUNTIME_VERSION,
    inspect_local_harbor_task,
)


@dataclass
class TaskRewardConfig:
    """CORAL's objective selection for a Harbor-backed task."""

    primary: str = ""
    direction: str = "maximize"

    def __post_init__(self) -> None:
        if not self.primary.strip():
            raise ValueError("task.reward.primary must be a non-empty Harbor reward key")
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(
                f"task.reward.direction must be 'maximize' or 'minimize', got {self.direction!r}"
            )


@dataclass
class TaskConfig:
    """Task definition within a CORAL project."""

    name: str = MISSING
    description: str = MISSING
    tips: str = ""
    source: str = ""
    reward: TaskRewardConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.reward, dict):
            self.reward = TaskRewardConfig(**self.reward)


@dataclass
class ParallelGraderConfig:
    """Parallel evaluation in the grader daemon.

    The daemon always routes pending attempts through a worker pool of this
    size. ``max_workers=1`` (the default) is serial — same behavior as before
    the pool existed. Set higher only when the grader is concurrency-safe
    (pure Python, sandboxed swebench runs, etc.); the daemon does not enforce
    safety, so a misconfigured value with a non-safe grader is the user's
    responsibility.
    """

    max_workers: int = 1


@dataclass
class GraderConfig:
    """Grader configuration."""

    entrypoint: str = (
        ""  # "module.path:ClassName" — required; resolved inside .coral/private/grader_venv/
    )
    setup: list[str] = field(
        default_factory=list
    )  # shell commands run in .coral/private/grader_venv/ before agents start
    timeout: int = 300  # eval timeout in seconds (0 = no limit)
    args: dict[str, Any] = field(default_factory=dict)
    private: list[str] = field(
        default_factory=list
    )  # files/dirs copied to .coral/ (hidden from agents)
    direction: str = "maximize"  # "maximize" or "minimize"
    # Producer-side queue cap. Reject `coral eval` when an agent already has
    # this many ungraded submissions in flight. 0 = unlimited (legacy behavior).
    # Default 1: an agent can only enqueue a fresh attempt once the prior one
    # is graded, which prevents runaway pending floods when the grader is slow.
    max_pending_per_agent: int = 1
    parallel: ParallelGraderConfig = field(default_factory=ParallelGraderConfig)

    def __post_init__(self) -> None:
        if self.max_pending_per_agent < 0:
            raise ValueError(
                f"grader.max_pending_per_agent must be >= 0, got {self.max_pending_per_agent}"
            )
        # SubprocessGrader serializes GraderConfig via dataclasses.asdict and
        # rebuilds with `GraderConfig(**payload)`, which leaves `parallel` as a
        # plain dict. Coerce here so validation and downstream attribute access
        # work for both real callers and the worker reconstruction path.
        if isinstance(self.parallel, dict):
            self.parallel = ParallelGraderConfig(**self.parallel)
        if self.parallel.max_workers < 1:
            raise ValueError(
                f"grader.parallel.max_workers must be >= 1, got {self.parallel.max_workers}"
            )


@dataclass
class HeartbeatActionConfig:
    """Configuration for a single heartbeat action.

    Trigger-specific knobs (e.g. ``epsilon`` for plateau) go under
    ``options``. The schema is validated at runtime against the trigger; see
    :class:`coral.agent.heartbeat.PlateauOptions`.
    """

    name: str = MISSING  # e.g. "reflect", "consolidate", "pivot"
    every: int = MISSING  # trigger every N evals (interval) or stall threshold (plateau)
    is_global: bool = False  # True = use global eval count, False = per-agent
    trigger: str = "interval"  # "interval" or "plateau"
    prompt: str = ""  # custom prompt; if empty, falls back to built-in DEFAULT_PROMPTS
    # Trigger-specific options. For ``trigger="plateau"`` the recognized key
    # is ``epsilon`` (minimum delta over the prior plateau-anchor that counts
    # as improvement; default 0.0 = legacy strict-> behavior). Unknown keys
    # raise at load time.
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class GatewayConfig:
    """LiteLLM gateway configuration for intercepting agent model traffic."""

    enabled: bool = False
    port: int = 4000
    config: str = ""  # path to litellm_config.yaml
    api_key: str = ""  # LiteLLM master key (auto-generated if empty)


@dataclass
class WarmStartConfig:
    """Warm-start configuration: optional research phase before the main coding loop."""

    enabled: bool = False


@dataclass
class SandboxConfig:
    """OS-level agent sandboxing via a pluggable provider (default: ``srt``).

    When enabled, every agent subprocess is wrapped by the configured
    sandbox provider (see ``coral.sandbox``), which enforces filesystem and
    network boundaries — uniformly across all runtimes, unlike the
    per-runtime permission settings, which are advisory. With the built-in
    ``srt`` provider (Anthropic's sandbox-runtime — Seatbelt on macOS,
    bubblewrap on Linux), reads are confined to this run: the agent sees
    its own worktree, the run repo, shared ``.coral/`` state (minus
    ``private/``), the surfaced grader source, and the toolchain's home
    dotdirs — not other runs, sibling agents' worktrees, or host secrets
    (``~/.ssh``, ``~/.aws``, ...). Writes are allow-listed to the same
    slice plus /tmp.

    Composes with tmux/local sessions; mutually exclusive with
    ``agents.isolate_user`` (and thus ``run.session=docker``) — both solve
    the same isolation problem.

    ``provider``: built-in name (``srt``) or a custom
    ``module.path:ClassName`` entrypoint implementing
    :class:`coral.sandbox.SandboxProvider` — the hook for hosted backends
    like e2b or modal.

    ``network``:
    - ``"open"`` (default): full network access via an allow-all proxy the
      manager runs in-process. srt refuses a wildcard allowlist; an external
      proxy owning the filtering policy is its sanctioned escape hatch.
      Traffic is proxy-mediated, so raw UDP/ICMP and proxy-ignorant clients
      won't work.
    - ``"allowlist"``: srt's own filtering proxies restrict traffic to
      ``allowed_domains`` (srt syntax, e.g. "github.com", "*.npmjs.org").

    The srt provider requires the sandbox-runtime npm package
    (npm install -g @anthropic-ai/sandbox-runtime), invoked via
    ``srt_command`` (default ``npx --no-install srt``); Linux additionally
    needs bubblewrap, socat, and ripgrep.
    """

    enabled: bool = False
    provider: str = "srt"  # built-in name or "module.path:ClassName" entrypoint
    network: str = "open"  # "open" or "allowlist"
    # How to invoke srt. The npx default also finds npm -g installs whose
    # global bin dir is not on PATH; --no-install stops npx from fetching
    # the unrelated "srt" registry package when sandbox-runtime is missing.
    # Override with e.g. ["srt"] or ["/usr/local/bin/srt"].
    srt_command: list[str] = field(default_factory=lambda: ["npx", "--no-install", "srt"])
    allowed_domains: list[str] = field(default_factory=list)  # allowlist mode only
    deny_read: list[str] = field(default_factory=list)  # extra paths hidden from agents
    allow_read: list[str] = field(default_factory=list)  # extra readable paths (e.g. ~/.aws)
    allow_write: list[str] = field(default_factory=list)  # extra writable paths
    # Escape hatch: deep-merged last into the generated srt settings JSON
    # (e.g. {"enableWeakerNestedSandbox": true} inside containers).
    extra_settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("agents.sandbox.provider must be a non-empty string")
        if self.network not in ("open", "allowlist"):
            raise ValueError(
                f"agents.sandbox.network must be 'open' or 'allowlist', got {self.network!r}"
            )
        if not self.srt_command:
            raise ValueError("agents.sandbox.srt_command must be a non-empty command list")


@dataclass
class AgentAssignmentConfig:
    """Per-assignment override of runtime/model for mix-and-match multi-agent runs.

    When ``agents.assignments`` is set, it overrides ``agents.count``: the total
    number of agents spawned is the sum of ``count`` across every assignment.
    Empty string fields inherit from the top-level ``agents.*`` defaults.
    Each assignment can override:
    - ``runtime``:        the agent runtime (claude_code / codex / opencode / ...)
    - ``model``:          model passed to that runtime
    - ``count``:          how many agents of this kind to spawn (default 1)
    - ``runtime_options`` extra options forwarded to that runtime's ``start()``
    """

    runtime: str = ""  # empty -> inherit from agents.runtime
    model: str = ""  # empty -> inherit from agents.model (or runtime default)
    count: int = 1
    runtime_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"agents.assignments[].count must be >= 1, got {self.count}")


@dataclass
class AgentConfig:
    """Agent spawning configuration."""

    count: int = 1
    runtime: str = "claude_code"
    model: str = "sonnet"
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    warmstart: WarmStartConfig = field(default_factory=WarmStartConfig)
    runtime_options: dict[str, Any] = field(default_factory=dict)
    # OS-user isolation: when set (e.g. "agent"), the agent subprocess is run as
    # this unprivileged user while the manager/grader stay root. The agent's
    # worktree + shared state + repo are chowned to it; ``.coral/private/``
    # (grader venv, answer keys) is kept root-owned mode 700 so the agent
    # cannot read it even via Bash. Requires CORAL to run as root.
    # This field is the HOST opt-in (empty = no isolation). In CORAL's Docker
    # session this value is forced to the image's ``agent`` user regardless of
    # what is set here.
    isolate_user: str = ""
    # OS-level sandboxing via sandbox-runtime (`srt`) — the lightweight,
    # host-native alternative to isolate_user/Docker. See SandboxConfig.
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    # Mix-and-match: when non-empty, each entry spawns its own runtime/model
    # combo. ``agents.count`` is ignored (total = sum of assignment counts).
    # Empty fields on an assignment inherit the agents.* defaults below.
    assignments: list[AgentAssignmentConfig] = field(default_factory=list)
    # Max agent turns per session before the runtime exits and the manager
    # restarts the agent (preserving context via --resume). 0 = no cap, let
    # the underlying CLI run until it exits naturally.
    max_turns: int = 0
    # Stall watchdog: restart an agent that produces no output for this many
    # seconds. 0 disables the watchdog. Default 1200s (20 min) catches deadlocks
    # faster than the prior 3600s while still being well above legitimate quiet
    # periods (long tool calls, grader queue waits — the latter is exempted).
    timeout: int = 1200
    heartbeat: list[HeartbeatActionConfig] = field(
        default_factory=lambda: [
            HeartbeatActionConfig(name="reflect", every=1),
            HeartbeatActionConfig(name="consolidate", every=10, is_global=True),
            HeartbeatActionConfig(name="pivot", every=5, trigger="plateau"),
            HeartbeatActionConfig(name="lint_wiki", every=10, is_global=True),
        ]
    )
    skills: list[str] = field(default_factory=list)  # skill dirs copied to .coral/public/skills/
    research: bool = True  # enable web search / literature review step in workflow
    stagger_seconds: int = 0  # delay between spawning each agent (rate-limit backpressure)

    # Reliability: crash-burst circuit breaker.
    # When an agent exits repeatedly in a short window with no clean-exit marker,
    # the manager pauses it instead of respawning into a tight loop.
    # 0 in any of the three knobs disables the breaker entirely.
    restart_burst_threshold: int = 3  # crashes within window before pausing the agent
    restart_burst_window: int = 30  # seconds; sliding window for crash counting
    restart_pause_seconds: int = (
        300  # how long the paused state holds before restart attempts resume
    )

    # Reliability: grader-queue exemption for stall detection.
    # Skip stall checks for an agent whose latest attempt is pending grading,
    # but only if the grader process is alive and the pending attempt is not stale.
    grader_pending_max_age: int = 1800  # seconds; older pending attempts no longer exempt

    # Reliability: minimum runtime in seconds before an exit_code==0 is considered "clean"
    # for runtimes that lack a stable terminal marker (codex/opencode/kiro).
    min_clean_runtime_seconds: int = 60

    def __post_init__(self) -> None:
        # Direct construction (tests, SubprocessGrader-style round-trips) may
        # leave nested sandbox config as a plain dict; coerce like RunConfig.stop.
        if isinstance(self.sandbox, dict):
            self.sandbox = SandboxConfig(**self.sandbox)
        if self.sandbox.enabled and self.isolate_user:
            raise ValueError(
                "agents.sandbox and agents.isolate_user are mutually exclusive — "
                "both isolate agents from .coral/private/; use one or the other."
            )
        # Reject negative values for the new reliability knobs;
        # 0 is treated as "disabled" for the same fields where it makes sense.
        for field_name in (
            "restart_burst_threshold",
            "restart_burst_window",
            "restart_pause_seconds",
            "grader_pending_max_age",
            "min_clean_runtime_seconds",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"agents.{field_name} must be >= 0, got {value}")
        # If the breaker is enabled at all, the pause must outlast the burst window;
        # otherwise the breaker can re-arm before the burst counter has cleared.
        if (
            self.restart_burst_threshold > 0
            and self.restart_burst_window > 0
            and 0 < self.restart_pause_seconds < self.restart_burst_window
        ):
            raise ValueError(
                "agents.restart_pause_seconds must be >= agents.restart_burst_window "
                f"(got pause={self.restart_pause_seconds}, window={self.restart_burst_window})"
            )

    def heartbeat_interval(self, name: str) -> int:
        """Get the interval for a heartbeat action by name."""
        for action in self.heartbeat:
            if action.name == name:
                return action.every
        raise KeyError(f"No heartbeat action named {name!r}")


@dataclass
class SharingConfig:
    """What shared state is enabled."""

    attempts: bool = True
    notes: bool = True
    skills: bool = True


@dataclass
class WorkspaceConfig:
    """Workspace layout configuration."""

    results_dir: str = "./results"
    repo_path: str = "."
    setup: list[str] = field(default_factory=list)  # shell commands to run before agents start
    # Ignored if results_dir is set
    base_dir: str = ""
    run_dir: str = ""  # if set, use this exact run directory instead of generating one


@dataclass
class RunStopConfig:
    """Optional run-level auto-stop conditions."""

    score_threshold: float | None = None
    max_real_attempts: int | None = None

    def __post_init__(self) -> None:
        if self.max_real_attempts is not None and self.max_real_attempts <= 0:
            raise ValueError(
                f"run.stop.max_real_attempts must be > 0, got {self.max_real_attempts}"
            )


@dataclass
class RunConfig:
    """Runtime flags for a CORAL session."""

    verbose: bool = False
    ui: bool = False
    session: str = "tmux"  # "local", "tmux", or "docker"
    docker_image: str = ""  # empty = auto-build from project Dockerfile
    stop: RunStopConfig = field(default_factory=RunStopConfig)

    def __post_init__(self) -> None:
        if isinstance(self.stop, dict):
            self.stop = RunStopConfig(**self.stop)


@dataclass
class MigrationConfig:
    """Agent migration between islands.

    Ignored in single-island mode (``islands.count == 1``).
    """

    enabled: bool = True
    every: int = 50  # global evals between migration cycles
    rank_window: int = 20  # "best agent" judged by max-over-last-N evals
    min_evals: int = 3  # candidate must have >= N attempts to be eligible
    dest_weighting: str = "weakest"  # weakest | uniform | round_robin | score
    max_per_cycle: int = 2
    # Min global evals before the same agent may migrate again. Attempt
    # records move with the migrant, so without this guard min_evals is
    # satisfied on arrival and the same top agent can ping-pong every cycle.
    # 0 disables the guard.
    remigration_cooldown: int = 100
    notify_island: bool = True
    # Standard post-migration phase: interrupt+resume live bystanders on the
    # affected islands so launch-injected per-agent state follows the new
    # partition immediately instead of at their next natural restart (see
    # AgentManager._migration_resync_ops for the op registry). Sessions are
    # resumed, so no work is lost. A no-op when no resync ops apply — today
    # the only op is the agents.sandbox boundary refresh.
    resync_bystanders: bool = True

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError(f"islands.migration.every must be >= 1, got {self.every}")
        if self.rank_window < 1:
            raise ValueError(f"islands.migration.rank_window must be >= 1, got {self.rank_window}")
        if self.rank_window > self.every:
            raise ValueError(
                f"islands.migration.rank_window ({self.rank_window}) must be "
                f"<= islands.migration.every ({self.every})"
            )
        if self.min_evals < 1:
            raise ValueError(f"islands.migration.min_evals must be >= 1, got {self.min_evals}")
        if self.dest_weighting not in {"score", "uniform", "round_robin", "weakest"}:
            raise ValueError(
                "islands.migration.dest_weighting must be one of "
                f"{{score, uniform, round_robin, weakest}}, got {self.dest_weighting!r}"
            )
        if self.max_per_cycle < 1:
            raise ValueError(
                f"islands.migration.max_per_cycle must be >= 1, got {self.max_per_cycle}"
            )
        if self.remigration_cooldown < 0:
            raise ValueError(
                "islands.migration.remigration_cooldown must be >= 0, "
                f"got {self.remigration_cooldown}"
            )


@dataclass
class IslandsConfig:
    """Multi-island shared-state partitioning.

    ``count = 1`` (the default) preserves today's single-island layout exactly
    — no ``.coral/islands/`` directory is created and no migration code paths
    are exercised.
    """

    count: int = 1
    migration: MigrationConfig = field(default_factory=MigrationConfig)

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"islands.count must be >= 1, got {self.count}")
        # OmegaConf round-trip can leave migration as a dict
        if isinstance(self.migration, dict):
            self.migration = MigrationConfig(**self.migration)


@dataclass
class CoralConfig:
    """Top-level project configuration."""

    task: TaskConfig = field(default_factory=TaskConfig)
    grader: GraderConfig = field(default_factory=GraderConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    islands: IslandsConfig = field(default_factory=IslandsConfig)
    sharing: SharingConfig = field(default_factory=SharingConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    run: RunConfig = field(default_factory=RunConfig)
    task_dir: Path | None = None  # internal: directory containing task.yaml

    @classmethod
    def from_yaml(cls, path: str | Path) -> CoralConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data, base_dir=Path(path).parent)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> CoralConfig:
        data = dict(data)
        # A top-level `preset:` names a built-in preset or points at a local
        # YAML file. Its keys form a layer between the schema defaults and the
        # task's own keys: schema < preset < task. The task always wins on any
        # key it sets explicitly. Stacking is not supported (a preset may not
        # itself declare `preset:`).
        preset_ref = data.pop("preset", None)
        preset_data = _load_preset(preset_ref, base_dir) if preset_ref else None

        schema = OmegaConf.structured(cls)
        # Merge preset under task as raw dicts first, then preprocess the
        # combined picture so legacy-key normalization and runtime/model
        # defaulting see the final merged values (e.g. runtime from preset +
        # model from task).
        layers = [OmegaConf.create(preset_data)] if preset_data is not None else []
        layers.append(OmegaConf.create(data))
        combined = OmegaConf.merge(*layers) if len(layers) > 1 else layers[0]
        combined_dict: dict[str, Any] = OmegaConf.to_container(combined)  # type: ignore[assignment]
        combined_dict = _preprocess(combined_dict, base_dir=base_dir)
        merged = OmegaConf.merge(schema, OmegaConf.create(combined_dict))
        cfg: CoralConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
        _validate_harbor_runtime_config(cfg)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        sc = OmegaConf.structured(self)
        container: dict[str, Any] = OmegaConf.to_container(sc, resolve=True)  # type: ignore[assignment]
        # Remove internal-only fields
        container.pop("task_dir", None)
        task_data = container.get("task", {})
        if not task_data.get("source"):
            task_data.pop("source", None)
        if task_data.get("reward") is None:
            task_data.pop("reward", None)
        # Serialize heartbeat is_global as "global" for YAML compat
        for h in container.get("agents", {}).get("heartbeat", []):
            h["global"] = h.pop("is_global", False)
        return container

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def merge_dotlist(cls, config: CoralConfig, dotlist: list[str]) -> CoralConfig:
        """Merge CLI dotlist overrides into an existing config."""
        if not dotlist:
            return config
        base = OmegaConf.structured(config)
        overrides = OmegaConf.from_dotlist(dotlist)
        merged = OmegaConf.merge(base, overrides)
        cfg: CoralConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
        if cfg.task.source:
            if cfg.task.reward is None:
                raise ValueError("Harbor-backed task requires task.reward")
            cfg.grader.direction = cfg.task.reward.direction
            cfg.grader.args["primary_reward"] = cfg.task.reward.primary
        _validate_harbor_runtime_config(cfg)
        return cfg


def _validate_harbor_runtime_config(config: CoralConfig) -> None:
    if config.task.source and config.run.session == "docker":
        raise ValueError(
            "Harbor-backed tasks require CORAL to run on the host; "
            "run.session=docker would require unsupported Docker-in-Docker"
        )


def _apply_binding(entry: dict[str, Any], binding: Any) -> None:
    """Fill an agents/assignment dict's empty fields from a binding, in place.

    Precedence: explicit task fields win over binding fields; binding fields
    win over runtime defaults (which are filled later in ``_preprocess``).
    """
    from coral.agent.registry import default_command_for_runtime

    if not entry.get("runtime"):
        entry["runtime"] = binding.runtime
    if not entry.get("model") and binding.model:
        entry["model"] = binding.model

    # runtime_options: binding provides the base, explicit task options override.
    merged: dict[str, Any] = dict(binding.runtime_options or {})
    merged.update(entry.get("runtime_options") or {})

    # A binding role seed compiles into runtime_options.role_file, unless the
    # task already pinned one explicitly.
    if binding.role_file and "role_file" not in merged:
        merged["role_file"] = binding.role_file

    # Forward a custom CLI command only when it diverges from the runtime's
    # default binary. Runtimes that honor a command path (e.g. cursor_agent)
    # pick it up; the common default case stays out of runtime_options.
    if binding.command and "command" not in merged:
        default_cmd = default_command_for_runtime(binding.runtime)
        if binding.command != default_cmd:
            merged["command"] = binding.command

    if merged:
        entry["runtime_options"] = merged


def _expand_bindings(agents_data: dict[str, Any]) -> None:
    """Resolve ``agents.binding`` and ``agents.assignments[].binding`` in place.

    Bindings are looked up in the user-level bindings file. The ``binding`` key
    is removed after expansion so it never reaches the structured schema.
    """
    assignments = agents_data.get("assignments")
    top_binding = agents_data.get("binding")
    assignment_bindings = (
        [a.get("binding") for a in assignments if isinstance(a, dict)]
        if isinstance(assignments, list)
        else []
    )

    if top_binding is None and not any(assignment_bindings):
        # Nothing references a binding — don't touch the file at all so configs
        # and tests that never opt in are completely unaffected.
        agents_data.pop("binding", None)
        return

    from coral.user_agents import load_store

    store = load_store()

    def _lookup(name: str) -> Any:
        binding = store.get(name)
        if binding is None:
            available = ", ".join(sorted(store.bindings)) or "(none defined)"
            raise ValueError(
                f"agents.binding {name!r} is not defined in {store.path}. "
                f"Available bindings: {available}. "
                f"Create one with `coral setup agent --name {name}`."
            )
        return binding

    if top_binding is not None:
        agents_data.pop("binding", None)
        _apply_binding(agents_data, _lookup(str(top_binding)))

    if isinstance(assignments, list):
        for entry in assignments:
            if not isinstance(entry, dict):
                continue
            name = entry.pop("binding", None)
            if name is not None:
                _apply_binding(entry, _lookup(str(name)))


def _builtin_presets_dir() -> Path:
    """Directory of bundled preset YAMLs shipped with CORAL."""
    return Path(__file__).parent / "template" / "presets"


def _resolve_preset_path(ref: str, base_dir: Path | None) -> Path:
    """Resolve a `preset:` string to a YAML file path.

    A bare name (no path separator, no .yaml/.yml suffix) refers to a built-in
    preset under ``coral/template/presets/``. Anything else is treated as a
    filesystem path: absolute as-is, relative resolved against ``base_dir``
    (the directory holding task.yaml) or the cwd as a fallback.
    """
    looks_like_path = (
        "/" in ref or "\\" in ref or ref.endswith((".yaml", ".yml")) or ref.startswith(".")
    )
    if not looks_like_path:
        path = _builtin_presets_dir() / f"{ref}.yaml"
        if not path.exists():
            available = sorted(p.stem for p in _builtin_presets_dir().glob("*.yaml"))
            raise ValueError(
                f"Unknown preset {ref!r}. Built-in presets: "
                f"{', '.join(available) or '(none)'}. "
                f"To use a local file, pass a path ending in .yaml."
            )
        return path

    path = Path(ref)
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    if not path.exists():
        raise ValueError(f"Preset file not found: {path}")
    return path


def _load_preset(ref: Any, base_dir: Path | None) -> dict[str, Any]:
    """Load and validate a preset referenced by a task's `preset:` key."""
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError(f"preset must be a non-empty string, got {ref!r}")
    path = _resolve_preset_path(ref.strip(), base_dir)
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Preset {path} must contain a YAML mapping, got {type(loaded).__name__}")
    if "preset" in loaded:
        raise ValueError(
            f"Preset {path} declares its own 'preset:' key — preset stacking is not supported."
        )
    # A preset is config defaults only; it must not carry task identity.
    loaded.pop("task_dir", None)
    return loaded


def _configure_harbor_task(
    data: dict[str, Any],
    *,
    base_dir: Path | None,
) -> None:
    """Hydrate the local Harbor source form into CORAL's runtime config.

    User-authored ``task.yaml`` keeps Harbor-owned fields out of CORAL.  The
    resolved name, instruction, digest, and built-in grader wiring are added
    only to the in-memory/run snapshot used by CORAL.
    """
    task_data = data.get("task")
    if not isinstance(task_data, dict):
        return
    task_data = dict(task_data)
    data["task"] = task_data

    source = task_data.get("source")
    if source in (None, ""):
        if task_data.get("reward") not in (None, {}):
            raise ValueError("task.reward is only valid when task.source is set")
        return
    if not isinstance(source, str) or not source.strip():
        raise ValueError("task.source must be a non-empty string")
    task_data["source"] = source.strip()

    raw_grader_data = data.get("grader")
    if raw_grader_data is None:
        grader_data: dict[str, Any] = {}
    elif not isinstance(raw_grader_data, dict):
        raise ValueError("grader must be a mapping")
    else:
        grader_data = dict(raw_grader_data)
    grader_args = grader_data.get("args") or {}
    hydrated = (
        grader_data.get("entrypoint") == HARBOR_GRADER_ENTRYPOINT
        and isinstance(grader_args, dict)
        and grader_args.get("harbor_adapter") == HARBOR_ADAPTER_MARKER
    )

    reward_data = task_data.get("reward")
    if not isinstance(reward_data, dict):
        raise ValueError("Harbor-backed task requires task.reward with primary and direction")
    primary = reward_data.get("primary")
    direction = reward_data.get("direction", "maximize")
    if not isinstance(primary, str) or not primary.strip():
        raise ValueError("task.reward.primary must be a non-empty Harbor reward key")
    if direction not in ("maximize", "minimize"):
        raise ValueError(
            f"task.reward.direction must be 'maximize' or 'minimize', got {direction!r}"
        )
    reward_data = {"primary": primary.strip(), "direction": direction}
    task_data["reward"] = reward_data

    if hydrated:
        grader_args = dict(grader_args)
        required = ("name", "description")
        missing = [field_name for field_name in required if not task_data.get(field_name)]
        if missing:
            raise ValueError(
                "Resolved Harbor run config is missing task metadata: " + ", ".join(missing)
            )
        grader_data["direction"] = direction
        grader_args["primary_reward"] = primary.strip()
        grader_data["args"] = grader_args
        data["grader"] = grader_data
        return

    duplicated_task_fields = [
        field_name for field_name in ("name", "description", "tips") if field_name in task_data
    ]
    if duplicated_task_fields:
        raise ValueError(
            "Harbor-backed task.yaml must not duplicate Harbor-owned fields: "
            + ", ".join(f"task.{field_name}" for field_name in duplicated_task_fields)
        )

    unsupported_grader_fields = sorted(
        set(grader_data) - {"timeout", "max_pending_per_agent", "parallel"}
    )
    if unsupported_grader_fields:
        raise ValueError(
            "Harbor-backed task.yaml cannot define portable grader fields: "
            + ", ".join(f"grader.{field_name}" for field_name in unsupported_grader_fields)
        )

    raw_workspace_data = data.get("workspace")
    if raw_workspace_data is None:
        workspace_data: dict[str, Any] = {}
    elif not isinstance(raw_workspace_data, dict):
        raise ValueError("workspace must be a mapping")
    else:
        workspace_data = raw_workspace_data
    unsupported_workspace_fields = sorted(
        field_name
        for field_name in ("repo_path", "setup")
        if field_name in workspace_data and workspace_data[field_name] not in (None, "", ".", [])
    )
    if unsupported_workspace_fields:
        raise ValueError(
            "Initial Harbor compatibility creates an empty CORAL workspace and does not "
            "support: "
            + ", ".join(f"workspace.{field_name}" for field_name in unsupported_workspace_fields)
        )

    descriptor = inspect_local_harbor_task(source, base_dir=base_dir)
    task_data["name"] = descriptor.name
    task_data["description"] = descriptor.instruction
    task_data["tips"] = ""

    grader_data["entrypoint"] = HARBOR_GRADER_ENTRYPOINT
    grader_data["direction"] = direction
    grader_data["args"] = {
        "harbor_adapter": HARBOR_ADAPTER_MARKER,
        "harbor_runtime_version": HARBOR_RUNTIME_VERSION,
        "harbor_schema_version": descriptor.schema_version,
        "harbor_task_digest": descriptor.digest,
        "harbor_task_name": descriptor.name,
        "harbor_task_package_version": descriptor.package_version,
        "harbor_task_subdir": HARBOR_PRIVATE_TASK_DIR,
        "primary_reward": primary.strip(),
    }
    data["task"] = task_data
    data["grader"] = grader_data


def _preprocess(
    data: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Transform legacy keys and normalize heartbeat config before OmegaConf merge."""
    _configure_harbor_task(data, base_dir=base_dir)

    # Reject removed grader.type / grader.module fields with migration guidance.
    grader_data = data.get("grader")
    if isinstance(grader_data, dict):
        legacy_grader_keys = [k for k in ("type", "module") if k in grader_data]
        if legacy_grader_keys:
            raise ValueError(
                f"grader.{' / grader.'.join(legacy_grader_keys)} is removed. "
                f"Use grader.entrypoint = 'your_pkg.module:Grader' (and grader.setup "
                f"to install the package). See docs/guides/custom-grader."
            )

    agents_data = data.get("agents", {})
    if not isinstance(agents_data, dict):
        return data

    # Make a copy so we don't mutate the original
    agents_data = dict(agents_data)

    # Expand user-level agent bindings (agents.binding / assignments[].binding)
    # into concrete runtime / model / runtime_options fields before anything
    # else looks at them. This keeps bindings a pure preset layer over the
    # existing runtime/model/assignment system.
    _expand_bindings(agents_data)

    heartbeat_raw = agents_data.pop("heartbeat", None)
    old_reflect = agents_data.pop("reflect_every", None)
    old_heartbeat = agents_data.pop("heartbeat_every", None)

    if heartbeat_raw is not None:
        specified = {h["name"] for h in heartbeat_raw}
        for dflt in AgentConfig().heartbeat:
            if dflt.name not in specified:
                heartbeat_raw.append(
                    {
                        "name": dflt.name,
                        "every": dflt.every,
                        "global": dflt.is_global,
                        "trigger": dflt.trigger,
                    }
                )
        agents_data["heartbeat"] = [
            {
                "name": h["name"],
                "every": h["every"],
                "is_global": h.get("global", False),
                "trigger": h.get("trigger", "interval"),
                "prompt": h.get("prompt", ""),
                "options": h.get("options", {}),
            }
            for h in heartbeat_raw
        ]
    elif old_reflect is not None or old_heartbeat is not None:
        agents_data["heartbeat"] = [
            {
                "name": "reflect",
                "every": old_reflect if old_reflect is not None else 1,
                "is_global": False,
            },
            {
                "name": "consolidate",
                "every": old_heartbeat if old_heartbeat is not None else 10,
                "is_global": False,
            },
        ]

    # If runtime is set but model is not, use the runtime-specific default.
    # Custom-entrypoint runtimes ('module.path:ClassName') have no default —
    # require the user to set agents.model explicitly so a footgun like the
    # builtin "sonnet" default doesn't silently land on a foreign runtime.
    if "runtime" in agents_data and "model" not in agents_data:
        from coral.agent.registry import default_model_for_runtime

        rt = agents_data["runtime"]
        default_model = default_model_for_runtime(rt)
        if default_model:
            agents_data["model"] = default_model
        elif isinstance(rt, str) and ":" in rt:
            raise ValueError(
                f"agents.runtime={rt!r} is a custom runtime entrypoint; "
                f"set agents.model explicitly in task.yaml."
            )

    # Normalize assignments: fill in missing model defaults from the assignment's
    # runtime so each entry stores a concrete model. Empty fields are kept as ""
    # (will inherit from the top-level agents.* defaults at resolve time).
    assignments_raw = agents_data.get("assignments")
    if isinstance(assignments_raw, list):
        from coral.agent.registry import default_model_for_runtime

        normalized: list[dict[str, Any]] = []
        for entry in assignments_raw:
            if not isinstance(entry, dict):
                continue
            entry = dict(entry)
            if entry.get("runtime") and not entry.get("model"):
                m = default_model_for_runtime(entry["runtime"])
                if m:
                    entry["model"] = m
            normalized.append(entry)
        agents_data["assignments"] = normalized

    data["agents"] = agents_data

    # Remove task_dir if present in raw data (it's internal-only)
    data.pop("task_dir", None)

    return data
