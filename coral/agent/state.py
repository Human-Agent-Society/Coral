"""Per-agent reliability state: crash history events and persisted PAUSED markers.

The manager records `RestartEvent`s in memory and persists a small JSON file at
`<coral_dir>/public/agent_state.json` whenever an agent transitions into or out
of PAUSED. The file is written atomically (tempfile + rename) so concurrent
readers (e.g. `coral status`) never see a partial document.

v1 covers the write side only; honoring persisted state across `coral resume`
is deferred to a follow-up patch.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Schema version embedded in the persisted JSON so future readers can migrate.
AGENT_STATE_SCHEMA_VERSION = 1


@dataclass
class RestartEvent:
    """A single observed agent exit, used to populate the crash-burst sliding window.

    Only "no_result" / "session_error" exits should ever be appended; "clean" exits
    are excluded so legitimate `max_turns` completions do not trip the breaker.
    """

    timestamp: float  # seconds since epoch (monotonic against datetime.now().timestamp())
    exit_code: int | None
    log_path: str  # absolute path to the agent's stream-json log at the time of exit
    classification: str  # "no_result" | "session_error" (clean exits are not recorded)


@dataclass
class AgentRuntimeState:
    """Persisted reliability state for a single agent.

    `state` is one of "active", "paused".
    `paused_until` is the wall-clock epoch second the pause expires; it is None
    when the agent is not currently paused.
    `sandbox` is the sandbox provider name the agent runs under
    (`agents.sandbox.provider`, e.g. "srt"); None when unsandboxed.
    """

    state: str = "active"
    paused_until: float | None = None
    pause_count: int = 0
    last_fault_at: str | None = None  # ISO-8601 UTC timestamp of most recent fault dump
    sandbox: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentRuntimeState:
        return cls(
            state=str(data.get("state", "active")),
            paused_until=data.get("paused_until"),
            pause_count=int(data.get("pause_count", 0)),
            last_fault_at=data.get("last_fault_at"),
            sandbox=data.get("sandbox"),
        )


@dataclass
class AgentStateDocument:
    """The full document persisted at `<coral_dir>/public/agent_state.json`."""

    schema_version: int = AGENT_STATE_SCHEMA_VERSION
    updated_at: str = ""
    agents: dict[str, AgentRuntimeState] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "updated_at": self.updated_at,
            "agents": {agent_id: rs.to_dict() for agent_id, rs in self.agents.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentStateDocument:
        agents_raw = data.get("agents", {}) or {}
        return cls(
            schema_version=int(data.get("schema_version", AGENT_STATE_SCHEMA_VERSION)),
            updated_at=str(data.get("updated_at", "")),
            agents={
                agent_id: AgentRuntimeState.from_dict(payload)
                for agent_id, payload in agents_raw.items()
            },
        )


def state_file_path(coral_dir: str | Path) -> Path:
    """Return the canonical location of the per-run agent state document."""
    return Path(coral_dir) / "public" / "agent_state.json"


def write_agent_state(coral_dir: str | Path, document: AgentStateDocument) -> Path:
    """Atomically persist the agent state document.

    Uses tempfile + os.replace to ensure readers never observe a partial JSON
    document. Returns the path of the persisted file.
    """
    target = state_file_path(coral_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    document.updated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    payload = json.dumps(document.to_dict(), indent=2, sort_keys=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".agent_state.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    return target


def read_agent_state(coral_dir: str | Path) -> AgentStateDocument:
    """Best-effort read of the persisted state.

    Missing or malformed files yield an empty document; callers fall back to
    log-inference behavior in that case (current pre-patch behavior).
    """
    target = state_file_path(coral_dir)
    if not target.exists():
        return AgentStateDocument()
    try:
        with open(target, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return AgentStateDocument()
    if not isinstance(raw, dict):
        return AgentStateDocument()
    return AgentStateDocument.from_dict(raw)
