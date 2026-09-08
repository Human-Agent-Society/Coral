"""Read-only checks of note evidence against public attempt snapshots.

An audit checks recorded observations, never the truth of a natural-language
claim. Author-written status/verified fields are deliberately not inputs to
the checks. Nothing is cached or written back to the shared knowledge base.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from coral.hub._island import island_root

_HASH = re.compile(r"[0-9a-fA-F]{4,64}\Z")
_COMPLETED = {"improved", "baseline", "regressed", "reverted"}
_CONTEXT_FIELDS = ("task", "grader", "inputs", "direction")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _number(value: Any) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class _Attempts:
    """One island's record index, including archived evidence and corrupt files."""

    def __init__(self, root: Path):
        self.paths = {p.stem: p for p in (root / "attempts").glob("*.json")}
        self.snapshots: dict[str, tuple[dict | None, str | None]] = {}

    def resolve(self, ref: Any) -> tuple[dict | None, dict, str | None]:
        if not isinstance(ref, str) or not _HASH.fullmatch(ref):
            return None, _check("failed", "invalid_reference", "Expected an attempt hash."), None
        ref = ref.lower()
        matches = [h for h in self.paths if h == ref]
        if not matches:
            matches = [h for h in self.paths if h.startswith(ref)]
        if not matches:
            return None, _check("failed", "evidence_missing", f"Attempt {ref} was not found."), None
        if len(matches) != 1:
            return (
                None,
                _check("failed", "evidence_ambiguous", f"Attempt {ref} is ambiguous."),
                None,
            )
        key = matches[0]
        if key not in self.snapshots:
            digest = None
            record = None
            try:
                raw = self.paths[key].read_bytes()
                digest = _digest(raw)
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and parsed.get("commit_hash") == key:
                    record = parsed
            except (OSError, ValueError, UnicodeDecodeError):
                pass
            self.snapshots[key] = record, digest
        record, digest = self.snapshots[key]
        if record is None:
            return (
                None,
                _check("failed", "evidence_invalid", f"Attempt {key} is unreadable or malformed."),
                digest,
            )
        if (
            not isinstance(record.get("status"), str)
            or record["status"] not in _COMPLETED
            or not _number(record.get("score"))
        ):
            return (
                None,
                _check(
                    "failed", "evidence_incomplete", f"Attempt {key} has no completed finite score."
                ),
                digest,
            )
        return record, _check("passed", "evidence_found", f"Completed attempt {key}."), digest


def _compare_context(baseline: dict, result: dict) -> tuple[dict, str | None]:
    """Compare explicitly recorded conditions; never infer them from today's config."""
    bm = baseline.get("metadata")
    rm = result.get("metadata")
    bm = bm if isinstance(bm, dict) else {}
    rm = rm if isinstance(rm, dict) else {}
    bc = bm.get("evaluation_context")
    rc = rm.get("evaluation_context")
    bc = bc if isinstance(bc, dict) else {}
    rc = rc if isinstance(rc, dict) else {}
    differences = []
    missing = []
    for field in _CONTEXT_FIELDS:
        left, right = bc.get(field), rc.get(field)
        if (
            not isinstance(left, str)
            or not left.strip()
            or not isinstance(right, str)
            or not right.strip()
        ):
            missing.append(field)
        elif left != right:
            differences.append(field)
    left_mode, right_mode = bm.get("budget_class"), rm.get("budget_class")
    if left_mode not in ("real", "tune") or right_mode not in ("real", "tune"):
        missing.append("budget_class")
    elif left_mode != right_mode:
        differences.append("budget_class")
    direction = rc.get("direction")
    if direction not in ("maximize", "minimize"):
        missing.append("valid direction")
        direction = None
    if differences:
        return _check(
            "failed", "incomparable", "Different recorded conditions: " + ", ".join(differences)
        ), direction
    if missing:
        return _check(
            "unchecked", "context_missing", "Missing recorded conditions: " + ", ".join(missing)
        ), direction
    return _check(
        "passed", "context_matched", "Recorded task, grader, inputs, direction and mode match."
    ), direction


def _audit(
    entry: dict, attempts: _Attempts, direction: str | None, config_hash: str | None
) -> dict:
    blocked = _check(
        "unchecked",
        "evidence_required",
        "Requires completed result and explicit baseline evidence.",
    )
    checks = {
        "evidence": _check(
            "unchecked", "evidence_unspecified", "No structured experiment evidence."
        ),
        "numeric": dict(blocked),
        "comparability": dict(blocked),
        "replication": _check(
            "unchecked",
            "replication_unchecked",
            "Independent repeat evaluations are not tracked by this audit.",
        ),
    }
    report: dict[str, Any] = {
        "version": 1,
        "note_sha256": entry.get("content_sha256")
        or _digest(json.dumps(entry, sort_keys=True, default=str).encode()),
        "config_sha256": config_hash,
        "evidence_sha256": {},
        "checks": checks,
    }
    evidence = entry.get("evidence")
    if not evidence:
        return report
    if not isinstance(evidence, dict):
        checks["evidence"] = _check("failed", "evidence_invalid", "Evidence must be a mapping.")
        return report
    resolved = {}
    problems = []
    for field in ("attempt", "baseline"):
        ref = evidence.get(field)
        if ref is None or ref == "":
            problems.append(
                _check(
                    "unchecked",
                    f"{field}_unspecified",
                    f"Missing evidence.{field}; based_on is not a comparison baseline.",
                )
            )
            continue
        record, check, digest = attempts.resolve(ref)
        if digest:
            report["evidence_sha256"][field] = digest
        if record is None:
            problems.append(check)
        else:
            resolved[field] = record
    if problems:
        checks["evidence"] = next((p for p in problems if p["status"] == "failed"), problems[0])
        return report
    baseline, result = resolved["baseline"], resolved["attempt"]
    checks["evidence"] = _check(
        "passed",
        "evidence_found",
        "Result and explicit baseline resolve to completed scored records.",
    )
    checks["comparability"], recorded_direction = _compare_context(baseline, result)
    delta = result["score"] - baseline["score"]
    if not _number(delta):
        checks["numeric"] = _check("failed", "delta_invalid", "Computed score delta is not finite.")
        return report
    report["actual_delta"] = delta
    direction = recorded_direction or direction
    report["direction"] = direction
    report["direction_source"] = (
        "attempt" if recorded_direction else "run_config" if direction else None
    )
    # Improvement is only meaningful after comparison conditions have matched.
    if checks["comparability"]["status"] == "passed":
        report["improved"] = delta < 0 if direction == "minimize" else delta > 0
    claimed = evidence.get("score_delta")
    if claimed is None:
        checks["numeric"] = _check(
            "unchecked", "delta_unspecified", "Missing evidence.score_delta."
        )
    elif not _number(claimed):
        checks["numeric"] = _check(
            "failed",
            "delta_invalid",
            "score_delta must be a finite number, not a string or boolean.",
        )
    elif not math.isclose(claimed, delta, rel_tol=1e-6, abs_tol=1e-9):
        checks["numeric"] = _check(
            "failed",
            "delta_mismatch",
            f"Claimed {claimed:+g}; recorded result minus baseline is {delta:+g}.",
        )
    else:
        checks["numeric"] = _check(
            "passed",
            "delta_matched",
            f"Recorded result minus baseline is {delta:+g}; this does not establish causality.",
        )
    return report


def attach_audits(
    coral_dir: str | Path, entries: list[dict], island_id: str | int | None = None
) -> None:
    """Attach separate system reports to parsed entries without modifying files.

    In the global view, each note resolves evidence only in its own island.
    Snapshots are loaded once per request and never reused by later requests.
    """
    coral_dir = Path(coral_dir)
    direction = None
    config_hash = None
    try:
        raw = (coral_dir / "config.yaml").read_bytes()
        config_hash = _digest(raw)
        config = yaml.safe_load(raw)
        grader = config.get("grader") if isinstance(config, dict) else None
        value = grader.get("direction") if isinstance(grader, dict) else None
        if value in ("maximize", "minimize"):
            direction = value
    except (OSError, ValueError, yaml.YAMLError):
        pass
    indexes: dict[Path, _Attempts] = {}
    for entry in entries:
        if entry.get("category") == "raw":
            continue
        root = island_root(coral_dir, entry.get("island_id", island_id))
        if root not in indexes:
            indexes[root] = _Attempts(root)
        entry["audit"] = _audit(entry, indexes[root], direction, config_hash)


def format_audits(entries: list[dict]) -> str:
    """Human-readable audit output, including partial and unavailable checks."""
    if not entries:
        return "No notes to audit."
    lines = ["System evidence audit (read-only; author status is not verification)."]
    for entry in entries:
        name = entry.get("relative_path") or entry.get("filename", "notes.md")
        if entry.get("island_id") is not None:
            name = f"{entry['island_id']}/{name}"
        lines.append(f"\n{name} — {entry.get('title', '')}")
        for dimension, check in entry["audit"]["checks"].items():
            lines.append(f"  {dimension}: {check['status']} [{check['code']}] {check['message']}")
        if "improved" in entry["audit"]:
            outcome = "improved" if entry["audit"]["improved"] else "did not improve"
            lines.append(f"  Recorded score {outcome} ({entry['audit']['direction']}).")
    return "\n".join(lines)
