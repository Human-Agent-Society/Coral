"""Read/list/search notes from .coral/public/notes/ directory.

Notes are individual Markdown files with optional YAML frontmatter. Beyond the
two legacy fields (``creator``/``created``), notes may carry a *structured
trace* schema so the framework — not just the reading agent — can filter,
relate, and verify them:

    ---
    creator: island-0-agent-2
    created: 2026-03-14T17:35:00-00:00
    type: experiment            # experiment | hypothesis | dead_end | open_question | synthesis
    claim: "matmul inner-loop tiling at tile=32 improves score"
    based_on: a3f9c2            # attempt this builds on (provenance)
    evidence:
      attempt: 7b1e4d           # the graded artifact behind the claim
      score_delta: -0.03        # 0.42 -> 0.39
      verified: true
    confidence: medium                # low | medium | high
    status: confirmed           # confirmed | refuted | untested
    supersedes: [research/old-idea.md]
    touched: [matmul.cu]
    ---
    # Title of the note
    Body text with findings, numbers, conclusions...

All fields are optional at parse time so legacy data still loads. Missing
``creator`` is surfaced explicitly as the sentinel ``unknown`` (see
``UNATTRIBUTED_CREATOR``) so it shows up loudly in list views instead of being
silently filtered out of team aggregations; :func:`notes_unattributed` lists
the offending files for audit / lint. The legacy single ``notes.md`` (##
headings) format is also supported.
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from coral.hub._island import all_view_roots, island_root

# Sentinel surfaced when a note has no parseable ``creator:`` field. Kept
# distinct from the empty string so the absence is loud — the dashboard, the
# `coral notes` CLI, and the consolidate roster all show ``unknown`` next to
# such notes instead of rendering them as anonymous-but-present. The string
# matches the convention already used by ``coral.hub.skills`` so the two
# subsystems agree on how to spell "no author."
UNATTRIBUTED_CREATOR = "unknown"

# Structured-trace frontmatter fields surfaced (beyond creator/created) so the
# API/UI and aggregation/verification passes can act on them.
_TRACE_FIELDS = (
    "type",
    "claim",
    "status",
    "confidence",
    "based_on",
    "evidence",
    "supersedes",
    "refutes",
    "touched",
    "tags",
    "next",
)


def _jsonsafe(value: Any) -> Any:
    """Coerce YAML-parsed values (datetimes, nested dicts/lists) into a
    JSON-serializable shape. A bare ``created: 2026-03-14`` parses to a
    ``date``/``datetime`` under real YAML; the API layer must not choke on it.
    """
    if hasattr(value, "isoformat"):  # date / datetime / time
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonsafe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonsafe(v) for v in value]
    return value


def _notes_dir(coral_dir: str | Path, island_id: str | int | None = None) -> Path:
    """Resolve the notes directory without creating it during read-only queries."""
    return island_root(coral_dir, island_id) / "notes"


_SYSTEM_NOTE_FILENAMES = {"notes.md", "index.md"}
_SYSTEM_NOTE_TOP_LEVEL_DIRS = {"raw"}


def _is_user_note(p: Path, notes_dir: Path) -> bool:
    """Whether a markdown file under notes/ should be treated as a user-authored note.

    Excludes the legacy single-file ``notes.md``, generated ``index.md``, raw
    source captures, and files whose name starts with ``_`` (convention for
    aggregate/system-managed files like ``_connections.md`` and
    ``_open-questions.md``).
    """
    if p.name in _SYSTEM_NOTE_FILENAMES or p.name.startswith("_"):
        return False
    rel = p.relative_to(notes_dir)
    return not (rel.parts and rel.parts[0] in _SYSTEM_NOTE_TOP_LEVEL_DIRS)


def _iter_user_note_files(notes_dir: Path) -> list[Path]:
    """Return user note markdown files, pruning system directories while walking."""
    if not notes_dir.is_dir():
        return []
    files: list[Path] = []
    for root, dirs, names in os.walk(notes_dir):
        root_path = Path(root)
        if root_path == notes_dir:
            dirs[:] = sorted(d for d in dirs if d not in _SYSTEM_NOTE_TOP_LEVEL_DIRS)
        else:
            dirs.sort()
        for name in sorted(names):
            if name.endswith(".md"):
                p = root_path / name
                if _is_user_note(p, notes_dir):
                    files.append(p)
    return files


def _iter_raw_source_files(notes_dir: Path) -> list[Path]:
    """Return raw source captures under notes/raw/ (excluding ``_``-prefixed meta).

    Raw sources are pruned from _iter_user_note_files on purpose — they are
    immutable reference material, not agent-authored notes, and other consumers
    (index generation, dedup, grounding) rely on that exclusion. This is the
    opt-in path for surfacing them in read-only display (the dashboard), never
    treating them as user notes.
    """
    raw_dir = notes_dir / "raw"
    if not raw_dir.is_dir():
        return []
    return sorted(p for p in raw_dir.rglob("*.md") if p.is_file() and not p.name.startswith("_"))


def _lenient_frontmatter(front: str) -> dict[str, Any]:
    """Flat ``key: value`` parse — the pre-YAML fallback for malformed blocks."""
    meta: dict[str, Any] = {}
    for line in front.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
    return meta


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown. Returns (metadata, body).

    Uses a real YAML parser so structured-trace fields (lists, nested dicts
    like ``evidence:``) round-trip. Falls back to a lenient line-by-line parse
    if the block isn't valid YAML, so a malformed frontmatter never drops the
    note.
    """
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            front = text[3:end].strip()
            body = text[end + 3 :].strip()
            try:
                meta = yaml.safe_load(front)
            except yaml.YAMLError:
                meta = None
            if not isinstance(meta, dict):
                meta = _lenient_frontmatter(front)
            return meta, body
    return {}, text


def _parse_legacy_entries(text: str) -> list[dict[str, Any]]:
    """Parse legacy notes.md (## [date] title format) into entries."""
    pattern = re.compile(r"^## ", re.MULTILINE)
    parts = pattern.split(text)
    entries = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"\[([^\]]*)\]\s*(.*)", part, re.DOTALL)
        if m:
            date = m.group(1).strip()
            rest = m.group(2)
            title_line, _, body = rest.partition("\n")
            title = title_line.strip()
            body = body.strip()
        else:
            title_line, _, body = part.partition("\n")
            date = ""
            title = title_line.strip()
            body = body.strip()

        entries.append(
            {
                "date": date,
                "title": title,
                "body": body,
                "creator": UNATTRIBUTED_CREATOR,
                "filename": "notes.md",
            }
        )
    return entries


def _parse_note_file(path: Path) -> dict[str, Any]:
    """Parse a single note .md file into an entry dict."""
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    meta, body = _parse_frontmatter(text)

    # Title: prefer a body `# heading`, then a frontmatter `title:`, then the
    # filename. Raw source captures often carry their title only in frontmatter.
    title = ""
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("# "):
            title = line[2:].strip()
            break
    if not title:
        title = str(meta.get("title", "") or "").strip()
    if not title:
        title = path.stem.replace("-", " ").replace("_", " ").title()

    creator_raw = str(meta.get("creator", "") or "").strip()
    entry: dict[str, Any] = {
        "content_sha256": hashlib.sha256(raw).hexdigest(),
        "date": str(meta.get("created", "") or ""),
        "title": title,
        "body": body,
        "creator": creator_raw or UNATTRIBUTED_CREATOR,
        "filename": path.name,
        "_mtime": os.path.getmtime(path),
        "_path": path,  # full path, used to compute relative path later
    }
    # Surface structured-trace fields when present (JSON-safe for the API).
    for key in _TRACE_FIELDS:
        val = meta.get(key)
        if val not in (None, "", [], {}):
            entry[key] = _jsonsafe(val)

    # Full frontmatter passthrough (JSON-safe), so the dashboard can show every
    # field an agent wrote — raw sources and research notes use different
    # vocabularies, and a curated allowlist silently drops whatever it misses.
    # Keyed as ``frontmatter`` so it never collides with derived entry fields.
    frontmatter = {str(k): _jsonsafe(v) for k, v in meta.items() if v not in (None, "", [], {})}
    if frontmatter:
        entry["frontmatter"] = frontmatter
    return entry


def _first_present(meta: dict[str, Any], keys: tuple[str, ...]) -> str:
    """First non-empty value among ``keys`` (raw sources use varied field names)."""
    for k in keys:
        v = str(meta.get(k, "") or "").strip()
        if v:
            return v
    return ""


def _parse_raw_source_file(path: Path) -> dict[str, Any]:
    """Parse a raw/ source capture, surfacing its provenance frontmatter.

    Raw sources use a source vocabulary (``source_url`` / ``source_type`` /
    ``captured`` / ``retrieved_by`` / ``also_confirmed_by``) rather than the
    note schema, so _parse_note_file would drop the one field that matters most
    for a source — where it came from. Surface those, and map capture/retriever
    into the shared ``date`` / ``creator`` slots the dashboard already renders.
    """
    entry = _parse_note_file(path)
    meta, _ = _parse_frontmatter(path.read_text(encoding="utf-8"))

    url = _first_present(meta, ("source_url", "url"))
    if url:
        entry["source_url"] = url
    stype = _first_present(meta, ("source_type", "type"))
    if stype:
        entry["source_type"] = stype
    captured = _first_present(meta, ("captured", "fetched"))
    if captured and not entry.get("date"):
        entry["date"] = captured
    who = _first_present(meta, ("retrieved_by", "captured_by"))
    if who and entry.get("creator") in (UNATTRIBUTED_CREATOR, "", None):
        entry["creator"] = who
    confirmed = meta.get("also_confirmed_by")
    if confirmed not in (None, "", [], {}):
        entry["also_confirmed_by"] = _jsonsafe(confirmed)
    return entry


def _collect_from_dir(directory: Path) -> list[dict[str, Any]]:
    """Collect note entries from a directory, including subdirectories."""
    if not directory.is_dir():
        return []

    md_files = _iter_user_note_files(directory)

    if md_files:
        entries = [_parse_note_file(f) for f in md_files]
        legacy = directory / "notes.md"
        if legacy.exists() and legacy.stat().st_size > 0:
            entries.extend(_parse_legacy_entries(legacy.read_text(encoding="utf-8")))
        return entries

    legacy = directory / "notes.md"
    if legacy.exists() and legacy.stat().st_size > 0:
        return _parse_legacy_entries(legacy.read_text(encoding="utf-8"))

    return []


def _sort_key(entry: dict[str, Any]) -> datetime:
    """Return a datetime for sorting. Parses the frontmatter date string,
    falling back to file mtime if unavailable or unparseable."""
    date_str = entry.get("date", "")
    if date_str:
        try:
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except (ValueError, TypeError):
            pass
    mtime = entry.get("_mtime")
    if mtime is not None:
        return datetime.fromtimestamp(mtime, tz=UTC)
    return datetime.min.replace(tzinfo=UTC)


def _filter_notes_by_status(
    entries: list[dict[str, Any]],
    status: str | None,
) -> list[dict[str, Any]]:
    """Return entries whose parsed status matches the requested value."""
    if status is None:
        return entries

    expected = status.strip().casefold()
    return [
        entry
        for entry in entries
        if "status" in entry and str(entry["status"]).strip().casefold() == expected
    ]


def list_notes(
    coral_dir: str | Path,
    island_id: str | int | None = None,
    *,
    include_raw: bool = False,
    status: str | None = None,
    audit: bool = False,
) -> list[dict[str, Any]]:
    """List all note entries from the notes directory.

    Reads individual .md files. Falls back to legacy notes.md format.
    Also checks the legacy 'insights/' directory for backward compatibility.

    With ``island_id=None`` in multi-island mode, aggregates notes from
    every island so ``coral notes`` shows the whole team's research.

    ``include_raw`` additionally surfaces immutable ``raw/`` source captures
    (category ``"raw"``) for read-only display. It defaults to False so
    agent-facing consumers (``coral notes``, search, recent-note summaries)
    keep seeing only authored notes; only the dashboard opts in.

    ``status`` filters on the parsed frontmatter value after trimming
    surrounding whitespace and normalizing case. Notes without a status are
    excluded only when this filter is active.

    ``audit`` attaches fresh system evidence checks, separate from author fields.
    """
    if audit:
        from coral.hub.notes_audit import attach_audits

        entries = list_notes(coral_dir, island_id, include_raw=include_raw, status=status)
        attach_audits(coral_dir, entries, island_id)
        return entries
    coral_dir = Path(coral_dir)
    if island_id is not None or not (coral_dir / "islands").exists():
        entries = _list_notes_single(coral_dir, island_id, include_raw=include_raw)
        return _filter_notes_by_status(entries, status)

    entries = []
    for view_root in all_view_roots(coral_dir):
        sub = _list_notes_single(
            coral_dir, island_id=view_root.name, clean=False, include_raw=include_raw
        )
        for entry in sub:
            entry["island_id"] = view_root.name
        entries.extend(sub)
    entries.sort(key=_sort_key)
    _clean_note_entries(entries)
    return _filter_notes_by_status(entries, status)


def _list_notes_single(
    coral_dir: Path,
    island_id: str | int | None,
    *,
    clean: bool = True,
    include_raw: bool = False,
) -> list[dict[str, Any]]:
    notes_dir = _notes_dir(coral_dir, island_id)
    entries = _collect_from_dir(notes_dir)

    # Also read from insights/ directory if present
    insights_dir = island_root(coral_dir, island_id) / "insights"
    if insights_dir.is_dir():
        seen = {e["filename"] for e in entries}
        for e in _collect_from_dir(insights_dir):
            if e["filename"] not in seen:
                entries.append(e)

    # Opt-in: surface raw source captures for read-only display. Parsed the same
    # way as notes, so _clean_note_entries derives category="raw" from the path.
    if include_raw:
        entries.extend(_parse_raw_source_file(f) for f in _iter_raw_source_files(notes_dir))

    entries.sort(key=_sort_key)

    if clean:
        _clean_note_entries(entries)
    return entries


def _clean_note_entries(entries: list[dict[str, Any]]) -> None:
    """Add display path/category fields and remove internal sort fields in place."""
    for entry in entries:
        entry.pop("_mtime", None)
        full_path = entry.pop("_path", None)
        if full_path:
            rel_path = Path(full_path)
            try:
                reversed_idx = list(reversed(rel_path.parts)).index("notes")
                notes_idx = len(rel_path.parts) - reversed_idx - 1
                rel = str(Path(*rel_path.parts[notes_idx + 1 :]))
            except ValueError:
                rel = rel_path.name
            entry["relative_path"] = rel
            # Categorize by top-level directory
            parts = rel.split(os.sep)
            if len(parts) > 1:
                entry["category"] = parts[0]  # raw, research, experiments, etc.
            else:
                entry["category"] = "other"
        else:
            entry["relative_path"] = entry.get("filename", "")
            entry["category"] = "other"


def search_notes(
    coral_dir: str | Path,
    query: str,
    island_id: str | int | None = None,
    *,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Search notes by keyword and optional status."""
    query_lower = query.lower()
    results = []
    for entry in list_notes(coral_dir, island_id=island_id, status=status):
        full_text = f"{entry['title']} {entry['body']}".lower()
        if query_lower in full_text:
            results.append(entry)
    return results


def get_recent_notes(
    coral_dir: str | Path,
    n: int = 5,
    island_id: str | int | None = None,
    *,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Filter notes, then return the last N matches."""
    entries = list_notes(coral_dir, island_id=island_id, status=status)
    return entries[-n:] if len(entries) > n else entries


def format_notes_list(entries: list[dict[str, Any]]) -> str:
    """Format note entries for terminal display.

    The ``creator`` field is always rendered — notes without a ``creator:``
    frontmatter field display as ``(unknown)`` so an agent who forgot the
    field sees the gap in `coral notes` output immediately instead of
    discovering it later via silent exclusion from team-level views.
    """
    if not entries:
        return "No notes yet."
    lines = []
    for i, e in enumerate(entries, 1):
        date_str = f"[{e['date']}] " if e.get("date") else ""
        creator = e.get("creator") or UNATTRIBUTED_CREATOR
        lines.append(f"  {i}. {date_str}{e['title']} ({creator})")
    return "\n".join(lines)


def read_note(
    coral_dir: str | Path,
    index: int,
    island_id: str | int | None = None,
) -> str | None:
    """Read a specific note entry by index (1-based)."""
    entries = list_notes(coral_dir, island_id=island_id)
    if 1 <= index <= len(entries):
        e = entries[index - 1]
        return e["body"]
    return None


def read_all_notes(
    coral_dir: str | Path,
    island_id: str | int | None = None,
) -> str:
    """Read all notes concatenated."""
    entries = list_notes(coral_dir, island_id=island_id)
    if not entries:
        return ""
    parts = []
    for e in entries:
        parts.append(e["body"])
    return "\n\n---\n\n".join(parts)


def notes_by(
    coral_dir: str | Path,
    island_id: str | int | None,
    agent_id: str,
) -> list[Path]:
    """Return absolute paths of notes whose frontmatter `creator` matches agent_id.

    Notes without a `creator:` field (e.g. legacy notes, the bundled
    notes.md) are excluded — they cannot be safely attributed and should
    stay on the source island when their author migrates. Use
    :func:`notes_unattributed` to surface them explicitly for audit /
    lint passes.
    """
    notes_dir = _notes_dir(coral_dir, island_id)
    matched: list[Path] = []
    for md_file in _iter_user_note_files(notes_dir):
        try:
            text = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        meta, _ = _parse_frontmatter(text)
        if meta.get("creator") == agent_id:
            matched.append(md_file)
    return matched


def notes_unattributed(
    coral_dir: str | Path,
    island_id: str | int | None,
) -> list[Path]:
    """Return absolute paths of user-authored notes missing a ``creator:`` field.

    A note that lands here is invisible to every team-level process that
    filters by author (``notes_by``, the consolidate roster, the librarian
    subagent, migration attribution). The list view still shows the file —
    tagged ``(unknown)`` via :func:`format_notes_list` — so the gap can be
    fixed by appending a ``creator:`` line to the file's frontmatter.
    """
    notes_dir = _notes_dir(coral_dir, island_id)
    missing: list[Path] = []
    for md_file in _iter_user_note_files(notes_dir):
        try:
            text = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        meta, _ = _parse_frontmatter(text)
        creator = str(meta.get("creator", "") or "").strip()
        if not creator:
            missing.append(md_file)
    return missing


# --------------------------------------------------------------------------- #
# Structured-trace graph                                                      #
# --------------------------------------------------------------------------- #

# Markdown link `[text](some/path.md)` and wiki link `[[name]]` to another note.
_MD_LINK_RE = re.compile(r"\]\(\s*([^)\s]+?\.md)\s*\)")
_WIKI_LINK_RE = re.compile(r"\[\[\s*([^\]]+?)\s*\]\]")


def _as_list(value: Any) -> list[str]:
    """Normalize a frontmatter field into a list of strings.

    Accepts a YAML list, a single scalar, or a comma-separated string (the
    shape the legacy flat-frontmatter fallback produces).
    """
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [s.strip() for s in str(value).split(",") if s.strip()]


def _node_id(entry: dict[str, Any]) -> str | None:
    return entry.get("relative_path") or entry.get("filename") or None


def notes_graph(
    coral_dir: str | Path,
    island_id: str | int | None = None,
) -> dict[str, Any]:
    """Build a node/edge graph of notes and the connections between them.

    Mirrors ``hub.attempts``' DAG shape so the dashboard can render it the same
    way: ``{"nodes": [...], "edges": [{"from", "to", "kind"}]}``.

    Nodes are notes (``id`` = relative path). Edges come from:
      - typed frontmatter links: ``supersedes`` / ``refutes`` (note → note),
      - markdown/wiki links in the body pointing at another note (``references``).

    The ``references`` edges work on existing free-text notes (the reflect
    heartbeat already has agents write ``Based on: [research/x.md](...)``), so
    the graph is populated even before the structured schema is adopted.
    """
    entries = list_notes(coral_dir, island_id=island_id)

    # Index every spelling an author might use to reference a note → canonical id.
    index: dict[str, str] = {}
    for e in entries:
        nid = _node_id(e)
        if not nid:
            continue
        for key in {nid, e.get("filename", ""), Path(nid).name, Path(nid).stem}:
            if key:
                index.setdefault(str(key), nid)

    def _resolve(ref: str) -> str | None:
        ref = str(ref).strip().lstrip("./")
        for key in (ref, Path(ref).name, Path(ref).stem):
            if key in index:
                return index[key]
        return None

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for e in entries:
        nid = _node_id(e)
        if not nid:
            continue
        nodes.append(
            {
                "id": nid,
                "title": e.get("title", nid),
                "type": e.get("type") or e.get("category") or "note",
                "status": e.get("status"),
                "confidence": e.get("confidence"),
                "creator": e.get("creator") or UNATTRIBUTED_CREATOR,
                "island_id": e.get("island_id"),
                "date": e.get("date", ""),
                "based_on": e.get("based_on"),
            }
        )

        body = e.get("body", "") or ""
        links: list[tuple[str, str]] = []
        links += [("supersedes", t) for t in _as_list(e.get("supersedes"))]
        links += [("refutes", t) for t in _as_list(e.get("refutes"))]
        links += [("references", t) for t in _MD_LINK_RE.findall(body)]
        links += [("references", t) for t in _WIKI_LINK_RE.findall(body)]

        for kind, target in links:
            tid = _resolve(target)
            if not tid or tid == nid:
                continue
            key = (nid, tid, kind)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append({"from": nid, "to": tid, "kind": kind})

    return {"nodes": nodes, "edges": edges}
