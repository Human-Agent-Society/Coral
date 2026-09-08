#!/usr/bin/env python3
"""lint.py — Mechanize the create-notes self-audit checklist.

Usage:
    python lint.py [NOTES_DIR_OR_FILE ...] [--strict] [--quiet] [--no-index]

What it checks (per note):
- Frontmatter is parseable and contains a non-blank ``creator:``
- ``created:`` is present and looks like an ISO-8601 date / datetime
- ``type:`` (if present) is one of the documented vocabulary
- ``status:`` (if present) is one of the documented vocabulary
- ``confidence:`` (if present) is one of ``low | medium | high``
- ``status: confirmed`` is paired with ``evidence.verified: true`` — a
  confirmed claim with no verified evidence is the most common silent
  inconsistency the knowledge graph hides
- Filename is lowercase kebab-case, no spaces, no agent_id (except for
  the per-agent ``focus-*`` and ``migration_*`` exceptions)
- The note's filename is referenced in the same ``notes/index.md``
- The note's ``type:`` matches the directory it lives in (experiments/
  → experiment, _synthesis/ → synthesis, focus/ → hypothesis,
  infra/ → experiment)

Defaults to the bundled location ``.coral/public/notes`` if no path is
given. Pass one or more files to lint just those.

Advisory by default — exits 0 even if there are warnings, so a tight
agent loop isn't blocked by a stylistic gap. Pass ``--strict`` for a
non-zero exit code when any warning is found (CI / pre-commit use).

Self-contained: no imports from coral.* so the script ships intact
inside .coral/public/skills/create-notes/scripts/ on every island.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_NOTES_DIR = ".coral/public/notes"

TYPE_VOCAB = {"experiment", "hypothesis", "dead_end", "open_question", "synthesis"}
STATUS_VOCAB = {"confirmed", "refuted", "untested"}
CONFIDENCE_VOCAB = {"low", "medium", "high"}

# Map directory / filename pattern → expected `type:` value. The first
# match wins, so order matters: focus-*.md must beat the bare "other"
# fallback for top-level notes.
PATH_TYPE_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(^|/)experiments/"), "experiment"),
    (re.compile(r"(^|/)_synthesis/"), "synthesis"),
    (re.compile(r"(^|/)infra/"), "experiment"),
    (re.compile(r"(^|/)focus/"), "hypothesis"),
    (re.compile(r"(^|/)focus-[^/]+\.md$"), "hypothesis"),
]

# Filenames that are exempt from the "no agent id, no leading _" rules.
EXEMPT_FILENAMES = re.compile(r"^(focus-.+|migration_.+|_.+)\.md$")


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML-ish frontmatter. Yaml-free for portability (the bundled
    skill must run in a venv that may not have PyYAML)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    front = text[3:end].strip()
    body = text[end + 3 :].strip()
    meta: dict[str, Any] = {}
    current_key: str | None = None
    for raw in front.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        # Nested-dict child like "  attempt: 7b1e4d" under "evidence:"
        if line.startswith((" ", "\t")) and current_key and isinstance(meta.get(current_key), dict):
            stripped = line.strip()
            if ":" in stripped:
                k, _, v = stripped.partition(":")
                meta[current_key][k.strip()] = _coerce(v.strip())
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        current_key = key
        if val == "":
            # Could be a nested dict opener (next lines indented) or just blank.
            meta[key] = {}
        elif val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            meta[key] = [s.strip() for s in inner.split(",")] if inner else []
        else:
            meta[key] = _coerce(val)
    return meta, body


def _coerce(val: str) -> Any:
    """Best-effort scalar coercion (bool / int / float / strip quotes)."""
    if val.lower() in {"true", "yes"}:
        return True
    if val.lower() in {"false", "no"}:
        return False
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


# System-managed files that live under notes/ but aren't individual
# author-written notes (and so have no frontmatter to lint). Mirrors the
# split in coral.hub.notes._is_user_note, plus index.md which is owned
# by organize-files.
SYSTEM_FILENAMES = {"notes.md", "index.md"}
SYSTEM_TOP_LEVEL_DIRS = {"raw"}


def _is_user_note(p: Path, notes_root: Path) -> bool:
    if p.name in SYSTEM_FILENAMES or p.name.startswith("_"):
        return False
    rel = p.relative_to(notes_root)
    return not (rel.parts and rel.parts[0] in SYSTEM_TOP_LEVEL_DIRS)


def _iter_user_note_files(notes_root: Path) -> list[Path]:
    files: list[Path] = []
    for root, dirs, names in os.walk(notes_root):
        root_path = Path(root)
        if root_path == notes_root:
            dirs[:] = sorted(d for d in dirs if d not in SYSTEM_TOP_LEVEL_DIRS)
        else:
            dirs.sort()
        for name in sorted(names):
            if name.endswith(".md"):
                p = root_path / name
                if _is_user_note(p, notes_root):
                    files.append(p)
    return files


def _iso_parseable(value: Any) -> bool:
    if not value:
        return False
    s = str(value).strip()
    if not s:
        return False
    # datetime.fromisoformat accepts "2026-06-25" and "2026-06-25T14:32:00+00:00"
    # but not the trailing Z. Strip it for the check.
    s_norm = s.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(s_norm)
        return True
    except ValueError:
        return False


def _index_entries(notes_root: Path) -> set[str]:
    """Read notes/index.md and return the set of paths/filenames it mentions."""
    index = notes_root / "index.md"
    if not index.exists():
        return set()
    text = index.read_text(encoding="utf-8", errors="replace")
    # Any `.md` reference, link or bare path. Don't be precise about format —
    # we're just checking "did the author mention this filename anywhere."
    return set(re.findall(r"([\w\-./]+\.md)", text))


def _check_filename(path: Path) -> list[str]:
    name = path.name
    warnings: list[str] = []
    if EXEMPT_FILENAMES.match(name):
        return warnings
    if " " in name:
        warnings.append("filename contains a space — use kebab-case")
    if name != name.lower():
        warnings.append("filename has uppercase characters — use kebab-case")
    if "_" in name.replace(".md", ""):
        warnings.append("filename uses snake_case — use kebab-case (dashes)")
    return warnings


def _expected_type_for(rel_path: str) -> str | None:
    for pat, expected in PATH_TYPE_HINTS:
        if pat.search(rel_path):
            return expected
    return None


def _lint_note(
    path: Path,
    notes_root: Path,
    index_refs: set[str],
    *,
    check_index: bool,
) -> list[str]:
    warnings: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"could not read file: {exc}"]

    meta, _body = _parse_frontmatter(text)

    creator = str(meta.get("creator", "") or "").strip()
    if not creator:
        warnings.append(
            "missing `creator:` — note will appear as `(unknown)` and be "
            "skipped by team-level views"
        )

    created = meta.get("created")
    if not created:
        warnings.append("missing `created:` — list/graph views fall back to file mtime")
    elif not _iso_parseable(created):
        warnings.append(f"`created: {created!r}` is not ISO-8601 parseable")

    type_ = meta.get("type")
    if type_ is not None and type_ not in TYPE_VOCAB:
        warnings.append(f"`type: {type_!r}` is not in the vocabulary {sorted(TYPE_VOCAB)}")

    rel = path.relative_to(notes_root).as_posix()
    expected = _expected_type_for(rel)
    if expected and type_ and type_ != expected:
        warnings.append(
            f"`type: {type_!r}` disagrees with path — `{rel}` typically uses `type: {expected}`"
        )

    status = meta.get("status")
    if status is not None and status not in STATUS_VOCAB:
        warnings.append(f"`status: {status!r}` is not in the vocabulary {sorted(STATUS_VOCAB)}")

    confidence = meta.get("confidence")
    if confidence is not None:
        if isinstance(confidence, int | float) and not isinstance(confidence, bool):
            warnings.append(
                f"`confidence: {confidence}` is a number — the schema is now "
                f"{sorted(CONFIDENCE_VOCAB)}; map low (<0.4) / medium / high (>0.7)"
            )
        elif confidence not in CONFIDENCE_VOCAB:
            warnings.append(
                f"`confidence: {confidence!r}` is not in the vocabulary {sorted(CONFIDENCE_VOCAB)}"
            )

    evidence = meta.get("evidence")
    if isinstance(evidence, dict) and "score_delta" in evidence and not evidence.get("baseline"):
        warnings.append(
            "`evidence.score_delta` needs an explicit `evidence.baseline` — "
            "`based_on` is provenance, not the comparison baseline; run `coral notes --audit`"
        )
    if status == "confirmed":
        verified = isinstance(evidence, dict) and evidence.get("verified") is True
        if not verified:
            warnings.append(
                "`status: confirmed` without `evidence.verified: true` — a "
                "confirmed claim without verified evidence renders ambiguously "
                "in the knowledge graph"
            )

    warnings.extend(_check_filename(path))

    if check_index and index_refs:
        if not any(token in index_refs for token in (rel, path.name)):
            warnings.append(
                "not referenced in `notes/index.md` — add a one-line entry so "
                "the next agent finds this note"
            )

    return warnings


def _collect_targets(args_paths: list[str]) -> list[tuple[Path, Path]]:
    """Resolve user inputs into (note_file, notes_root) pairs."""
    inputs = args_paths or [DEFAULT_NOTES_DIR]
    pairs: list[tuple[Path, Path]] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for f in _iter_user_note_files(p):
                pairs.append((f, p))
        elif p.is_file() and p.suffix == ".md":
            # Walk up to find a directory named "notes"; fall back to parent.
            root = p.parent
            for ancestor in (p, *p.parents):
                if ancestor.name == "notes":
                    root = ancestor
                    break
            if _is_user_note(p, root):
                pairs.append((p, root))
        else:
            print(f"warning: {raw} is neither a directory nor a .md file", file=sys.stderr)
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="*", help="Notes directory or specific .md files")
    ap.add_argument("--strict", action="store_true", help="Exit 1 if any warnings")
    ap.add_argument("--quiet", action="store_true", help="Only print files with warnings")
    ap.add_argument(
        "--no-index", action="store_true", help="Skip the index.md cross-reference check"
    )
    args = ap.parse_args()

    pairs = _collect_targets(args.paths)
    if not pairs:
        print(f"no notes found under {args.paths or [DEFAULT_NOTES_DIR]}")
        return 0

    index_cache: dict[Path, set[str]] = {}
    total_warnings = 0
    files_with_warnings = 0

    for note_file, notes_root in pairs:
        if notes_root not in index_cache:
            index_cache[notes_root] = _index_entries(notes_root)
        warnings = _lint_note(
            note_file,
            notes_root,
            index_cache[notes_root],
            check_index=not args.no_index,
        )
        rel = note_file.relative_to(notes_root).as_posix()
        if warnings:
            files_with_warnings += 1
            total_warnings += len(warnings)
            print(f"\n{rel}")
            for w in warnings:
                print(f"  - {w}")
        elif not args.quiet:
            print(f"\n{rel}\n  ok")

    print(
        f"\n{len(pairs)} note(s) checked, {files_with_warnings} with warnings, "
        f"{total_warnings} warning(s) total"
    )
    if args.strict and total_warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
