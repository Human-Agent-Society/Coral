"""Bundled prompts and subagent templates instruct agents to stamp `creator:`.

Migration filters notes/skills by frontmatter `creator: <agent_id>`. If the
canonical heartbeat prompt and the bundled subagent / skill-creator templates
do not tell agents to stamp it, migration will silently drop their work.
This test is the regression gate for that instruction surviving future
prompt edits.
"""

import importlib.util
import tempfile
from pathlib import Path
from types import ModuleType

COMMON_INSTRUCTION_KEYWORDS = ["creator:", "frontmatter"]


def _load_script(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_prompt(path: Path) -> None:
    text = path.read_text(encoding="utf-8").lower()
    for kw in COMMON_INSTRUCTION_KEYWORDS:
        assert kw in text, f"{path} must mention {kw!r} so agents stamp the creator field"


def test_consolidate_prompt_instructs_creator_stamping():
    _check_prompt(Path("coral/hub/prompts/consolidate.md"))


def test_librarian_template_instructs_creator_stamping():
    _check_prompt(Path("coral/template/agents/librarian.md"))


def test_skill_creator_template_instructs_creator_stamping():
    _check_prompt(Path("coral/template/skills/skill-creator/SKILL.md"))


def test_bundled_skill_md_files_have_no_creator_frontmatter():
    """Bundled skills must not have `creator:` in their SKILL.md frontmatter, so
    migration's `skills_by` filter correctly excludes them."""
    bundled = Path("coral/template/skills").rglob("SKILL.md")
    for skill_md in bundled:
        text = skill_md.read_text(encoding="utf-8")
        # Only inspect frontmatter (first --- block); body may legitimately mention "creator"
        if not text.startswith("---"):
            continue
        end = text.find("---", 3)
        front = text[3:end]
        assert "\ncreator:" not in front and not front.startswith("creator:"), (
            f"{skill_md} has stray `creator:` in frontmatter — would migrate as agent-authored"
        )


def test_create_notes_scripts_skip_index_and_raw_sources():
    lint = _load_script(Path("coral/template/skills/create-notes/scripts/lint.py"))
    unattributed = _load_script(Path("coral/template/skills/create-notes/scripts/unattributed.py"))

    with tempfile.TemporaryDirectory() as d:
        notes_dir = Path(d) / "notes"
        raw_dir = notes_dir / "raw"
        research_dir = notes_dir / "research"
        synthesis_dir = notes_dir / "_synthesis"
        raw_dir.mkdir(parents=True)
        research_dir.mkdir()
        synthesis_dir.mkdir()
        (notes_dir / "index.md").write_text(
            "# Index\n\n- [Research](research/useful.md)", encoding="utf-8"
        )
        (raw_dir / "paper.md").write_text("# Raw source without frontmatter\n", encoding="utf-8")
        (research_dir / "useful.md").write_text(
            "# User note without frontmatter\n", encoding="utf-8"
        )
        (synthesis_dir / "team-roster.md").write_text(
            "# Synthesis note without frontmatter\n", encoding="utf-8"
        )

        lint_targets = lint._collect_targets([str(notes_dir)])
        assert [(p.relative_to(root).as_posix(), root) for p, root in lint_targets] == [
            ("_synthesis/team-roster.md", notes_dir),
            ("research/useful.md", notes_dir),
        ]
        assert lint._collect_targets([str(notes_dir / "index.md")]) == []
        assert lint._collect_targets([str(raw_dir / "paper.md")]) == []
        unattributed_targets = [
            p.relative_to(notes_dir).as_posix()
            for p in unattributed._iter_user_note_files(notes_dir)
        ]
        assert unattributed_targets == ["_synthesis/team-roster.md", "research/useful.md"]


def test_note_lint_requires_explicit_comparison_baseline_and_boolean_verified(tmp_path):
    lint = _load_script(Path("coral/template/skills/create-notes/scripts/lint.py"))
    note = tmp_path / "experiment.md"
    text = (
        "---\ncreator: agent\ncreated: 2026-09-08T00:00:00Z\n"
        "type: experiment\nstatus: confirmed\nevidence:\n"
        '  attempt: "bbbbbbbb"\n  score_delta: 2\n  verified: "false"\n'
        "---\n# Experiment\n"
    )
    note.write_text(text)
    warnings = lint._lint_note(note, tmp_path, set(), check_index=False)
    assert any("evidence.baseline" in warning for warning in warnings)
    assert any("without `evidence.verified: true`" in warning for warning in warnings)
    note.write_text(
        text.replace('verified: "false"', "verified: true").replace(
            "  score_delta:", '  baseline: "aaaaaaaa"\n  score_delta:'
        )
    )
    assert lint._lint_note(note, tmp_path, set(), check_index=False) == []
