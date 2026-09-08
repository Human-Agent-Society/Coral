"""Evidence checks must not turn author assertions into system verification."""

import argparse
import json
from types import SimpleNamespace

import pytest
import yaml

from coral.cli import query
from coral.hub.attempts import write_attempt
from coral.hub.notes import list_notes
from coral.types import Attempt
from coral.web.api import get_notes

BASE = "a" * 40
RESULT = "b" * 40


def _attempt(root, commit, score, *, island=None, context=True, direction="maximize", **changes):
    metadata = {"budget_class": "real"}
    if context:
        metadata["evaluation_context"] = {
            "task": "task-v1",
            "grader": "grader-v1",
            "inputs": "dataset-v1",
            "direction": direction,
        }
    fields = dict(
        commit_hash=commit,
        agent_id="agent-1",
        title="Experiment",
        score=score,
        status="baseline",
        parent_hash=None,
        timestamp="2026-09-08T01:00:00Z",
        metadata=metadata,
    )
    fields.update(changes)
    return write_attempt(root, Attempt(**fields), island_id=island)


def _note(root, *, island=None, evidence=None, filename="experiment.md", **fields):
    view = root / "public" if island is None else root / "islands" / island
    notes = view / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    meta = {
        "creator": "agent-1",
        "type": "experiment",
        "status": "confirmed",
        "confidence": "high",
        "evidence": evidence
        if evidence is not None
        else {
            "attempt": RESULT[:8],
            "baseline": BASE[:8],
            "score_delta": 2,
            "verified": True,
        },
        **fields,
    }
    path = notes / filename
    path.write_text("---\n" + yaml.safe_dump(meta) + "---\n# Experiment\nMy causal explanation.\n")
    return path


def _report(root, island=None):
    return list_notes(root, island_id=island, audit=True)[0]["audit"]


@pytest.fixture
def run(tmp_path):
    root = tmp_path / ".coral"
    _attempt(root, BASE, 10)
    _attempt(root, RESULT, 12)
    _note(root)
    return root


def test_verified_flag_does_not_verify_claim_and_audit_is_read_only(run):
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in run.rglob("*") if p.is_file()}
    entries = list_notes(run, audit=True)
    report = entries[0]["audit"]
    assert entries[0]["status"] == "confirmed"
    assert entries[0]["evidence"]["verified"] is True
    assert report["checks"]["numeric"]["status"] == "passed"
    assert report["checks"]["comparability"]["status"] == "passed"
    assert report["checks"]["replication"]["status"] == "unchecked"
    assert report["improved"] is True
    assert before == {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in run.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("direction,expected", [("maximize", True), ("minimize", False)])
def test_score_direction_does_not_flip_raw_delta(run, direction, expected):
    _attempt(run, BASE, 10, direction=direction)
    _attempt(run, RESULT, 12, direction=direction)
    report = _report(run)
    assert report["actual_delta"] == 2
    assert report["improved"] is expected
    assert report["direction_source"] == "attempt"


def test_minimize_negative_delta_and_float_tolerance(run):
    _attempt(run, BASE, 0.3, direction="minimize")
    _attempt(run, RESULT, 0.1, direction="minimize")
    _note(run, evidence={"attempt": RESULT, "baseline": BASE, "score_delta": -0.2})
    report = _report(run)
    assert report["checks"]["numeric"]["code"] == "delta_matched"
    assert report["improved"] is True


@pytest.mark.parametrize(
    "delta,code",
    [
        (-2, "delta_mismatch"),
        (True, "delta_invalid"),
        ("2", "delta_invalid"),
        (float("nan"), "delta_invalid"),
        (float("inf"), "delta_invalid"),
        (None, "delta_unspecified"),
    ],
)
def test_false_or_invalid_claimed_delta(run, delta, code):
    _note(run, evidence={"attempt": RESULT, "baseline": BASE, "score_delta": delta})
    assert _report(run)["checks"]["numeric"]["code"] == code


@pytest.mark.parametrize(
    "ref,code",
    [
        ("cccccccc", "evidence_missing"),
        ("../private/key", "invalid_reference"),
        (42, "invalid_reference"),
        ([RESULT], "invalid_reference"),
    ],
)
def test_bad_evidence_references(run, ref, code):
    _note(run, evidence={"attempt": ref, "baseline": BASE, "verified": True})
    assert _report(run)["checks"]["evidence"]["code"] == code


def test_ambiguous_prefix_and_exact_hash(run):
    _attempt(run, "b" * 39 + "c", 14)
    assert _report(run)["checks"]["evidence"]["code"] == "evidence_ambiguous"
    _note(run, evidence={"attempt": RESULT, "baseline": BASE, "score_delta": 2})
    assert _report(run)["checks"]["numeric"]["status"] == "passed"


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "pending"},
        {"status": "crashed"},
        {"status": "timeout"},
        {"status": []},
        {"score": None},
        {"score": float("nan")},
        {"score": True},
    ],
)
def test_unusable_evaluations(run, changes):
    _attempt(run, RESULT, changes.pop("score", 12), **changes)
    assert _report(run)["checks"]["evidence"]["code"] == "evidence_incomplete"


@pytest.mark.parametrize("raw", ["{", "[]", '{"commit_hash": "wrong"}'])
def test_corrupt_attempt_does_not_crash_audit(run, raw):
    (run / "public" / "attempts" / f"{RESULT}.json").write_text(raw)
    assert _report(run)["checks"]["evidence"]["code"] == "evidence_invalid"


def test_archived_evidence_still_resolves(run):
    path = run / "public" / "attempts" / f"{BASE}.json"
    record = json.loads(path.read_text())
    record["metadata"]["archived"] = True
    path.write_text(json.dumps(record))
    assert _report(run)["checks"]["evidence"]["status"] == "passed"


def test_missing_baseline_is_not_inferred_from_based_on_or_parent(run):
    _attempt(run, RESULT, 12, parent_hash=BASE)
    _note(run, based_on=[BASE], evidence={"attempt": RESULT, "score_delta": 2, "verified": True})
    report = _report(run)
    assert report["checks"]["evidence"]["code"] == "baseline_unspecified"
    assert report["checks"]["numeric"]["status"] == "unchecked"


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", "other"),
        ("grader", "v2"),
        ("inputs", "other-inputs"),
        ("direction", "minimize"),
        ("budget_class", "tune"),
    ],
)
def test_incomparable_conditions_do_not_hide_numeric_check(run, field, value):
    path = run / "public" / "attempts" / f"{RESULT}.json"
    record = json.loads(path.read_text())
    target = (
        record["metadata"] if field == "budget_class" else record["metadata"]["evaluation_context"]
    )
    target[field] = value
    path.write_text(json.dumps(record))
    report = _report(run)
    assert report["checks"]["comparability"]["code"] == "incomparable"
    assert report["checks"]["numeric"]["status"] == "passed"
    assert "improved" not in report


@pytest.mark.parametrize(
    "metadata", [None, [], {}, {"budget_class": [], "evaluation_context": {"direction": []}}]
)
def test_missing_or_invalid_context_is_unchecked(run, metadata):
    _attempt(run, RESULT, 12, metadata=metadata)
    (run / "config.yaml").write_text("grader:\n  direction: minimize\n")
    report = _report(run)
    assert report["checks"]["comparability"]["code"] == "context_missing"
    assert report["checks"]["numeric"]["status"] == "passed"
    assert report["direction"] == "minimize"
    assert "improved" not in report


def test_legacy_and_empty_run_are_read_only(tmp_path):
    root = tmp_path / ".coral"
    assert list_notes(root, audit=True) == []
    assert not root.exists()
    notes = root / "public" / "notes"
    notes.mkdir(parents=True)
    (notes / "notes.md").write_text("## Old observation\nThis worked once.\n")
    report = _report(root)
    assert all(c["status"] == "unchecked" for c in report["checks"].values())


def test_new_snapshot_after_note_evidence_or_config_change(run):
    first = _report(run)
    path = run / "public" / "notes" / "experiment.md"
    path.write_text(path.read_text() + "New explanation.\n")
    second = _report(run)
    assert first["note_sha256"] != second["note_sha256"]
    _attempt(run, RESULT, 11)
    third = _report(run)
    assert second["evidence_sha256"]["attempt"] != third["evidence_sha256"]["attempt"]
    assert third["checks"]["numeric"]["code"] == "delta_mismatch"
    (run / "config.yaml").write_text("grader:\n  direction: minimize\n")
    assert _report(run)["config_sha256"] != third["config_sha256"]


def test_island_scope_including_global_view(tmp_path):
    root = tmp_path / ".coral"
    (root / "islands").mkdir(parents=True)
    for island, score in [("avalon", 12), ("atlantis", 8)]:
        _attempt(root, BASE, 10, island=island)
        _attempt(root, RESULT, score, island=island)
        _note(root, island=island)
    local = list_notes(root, island_id="avalon", audit=True)
    assert len(local) == 1
    assert local[0]["audit"]["checks"]["numeric"]["status"] == "passed"
    global_notes = {e["island_id"]: e for e in list_notes(root, audit=True)}
    assert global_notes["atlantis"]["audit"]["checks"]["numeric"]["code"] == "delta_mismatch"
    (root / "islands" / "avalon" / "attempts" / f"{RESULT}.json").unlink()
    assert _report(root, "avalon")["checks"]["evidence"]["code"] == "evidence_missing"


def test_cli_audit_filters_and_preserves_status(run, monkeypatch, capsys):
    monkeypatch.setattr(query, "find_coral_dir_and_island", lambda *args: (run, None))
    query.cmd_notes(argparse.Namespace(audit=True, search="causal", recent=1))
    output = capsys.readouterr().out
    assert "System evidence audit" in output
    assert "delta_matched" in output
    assert "replication_unchecked" in output
    assert "experiment.md" in output
    with pytest.raises(SystemExit, match="cannot be combined"):
        query.cmd_notes(argparse.Namespace(audit=True, read="1"))


async def test_dashboard_separates_author_status_from_system_audit(run):
    raw = run / "public" / "notes" / "raw"
    raw.mkdir()
    (raw / "source.md").write_text("# Source\nUnstructured source.\n")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(coral_dir=run)))
    response = await get_notes(request)
    entries = json.loads(response.body)
    note = next(e for e in entries if e["category"] != "raw")
    assert note["status"] == "confirmed"
    assert note["audit"]["checks"]["replication"]["status"] == "unchecked"
    assert "audit" not in next(e for e in entries if e["category"] == "raw")


def test_audit_composes_with_status_filter(run, monkeypatch, capsys):
    _note(run, filename="untested.md", status="untested")
    entries = list_notes(run, status=" CONFIRMED ", audit=True)
    assert [e["filename"] for e in entries] == ["experiment.md"]
    assert entries[0]["audit"]["checks"]["numeric"]["status"] == "passed"
    monkeypatch.setattr(query, "find_coral_dir_and_island", lambda *args: (run, None))
    query.cmd_notes(argparse.Namespace(audit=True, status="confirmed", search="causal", recent=1))
    output = capsys.readouterr().out
    assert "experiment.md" in output
    assert "untested.md" not in output
