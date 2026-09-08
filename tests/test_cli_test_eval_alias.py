"""Tests for the hidden ``coral test-eval`` -> ``coral validate`` alias.

The alias parser must accept the same arguments as ``validate``; otherwise it
cannot dispatch (``cmd_validate`` reads ``args.path``).
"""

import sys

import pytest

import coral.cli as cli_module
import coral.cli.author as author_module


def _capture_cmd_validate(monkeypatch, captured: dict) -> None:
    def fake_cmd_validate(args):
        captured["path"] = args.path
        captured["json"] = getattr(args, "json", False)

    monkeypatch.setattr(author_module, "cmd_validate", fake_cmd_validate)


def test_test_eval_alias_forwards_path(monkeypatch):
    captured: dict = {}
    _capture_cmd_validate(monkeypatch, captured)
    monkeypatch.setattr(sys, "argv", ["coral", "test-eval", "my-task"])

    cli_module.main()

    assert captured == {"path": "my-task", "json": False}


def test_test_eval_alias_accepts_json_flag(monkeypatch):
    captured: dict = {}
    _capture_cmd_validate(monkeypatch, captured)
    monkeypatch.setattr(sys, "argv", ["coral", "test-eval", "my-task", "--json"])

    cli_module.main()

    assert captured == {"path": "my-task", "json": True}


def test_test_eval_alias_without_path_exits_with_usage_error(monkeypatch, capsys):
    """A missing positional must be an argparse usage error, not a traceback."""
    captured: dict = {}
    _capture_cmd_validate(monkeypatch, captured)
    monkeypatch.setattr(sys, "argv", ["coral", "test-eval"])

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()

    assert excinfo.value.code == 2
    assert captured == {}
