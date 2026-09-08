"""Tests for agent identification from auth headers in the gateway middleware."""

import logging
from pathlib import Path

import pytest

from coral.gateway.middleware import CoralGatewayMiddleware


@pytest.fixture
def middleware(tmp_path: Path) -> CoralGatewayMiddleware:
    async def dummy_app(scope, receive, send):  # pragma: no cover - never called
        raise AssertionError("app should not be called in these tests")

    return CoralGatewayMiddleware(dummy_app, log_dir=tmp_path, master_key="master")


def test_matching_key_returns_agent(middleware: CoralGatewayMiddleware, tmp_path: Path) -> None:
    middleware.register_agent("agent-1", tmp_path, "key-1")
    middleware.register_agent("agent-2", tmp_path, "key-2")

    info = middleware._get_agent_info("Bearer key-2")

    assert info is not None
    assert info.agent_id == "agent-2"


def test_missing_header_falls_back_to_sole_agent_without_warning(
    middleware: CoralGatewayMiddleware,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    middleware.register_agent("agent-1", tmp_path, "key-1")

    with caplog.at_level(logging.WARNING, logger="coral.gateway.middleware"):
        info = middleware._get_agent_info("")

    assert info is not None
    assert info.agent_id == "agent-1"
    assert not caplog.records


def test_mismatched_key_falls_back_to_sole_agent_with_warning(
    middleware: CoralGatewayMiddleware,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    middleware.register_agent("agent-1", tmp_path, "key-1")

    with caplog.at_level(logging.WARNING, logger="coral.gateway.middleware"):
        info = middleware._get_agent_info("Bearer stale-key")

    assert info is not None
    assert info.agent_id == "agent-1"
    assert len(caplog.records) == 1
    assert "does not match any registered proxy key" in caplog.records[0].message
    # The unrecognized key itself must not leak into the log
    assert "stale-key" not in caplog.records[0].message


def test_mismatched_key_warns_only_once_per_key(
    middleware: CoralGatewayMiddleware,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    middleware.register_agent("agent-1", tmp_path, "key-1")

    with caplog.at_level(logging.WARNING, logger="coral.gateway.middleware"):
        middleware._get_agent_info("Bearer stale-key")
        middleware._get_agent_info("Bearer stale-key")
        middleware._get_agent_info("Bearer other-stale-key")

    assert len(caplog.records) == 2


def test_mismatched_key_with_multiple_agents_returns_none(
    middleware: CoralGatewayMiddleware, tmp_path: Path
) -> None:
    middleware.register_agent("agent-1", tmp_path, "key-1")
    middleware.register_agent("agent-2", tmp_path, "key-2")

    assert middleware._get_agent_info("Bearer stale-key") is None
    assert middleware._get_agent_info("") is None
