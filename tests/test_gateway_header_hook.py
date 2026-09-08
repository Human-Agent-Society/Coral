"""Tests for the optional header_provider hook in the gateway middleware."""

import logging
from pathlib import Path
from typing import Any

import pytest

from coral.gateway.middleware import CoralGatewayMiddleware


class FakeApp:
    """Minimal ASGI app that captures the scope and returns a JSON body."""

    def __init__(self) -> None:
        self.captured_scope: dict | None = None

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        self.captured_scope = scope
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"ok": true}'})


def _make_scope() -> dict:
    return {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [
            (b"authorization", b"Bearer key-1"),
            (b"content-type", b"application/json"),
        ],
    }


async def _run_request(middleware: CoralGatewayMiddleware, scope: dict) -> list[dict]:
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b'{"model": "m"}', "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    await middleware(scope, receive, send)
    return sent


def _headers(app: FakeApp) -> dict[bytes, bytes]:
    assert app.captured_scope is not None
    return dict(app.captured_scope["headers"])


async def test_no_provider_keeps_existing_behavior(tmp_path: Path) -> None:
    app = FakeApp()
    middleware = CoralGatewayMiddleware(app, log_dir=tmp_path, master_key="master")
    middleware.register_agent("agent-1", tmp_path, "key-1")

    sent = await _run_request(middleware, _make_scope())

    headers = _headers(app)
    assert headers[b"authorization"] == b"Bearer master"
    assert headers[b"x-coral-agent-id"] == b"agent-1"
    assert b"x-coral-session-id" in headers
    assert sent[-1]["body"] == b'{"ok": true}'


async def test_provider_headers_are_appended(tmp_path: Path) -> None:
    app = FakeApp()
    seen: list[tuple] = []

    def provider(agent_info, scope):
        seen.append((agent_info, scope))
        return [(b"x-experiment-id", b"exp-42")]

    middleware = CoralGatewayMiddleware(
        app, log_dir=tmp_path, master_key="master", header_provider=provider
    )
    middleware.register_agent("agent-1", tmp_path, "key-1")

    await _run_request(middleware, _make_scope())

    assert _headers(app)[b"x-experiment-id"] == b"exp-42"
    assert len(seen) == 1
    agent_info, scope = seen[0]
    assert agent_info is not None
    assert agent_info.agent_id == "agent-1"
    assert scope["path"] == "/v1/chat/completions"


async def test_provider_gets_none_for_unknown_agent(tmp_path: Path) -> None:
    app = FakeApp()
    seen: list[Any] = []

    def provider(agent_info, scope):
        seen.append(agent_info)
        return []

    middleware = CoralGatewayMiddleware(
        app, log_dir=tmp_path, master_key="master", header_provider=provider
    )
    # Two agents registered and a non-matching key -> no identification
    middleware.register_agent("agent-1", tmp_path, "key-a")
    middleware.register_agent("agent-2", tmp_path, "key-b")

    await _run_request(middleware, _make_scope())

    assert seen == [None]


async def test_raising_provider_does_not_break_request(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    app = FakeApp()

    def provider(agent_info, scope):
        raise RuntimeError("boom")

    middleware = CoralGatewayMiddleware(
        app, log_dir=tmp_path, master_key="master", header_provider=provider
    )
    middleware.register_agent("agent-1", tmp_path, "key-1")

    with caplog.at_level(logging.WARNING, logger="coral.gateway.middleware"):
        sent = await _run_request(middleware, _make_scope())

    # Request still went through with the standard headers
    headers = _headers(app)
    assert headers[b"x-coral-agent-id"] == b"agent-1"
    assert sent[-1]["body"] == b'{"ok": true}'
    assert any("header_provider raised" in r.message for r in caplog.records)


async def test_invalid_provider_result_is_skipped_atomically(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    app = FakeApp()

    def provider(agent_info, scope):
        # First pair valid, second is str, which is not a valid header value
        return [(b"x-good", b"1"), (b"x-bad", "not-bytes")]

    middleware = CoralGatewayMiddleware(
        app, log_dir=tmp_path, master_key="master", header_provider=provider
    )
    middleware.register_agent("agent-1", tmp_path, "key-1")

    with caplog.at_level(logging.WARNING, logger="coral.gateway.middleware"):
        await _run_request(middleware, _make_scope())

    # Neither header applied: partial application would be worse than none
    headers = _headers(app)
    assert b"x-good" not in headers
    assert b"x-bad" not in headers
    assert any("header_provider raised" in r.message for r in caplog.records)
