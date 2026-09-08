"""Tests for SSE response assembly in the gateway middleware."""

import json

from coral.gateway.middleware import _assemble_response


def _sse(events: list[dict], with_event_lines: bool = False) -> bytes:
    """Render events as an SSE stream body."""
    lines = []
    for event in events:
        if with_event_lines:
            lines.append(f"event: {event.get('type', 'message')}")
        lines.append(f"data: {json.dumps(event)}")
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def test_non_sse_json_body_is_parsed() -> None:
    body = json.dumps({"id": "resp_1", "object": "chat.completion"}).encode()
    assert _assemble_response(body) == {"id": "resp_1", "object": "chat.completion"}


def test_empty_body_returns_none() -> None:
    assert _assemble_response(b"") is None


def test_chat_completions_stream_assembly() -> None:
    events = [
        {
            "id": "chatcmpl-1",
            "model": "gpt-x",
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1",
            "choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}],
        },
        {"id": "chatcmpl-1", "choices": [], "usage": {"total_tokens": 7}},
    ]
    raw = _sse(events) + b"data: [DONE]\n"

    assembled = _assemble_response(raw)

    assert assembled == {
        "id": "chatcmpl-1",
        "model": "gpt-x",
        "content": "Hello world",
        "finish_reason": "stop",
        "usage": {"total_tokens": 7},
    }


def test_responses_api_stream_assembly() -> None:
    events = [
        {"type": "response.output_text.delta", "delta": "Hi"},
        {"type": "response.output_text.delta", "delta": " there"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_9",
                "model": "gpt-x",
                "status": "completed",
                "usage": {"total_tokens": 11},
            },
        },
    ]

    assembled = _assemble_response(_sse(events))

    assert assembled == {
        "id": "resp_9",
        "model": "gpt-x",
        "content": "Hi there",
        "status": "completed",
        "usage": {"total_tokens": 11},
    }


def test_anthropic_messages_stream_assembly() -> None:
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "model": "claude-x",
                "usage": {"input_tokens": 25, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " world"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 6},
        },
        {"type": "message_stop"},
    ]

    # The Anthropic SDK emits "event: <type>" lines before each data line,
    # so the stream starts with "event: message_start", not "data:".
    assembled = _assemble_response(_sse(events, with_event_lines=True))

    assert assembled == {
        "id": "msg_1",
        "model": "claude-x",
        "content": "Hello world",
        "finish_reason": "end_turn",
        "usage": {"input_tokens": 25, "output_tokens": 6},
    }


def test_anthropic_non_text_deltas_are_ignored() -> None:
    events = [
        {
            "type": "message_start",
            "message": {"id": "msg_2", "model": "claude-x", "usage": {"input_tokens": 3}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"city": "SF"}'},
        },
        {"type": "ping"},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}},
    ]

    assembled = _assemble_response(_sse(events, with_event_lines=True))

    assert "content" not in assembled
    assert assembled["finish_reason"] == "tool_use"


def test_invalid_json_frames_are_skipped() -> None:
    raw = (
        b"data: not-json\n\n"
        b'data: {"id": "chatcmpl-2", "choices": '
        b'[{"delta": {"content": "ok"}, "finish_reason": "stop"}]}\n\n'
        b"data: [DONE]\n"
    )

    assembled = _assemble_response(raw)

    assert assembled["content"] == "ok"
    assert assembled["finish_reason"] == "stop"
