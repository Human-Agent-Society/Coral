#!/usr/bin/env python3
"""Score explanatory discovery knowledge with the fixed base model."""
from __future__ import annotations

import http.client
import json
import os
import socket
import threading
import time
import urllib.parse
from typing import Any


_DNS_LOCK = threading.Lock()
_PINNED: dict[str, str] = {}


def _pinned_ip(host: str) -> str:
    """Resolve once and reuse. Bursts of fresh lookups exhaust the sandbox resolver."""
    with _DNS_LOCK:
        ip = _PINNED.get(host)
        if ip is None:
            ip = socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
            _PINNED[host] = ip
        return ip


class _PinnedHTTPS(http.client.HTTPSConnection):
    def connect(self) -> None:
        sock = socket.create_connection((_pinned_ip(self.host), self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _chat(messages, *, timeout: float = 180.0) -> str:
    """One OpenAI-compatible /chat/completions call; returns the assistant text."""
    api_key = os.environ.get("DISCOVERYWORLD_LLM_API_KEY", "")
    base = urllib.parse.urlsplit(
        os.environ.get("DISCOVERYWORLD_LLM_API_BASE", "https://api.openai.com/v1")
    )
    model = os.environ.get("DISCOVERYWORLD_LLM_MODEL", "gpt-4o")
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode()
    path = base.path.rstrip("/") + "/chat/completions"
    last = None
    for attempt in range(3):
        conn = _PinnedHTTPS(base.hostname, base.port or 443, timeout=timeout)
        try:
            conn.request("POST", path, body=body, headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            })
            resp = conn.getresponse()
            payload = json.loads(resp.read())
            if resp.status != 200:
                raise RuntimeError(f"model HTTP {resp.status}: {json.dumps(payload)[:300]}")
            return payload["choices"][0]["message"].get("content") or ""
        except Exception as exc:
            last = exc
            with _DNS_LOCK:
                _PINNED.pop(base.hostname, None)
            time.sleep(1.0 + attempt)
        finally:
            conn.close()
    raise RuntimeError(f"model call failed after 3 attempts: {last}")


def _knowledge_text(thoughts: list[str], limit: int = 60000) -> str:
    text = "\n".join(
        f"Step {index}: {thought.strip()}"
        for index, thought in enumerate(thoughts)
        if thought.strip()
    )
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...[middle truncated]...\n" + text[-limit // 2 :]


def evaluate_knowledge(
    *,
    task_description: str,
    critical_questions: list[str],
    thoughts: list[str],
    timeout: float = 200.0,
) -> tuple[float, int, int, str | None]:
    """Return (normalized score, correct, total, error) without exposing questions."""
    total = len(critical_questions)
    if total == 0:
        return 0.0, 0, 0, "scenario supplied no critical questions"

    prompt = f"""Act as the explanatory-knowledge evaluator from the DiscoveryWorld paper.
Judge only whether the agent's stated knowledge explicitly reflects each critical
discovery. Do not award credit merely because an action succeeded. For every
numbered question, return 1 only when the supplied knowledge clearly supports it;
otherwise return 0.

Task description:
{task_description}

Agent knowledge (concatenated ReAct thoughts):
{_knowledge_text(thoughts) or "[No explicit knowledge was recorded.]"}

Critical questions:
{json.dumps(list(enumerate(critical_questions)), ensure_ascii=False)}

Return exactly one JSON object with this shape:
{{"evaluations":[{{"index":0,"evaluation":0,"explanation":"brief reason"}}]}}
Include every index exactly once. evaluation must be integer 0 or 1."""

    last_error: str | None = None
    for _ in range(2):
        try:
            text = _chat(
                [
                    {"role": "system", "content": "You are a strict grader. Reply with one JSON object."},
                    {"role": "user", "content": prompt},
                ],
                timeout=timeout,
            )
            answer: dict[str, Any] = json.loads(text)
            rows = answer.get("evaluations")
            if not isinstance(rows, list):
                raise ValueError("judge response lacks evaluations list")
            values: dict[int, int] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                index = row.get("index")
                value = row.get("evaluation")
                if (
                    isinstance(index, int)
                    and 0 <= index < total
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                    and value in (0, 1)
                ):
                    values[index] = value
            if len(values) != total:
                raise ValueError(
                    f"judge returned {len(values)} valid evaluations for {total} questions"
                )
            correct = sum(values.values())
            return correct / total, correct, total, None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return 0.0, 0, total, last_error
