#!/usr/bin/env python3
"""Starter ReAct harness for the line-oriented DiscoveryWorld protocol.

stdin receives one JSON object per line:
  {"type": "init", ...}
  {"type": "observation", ...}

stdout must contain exactly one JSON action object for each observation.
Diagnostic output belongs on stderr.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.parse
from typing import Any

SYSTEM_PROMPT = """You are playing a video game about making scientific discoveries. The game is
in the style of a 2D top-down RPG. As input you get information from the user interface (provided
as JSON) that describes your location, inventory, objects in front of you, the result of your last
action, and the task you are assigned to complete. The actions you can take are limited to the set
the game defines; they are described below. The game is played step-by-step: at each step you get
the input and output a single action to take next.

Navigation. Moving forward moves you in the direction you are facing. Teleporting is much faster
and less error-prone than moving square by square: you can teleport to a named location with
{"action": "TELEPORT_TO_LOCATION", "arg1": "school"}, or to any object by its UUID with
TELEPORT_TO_OBJECT -- even objects that are not in the accessible list. Read the whole object list
in the observation, not just the accessible ones.

Interaction. You can only act on objects that are in your inventory or directly one square in
front of you, in the direction you are facing. For two-argument actions such as USE and PUT, one
object must be in your inventory and you must be next to the other.

Response format. Respond with JSON only, one action object. Include a "thought" key describing your
reasoning and tracking what you have learned about the world, e.g.
{"action": "USE", "arg1": 5, "arg2": 12, "thought": "I have the seed in hand and need to plant it."}
If a dialog is active, respond with {"chosen_dialog_option_int": N}.

Hints.
- If recent actions failed, think about why before repeating them.
- If you do not see what you are looking for, consider teleporting to another location.
- Use verbose "thought" statements to maintain progress, running hypotheses, and what you know.
- If you find yourself waiting for something to happen, check whether it has already happened.
- If you are waiting a long time you may be stuck: reexamine the task, reassess where you are, and
  make a new plan.
- When you believe you are done, submit with {"action": "SUBMIT", "arg1": "Task completed!"}. If you
  cannot make further progress, submit with arg1 explaining why."""


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
            "max_tokens": 400,
            "stop": ["```\n"],
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


MAX_OUTPUT_CHARS = 10000   # whole user message, as in the reference config


def parse_action(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        try:
            value = json.loads(fenced.group(1))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            value = json.loads(text[start : end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {"action": "WAIT", "thought": "Model output was not valid JSON."}


class ReActHarness:
    def __init__(self, init: dict[str, Any]) -> None:
        self.task = init["task"]
        self.action_help = init["action_help"]
        self.known_actions = init["known_actions"]
        self.history: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def _model_turn(self, prompt: str) -> dict[str, Any]:
        text = _chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        return parse_action(text)

    @staticmethod
    def _render(history, budget: int) -> str:
        """Newest first until the budget runs out, then re-ordered oldest to newest."""
        out: list[str] = []
        used = 0
        for action, observation in reversed(history):
            for label, value in (("Observation", observation), ("Action", action)):
                block = f"{label}:\n```json\n{json.dumps(value, ensure_ascii=False)}\n```\n\n"
                used += len(block)
                if used > budget:
                    return "".join(reversed(out))
                out.append(block)
        return "".join(reversed(out))

    def act(self, message: dict[str, Any]) -> dict[str, Any]:
        observation = message["observation"]
        head = (
            f"Task:\n{self.task}\n\n"
            f"Action help:\n{self.action_help}\n\n"
            f"Known actions:\n{json.dumps(self.known_actions, ensure_ascii=False)}\n\n"
            f"Teleport locations:\n"
            f"{json.dumps(message.get('teleport_locations', {}), ensure_ascii=False)}\n\n"
            f"Current observation:\n{json.dumps(observation, ensure_ascii=False)}\n\n"
            "History of action-observations:\n"
        )
        rendered = self._render(self.history, MAX_OUTPUT_CHARS - len(head))
        trimmed = len(rendered) < len(self._render(self.history, 10 ** 9))
        prompt = head + ("[TRIMMED HISTORY]\n\n" if trimmed else "") + rendered
        action = self._model_turn(prompt)
        self.history.append((action, observation))
        return action


def main() -> None:
    harness: ReActHarness | None = None
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if message.get("type") == "init":
                harness = ReActHarness(message)
                continue
            if message.get("type") != "observation" or harness is None:
                continue
            action = harness.act(message)
        except Exception as exc:
            print(f"harness error: {type(exc).__name__}: {exc}", file=sys.stderr)
            action = {"action": "WAIT", "thought": "Harness recovered from an error."}
        print(json.dumps(action, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
