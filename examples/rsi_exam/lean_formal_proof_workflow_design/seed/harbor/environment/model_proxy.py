"""Trusted GPT-5.4 proxy with parent-owned per-problem accounting."""
from __future__ import annotations

import os
import math
import socket
import time

# A-only: the backend has no AAAA record and the dual-stack lookup fails behind the
# egress sidecar, silently degrading every call into the caller's fallback.
_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_v4(host, port, family=0, *args, **kwargs):
    if family in (0, socket.AF_UNSPEC):
        family = socket.AF_INET
    return _getaddrinfo(host, port, family, *args, **kwargs)


if os.environ.get("LEAN_LLM_DUAL_STACK", "") != "1":
    socket.getaddrinfo = _getaddrinfo_v4


PINNED_MODEL = "gpt-5.4"
PINNED_REVISION = "gpt-5.4-2026-03-05"


class BudgetExceeded(RuntimeError):
    pass


class ModelInfrastructureError(RuntimeError):
    """The fixed model service is missing, unreachable, or violates its contract."""


class ModelRequestError(ValueError):
    """The submitted workflow constructed a request rejected by the fixed endpoint."""


class ModelProxy:
    def __init__(self, max_calls, max_tokens):
        self.max_calls = int(max_calls)
        self.max_tokens = int(max_tokens)
        self.num_calls = 0
        self.total_tokens = 0
        self.calls_exceeded = False
        self.tokens_exceeded = False
        self.model = PINNED_MODEL
        self.base_url = (
            os.environ.get("LEAN_GPT54_MODEL_BASE_URL")
            or os.environ.get("AUTOHARNESS_BASE_URL", "")
        ).strip().strip('"')
        self.api_key = (
            os.environ.get("LEAN_GPT54_MODEL_API_KEY")
            or os.environ.get("AUTOHARNESS_API_KEY", "")
        ).strip().strip('"')
        self.revision = PINNED_REVISION
        self.temperature = 0.0
        self.seed = 0
        self._client = None
        missing = [name for name, value in (
            ("LEAN_GPT54_MODEL_NAME", self.model),
            ("LEAN_GPT54_MODEL_BASE_URL", self.base_url),
            ("LEAN_GPT54_MODEL_API_KEY", self.api_key),
            ("LEAN_GPT54_MODEL_REVISION", self.revision),
        ) if not value]
        if missing:
            raise ModelInfrastructureError(
                "fixed model configuration is incomplete: " + ", ".join(missing)
            )
        if self.model != PINNED_MODEL or self.revision != PINNED_REVISION:
            raise ModelInfrastructureError("fixed model identity or revision does not match the task")
        if not math.isfinite(self.temperature) or self.temperature != 0.0 or self.seed != 0:
            raise ModelInfrastructureError("fixed model temperature or seed does not match the task")

    # Transport failures (DNS, connect, reset) are retried with a backoff long
    # enough to outlast a stalled resolver; config failures are not.
    _RETRY_BACKOFF_SEC = (5, 20, 40)
    # Short: a call runs inside the per-question wall alarm, so waiting is paid for.
    _CALL_BACKOFF_SEC = (1, 3, 6)

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=170.0)
            except Exception as exc:
                raise ModelInfrastructureError(
                    f"fixed GPT-5.4 client import/construction failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        return self._client

    @staticmethod
    def _is_transport_failure(exc):
        if getattr(exc, "status_code", None) in {429, 500, 502, 503, 504}:
            return True
        name = type(exc).__name__
        return name in {"APIConnectionError", "APITimeoutError", "ConnectionError",
                        "URLError", "gaierror", "OSError", "ConnectError",
                        "ReadTimeout", "ConnectTimeout", "RemoteProtocolError"}

    def healthcheck(self):
        """Verify listing plus one tiny completion/usage response before sealed-data access.

        Retried: this is the first outbound connection of the run, so a stalled
        resolver here would otherwise abort a run whose endpoint is perfectly fine.
        """
        last = None
        for attempt in range(len(self._RETRY_BACKOFF_SEC) + 1):
            try:
                return self._healthcheck_once()
            except ModelInfrastructureError as exc:
                last = exc
                if not self._is_transport_failure(exc.__cause__ or exc):
                    raise
                if attempt < len(self._RETRY_BACKOFF_SEC):
                    time.sleep(self._RETRY_BACKOFF_SEC[attempt])
        raise last

    def _healthcheck_once(self):
        try:
            listing = self._get_client().models.list()
            model_ids = {str(item.id) for item in listing.data}
            if self.model not in model_ids and self.revision not in model_ids:
                raise ModelInfrastructureError(
                    "fixed GPT-5.4 endpoint exposes neither the alias nor pinned snapshot"
                )
            self([{"role": "user", "content": "Return only: by trivial"}], max_tokens=8)
        except ModelInfrastructureError:
            raise
        except Exception as exc:
            raise ModelInfrastructureError(
                f"fixed model endpoint health check failed ({type(exc).__name__})"
            ) from exc
        return {"model": self.model, "revision": self.revision,
                "endpoint_verified": True, "completion_verified": True}

    def __call__(self, messages, max_tokens=1024, stop=None):
        self.num_calls += 1
        if self.num_calls > self.max_calls:
            self.calls_exceeded = True
            raise BudgetExceeded(f"exceeded max model calls ({self.max_calls})")
        max_tokens = int(max_tokens)
        if max_tokens < 1 or max_tokens > self.max_tokens:
            raise ValueError("invalid max_tokens")
        for attempt in range(len(self._CALL_BACKOFF_SEC) + 1):
            try:
                text, used = self._request(messages, max_tokens, stop)
                break
            except (ModelInfrastructureError, ModelRequestError, TimeoutError, ValueError):
                raise
            except Exception as exc:
                if getattr(exc, "status_code", None) in {400, 413, 422}:
                    raise ModelRequestError("fixed model rejected the submitted request") from exc
                if getattr(exc, "status_code", None) in {401, 403, 404}:
                    raise ModelInfrastructureError(
                        f"fixed model endpoint rejected the credentials or model id: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                if self._is_transport_failure(exc) and attempt < len(self._CALL_BACKOFF_SEC):
                    time.sleep(self._CALL_BACKOFF_SEC[attempt])
                    continue
                raise ModelInfrastructureError(
                    f"fixed model request failed: {type(exc).__name__}: {exc}"
                ) from exc
        self.total_tokens += used
        if self.total_tokens > self.max_tokens:
            self.tokens_exceeded = True
            raise BudgetExceeded(f"exceeded total model tokens ({self.max_tokens})")
        return self._apply_stop(text, stop)

    def _request(self, messages, max_tokens, stop):
        """One attempt. Transport failures propagate raw so __call__ can classify them."""
        result = self._get_client().chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=self.temperature,
            seed=self.seed,
            max_completion_tokens=max_tokens,
            stop=list(stop) if stop else None,
        )
        response_model = str(getattr(result, "model", ""))
        if response_model != self.revision:
            raise ModelInfrastructureError(
                "fixed GPT-5.4 response revision does not match the task"
            )
        text = (result.choices[0].message.content or "").strip()
        usage = getattr(result, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int) \
                or completion_tokens < 1:
            raise ModelInfrastructureError(
                "fixed GPT-5.4 response omitted trustworthy generated-token usage"
            )
        if not text:
            raise ModelInfrastructureError("fixed model returned an empty response")
        return text, completion_tokens

    @staticmethod
    def _apply_stop(text, stop):
        for marker in stop or ():
            if marker and marker in text:
                text = text.split(marker, 1)[0]
        return text
