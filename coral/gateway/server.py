"""Gateway server embedding LiteLLM with CORAL middleware."""

from __future__ import annotations

import asyncio
import logging
import secrets
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn
from litellm.proxy.proxy_server import app as litellm_app
from litellm.proxy.proxy_server import initialize

from coral.gateway.middleware import CoralGatewayMiddleware, HeaderProvider

logger = logging.getLogger(__name__)

HEALTH_CHECK_INTERVAL = 1.0  # seconds between polls
HEALTH_CHECK_TIMEOUT = 60.0  # total wait before giving up


class GatewayManager:
    """Manage an embedded LiteLLM proxy with CORAL interception middleware.

    Initializes LiteLLM's proxy app, wraps it with CoralGatewayMiddleware,
    and runs it via uvicorn in a background thread.
    """

    def __init__(
        self,
        port: int,
        config_path: str,
        api_key: str = "",
        log_dir: Path | None = None,
        header_provider: HeaderProvider | None = None,
    ) -> None:
        self.port = port
        self.config_path = config_path
        self.api_key = api_key or f"sk-coral-{secrets.token_hex(8)}"
        self.log_dir = log_dir or Path(".coral/gateway")
        self.header_provider = header_provider
        self._server_thread: threading.Thread | None = None
        self._server: object | None = None  # uvicorn.Server
        self._middleware: object | None = None  # CoralGatewayMiddleware

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    def register_agent(self, agent_id: str, worktree_path: Path) -> str:
        """Register an agent and return its unique proxy API key.

        The proxy key is used by the middleware to identify which agent
        made each request, enabling per-agent commit hash tracking.
        """
        from coral.gateway.middleware import CoralGatewayMiddleware

        proxy_key = f"sk-coral-{agent_id}-{secrets.token_hex(4)}"
        if not isinstance(self._middleware, CoralGatewayMiddleware):
            raise RuntimeError("gateway middleware is not initialized; call start() first")
        self._middleware.register_agent(agent_id, worktree_path, proxy_key)
        logger.info(f"Registered {agent_id} with gateway (key: {proxy_key[:20]}...)")
        return proxy_key

    def start(self) -> None:
        """Initialize LiteLLM, add middleware, and start uvicorn in a thread."""
        # Fail fast if port is already in use (e.g. stale gateway from previous run)
        self._check_port_available()

        logger.info(f"Starting gateway on port {self.port}")
        logger.info(f"LiteLLM config: {self.config_path}")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                initialize(
                    config=self.config_path,
                )
            )
        finally:
            loop.close()

        # Wrap with CORAL middleware
        middleware = CoralGatewayMiddleware(
            app=litellm_app,
            log_dir=self.log_dir,
            master_key=self.api_key,
            header_provider=self.header_provider,
        )
        self._middleware = middleware

        # Configure uvicorn
        # Disable ASGI lifespan so LiteLLM's proxy_startup_event doesn't run.
        # That handler tries to check licenses, connect to databases, etc. which
        # can hang in Docker.  We already called initialize() above, so the
        # model list / router is ready.
        config = uvicorn.Config(
            app=middleware,
            host="0.0.0.0",
            port=self.port,
            log_level="info",
            lifespan="off",
        )
        server = uvicorn.Server(config)
        self._server = server

        # Run in background thread
        self._server_thread = threading.Thread(
            target=server.run,
            daemon=True,
            name="coral-gateway",
        )
        self._server_thread.start()

        # Wait for healthy
        self._wait_healthy()
        logger.info(f"Gateway ready at {self.url}")

    def stop(self) -> None:
        """Shut down the gateway server."""
        if self._server is None:
            return
        logger.info("Stopping gateway...")
        try:
            self._server.should_exit = True  # type: ignore[union-attr]
        except Exception as e:
            logger.warning(f"Error stopping gateway: {e}")
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=10)
        self._server = None
        self._server_thread = None
        logger.info("Gateway stopped.")

    def _check_port_available(self) -> None:
        """Raise early if the gateway port is already in use."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", self.port))
        except OSError:
            raise RuntimeError(
                f"Port {self.port} is already in use. "
                f"A stale gateway may still be running — try `coral stop` or "
                f"`lsof -i :{self.port}` to find the process."
            )
        finally:
            sock.close()

    def _wait_healthy(self) -> None:
        """Poll /health/readiness until 200 or timeout."""
        health_url = f"{self.url}/health/readiness"
        deadline = time.monotonic() + HEALTH_CHECK_TIMEOUT
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            time.sleep(HEALTH_CHECK_INTERVAL)
        raise RuntimeError(
            f"Gateway did not become healthy within {HEALTH_CHECK_TIMEOUT}s on port {self.port}."
        )
