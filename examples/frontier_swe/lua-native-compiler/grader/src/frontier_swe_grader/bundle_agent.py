"""Harbor agent that uploads a deterministic bundle instead of invoking an LLM."""

from __future__ import annotations

import tempfile
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .bundle import materialize_bundle


class BundleAgent(BaseAgent):
    """Materialize the CORAL candidate into the official task workspace."""

    def __init__(
        self,
        *args: object,
        candidate_path: str,
        target_dir: str,
        prepare_command: str = "",
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._candidate_path = Path(candidate_path)
        self._target_dir = target_dir
        self._prepare_command = prepare_command

    @staticmethod
    def name() -> str:
        return "frontier-swe-bundle"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del instruction, context
        candidate = self._candidate_path.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="frontier-swe-candidate-") as temporary:
            source_dir = Path(temporary)
            materialize_bundle(candidate, source_dir)
            await environment.upload_dir(source_dir=source_dir, target_dir=self._target_dir)
        if self._prepare_command:
            result = await environment.exec(command=self._prepare_command, timeout_sec=600)
            if result.return_code != 0:
                raise RuntimeError(
                    f"candidate preparation failed with exit {result.return_code}: "
                    f"{result.stderr[-2000:]}"
                )
