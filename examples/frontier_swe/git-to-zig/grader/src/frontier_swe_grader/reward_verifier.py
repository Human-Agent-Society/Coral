"""Harbor verifier adapter that selects the scalar from Frontier-SWE details."""

from __future__ import annotations

import json

from harbor.verifier.verifier import Verifier, VerifierOutputParseError


class FrontierSWEVerifier(Verifier):
    """Run Harbor's verifier unchanged, but return only its primary reward."""

    def _parse_reward_json(self) -> dict[str, float | int]:
        try:
            payload = json.loads(self.trial_paths.reward_json_path.read_text())
            value = payload.get("reward", payload.get("score"))
            return {"reward": float(value)}
        except (AttributeError, OSError, TypeError, ValueError) as error:
            raise VerifierOutputParseError(
                f"Frontier-SWE reward JSON has no numeric reward or score: "
                f"{self.trial_paths.reward_json_path}"
            ) from error
