"""Lift-guided meta-evolve attribution and recommendation helpers."""

from __future__ import annotations

import math
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from coral.config import MetaEvolveConfig
from coral.types import BUDGET_CLASS_REAL, Attempt


@dataclass(frozen=True)
class MetaEvolveAttribution:
    """Operator and mutation labels attached to one attempt."""

    operator: str
    mutation: str


@dataclass(frozen=True)
class LiftSummary:
    """Count and mean directional lift for one evidence group."""

    count: int
    mean_lift: float


@dataclass(frozen=True)
class MetaEvolveStats:
    """Lift statistics reconstructed from durable attempt records."""

    arms: dict[MetaEvolveAttribution, LiftSummary]
    operators: dict[str, LiftSummary]
    skipped_attempts: int


@dataclass(frozen=True)
class MetaEvolveRecommendation:
    """One deterministic arm recommendation and its supporting summaries."""

    attribution: MetaEvolveAttribution
    selection_mode: Literal["initial-exploration", "upper-confidence"]
    arm_summary: LiftSummary
    operator_summary: LiftSummary


def validate_attribution(
    config: MetaEvolveConfig,
    *,
    operator: str | None,
    mutation: str | None,
    tune: bool,
) -> MetaEvolveAttribution | None:
    """Validate CLI attribution before an eval creates a Git commit."""
    if (operator is None) != (mutation is None):
        raise ValueError("--operator and --mutation must be provided together")
    if operator is None or mutation is None:
        if config.enabled and not tune:
            raise ValueError(
                "enabled agents.meta_evolve requires --operator and --mutation for every real eval"
            )
        return None

    normalized_operator = operator.strip()
    normalized_mutation = mutation.strip()
    if not normalized_operator or not normalized_mutation:
        raise ValueError("--operator and --mutation must be non-empty")

    attribution = MetaEvolveAttribution(
        operator=normalized_operator,
        mutation=normalized_mutation,
    )
    if config.enabled:
        configured = {(arm.operator, arm.mutation) for arm in config.arms}
        if (attribution.operator, attribution.mutation) not in configured:
            raise ValueError(
                "meta-evolve attribution must match a configured arm: "
                f"{attribution.operator}/{attribution.mutation}"
            )
    return attribution


def attempt_attribution(attempt: Attempt) -> MetaEvolveAttribution | None:
    """Read valid namespaced attribution from an attempt, tolerating legacy data."""
    if not isinstance(attempt.metadata, dict):
        return None
    raw = attempt.metadata.get("meta_evolve")
    if not isinstance(raw, dict):
        return None
    operator = raw.get("operator")
    mutation = raw.get("mutation")
    if not isinstance(operator, str) or not isinstance(mutation, str):
        return None
    operator = operator.strip()
    mutation = mutation.strip()
    if not operator or not mutation:
        return None
    return MetaEvolveAttribution(operator=operator, mutation=mutation)


def attribution_metadata(
    attribution: MetaEvolveAttribution,
) -> dict[str, dict[str, str]]:
    """Serialize attribution under its Attempt.metadata namespace."""
    return {
        "meta_evolve": {
            "operator": attribution.operator,
            "mutation": attribution.mutation,
        }
    }


def configured_attributions(
    config: MetaEvolveConfig,
) -> list[MetaEvolveAttribution]:
    """Return configured arms in their deterministic selection order."""
    return [
        MetaEvolveAttribution(operator=arm.operator, mutation=arm.mutation) for arm in config.arms
    ]


def _summary(total: float, count: int) -> LiftSummary:
    return LiftSummary(count=count, mean_lift=total / count if count else 0.0)


def build_stats(
    attempts: Sequence[Attempt],
    *,
    arms: Sequence[MetaEvolveAttribution],
    minimize: bool,
) -> MetaEvolveStats:
    """Aggregate eligible parent-relative lift observations."""
    arm_totals = {arm: 0.0 for arm in arms}
    arm_counts = {arm: 0 for arm in arms}
    operator_totals = {arm.operator: 0.0 for arm in arms}
    operator_counts = {arm.operator: 0 for arm in arms}
    configured = set(arms)
    by_hash = {attempt.commit_hash: attempt for attempt in attempts}
    skipped_attempts = 0

    for attempt in attempts:
        attribution = attempt_attribution(attempt)
        parent = by_hash.get(attempt.parent_hash) if attempt.parent_hash else None
        if (
            attribution is None
            or attempt.archived
            or attempt.score is None
            or attempt.budget_class != BUDGET_CLASS_REAL
            or attribution not in configured
        ):
            skipped_attempts += 1
            continue
        if (
            parent is None
            or not isinstance(parent.metadata, dict)
            or parent.archived
            or parent.score is None
            or parent.budget_class != BUDGET_CLASS_REAL
            or parent.agent_id != attempt.agent_id
        ):
            skipped_attempts += 1
            continue
        lift = (
            float(parent.score) - float(attempt.score)
            if minimize
            else float(attempt.score) - float(parent.score)
        )
        arm_totals[attribution] += lift
        arm_counts[attribution] += 1
        operator_totals[attribution.operator] += lift
        operator_counts[attribution.operator] += 1

    return MetaEvolveStats(
        arms={arm: _summary(arm_totals[arm], arm_counts[arm]) for arm in arms},
        operators={
            operator: _summary(total, operator_counts[operator])
            for operator, total in operator_totals.items()
        },
        skipped_attempts=skipped_attempts,
    )


def _recommendation(
    attribution: MetaEvolveAttribution,
    *,
    stats: MetaEvolveStats,
    selection_mode: Literal["initial-exploration", "upper-confidence"],
) -> MetaEvolveRecommendation:
    return MetaEvolveRecommendation(
        attribution=attribution,
        selection_mode=selection_mode,
        arm_summary=stats.arms[attribution],
        operator_summary=stats.operators.get(
            attribution.operator,
            LiftSummary(count=0, mean_lift=0.0),
        ),
    )


def recommend_arm(
    *,
    stats: MetaEvolveStats,
    arms: Sequence[MetaEvolveAttribution],
    exploration_weight: float,
) -> MetaEvolveRecommendation:
    """Select an arm by ordered initial exploration, then deterministic UCB."""
    if not arms:
        raise ValueError("meta-evolve recommendation requires at least one arm")

    for arm in arms:
        if stats.arms[arm].count == 0:
            return _recommendation(
                arm,
                stats=stats,
                selection_mode="initial-exploration",
            )

    total_observations = sum(stats.arms[arm].count for arm in arms)
    scores = [
        stats.arms[arm].mean_lift
        + exploration_weight * math.sqrt(math.log(total_observations) / stats.arms[arm].count)
        for arm in arms
    ]
    index = max(range(len(arms)), key=lambda candidate: (scores[candidate], -candidate))
    return _recommendation(
        arms[index],
        stats=stats,
        selection_mode="upper-confidence",
    )


def render_bootstrap_guidance(config: MetaEvolveConfig) -> str:
    """Render the opt-in attribution contract for generated agent instructions."""
    if not config.enabled:
        return ""

    arms = "\n".join(f"- `{arm.operator}` / `{arm.mutation}`" for arm in config.arms)
    example = config.arms[0]
    example_command = (
        'coral eval -m "what you changed and why" '
        f"--operator {shlex.quote(example.operator)} "
        f"--mutation {shlex.quote(example.mutation)}"
    )
    return (
        "## Meta-evolve attribution\n\n"
        "This run tracks directional lift by configured operator/mutation arm. "
        "Every real eval must use one of these pairs:\n\n"
        f"{arms}\n\n"
        "Attribute a real eval with both flags:\n\n"
        "```bash\n"
        f"{example_command}\n"
        "```\n\n"
        "Until a pivot supplies a recommendation, use the first unobserved arm "
        "in configuration order. "
        "Tune evals may omit attribution and never update lift statistics. "
        "A pivot heartbeat may recommend the next arm; the recommendation is advisory.\n\n"
    )


def render_recommendation(
    recommendation: MetaEvolveRecommendation,
    *,
    stats: MetaEvolveStats,
) -> str:
    """Render one auditable recommendation for a pivot heartbeat."""
    attribution = recommendation.attribution
    arm = recommendation.arm_summary
    operator = recommendation.operator_summary
    command = (
        'coral eval -m "what you changed and why" '
        f"--operator {shlex.quote(attribution.operator)} "
        f"--mutation {shlex.quote(attribution.mutation)}"
    )
    return (
        "## Meta-evolve recommendation\n\n"
        f"Recommended operator: {attribution.operator}\n"
        f"Recommended mutation: {attribution.mutation}\n"
        f"Selection: {recommendation.selection_mode}\n"
        f"Arm evidence: {arm.count} observation(s), mean directional lift {arm.mean_lift:+.4f}\n"
        f"Operator evidence: {operator.count} observation(s), "
        f"mean directional lift {operator.mean_lift:+.4f}\n"
        f"Skipped or unattributed attempts: {stats.skipped_attempts}\n\n"
        "This is an advisory recommendation. Attribute the next real eval with:\n\n"
        f"```bash\n{command}\n```"
    )
