"""Tests for lift-guided meta-evolve attribution and selection."""

from __future__ import annotations

import subprocess
import sys

import pytest

from coral.agent.meta_evolve import (
    LiftSummary,
    MetaEvolveAttribution,
    MetaEvolveRecommendation,
    MetaEvolveStats,
    attempt_attribution,
    attribution_metadata,
    build_stats,
    configured_attributions,
    recommend_arm,
    validate_attribution,
)
from coral.config import MetaEvolveArmConfig, MetaEvolveConfig
from coral.types import Attempt


def _enabled_config() -> MetaEvolveConfig:
    return MetaEvolveConfig(
        enabled=True,
        arms=[
            MetaEvolveArmConfig(operator="prompt", mutation="rewrite"),
            MetaEvolveArmConfig(operator="implementation", mutation="replace"),
        ],
    )


def _attempt(
    metadata: dict | None = None,
    *,
    commit_hash: str = "a" * 40,
    agent_id: str = "agent-1",
    score: float | None = 2.0,
    parent_hash: str | None = "b" * 40,
) -> Attempt:
    return Attempt(
        commit_hash=commit_hash,
        agent_id=agent_id,
        title="test attempt",
        score=score,
        status="improved",
        parent_hash=parent_hash,
        timestamp="2026-08-29T00:00:00+00:00",
        metadata=dict(metadata or {}),
    )


def test_validate_attribution_requires_paired_values():
    with pytest.raises(ValueError, match="together"):
        validate_attribution(
            MetaEvolveConfig(),
            operator="prompt",
            mutation=None,
            tune=False,
        )


def test_validate_attribution_requires_configured_arm_when_enabled():
    with pytest.raises(ValueError, match="configured arm"):
        validate_attribution(
            _enabled_config(),
            operator="prompt",
            mutation="unknown",
            tune=False,
        )


def test_validate_attribution_requires_real_attempt_tag_when_enabled():
    with pytest.raises(ValueError, match="requires --operator and --mutation"):
        validate_attribution(
            _enabled_config(),
            operator=None,
            mutation=None,
            tune=False,
        )


def test_validate_attribution_allows_untagged_tune_attempt():
    assert (
        validate_attribution(
            _enabled_config(),
            operator=None,
            mutation=None,
            tune=True,
        )
        is None
    )


def test_validate_attribution_normalizes_values():
    assert validate_attribution(
        _enabled_config(),
        operator=" prompt ",
        mutation=" rewrite ",
        tune=False,
    ) == MetaEvolveAttribution(operator="prompt", mutation="rewrite")


def test_validate_attribution_accepts_explicit_disabled_run_labels():
    assert validate_attribution(
        MetaEvolveConfig(),
        operator="research",
        mutation="literature-search",
        tune=False,
    ) == MetaEvolveAttribution(operator="research", mutation="literature-search")


def test_attempt_attribution_reads_nested_metadata():
    attempt = _attempt({"meta_evolve": {"operator": "prompt", "mutation": "rewrite"}})

    assert attempt_attribution(attempt) == MetaEvolveAttribution(
        operator="prompt",
        mutation="rewrite",
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"meta_evolve": "prompt/rewrite"},
        {"meta_evolve": {"operator": "prompt"}},
        {"meta_evolve": {"operator": " ", "mutation": "rewrite"}},
    ],
)
def test_attempt_attribution_ignores_missing_or_malformed_metadata(metadata):
    assert attempt_attribution(_attempt(metadata)) is None


def test_attempt_attribution_ignores_non_mapping_attempt_metadata():
    attempt = _attempt()
    attempt.metadata = None  # type: ignore[assignment]

    assert attempt_attribution(attempt) is None


def test_attribution_metadata_uses_nested_namespaced_shape():
    attribution = MetaEvolveAttribution(operator="prompt", mutation="rewrite")

    assert attribution_metadata(attribution) == {
        "meta_evolve": {"operator": "prompt", "mutation": "rewrite"}
    }


def test_eval_help_documents_meta_evolve_attribution():
    result = subprocess.run(
        [sys.executable, "-m", "coral.cli", "eval", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "--operator" in result.stdout
    assert "--mutation" in result.stdout


def _arms() -> list[MetaEvolveAttribution]:
    return [
        MetaEvolveAttribution(operator="prompt", mutation="rewrite"),
        MetaEvolveAttribution(operator="prompt", mutation="add-examples"),
        MetaEvolveAttribution(operator="implementation", mutation="replace"),
    ]


def _attribution(operator: str, mutation: str) -> dict:
    return {"meta_evolve": {"operator": operator, "mutation": mutation}}


def test_configured_attributions_preserves_configuration_order():
    assert configured_attributions(_enabled_config()) == [
        MetaEvolveAttribution(operator="prompt", mutation="rewrite"),
        MetaEvolveAttribution(operator="implementation", mutation="replace"),
    ]


def test_build_stats_aggregates_maximize_lift_by_arm_and_operator():
    parent = _attempt(commit_hash="0" * 40, score=5.0, parent_hash=None)
    rewrite = _attempt(
        _attribution("prompt", "rewrite"),
        commit_hash="1" * 40,
        score=7.5,
        parent_hash=parent.commit_hash,
    )
    examples = _attempt(
        _attribution("prompt", "add-examples"),
        commit_hash="2" * 40,
        score=6.5,
        parent_hash=rewrite.commit_hash,
    )

    stats = build_stats([parent, rewrite, examples], arms=_arms(), minimize=False)

    assert stats.arms[MetaEvolveAttribution("prompt", "rewrite")] == LiftSummary(
        count=1,
        mean_lift=2.5,
    )
    assert stats.arms[MetaEvolveAttribution("prompt", "add-examples")] == LiftSummary(
        count=1, mean_lift=-1.0
    )
    assert stats.arms[MetaEvolveAttribution("implementation", "replace")] == LiftSummary(
        count=0, mean_lift=0.0
    )
    assert stats.operators["prompt"] == LiftSummary(count=2, mean_lift=0.75)
    assert stats.operators["implementation"] == LiftSummary(
        count=0,
        mean_lift=0.0,
    )
    assert stats.skipped_attempts == 1


def test_build_stats_inverts_lift_for_minimize_direction():
    parent = _attempt(commit_hash="0" * 40, score=10.0, parent_hash=None)
    current = _attempt(
        _attribution("implementation", "replace"),
        commit_hash="1" * 40,
        score=7.5,
        parent_hash=parent.commit_hash,
    )

    stats = build_stats([parent, current], arms=_arms(), minimize=True)

    assert stats.arms[MetaEvolveAttribution("implementation", "replace")] == LiftSummary(
        count=1, mean_lift=2.5
    )


def test_build_stats_treats_zero_as_a_valid_score():
    parent = _attempt(commit_hash="0" * 40, score=0.0, parent_hash=None)
    current = _attempt(
        _attribution("prompt", "rewrite"),
        commit_hash="1" * 40,
        score=1.0,
        parent_hash=parent.commit_hash,
    )

    stats = build_stats([parent, current], arms=_arms(), minimize=False)

    assert stats.arms[MetaEvolveAttribution("prompt", "rewrite")] == LiftSummary(
        count=1,
        mean_lift=1.0,
    )


def test_build_stats_skips_parent_with_non_mapping_metadata():
    parent = _attempt(commit_hash="0" * 40, score=1.0, parent_hash=None)
    parent.metadata = None  # type: ignore[assignment]
    current = _attempt(
        _attribution("prompt", "rewrite"),
        commit_hash="1" * 40,
        score=2.0,
        parent_hash=parent.commit_hash,
    )

    stats = build_stats([parent, current], arms=_arms(), minimize=False)

    assert stats.arms[MetaEvolveAttribution("prompt", "rewrite")].count == 0
    assert stats.skipped_attempts == 2


@pytest.mark.parametrize(
    ("attempts", "reason"),
    [
        (
            [
                _attempt(
                    _attribution("prompt", "rewrite"),
                    commit_hash="1" * 40,
                    parent_hash="9" * 40,
                )
            ],
            "missing parent",
        ),
        (
            [
                _attempt(commit_hash="0" * 40, score=1.0, parent_hash=None),
                _attempt(
                    _attribution("prompt", "rewrite"),
                    commit_hash="1" * 40,
                    score=None,
                    parent_hash="0" * 40,
                ),
            ],
            "unscored current",
        ),
        (
            [
                _attempt(commit_hash="0" * 40, score=1.0, parent_hash=None),
                _attempt(
                    {
                        **_attribution("prompt", "rewrite"),
                        "budget_class": "tune",
                    },
                    commit_hash="1" * 40,
                    parent_hash="0" * 40,
                ),
            ],
            "tune current",
        ),
        (
            [
                _attempt(commit_hash="0" * 40, score=1.0, parent_hash=None),
                _attempt(
                    {
                        **_attribution("prompt", "rewrite"),
                        "budget_class": "grader_error",
                    },
                    commit_hash="1" * 40,
                    parent_hash="0" * 40,
                ),
            ],
            "grader-error current",
        ),
        (
            [
                _attempt(commit_hash="0" * 40, score=1.0, parent_hash=None),
                _attempt(
                    {**_attribution("prompt", "rewrite"), "archived": True},
                    commit_hash="1" * 40,
                    parent_hash="0" * 40,
                ),
            ],
            "archived current",
        ),
        (
            [
                _attempt(commit_hash="0" * 40, score=1.0, parent_hash=None),
                _attempt(
                    {"meta_evolve": "prompt/rewrite"},
                    commit_hash="1" * 40,
                    parent_hash="0" * 40,
                ),
            ],
            "malformed attribution",
        ),
        (
            [
                _attempt(commit_hash="0" * 40, score=1.0, parent_hash=None),
                _attempt(
                    _attribution("prompt", "unknown"),
                    commit_hash="1" * 40,
                    parent_hash="0" * 40,
                ),
            ],
            "unconfigured attribution",
        ),
        (
            [
                _attempt(
                    commit_hash="0" * 40,
                    agent_id="agent-2",
                    score=1.0,
                    parent_hash=None,
                ),
                _attempt(
                    _attribution("prompt", "rewrite"),
                    commit_hash="1" * 40,
                    agent_id="agent-1",
                    parent_hash="0" * 40,
                ),
            ],
            "cross-agent parent",
        ),
        (
            [
                _attempt(commit_hash="0" * 40, score=None, parent_hash=None),
                _attempt(
                    _attribution("prompt", "rewrite"),
                    commit_hash="1" * 40,
                    parent_hash="0" * 40,
                ),
            ],
            "unscored parent",
        ),
        (
            [
                _attempt(
                    {"budget_class": "tune"},
                    commit_hash="0" * 40,
                    score=1.0,
                    parent_hash=None,
                ),
                _attempt(
                    _attribution("prompt", "rewrite"),
                    commit_hash="1" * 40,
                    parent_hash="0" * 40,
                ),
            ],
            "tune parent",
        ),
        (
            [
                _attempt(
                    {"budget_class": "grader_error"},
                    commit_hash="0" * 40,
                    score=1.0,
                    parent_hash=None,
                ),
                _attempt(
                    _attribution("prompt", "rewrite"),
                    commit_hash="1" * 40,
                    parent_hash="0" * 40,
                ),
            ],
            "grader-error parent",
        ),
        (
            [
                _attempt(
                    {"archived": True},
                    commit_hash="0" * 40,
                    score=1.0,
                    parent_hash=None,
                ),
                _attempt(
                    _attribution("prompt", "rewrite"),
                    commit_hash="1" * 40,
                    parent_hash="0" * 40,
                ),
            ],
            "archived parent",
        ),
    ],
)
def test_build_stats_skips_ineligible_observations(attempts, reason):
    stats = build_stats(attempts, arms=_arms(), minimize=False)

    assert stats.arms[MetaEvolveAttribution("prompt", "rewrite")].count == 0, reason
    assert stats.skipped_attempts == len(attempts), reason


def _stats(
    *,
    rewrite: LiftSummary,
    examples: LiftSummary,
    implementation: LiftSummary | None = None,
) -> MetaEvolveStats:
    implementation = implementation or LiftSummary(count=0, mean_lift=0.0)
    return MetaEvolveStats(
        arms={
            MetaEvolveAttribution("prompt", "rewrite"): rewrite,
            MetaEvolveAttribution("prompt", "add-examples"): examples,
            MetaEvolveAttribution("implementation", "replace"): implementation,
        },
        operators={
            "prompt": LiftSummary(
                count=rewrite.count + examples.count,
                mean_lift=0.0,
            ),
            "implementation": implementation,
        },
        skipped_attempts=0,
    )


def test_recommend_arm_explores_unobserved_arms_in_configuration_order():
    arms = _arms()
    empty = LiftSummary(count=0, mean_lift=0.0)

    first = recommend_arm(
        stats=_stats(rewrite=empty, examples=empty),
        arms=arms,
        exploration_weight=1.0,
    )
    second = recommend_arm(
        stats=_stats(
            rewrite=LiftSummary(count=1, mean_lift=2.0),
            examples=empty,
        ),
        arms=arms,
        exploration_weight=1.0,
    )

    assert first == MetaEvolveRecommendation(
        attribution=MetaEvolveAttribution("prompt", "rewrite"),
        selection_mode="initial-exploration",
        arm_summary=empty,
        operator_summary=LiftSummary(count=0, mean_lift=0.0),
    )
    assert second.attribution == MetaEvolveAttribution("prompt", "add-examples")
    assert second.selection_mode == "initial-exploration"


def test_recommend_arm_exploits_higher_mean_with_equal_counts():
    recommendation = recommend_arm(
        stats=_stats(
            rewrite=LiftSummary(count=3, mean_lift=2.0),
            examples=LiftSummary(count=3, mean_lift=0.5),
            implementation=LiftSummary(count=3, mean_lift=-1.0),
        ),
        arms=_arms(),
        exploration_weight=1.0,
    )

    assert recommendation.attribution == MetaEvolveAttribution("prompt", "rewrite")
    assert recommendation.selection_mode == "upper-confidence"
    assert recommendation.arm_summary == LiftSummary(count=3, mean_lift=2.0)


def test_recommend_arm_applies_exploration_bonus_to_less_observed_arm():
    recommendation = recommend_arm(
        stats=_stats(
            rewrite=LiftSummary(count=9, mean_lift=1.0),
            examples=LiftSummary(count=1, mean_lift=0.7),
            implementation=LiftSummary(count=9, mean_lift=0.0),
        ),
        arms=_arms(),
        exploration_weight=2.0,
    )

    assert recommendation.attribution == MetaEvolveAttribution("prompt", "add-examples")


def test_recommend_arm_zero_exploration_weight_uses_mean_lift_only():
    recommendation = recommend_arm(
        stats=_stats(
            rewrite=LiftSummary(count=9, mean_lift=1.0),
            examples=LiftSummary(count=1, mean_lift=0.7),
            implementation=LiftSummary(count=9, mean_lift=0.0),
        ),
        arms=_arms(),
        exploration_weight=0.0,
    )

    assert recommendation.attribution == MetaEvolveAttribution("prompt", "rewrite")


def test_recommend_arm_breaks_equal_scores_by_configuration_order():
    recommendation = recommend_arm(
        stats=_stats(
            rewrite=LiftSummary(count=2, mean_lift=1.0),
            examples=LiftSummary(count=2, mean_lift=1.0),
            implementation=LiftSummary(count=2, mean_lift=1.0),
        ),
        arms=_arms(),
        exploration_weight=1.0,
    )

    assert recommendation.attribution == MetaEvolveAttribution("prompt", "rewrite")


def test_fixed_budget_recommendation_biases_toward_higher_lift_arm():
    arms = _arms()[:2]
    rewards = {
        arms[0]: 1.0,
        arms[1]: 0.0,
    }
    attempts: list[Attempt] = []
    choices: list[MetaEvolveAttribution] = []
    treatment_lift = 0.0

    for index in range(8):
        stats = build_stats(attempts, arms=arms, minimize=False)
        chosen = recommend_arm(
            stats=stats,
            arms=arms,
            exploration_weight=0.0,
        ).attribution
        reward = rewards[chosen]
        parent_hash = f"{index * 2:040x}"
        current_hash = f"{index * 2 + 1:040x}"
        attempts.extend(
            [
                _attempt(
                    commit_hash=parent_hash,
                    score=0.0,
                    parent_hash=None,
                ),
                _attempt(
                    _attribution(chosen.operator, chosen.mutation),
                    commit_hash=current_hash,
                    score=reward,
                    parent_hash=parent_hash,
                ),
            ]
        )
        choices.append(chosen)
        treatment_lift += reward

    baseline_lift = 4.0

    assert choices[:2] == arms
    assert choices.count(arms[0]) == 7
    assert treatment_lift == 7.0
    assert treatment_lift > baseline_lift
