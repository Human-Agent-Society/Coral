# Gradient-style meta-evolve

## Status

Proposed first implementation slice for Issue #74. This document selects a
prompt-level adaptive operator recommendation design. It does not introduce a
central optimizer that can force agents to mutate code in a particular way.

## Context

CORAL currently treats agents as the optimizers. Agents choose their own
strategies, submit commits through `coral eval`, and receive scores and
heartbeat prompts. The manager tracks per-agent score history and injects the
`pivot` heartbeat after a plateau, but it has no explicit operator catalogue,
mutation attribution, or central sampler.

Issue #74 asks CORAL to retain credit from prior mutations instead of repeatedly
rediscovering useful strategies through blind exploration. The first version
must fit the existing architecture:

- keep agents responsible for the final strategy choice;
- use existing attempt records as the durable evidence source;
- preserve legacy runs and default behavior;
- activate only through an explicit configuration flag;
- make the recommendation deterministic and auditable;
- permit a fixed-budget comparison with the current pivot behavior.

## Decision

Add an optional meta-evolve layer that attributes real attempts to configured
operator/mutation arms, estimates direction-aware lift from parent attempts,
and recommends an arm when the existing `pivot` heartbeat fires.

The recommendation is advisory. It is injected into the pivot prompt so the
agent can accept it, explain why another arm is more appropriate, or declare a
new hypothesis in its focus note. CORAL does not rewrite the agent's code or
silently change its task.

## Scope

The first version will:

1. define a task-configured set of discrete operator/mutation arms;
2. add explicit operator and mutation attribution to `coral eval`;
3. store attribution in `Attempt.metadata`;
4. compute per-arm and per-operator directional lift;
5. select a recommended arm with a deterministic upper-confidence rule;
6. inject the recommendation and its evidence into `pivot` heartbeats;
7. reconstruct state from attempts after resume or agent migration;
8. document an opt-in fixed-budget A/B procedure;
9. cover enabled and disabled behavior with deterministic tests.

## Non-goals

This slice will not:

- create a general mutation framework or plugin API;
- force an agent to use the recommended arm;
- infer operators from commit messages or LLM-generated prose;
- compare tune-mode scores with real-attempt scores;
- evolve the exploration coefficient or arm catalogue at runtime;
- combine models or implement Issue #75;
- claim empirical token savings without a real fixed-budget field run;
- change behavior when meta-evolve is disabled.

## Configuration

Add an optional configuration block under `agents`:

```yaml
agents:
  meta_evolve:
    enabled: true
    exploration_weight: 1.0
    arms:
      - operator: prompt
        mutation: rewrite-instructions
      - operator: prompt
        mutation: add-examples
      - operator: implementation
        mutation: replace-algorithm
```

The corresponding model is:

```text
MetaEvolveConfig
|- enabled: bool = false
|- exploration_weight: float = 1.0
`- arms: list[MetaEvolveArmConfig] = []

MetaEvolveArmConfig
|- operator: str
`- mutation: str
```

Validation rules:

- `exploration_weight` must be finite and non-negative;
- enabled mode requires at least two arms so selection is meaningful;
- `operator` and `mutation` must be non-empty after trimming;
- `(operator, mutation)` pairs must be unique;
- configuration order is stable and is used as the deterministic tie-breaker.

Disabled mode accepts legacy task files unchanged and does not add any prompt
or runtime output.

## Attempt attribution

Add paired CLI options:

```bash
coral eval -m "rewrite prompt" \
  --operator prompt \
  --mutation rewrite-instructions
```

The two options are atomic: either both are present or both are absent. In an
enabled meta-evolve run, every real attempt must provide a configured pair.
Validation happens before `git add` or commit creation so a typo cannot consume
an attempt or mutate the worktree history.

Tune attempts may carry a configured pair for diagnostic continuity, but they
are never used as lift observations. Grader-error attempts preserve attribution
for auditability and are also excluded from lift calculations.

Attribution is stored without changing the `Attempt` schema:

```json
{
  "metadata": {
    "meta_evolve": {
      "operator": "prompt",
      "mutation": "rewrite-instructions"
    }
  }
}
```

Old attempts and disabled runs have no `meta_evolve` key and continue to load
normally.

## Lift observations

An observation is eligible only when all of the following are true:

- the attempt is not archived;
- it is a scored `real` attempt;
- it has a complete configured operator/mutation pair;
- its `parent_hash` resolves to a scored, non-archived `real` attempt from the
  same agent;
- the current and parent scores come from the same run and grader direction.

The directional lift is:

```text
maximize: current_score - parent_score
minimize: parent_score - current_score
```

Positive lift is therefore always better. A numeric score of zero is valid;
`None` is not an observation.

Each eligible observation contributes to:

- the exact `(operator, mutation)` arm statistics;
- aggregate statistics for its `operator`.

Statistics retain observation count and mean lift. The implementation rebuilds
them from attempt JSON rather than writing a second mutable state file. Resume,
manager restart, and agent migration therefore share one evidence source.

Attempts whose parent is missing, unscored, tune-mode, grader-error, or from a
different agent remain visible as unattributed evidence but do not affect arm
weights.

## Recommendation algorithm

The first version uses a deterministic upper-confidence rule over configured
arms:

1. Recommend unobserved arms in configuration order until every arm has one
   eligible lift observation.
2. Afterwards compute:

   ```text
   arm_score = mean_lift
             + exploration_weight * sqrt(log(total_observations) / observation_count)
   ```

3. Select the arm with the largest `arm_score`.
4. Break ties by configuration order.

This is a cheap credit-assignment proxy for a discrete, non-differentiable
operator space. It exploits consistently positive lift while preserving an
exploration bonus for less-tested arms. It does not claim to be a differentiable
gradient optimizer.

The selection functions are pure and take attempts, direction, arm catalogue,
and exploration weight as inputs. They do not read global process state or use
Python's randomized hash order.

## Runtime integration

Add a focused module under `coral/agent/` responsible for:

- parsing attribution from attempt metadata;
- building lift observations and aggregate statistics;
- selecting the next arm;
- rendering an auditable recommendation block.

The existing manager flow remains the integration point:

```text
scored attempt
  -> existing real-attempt accounting
  -> existing heartbeat trigger evaluation
  -> pivot action fires
  -> rebuild this agent's meta-evolve statistics from attempts
  -> recommend one configured arm
  -> inject recommendation before the existing pivot prompt
  -> interrupt and resume the agent as today
```

Only a fired action named `pivot` receives the recommendation. Reflect,
consolidate, lint-wiki, custom interval actions, and runs without meta-evolve
keep their existing prompts byte-for-byte.

The recommendation block contains:

- the recommended operator and mutation;
- observation count and mean lift for that arm;
- aggregate operator count and mean lift;
- whether the choice is initial exploration or upper-confidence selection;
- a reminder to attribute the next real eval with the CLI options;
- the number of skipped/unattributed attempts, without treating them as zero
  reward.

## Agent bootstrap contract

When meta-evolve is enabled, generated CORAL guidance lists the configured arms
and the attribution command. Before the first pivot recommendation, the agent
uses the first unobserved arm in configuration order. After a pivot, the latest
recommendation is advisory but should be the default unless the agent records a
specific reason to deviate.

The disabled template output is unchanged. No global prompt is modified for
runs that do not opt in.

## Resume and multi-island behavior

Statistics are reconstructed from durable attempt files for the target agent,
not from the manager's in-memory score history. This provides:

- manager restart and `coral resume` continuity;
- compatibility with attempts created before meta-evolve was enabled;
- migration continuity when an agent's attempt records move between islands;
- isolation between agents and islands.

The collector filters by the full `agent_id` and uses CORAL's existing
cross-island attempt reader. It must not pool lift across agents because agents
may operate on different baselines or inherit different histories.

## Error handling

- Invalid configuration fails at task-config load with an actionable message.
- An unknown CLI pair fails before any commit is created.
- Supplying only one of `--operator` or `--mutation` is a CLI usage error.
- Missing or malformed historical attribution is skipped and counted.
- A missing or ineligible parent is skipped, not assigned zero lift.
- If no eligible observations exist, deterministic initial exploration still
  produces a recommendation.
- Recommendation failure must not crash the manager. It is logged and the
  original pivot prompt is delivered unchanged.

## Compatibility and migration

The feature is additive and disabled by default:

- existing task YAML files remain valid;
- existing `coral eval` commands remain valid in disabled runs;
- old attempt JSON remains readable;
- score, budget-class, plateau, and leaderboard behavior is unchanged;
- no persistent migration command is required.

Enabling the feature deliberately strengthens the real-eval contract by
requiring a configured operator/mutation pair. This is opt-in validation, not a
backward-compatibility shim.

## Observability

The initial user-visible surface is intentionally narrow:

- attribution is visible in attempt JSON and `coral show`;
- pivot prompts show the recommendation and supporting statistics;
- verbose manager logs identify the selected arm and selection mode.

A general `coral meta-evolve stats` command, dashboard visualization, and
cross-run analytics are deferred until field use establishes their value.

## Verification

### Unit tests

- configuration validation and disabled defaults;
- paired CLI option validation before commit;
- attempt metadata round-trip and legacy attempts;
- maximize and minimize directional lift;
- zero scores, missing parents, tune attempts, grader errors, archived attempts,
  malformed attribution, and cross-agent parents;
- per-arm and per-operator aggregation;
- deterministic unobserved-arm exploration;
- upper-confidence exploitation and exploration bonus;
- stable configuration-order tie-breaking.

### Integration tests

- disabled manager and generated guidance remain unchanged;
- enabled `coral eval` writes attribution;
- pivot injection contains one recommendation and the expected evidence;
- non-pivot heartbeat prompts are unchanged;
- resume rebuilds the same recommendation from attempt files;
- multi-island reads do not mix agents.

### Fixed-budget A/B harness

Use a deterministic synthetic sequence with a fixed number of attempts and
known arm reward streams:

- baseline: uniform arm rotation using the same budget;
- treatment: the upper-confidence recommendation;
- assertion: after mandatory initial exploration, treatment allocates more of
  the remaining budget to the higher-lift arm and produces a higher cumulative
  directional lift for the fixed fixture.

This test verifies the selection mechanism, not real-world token savings. A
real task A/B run is required before claiming operational efficiency gains.

### Repository checks

- focused meta-evolve, CLI, hook, manager, config, template, and type tests;
- full `uv run pytest tests/ -v`;
- `uv run ruff check .`;
- `uv run ruff format --check .`;
- focused mypy for the changed framework modules;
- `git diff --check upstream/dev...HEAD`.

## Expected file boundary

The implementation is expected to touch only:

- `coral/config.py`;
- `coral/types.py` or a small metadata helper;
- `coral/cli/__init__.py`, `coral/cli/eval.py`, and the narrow `coral show`
  rendering path in `coral/cli/query.py`;
- `coral/hooks/post_commit.py`;
- one focused module under `coral/agent/`;
- the narrow manager pivot integration;
- CORAL guidance generation/templates;
- CLI/config documentation;
- corresponding focused tests.

Changes to graders, the grader daemon, migration scheduling, web UI, example
tasks, model-combination logic, or unrelated heartbeats are outside scope.

## Acceptance criteria

The design is complete when:

1. disabled runs are behaviorally unchanged;
2. enabled real attempts carry validated configured attribution;
3. eligible attempts produce direction-aware lift observations;
4. selection preserves initial exploration and then uses observed lift;
5. only pivot prompts receive an auditable recommendation;
6. restart and migration reconstruct equivalent statistics;
7. fixed-budget deterministic tests demonstrate biased allocation without
   claiming empirical production gains;
8. all repository-required tests, lint, formatting, documentation, and human
   review gates are satisfied.
