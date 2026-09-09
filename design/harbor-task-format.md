# Harbor task compatibility and migration RFC

- **Status:** Phase 1 contract plus an implemented, bounded Gate B spike
- **Issue:** [#225](https://github.com/Human-Agent-Society/CORAL/issues/225)
- **Scope:** Phase 1 mapping plus a local, single-step, empty-workdir adapter;
  later migration gates remain deferred
- **CORAL baseline:** `88a2da15b0b3780357b473218b140403f1410a12`
  (`dev`, 2026-08-31)
- **Harbor baseline:** task schema `1.4`, Harbor `v0.22.0`, verified 2026-08-23

## Summary

CORAL should adopt the standard Harbor task directory as its portable task and
evaluation contract. CORAL should continue to own optimization orchestration:
agent runtimes, assignments, islands, shared state, attempt budgets, stop
conditions, worktree lifecycle, and the dashboard. Those settings must remain
outside Harbor's `task.toml`.

CORAL should keep `task.yaml` as its existing CLI and orchestration entrypoint.
A Harbor-backed `task.yaml` should reference exactly one local or registry
Harbor task instead of duplicating that task's instruction, metadata,
environment, or verifier fields.

The initial adapter targets Harbor task schema `1.4` and a pinned Harbor
`v0.22.0` runtime. This PR implements one deliberately narrow profile: a local,
single-step Linux container task whose workdir is empty at startup. Registry
resolution, pre-populated workdir export, datasets, multi-step tasks, declared
public artifacts, provider selection, and bulk migration remain later gates.

The implemented path translates Harbor's numeric reward mapping into CORAL's
`ScoreBundle` and preserves CORAL's private/public feedback boundary. It proves
that a CORAL candidate can be uploaded into a fresh Harbor environment and
verified without launching a second optimization agent. It does not claim
that all optional schema 1.4 capabilities or the complete migration in Issue
#225 are implemented.

## Implemented Gate B spike

The implementation in this PR adds the following bounded path:

1. A Harbor-backed `task.yaml` uses `task.source` plus
   `task.reward.primary`/`direction`. It rejects duplicated legacy task or
   portable grader fields.
2. CORAL resolves one local Harbor directory relative to `task.yaml`, requires
   schema `1.4`, records a content digest, derives the task name and instruction,
   and copies the complete Harbor task into `.coral/private/`.
3. The run starts from an empty Git workspace. Harbor `solution/`, `tests/`,
   task configuration, and verifier inputs never enter an agent worktree.
4. Evaluation launches exact `harbor==0.22.0` under isolated Python 3.12,
   starts a fresh Docker environment through Harbor's public `Task`/`Trial`
   APIs, verifies that the task workdir is dedicated and empty, and uploads
   only the candidate snapshot through a non-optimizing transfer agent.
5. Every numeric Harbor reward becomes a named CORAL `Score`; the configured
   primary key becomes `ScoreBundle.aggregated`, while direction remains a
   CORAL ranking concern and does not rewrite raw reward values.
6. Raw trial output stays under manager-only `harbor_runs/`. Agents receive
   only reward values, compatibility metadata, and a sanitized summary under
   `eval_logs/`.
7. The outer timeout fires early enough to interrupt the Harbor asyncio runner
   and give it a cleanup window before a final force-kill fallback.

The spike fails closed for registry references, non-1.4 schemas, multi-step or
Windows tasks, non-empty container workdirs, source/task symlinks, declared
public artifacts, mixed `seed/` or external starter repositories, and missing
or non-numeric primary rewards. CORAL's own `run.session=docker` is also
rejected because Harbor requires host Docker rather than Docker-in-Docker.
These are explicit capability limits, not silent partial support.

## Decision language

This document uses three statuses:

- **Existing** — behavior verified in current CORAL or current Harbor sources.
- **Proposed** — the recommended contract for Maintainer approval.
- **Open** — a decision or feasibility point that Phase 2 must resolve before
  the adapter API is treated as stable.

## Source baseline

### CORAL today

At the baseline above, CORAL has 397 `task*.yaml` or `task*.yml` files under
`examples/`.
`CoralConfig` combines two different concerns in one YAML document:

1. portable task and evaluation content:
   - `task.name`, `task.description`, and `task.tips`;
   - `grader.entrypoint`, setup commands, timeout, args, private paths, and
     score direction;
   - the visible starting repository selected through `workspace.repo_path`;
2. optimization-run orchestration:
   - agent runtime/model/bindings, assignments, skills, heartbeat, sandbox,
     gateway, and reliability controls;
   - islands, migration, and sharing;
   - run/session/UI/stop settings and result-directory layout.

The authoritative implementation is
[`coral/config.py`](../coral/config.py). Project creation copies `seed/` into a
CORAL-managed Git repository, copies grader-private inputs under
`.coral/private/`, and builds an isolated grader virtual environment; see
[`coral/workspace/project.py`](../coral/workspace/project.py) and
[`coral/workspace/grader_env.py`](../coral/workspace/grader_env.py).

Grader entrypoints return a [`ScoreBundle`](../coral/types.py), and finalized
attempts retain the aggregate score, public feedback, and selected metadata.
Current Harbor-backed examples do not load Harbor tasks as CORAL's canonical
task format. Instead, task-specific `TaskGrader` packages shell out to
`harbor run` and parse job output; see
[`examples/swebench-verified`](../examples/swebench-verified),
[`examples/terminal-bench`](../examples/terminal-bench), and the Harbor v0.13
result fixtures in [`tests/test_grader.py`](../tests/test_grader.py).

### Harbor today

The current official project is
[`harbor-framework/harbor`](https://github.com/harbor-framework/harbor), not the
older `corca-ai/harbor` URL referenced in Issue #225. The latest verified
release for this RFC is
[`v0.22.0`](https://github.com/harbor-framework/harbor/releases/tag/v0.22.0),
published on 2026-08-22.

The official [Task Structure](https://www.harborframework.com/docs/tasks)
defines:

```text
instruction.md
task.toml
environment/
solution/       # optional oracle/reference solution
tests/
```

The released v0.22.0 task schema default is `1.4`. `task.toml` separates task
metadata, agent/verifier timeouts, environment resources and network policy,
optional solution settings, and verifier isolation. The agent executes in a
Harbor environment; `tests/test.sh` or `test.bat` writes numeric rewards to
`/logs/verifier/reward.json` or `reward.txt`. Harbor exposes those numbers as
`VerifierResult.rewards: dict[str, float | int] | None` in the tagged
[`VerifierResult` model](https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/models/verifier/result.py).

Harbor's live documentation can advance ahead of an installed release. The
compatibility contract therefore uses the tagged v0.22.0 source and release
notes as authority; live documentation is a discovery aid, not the version
boundary.

Harbor's [Core Concepts](https://www.harborframework.com/docs/core-concepts)
distinguish a portable task, a dataset, an agent, a container environment, a
trial, and a job. Registry packages use `org/name@tag`; publishing validates
and stores the canonical task configuration and digest. See
[Publishing a task](https://www.harborframework.com/docs/tasks/publishing) and
[Tasks and Datasets](https://www.harborframework.com/docs/sharing/sharing).

These are current facts, not a compatibility promise across every Harbor
release. CORAL's existing v0.13 job-result parser is not evidence that it is
compatible with Harbor v0.22 task loading or trial execution.

## Proposed ownership boundary

### Harbor owns the portable task contract

**Proposed:** Harbor owns all information needed to understand and verify a
task independently of CORAL:

- task identity, description, authors, keywords, and task package version;
- human instruction;
- initial environment and resource/network requirements;
- optional oracle solution;
- verifier code, verifier environment, timeouts, numeric rewards, and declared
  artifacts;
- task schema version and registry packaging.

CORAL must not add private keys to Harbor's standard `task.toml` schema. If a
portable task is downloaded and run with Harbor alone, its task semantics and
verification must remain intact.

### `task.yaml` remains the CORAL optimization-run contract

**Proposed:** extend the existing `task.yaml` so it can reference one Harbor
task while continuing to contain CORAL orchestration and objective-selection
state:

```yaml
task:
  source: ./task                 # or org/name@versioned-tag
  reward:
    primary: reward
    direction: maximize

agents:
  runtime: codex
  model: gpt-5
  count: 4

islands:
  count: 2

sharing:
  attempts: true
  notes: true
  skills: true

workspace:
  results_dir: ./results

run:
  session: local
  stop:
    max_real_attempts: 100
```

The retained filename follows CORAL's current CLI and authoring workflow. The
field shape is **Implemented for the initial local profile**. The important
compatibility rule is a one-of contract:

- a legacy `task.yaml` contains the current inline task/grader fields and no
  `task.source`;
- a Harbor-backed `task.yaml` contains `task.source` and CORAL orchestration,
  but does not duplicate `task.name`, `task.description`, `task.tips`, or
  portable grader fields owned by the referenced Harbor task.

Validation must reject mixed configurations rather than guess which definition
wins. Keeping `task.yaml` does not create a CORAL fork of Harbor's schema:
Harbor still owns `task.toml`, while `task.yaml` remains a separate CORAL file
that points to it.

A local `task.source` is resolved relative to the directory containing
`task.yaml` and must identify one directory with a `task.toml`; it does not scan
for tasks or accept a dataset implicitly. The current implementation rejects
registry sources; Gate C reserves `org/name@tag` before digest pinning. This
keeps the selected local Harbor task deterministic for `coral validate`,
`start`, and `eval`.

Persisted CORAL configs must not use a mutable registry alias such as `latest`.
At run creation CORAL should resolve a local directory or `org/name@tag` to an
immutable task digest and record the original reference, digest, Harbor task
schema, task package version, and Harbor runtime version in the run metadata.

## Mapping from the current CORAL format

| Current CORAL asset or field | Harbor / new location | Status and notes |
| --- | --- | --- |
| `task.name` | `[task].name` in `task.toml` | **Proposed.** Harbor requires stable `org/name`; migration may need an explicit organization. |
| `task.description` | `instruction.md` plus `[task].description` | **Proposed.** Full agent-facing content goes in `instruction.md`; the TOML description stays short. |
| `task.tips` | `instruction.md` | **Proposed.** Merge into a clearly labelled guidance section; do not duplicate it in Harbor-backed `task.yaml`. |
| `seed/` | agent-visible starting workspace | **Open.** It must never map to Harbor `solution/`. See workspace materialization below. |
| `grader.entrypoint` | `tests/`, verifier config, or a temporary legacy bridge | **Proposed.** Canonical tasks use Harbor verification; generic `TaskGrader` remains transition-only. |
| `grader.setup` | verifier/environment image build | **Proposed.** Convert reproducible installs into Dockerfiles or verifier images; do not execute arbitrary legacy setup implicitly. |
| `grader.timeout` | `[verifier].timeout_sec` | **Proposed.** CORAL may impose a stricter outer infrastructure timeout but must record which layer fired. |
| `grader.private` | Harbor `tests/`, separate verifier environment, and declared verifier-only artifacts | **Proposed.** Never copy these paths into a CORAL agent worktree. |
| `grader.args` | task-specific Harbor metadata/config or migration code | **Open per field.** Do not dump arbitrary runtime args into `[metadata]` and call them portable. |
| `grader.direction` | `task.reward.direction` in `task.yaml` | **Proposed.** Preserve raw Harbor reward values; CORAL applies maximize/minimize when ranking attempts. |
| `grader.max_pending_per_agent` | CORAL orchestration config | **Proposed.** Queue policy is not task semantics. |
| `grader.parallel.max_workers` | CORAL orchestration config | **Proposed.** Harbor/provider concurrency is a separate adapter setting. |
| `workspace.repo_path` and `seed/` copying | workspace materializer | **Open.** Must produce the same agent-visible Git baseline as the Harbor environment. |
| `workspace.setup` | Harbor environment build where portable; CORAL run bootstrap otherwise | **Open.** Every command needs an owner and reproducibility rule. |
| `agents.*`, `islands.*`, `sharing.*`, `run.*` | `task.yaml` | **Proposed.** These stay entirely outside `task.toml`. |
| `ScoreBundle.scores` | one CORAL `Score` per Harbor reward key | **Proposed.** Preserve names and numeric values exactly. |
| `ScoreBundle.aggregated` | explicit primary reward or approved aggregation | **Proposed.** Never infer weights or average an arbitrary reward dictionary. |
| `ScoreBundle.feedback` | sanitized verifier summary | **Proposed.** Public feedback policy still applies. |
| agent-visible `eval_logs/` directories | selected Harbor logs and artifact manifest | **Proposed.** Preserve both single-island and per-island locations; store references and public copies without leaking verifier-only data. |
| `Attempt` status/budget class | Harbor result or exception classification | **Proposed.** Missing reward, verifier crash, setup failure, and timeout are grader infrastructure outcomes, not a score of zero. |

## Reward and attempt translation

Harbor's canonical verifier result is a mapping of numeric reward names to
values. The adapter should translate it without changing meaning:

```text
Harbor VerifierResult.rewards
  -> ScoreBundle.scores[name] = Score(value=value, name=name)
  -> ScoreBundle.aggregated = scores[configured_primary].value
```

**Proposed rules:**

1. `task.reward.primary` is required when more than one Harbor reward key can
   be returned.
2. If an approved aggregation is required, it is declared explicitly in
   CORAL orchestration and versioned with the run. The adapter does not invent
   weights or silently use the first dictionary key.
3. `direction` controls CORAL ranking and stop thresholds; it does not negate
   or rewrite the stored raw Harbor reward.
4. `rewards is None`, a missing primary key, task/environment build failure,
   verifier failure, and timeout produce a structured grader-error/timeout
   attempt. They do not produce a real attempt with score `0`.
5. Public feedback contains only a bounded verifier summary. The initial
   adapter keeps raw Harbor logs private and rejects declared public artifacts
   until an explicit visibility/copy policy is implemented.
6. The adapter records Harbor task digest, task version, schema version,
   Harbor runtime version, reward mapping, and the sanitized summary reference
   in attempt metadata. Provider metadata and artifact references remain
   deferred with provider/artifact support.

## Workspace materialization and evaluation flow

The central incompatibility is that CORAL evolves a Git worktree while Harbor
normally owns an agent trial inside an environment. Mapping `seed/` to
`solution/` would expose an oracle and is rejected.

**Implemented initial adapter boundary:**

```text
resolve local Harbor task path -> verify schema + pin digest
                               -> create empty agent-visible workspace
                               -> initialize CORAL repo/worktrees

candidate CORAL commit -> create fresh Harbor environment
                       -> upload candidate workspace to the agent workdir
                       -> run Harbor verifier without a second optimization agent
                       -> collect VerifierResult; retain raw logs privately
                       -> translate to ScoreBundle + Attempt
```

This PR proves candidate import and verification without launching a competing
optimization agent, using Harbor's public Python interfaces. Exporting a
pre-populated Harbor workdir into CORAL remains **Open**; the implemented
profile therefore requires Harbor's container workdir to start empty.

If that contract is unavailable, the fallback choices are:

1. add or upstream a Harbor API for prepare/upload/verify; or
2. keep the affected CORAL task on the legacy loader.

Running a no-op Harbor agent merely to reach the verifier is not recommended:
it adds lifecycle and logging ambiguity and makes cancellation ownership
unclear.

**Proposed record relationship:** one completed CORAL candidate evaluation maps
to one Harbor trial-like verifier result and one CORAL `Attempt`. A CORAL
optimization run spans many such evaluations; it is not itself a single Harbor
trial. Whether the adapter creates one Harbor job per evaluation or owns a
longer-lived job/session is **Open** and must not change the
one-result-per-attempt history contract.

### Non-containerized tasks

Harbor's current core task model is container-environment based. CORAL has many
host-worktree tasks. The initial canonical adapter should therefore support
only tasks whose Harbor environment can be reproduced by a supported provider.

**Proposed transition rule:** host-only tasks remain on the legacy CORAL loader
until one of these is accepted:

- an official Harbor environment provider that safely models the required host
  behavior; or
- a separately named CORAL compatibility profile with explicit portability and
  security limitations.

Consuming only the Harbor directory layout while bypassing Harbor environment
and verifier semantics must not be described as full Harbor compatibility.

## Security and visibility invariants

The adapter must preserve these invariants before any bulk migration:

1. `solution/`, `tests/`, verifier source, answer keys, and verifier-only
   environment variables are never copied into agent worktrees or
   `.coral/public/`.
2. Sensitive tasks use Harbor's separate verifier environment where feasible.
   A shared verifier environment is accepted only when the task declares that
   its tests and dependencies are safe from the agent.
3. Only explicitly declared public artifacts and sanitized feedback cross from
   Harbor verification into agent-visible CORAL state.
4. Private logs remain under `.coral/private/` or another manager-only path.
   Public eval logs contain an artifact manifest and approved copies, not an
   indiscriminate Harbor job directory.
5. Registry credentials, environment secrets, and provider tokens stay in the
   manager process and are never serialized into `task.yaml`, attempt JSON,
   CORAL.md, or PR/test output.
6. Task resolution is immutable for a run. A registry tag is resolved once and
   its digest is persisted before agents start.
7. Resuming a run uses the recorded digest and compatibility metadata; it does
   not re-resolve a mutable tag.

## Version and compatibility policy

**Implemented initial support:**

- Harbor runtime: exactly `v0.22.0` for Gate B, pinned in the isolated runner
  invocation;
- Harbor task schema: exactly `1.4`;
- registry task: deferred to Gate C;
- CORAL legacy task loader: retained during the migration window.

Schema acceptance is not blanket support for every optional Harbor feature.
The adapter must publish a capability profile and reject unsupported features
with an actionable validation error rather than ignore them.

| Harbor v0.22 capability | Initial status |
| --- | --- |
| single-step `instruction.md`, empty workdir, environment, verifier, and numeric rewards | **Implemented for the bounded Gate B spike** |
| declared public artifacts | **Deferred.** Reject rather than silently hide or publish them. |
| local task directory | **Implemented for the bounded Gate B spike** |
| registry task reference | **Proposed for Gate C**, after digest pinning is proven |
| dataset reference and task selection | **Deferred.** Build on the task adapter; do not make dataset selection implicit in `task.source`. |
| multi-step task | **Deferred.** Reject initially; a CORAL attempt is not a Harbor step, and reward/cancellation semantics need a separate mapping. |
| simulated-user and loaded-trajectory runs | **Deferred.** Add only with explicit history, privacy, and lifecycle mappings. |
| host-only environment | **Legacy loader** until an approved provider or compatibility profile exists. |

Package version, task schema version, task package version, and registry tag
are separate values and must be reported separately. Any new Harbor release or
task schema is unsupported until the adapter compatibility suite passes. The
supported set may then be widened one tested version at a time or expressed as
an upper-bounded range once CI continuously exercises that range. Patch updates
are not assumed compatible merely because the version is semver-like.

The current v0.13 parser tests remain useful only for the two legacy wrappers.
They do not define the canonical adapter's compatibility range.

## Validation layers

`coral validate` should eventually report distinct layers rather than collapse
them into one success message:

1. **Resolution** — local path or registry reference resolves to a pinned task.
2. **Harbor schema** — Harbor loads and validates the task directory.
3. **Materialization** — the agent-visible workspace can be created without
   private content.
4. **Verification smoke** — the initial workspace produces a structured Harbor
   verifier result or a structured failure.
5. **CORAL mapping** — reward selection, direction, feedback visibility, and
   attempt metadata are complete.
6. **Optional oracle/calibration** — when `solution/` or expected cases exist,
   their results match declared expectations.

Passing layers 1–4 proves structural executability. It does not prove that the
reward captures user intent, and the UI/CLI must not claim otherwise.

## Migration and rollout gates

This RFC refines, but does not reorder, the phases in Issue #225.

### Gate A — accept this compatibility contract

- Maintainers decide the Open items below.
- Replace obsolete Harbor repository links with current official sources.
- Agree on the first supported Harbor runtime and schema.

### Gate B — build a narrow loader/verification spike (partially implemented)

- local task directory only — **implemented**;
- one container-backed deterministic empty-workdir task — **implemented and
  exercised end-to-end**;
- no registry, no bulk migration, no CLI default change — **implemented**;
- private task staging, reward mapping, timeout/error classification, and a
  cancellation cleanup window — **implemented**;
- pre-populated workspace export, declared public artifacts, and the wider
  representative parity matrix — **not yet implemented**.

### Gate C — introduce the dual loader behind an explicit task source

- support a pinned local Harbor directory and a versioned registry reference
  resolved to a recorded content digest;
- select legacy or Harbor-backed loading from the validated, mutually exclusive
  `task.yaml` shape;
- preserve legacy `task.yaml` behavior when `task.source` is absent;
- persist task digest and compatibility metadata in every run;
- add CLI diagnostics without changing default authoring output.

### Gate D — pass the representative parity matrix

| Family | Representative CORAL task | Required evidence |
| --- | --- | --- |
| deterministic numeric maximize | `circle_packing` or `dna_design` proxy mode | same valid/invalid behavior, reward name/value, public feedback |
| minimize objective | `kernel_builder` or another current minimize task | same raw metric and leaderboard ordering without sign corruption |
| private inputs | `mnist` or `stanford_covid_vaccine` | no private path visible; equivalent score and failure feedback |
| rubric/LLM judge | `race-japan-elderly` or `apex-eggshell-skull` | judge config, multi-metric rewards, feedback and secret handling |
| GPU/resource-heavy | a current GPU example | resource declaration and timeout behavior on a supported provider |
| existing Harbor wrapper | `swebench-verified` | equivalent fixed-slice score, trajectories, logs, tune/real budget class |
| existing Harbor wrapper | `terminal-bench` | equivalent pass rate, timeouts, logs and failure classification |

Parity means repeated results within an agreed tolerance, equivalent validity
gates, preserved direction, and no reduction in feedback or security. One
successful baseline is not enough.

### Gate E — authoring and controlled migration

- `coral init` can scaffold a Harbor task plus a compatible `task.yaml` that
  references it;
- `coral validate`, `start`, and `eval` support the new source explicitly;
- migration tooling produces a report and refuses ambiguous field mappings;
- convert a small reviewed batch before any generated bulk migration;
- retain the legacy loader and warnings for a documented compatibility window.

### Gate F — deprecate only after downstream evidence

Deprecation starts only when representative built-ins, external usage, docs,
templates, and bundled skills use the new contract and rollback remains
possible. Removing `TaskGrader` as the canonical task-definition path does not
preclude a clearly named legacy/custom extension while Harbor lacks required
semantics.

## Open decisions for Maintainers

| Decision | Recommendation | Alternative | Status |
| --- | --- | --- | --- |
| CORAL orchestration entrypoint | keep `task.yaml` and add a Harbor-backed `task.source` mode | introduce a separate `coral.yaml` | **Implemented for the local spike** |
| first Harbor runtime range | start with exact `v0.22.0` and schema `1.4`; widen only after compatibility tests | target an older release matching current wrappers | **Implemented for the local spike** |
| local and registry references | support local paths first, then pinned `org/name@tag`; persist digest | registry-first | **Local implemented; registry deferred** |
| dataset references | defer until the task adapter is stable, then add an explicit dataset source and task selector | overload `task.source` with path, task, and dataset guessing | **Proposed** |
| mutable `latest` | reject in persisted run config | resolve silently on each start | **Proposed**; silent re-resolution breaks reproducibility |
| visible starter workspace | export Harbor agent workdir into a CORAL Git baseline, then upload candidate snapshots for verification | run CORAL agents inside Harbor environments | **Empty workdir implemented; pre-populated export remains Open** |
| non-container tasks | keep legacy until an explicit supported provider/profile exists | native verifier path using Harbor files only | **Open**; the alternative has reduced portability |
| multiple rewards | require a primary key and optional explicit aggregation | infer first key or average | **Primary-key mapping implemented; aggregation deferred** |
| minimize objectives | store raw reward; let CORAL ranking apply direction | negate reward in adapter | **Implemented** |
| custom/rubric graders | migrate to Harbor tests/RewardKit where possible; keep a transition escape hatch | preserve `TaskGrader` indefinitely as a second canonical system | **Open** |
| private verifier mode | prefer separate verifier environment for sensitive tasks | shared verifier with task-owned risk acceptance | **Proposed** |
| legacy removal timeline | evidence-based dual-loader window | immediate breaking migration | **Open** |
| Harbor multi-step tasks | reject in the initial profile and design a separate step/attempt mapping | treat CORAL attempts as Harbor steps | **Open**; the two lifecycles are not equivalent |

## Acceptance criteria for Phase 1

Phase 1 is complete when Maintainers have reviewed the Proposed decisions and
resolved or explicitly deferred every Open item needed for the next supported
profile. This PR now includes the bounded Gate B spike documented above, but
does not claim complete migration, representative score parity, registry
support, or compatibility with Harbor versions beyond `v0.22.0`.

For the configuration boundary, Phase 1 acceptance additionally requires
agreement that Harbor-backed and legacy `task.yaml` forms are mutually
exclusive, and that a Harbor-backed form references exactly one Harbor task
without copying portable task fields into CORAL configuration.

Subsequent implementation PRs should reference Issue #225 but must not use
`Fixes #225` until the complete migration and deprecation acceptance criteria
are satisfied.

## Primary references

- [CORAL Issue #225](https://github.com/Human-Agent-Society/CORAL/issues/225)
- [PR #251 configuration discussion](https://github.com/Human-Agent-Society/CORAL/pull/251#issuecomment-5392439317)
- [Harbor task structure](https://www.harborframework.com/docs/tasks)
- [Harbor core concepts](https://www.harborframework.com/docs/core-concepts)
- [Harbor task publishing](https://www.harborframework.com/docs/tasks/publishing)
- [Harbor task and dataset sharing](https://www.harborframework.com/docs/sharing/sharing)
- [Harbor v0.22.0 release](https://github.com/harbor-framework/harbor/releases/tag/v0.22.0)
- [Harbor v0.22.0 task configuration](https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/models/task/config.py)
- [Harbor v0.22.0 `VerifierResult`](https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/models/verifier/result.py)
- [CORAL configuration](../coral/config.py)
- [CORAL score and attempt types](../coral/types.py)
- [CORAL project materialization](../coral/workspace/project.py)
- [CORAL grader environment](../coral/workspace/grader_env.py)
- [CORAL subprocess grader](../coral/grader/subprocess_grader.py)
