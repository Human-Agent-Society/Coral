# RSI-Exam

CORAL adapter for the **35 public research tasks** in
[RSI-Exam](https://github.com/aiming-lab/RSI-Exam). Task data comes from
[RSI-Exam/RSI-Exam on Hugging Face](https://huggingface.co/datasets/RSI-Exam/RSI-Exam),
pinned to commit `66f54935eaa576e27dae446f74c2ce17875c14da`.
The other 53 tasks are not publicly available and are not included.

Each public task has its own `examples/rsi_exam/<task_name>/` directory,
with a `task.yaml`, `seed/`, packaged `grader/`, and private `rsi_task/`.
`_grader/` is the shared adapter source; its copies make each task standalone.
There is no suite-level CORAL task or aggregate score.

A lightweight starting point is `tidal_friction_inverse`, a small CPU task whose
submission implements `estimate_logf(case)` in `methods/main/solver.py`.
Its deliberately weak upstream baseline is retained unchanged. A zero reward
is a valid baseline result; infrastructure failures return no score.

## Run a task

Run from the CORAL repository root:

```bash
uv run examples/rsi_exam/prepare.py tidal_friction_inverse
uv run coral validate examples/rsi_exam/tidal_friction_inverse
uv run coral start -c examples/rsi_exam/tidal_friction_inverse/task.yaml
```

Requirements: `uv`, Python 3.12 (downloadable by uv), a running Docker daemon,
and Docker Compose v2 with a Linux kernel that supports Harbor’s nftables
egress isolation (`CONFIG_NFT_FIB_INET`). The tidal task’s agent image requests 2 CPUs and
12 GiB memory. Image builds need internet access. Evaluation uses upstream's
network restrictions. The first run downloads Harbor and builds two images;
subsequent runs use caches. Harbor 0.22.0 runs in a separate uv environment
because its LiteLLM dependency conflicts with CORAL's pinned version.

Tasks default to one Claude Code agent, except the stock return ranking config
below. Choose another configured runtime or binding with CORAL's normal
overrides, for example `agents.binding=my-codex`.

### Stock return ranking on a local Mac

`qlib_alpha_factor_icir/task.yaml` is configured for two `gpt-6-astra` Codex
agents in srt sandboxes, with a shared limit of 10 real evaluations. Its Docker
backend runs the unchanged upstream verifier in a fresh `--network none`
container, so it does not require Harbor's nftables support. It builds Linux
amd64 images because upstream pins a pyqlib release without Linux ARM wheels;
on Apple Silicon this requires Docker's amd64 emulation and can be slow.

Before agents start, workspace setup builds the public image and copies only
its public panels into ignored `data/` directories, then installs a local
Python environment. Agents run `.venv/bin/python selfcheck.py` in their srt
sandbox and submit with `.venv/bin/coral eval -m 'description'`. They do not
need Docker access. Local self-checks use native Python and may have numerical
differences from the Linux verifier. Hidden scoring still uses the original
Dockerfile, data generator, scoring code, and 30-minute verifier limit.

Only committed `methods/` files enter the verifier container; no host paths or
Docker sockets are mounted. Full diagnostics remain under `.coral/private/`.
Infrastructure failures raise grader errors and do not consume the real
evaluation budget. This repeated hidden evaluation mode differs from the
official benchmark's single final evaluation.

## Prepare task data

```bash
uv run examples/rsi_exam/prepare.py --list
uv run examples/rsi_exam/prepare.py game2048_policy_search
uv run coral validate examples/rsi_exam/game2048_policy_search
uv run coral start -c examples/rsi_exam/game2048_policy_search/task.yaml
```

All 35 task directories are checked in. Source code, instructions, initial
methods, and container configurations are vendored from the pinned dataset;
data assets listed in each `UPSTREAM.json` are downloaded on demand and ignored
by Git. `prepare.py` verifies their Git blob/LFS hashes and fills only missing
files. It preserves existing source files and edited baselines. The tidal task
already includes its full small source bundle, so preparation is a no-op.

To prepare from a local dataset snapshot, or prepare the full public suite:

```bash
uv run examples/rsi_exam/prepare.py game2048_policy_search --source /path/to/dataset
uv run examples/rsi_exam/prepare.py --all
```

`--all` prepares data; it does not start agents. The task data is hundreds of
megabytes, and GPU/model tasks may download much more during image builds.
Each task can be started independently with its own `task.yaml`.

The checked-in workspace paths assume commands run from the CORAL repository
root. `--output /path/to/new-tasks` can generate complete standalone copies at
a new location; those configs use absolute workspace paths. Existing tasks are
only hydrated, never regenerated in place. Offline new imports record
`local-source` and SHA-256 hashes rather than claiming a pinned revision.

## How agents experiment and submit

Inside a CORAL agent worktree:

```bash
# Extract baseline artifacts generated during the image build.
uv run rsi_runtime.py bootstrap

# Edit methods/main/solver.py locally, then run the original public evaluator.
uv run rsi_runtime.py run 'python /app/selfcheck.py'

# Use the command from instruction.md if a task uses a different evaluator.
uv run rsi_runtime.py run 'your training or evaluation command'

# Submit the final method after iterating on visible feedback.
coral eval -m 'final method'
```

Submission paths map directly: `/app/methods/main/solver.py` in the container
is `methods/main/solver.py` in the worktree. This also supports tasks declaring
`/app/submission` or `/app/experiment_log.md`. Public build inputs and support
code are under `harbor/environment/`; build-generated datasets remain in the
container. A visible command runs in `/app` and retrieves declared submission
artifacts, including trained outputs, into the worktree. Other generated files
are ephemeral. Its logs remain under `.rsi_runs/`, excluded from commits.

`coral eval` runs a transfer-only Harbor adapter: no second LLM agent is invoked.
It replaces the image's submission paths with the committed files, then Harbor
transfers the declared artifacts to a fresh **separate verifier container**.
Deleted files do not fall back to the baked-in baseline. Symlinks and special
files in submissions are rejected. Artifact destinations, scoring code,
resource settings, API environment declarations, and Compose overlays remain
upstream-controlled. The adapter currently accepts non-overlapping artifacts
below `/app` on the main service, which covers the pinned public task configs.

The score is the trial's `verifier_result.rewards.reward`, maximized without
renormalization or clipping. Some GPU tasks emit raw speedups above one. A
missing/non-finite reward, unfinished/failed trial, or shared verifier produces
no score. Private Harbor diagnostics are under `.coral/private/rsi_jobs/`;
only the numeric reward and a job identifier enter CORAL's public feedback.

`coral eval --tune` returns no score and never runs hidden tests. Use the visible
experiment command for development feedback instead.

## Protocol and isolation

The default config stops after **one finalized real attempt**. Perform local
visible experiments before that attempt. `coral validate` also evaluates the
seed on the hidden split, so use it as an operator preflight, not as an agent's
development loop.

You can allow repeated hidden evaluations with
`run.stop.max_real_attempts=30`, but that changes the experiment: it provides
feedback from the hidden split. Such results are not comparable to RSI-Exam's
single sealed evaluation. Even with one evaluation, this adapter is a CORAL
optimization workflow, not a reproduction of upstream's complete harness:
CORAL's host agent is not enclosed by Harbor's network boundary, its timeout
does not impose upstream's entire 12-hour/token-budget protocol, and no
official leaderboard-equivalence claim is made.

`rsi_task/` contains the unchanged upstream task, including **hidden tests**,
and is declared in `grader.private`. It is never copied into `seed/` or the
visible grader package. Do not give optimizing agents access to the source
dataset checkout or start them directly in this example directory. As with
other Docker-backed CORAL examples, the machine owner and processes with
unrestricted host/Docker access can inspect the containers and dataset; this
is not a hostile multi-tenant security boundary.

## GPU and API tasks

Use suitable Linux/NVIDIA hardware for GPU tasks and preserve the task's own
`environment/docker-compose.yaml` and `tests/docker-compose.yaml` overlays.
Set variables they require, such as `HARBOR_GPU_DEVICE` and `HARBOR_CPUSET`,
before starting CORAL. GPU timing scores require the upstream-calibrated
hardware. See the upstream
[GPU guide](https://github.com/aiming-lab/RSI-Exam/blob/main/docs/gpu_docker_no_network.md).

Four public tasks require task-side model API credentials. Export only the
variables needed by the selected task, following its `task.toml` and the
upstream [.env.local.example](https://github.com/aiming-lab/RSI-Exam/blob/main/.env.local.example):

| Task | Key |
| --- | --- |
| `locomo_longterm_memory` | `MEMORY_LLM_API_KEY` |
| `legal_matter_caseload_regulatory` | `APEX_LLM_API_KEY` |
| `lean_formal_proof_workflow_design` | `LEAN_LLM_API_KEY` |
| `discoveryworld_agent_harness_low2` | `DISCOVERYWORLD_LLM_API_KEY` |

For visible commands that call a model, explicitly allow the provider host:

```bash
uv run rsi_runtime.py run --allow-host api.openai.com 'python /app/selfcheck.py'
```

The original verifier host allowlists are retained. An endpoint change must
also be permitted by the task's network policy. Task-side credentials and
agent-runtime credentials are separate.

## Provenance and verification

Upstream files are MIT licensed; `UPSTREAM_LICENSE` preserves AIMING Lab's
copyright notice. `UPSTREAM.json` records the dataset revision and hashes of
every vendored file and the omitted data assets. Upstream scoring/data-generation
code is not rewritten. After editing `_grader/`, propagate the adapter with
`uv run python examples/rsi_exam/_scripts/sync_runtime.py`.

Review the hand-written integration in `_grader/`, `prepare.py`, the stock
return ranking workspace setup, and `tests/test_rsi_exam.py`. GitHub marks the
vendored upstream trees and generated adapter copies separately. Every task
remains standalone; generated copies are checked against the canonical source
by the test suite.

Offline adapter checks:

```bash
uv run pytest tests/test_rsi_exam.py -q
uv run ruff check examples/rsi_exam/_grader/src examples/rsi_exam/prepare.py tests/test_rsi_exam.py
```

These check the artifact boundary, deletions, private log placement, import
layout, pinned file integrity, and rejection of invalid trial results. Real
container evaluation requires the Docker/hardware/API prerequisites above;
offline tests alone do not verify benchmark scores.

Runtime verification on 2026-09-08:

- All 35 task layouts passed structural validation. The tests also check every
  vendored file against its pinned provenance manifest.
- Both unmodified tidal Docker images built. The visible self-check RMSE was
  `0.129427`; a fresh verifier with `--network none` returned baseline reward
  `0.0491640867`. This checks upstream code, not full Harbor orchestration.
  `coral validate examples/rsi_exam/tidal_friction_inverse` loaded the package
  but returned no score because the local Docker Desktop kernel lacks
  `CONFIG_NFT_FIB_INET` and Harbor rejects that network policy.
- A stock return ranking CORAL run using the Docker backend produced finite
  scores, including `0.261313`. Its srt workspaces could read public panels and
  import the scientific Python stack while private file access remained denied.
  The run was stopped by the operator before its 10-evaluation budget completed.
- The stock verifier can emit a non-finite reward. The adapter rejects it as a
  grader error instead of recording a fabricated numeric score. Apple Silicon
  emulation can make an evaluation approach its 30-minute limit.
- The other tasks, including GPU and task-side model API workflows, have not
  been exercised end to end. These checks do not establish official benchmark
  equivalence.

## Public tasks

Every row is an independent CORAL task under this directory. Run `prepare.py`
for the selected task before validation/start. The full names are also available
through `prepare.py --list`.

| Task | Description |
| --- | --- |
| [bbo_noisy_continuous](bbo_noisy_continuous/task.yaml) | Design a noise-aware optimizer for multimodal continuous functions under a fixed query budget and a comparable visible/sealed normalized anytime-final metric
| [bbo_simopt_inventory](bbo_simopt_inventory/task.yaml) | Design a query-efficient optimizer for a stochastic replenishment policy across varied cost and demand regimes
| [bigann_filtered_vector_search](bigann_filtered_vector_search/task.yaml) | Build a filtered-ANN index/search solver over the real Big-ANN yfcc-10M library (10M x 192 uint8 vectors, 200,386 tags, 108M tag entries); graded by wall-clock throughput of one batch call at…
| [citywide_signal_coordination](citywide_signal_coordination/task.yaml) | Coordinate signal timing across a full metropolitan road network (~600 signalised junctions) to minimise a weighted delay/teleport/stop cost under heavy demand; you submit a signal program /…
| [clevr_cogent_grpo_qwen2vl](clevr_cogent_grpo_qwen2vl/task.yaml) | Train Qwen2-VL-2B-Instruct with GRPO on CLEVR-CoGenT-A so it generalizes to hidden CLEVR-CoGenT-B and SuperCLEVR; graded by mean keyword-matching counting accuracy
| [cxr_ood_triage_policy](cxr_ood_triage_policy/task.yaml) | Improve a weak chest-X-ray triage policy over frozen classifier scores
| [device_iv_regime_extrapolation](device_iv_regime_extrapolation/task.yaml) | Power-device qualification sign-off: submit a prediction method that turns safe-window I-V/temperature measurements of a diode lot into terminal-current commitments at mission-profile extremes…
| [discoveryworld_agent_harness_low2](discoveryworld_agent_harness_low2/task.yaml) | Improve a ReAct harness over a fixed language model on low-baseline Space Sick and Chemistry Challenge tasks
| [eda_gate_sizing](eda_gate_sizing/task.yaml) | Choose a library cell per instance to minimize power under timing/DRV constraints on placed netlists (CircuitOps IR tables); submit an algorithm, graded on sealed hidden designs, lower is better
| [feeder_phase_impedance_inversion](feeder_phase_impedance_inversion/task.yaml) | Distribution-feeder model calibration before hosting-capacity studies: submit a method that jointly repairs per-customer phase labels, per-segment conductor impedances and the regulator tap from…
| [finscope_dcf_valuation](finscope_dcf_valuation/task.yaml) | Forecast forward value-drivers for a coverage universe from a messy fundamentals panel; graded by median absolute DCF valuation error on a sealed held-out tier
| [flashattention_varlen_feature_full_vjp_speedup](flashattention_varlen_feature_full_vjp_speedup/task.yaml) | Optimize six preregistered D160 noncausal-global ragged attention strata per panel with 48–96:1 GQA, ALiBi, softcap, non-dense packed views, and complete Q/K/V VJP
| [flashfftconv_multistream_gated_forward_speedup](flashfftconv_multistream_gated_forward_speedup/task.yaml) | Optimize independent multi-stream gated causal convolutions in the published FP16 D=768, length-2048 regime across multiple batch/stream/head factorizations on NVIDIA H100
| [game2048_policy_search](game2048_policy_search/task.yaml) | Improve a Python policy for seeded 2048 games, using public games for feedback and sealed same-distribution seeds for final scoring
| [johnson1991_leighton_graph_coloring](johnson1991_leighton_graph_coloring/task.yaml) | Minimize coloring conflicts at a fixed color budget on real, hard DIMACS/COLOR02 graphs (Leighton le450, Johnson DSJC, DSJR; n=450..1000)
| [lean_formal_proof_workflow_design](lean_formal_proof_workflow_design/task.yaml) | Design one verifier-guided workflow that proves fixed formal theorem statements with a pinned language model under per-problem inference, compiler-check, and wall-clock budgets
| [legal_matter_caseload_regulatory](legal_matter_caseload_regulatory/task.yaml) | Work six unrelated legal matters end-to-end on real case files (1,000+ documents: Senior Living Lending telemarketing/TCPA compliance, Harborview spinoff + cross-border data transfer, a…
| [locomo_longterm_memory](locomo_longterm_memory/task.yaml) | Redesign a minimal conversational memory system (LLM extraction + BM25-only retrieval + concise answering) so it generalizes: iterate freely on 4 visible multi-session conversations (812 QA); a…
| [lung_fewshot_celltype_annotation](lung_fewshot_celltype_annotation/task.yaml) | Improve a five-shot lung cell-type classifier using unlabeled scRNA-seq cells; graded by macro-F1 after replay on donor-disjoint sealed cells
| [mbff_banking_placement](mbff_banking_placement/task.yaml) | Optimize multibit flip-flop banking and placement to minimize a weighted power/area/timing/displacement cost on industrial-style testcases; graded on sealed hidden cases, lower is better
| [multiview_dense_reconstruction](multiview_dense_reconstruction/task.yaml) | Register several noisy, partially-overlapping 3D point sets that are not in a common coordinate system -- each view carries an unknown rigid pose relative to the first -- and fuse them, with…
| [paged_ragged_gqa_decode_speedup](paged_ragged_gqa_decode_speedup/task.yaml) | Optimize FP16/BF16 grouped-query decode attention over a paged, ragged KV cache on NVIDIA H100
| [pbmc_batch_correction](pbmc_batch_correction/task.yaml) | Produce a batch-integrated cell embedding from binary PBMC scATAC-seq peaks; graded by cell-type NMI (scIB Leiden protocol) on a sealed split
| [perturbed_cell_morphology_generation](perturbed_cell_morphology_generation/task.yaml) |  |
| [protein_ligand_cofolding_posebusters](protein_ligand_cofolding_posebusters/task.yaml) | Improve an inference-only protein-ligand co-folding pipeline; the isolated verifier re-runs submitted source on a sealed structural split and scores physically valid, geometrically accurate poses
| [qec_decoder_arena](qec_decoder_arena/task.yaml) | Decode quantum error-correction color codes
| [qlib_alpha_factor_icir](qlib_alpha_factor_icir/task.yaml) | Build a cross-sectional return-prediction model on an anonymized daily factor panel; graded by ICIR on a hidden later period
| [roadef_glass_cutting](roadef_glass_cutting/task.yaml) | Cut a batch of rectangular glass items out of the fewest 6000x3210 stock plates on a guillotine line, minimising wasted area, on an unseen unseen industrial instance
| [robotap_switch_budget_candidate_routing_optimization](robotap_switch_budget_candidate_routing_optimization/task.yaml) | Optimize semantic candidate routing and visibility under a four-switch temporal budget, scored by mean per-video Average Jaccard on a sealed split
| [splash_attention_strata_speedup](splash_attention_strata_speedup/task.yaml) | Given naive-XLA masked GQA attention on TPU v6e, write a faster kernel; graded by per-case speedup on sealed shape strata, correctness gated
| [tddn_contraction_planning](tddn_contraction_planning/task.yaml) | Deterministic inference-time planning for exact Tensor Decision Diagram Network contractions in quantum-circuit equivalence checking
| [teacher_student_math_posttraining](teacher_student_math_posttraining/task.yaml) | Post-train a fixed Qwen3-1.7B Base student with a fixed 8B teacher and a sanitized mathematical-reasoning corpus; submit merged model weights scored by sealed exact-answer accuracy
| [tidal_friction_inverse](tidal_friction_inverse/task.yaml) | Estimate a heterogeneous seabed friction field from sparse tide gauges and generalize to sealed gauges
| [trifinger_offline_rl_push_refined](trifinger_offline_rl_push_refined/task.yaml) | Learn a cube-pushing policy for the TriFinger robot from a fixed offline dataset (no live simulator access during training)
| [warehouse_robot_macro_routing](warehouse_robot_macro_routing/task.yaml) | Drive a warehouse cleanup robot (forward / turn / swap, plus a record-once / replay-many macro button) to sort every ball into its matching basket in as few button presses as possible, on an…
