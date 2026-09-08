# Examples

Example task configurations for CORAL. Each directory contains a `task.yaml` and any supporting files (graders, seed code, data).

To run any example:

```bash
coral start --config examples/<name>/task.yaml
```

Every task ships its grader as a standalone Python package — CORAL
bootstraps a fresh venv at `.coral/private/grader_venv/`, installs the
package via `grader.setup`, and runs evaluation in a worker subprocess
from there. No extra install step is required for the user.

## Task Structure

All tasks use the **packaged grader** layout:

```
my_task/
├── task.yaml              # references grader.entrypoint + grader.setup
├── seed/                  # initial codebase given to agents
│   └── solution.py
└── grader/                # standalone Python package
    ├── pyproject.toml
    └── src/my_task_grader/
        ├── __init__.py
        └── grader.py      # class Grader(TaskGrader): ...
```

### `task.yaml`

The central configuration file. All available fields (with defaults):

```yaml
task:
  name: "My Task"                    # Task name (required)
  description: |                     # Full problem description shown to agents (required)
    What the agent should do.
  tips: |                            # Hints shown to agents (timeouts, constraints, etc.)
    - Eval timeout is 120s.

grader:
  entrypoint: "my_task_grader.grader:Grader"   # required
  setup:                                        # shell commands run in .coral/private/grader_venv/
    - "uv pip install -e ./grader"
  timeout: 300                       # Max seconds per evaluation (default: 300)
  direction: maximize                # "maximize" or "minimize" (default: maximize)
  args:                              # Arbitrary kwargs accessible as self.args inside the grader
    program_file: "solution.py"
  private:                           # Files/dirs copied to .coral/private/ (hidden from agents)
    - "answers/"

agents:
  count: 1                           # Number of concurrent agents (default: 1)
  runtime: claude_code               # "claude_code", "opencode", or "codex" (default: claude_code)
  model: sonnet                      # Model name or path (default: sonnet)
  max_turns: 0                       # Max agent turns per session (default: 0 = no cap)
  timeout: 3600                      # Agent-level timeout in seconds (default: 3600)
  research: true                     # Enable web search / literature review (default: true)
  stagger_seconds: 0                 # Delay between spawning each agent (default: 0)
  gateway:                           # LiteLLM gateway for routing model traffic
    enabled: false
    port: 4000
    config: "./litellm_config.yaml"
    api_key: ""                      # Auto-generated if empty
  heartbeat:                         # Periodic agent self-reflection actions
    - name: reflect
      every: 1                       # Trigger every N evals
      global: false                  # false = per-agent count, true = global count
      trigger: interval              # "interval" or "plateau"
    - name: consolidate
      every: 10
      global: true
      trigger: interval
    - name: pivot
      every: 5
      trigger: plateau               # Triggers after N evals with no improvement

sharing:
  attempts: true                     # Share attempt history across agents (default: true)
  notes: true                        # Share notes across agents (default: true)
  skills: true                       # Share skills across agents (default: true)

workspace:
  repo_path: "./examples/my_task/seed"  # Path to seed directory (default: ".")
  results_dir: "./results"              # Where run outputs are stored (default: ./results)
  setup:                                # Shell commands run once per worktree before agents start
    - "uv pip install numpy scipy"

run:
  verbose: false                     # Verbose output (default: false)
  ui: false                          # Launch web dashboard (default: false)
  session: tmux                      # "local", "tmux", or "docker" (default: tmux)
  docker_image: ""                   # Custom docker image; empty = auto-build from docker/<runtime>/ (default: "")
```

### `grader/` (packaged)

A standalone Python package consumed by `grader.entrypoint`. Inherits from
`TaskGrader` and implements `evaluate()`. CORAL installs it into
`.coral/private/grader_venv/` so its dependencies stay separate from CORAL's
and from each agent's worktree env.

Hidden data the grader must keep from agents (answer keys, hidden test
fixtures) is declared under `grader.private` in task.yaml and read via
`self.private_dir` — CORAL copies those paths into `.coral/private/`, which
every agent runtime is denied read access to. Keep those paths **outside** the
`grader/` package (conventionally a sibling `taskdata/`, declared as
`taskdata`), because **everything inside `grader/` is visible to agents**: the
whole grader source is surfaced read-only at `<shared_dir>/grader/` so they can
read how they're scored, so a `grader.private` path inside `grader/` would leak
— `coral validate` errors on that. Non-secret bundled data (lookup tables,
helper modules) may live inside `grader/` and be read via `Path(__file__).parent`,
but it is visible — never put a secret there.

### `seed/`

The starting codebase that gets copied into each agent's git worktree. This is what agents see when they begin working. It should contain starter code (with a function signature the grader expects) and any data files the solution needs at runtime.

## Migrated examples

These examples serve as reference implementations of the packaged grader
form:

- `erdos/grader/` — minimal package, single grader file, numpy dep
- `drug_design/grader/` — package with bundled scorer modules + reference
  data files; demonstrates `[ml]` optional-deps for heavy dependencies

To inspect or hack on either, the package can be installed standalone for
local development:

```bash
uv pip install -e ./examples/erdos/grader
uv pip install -e ./examples/drug_design/grader[ml]
```

## Examples Overview

| Example | Description | Direction |
|---------|-------------|-----------|
| [circle_packing](#circle_packing) | Pack 26 circles in a unit square to maximize sum of radii | Maximize |
| [erdos](#erdos) | Erdos minimum overlap problem | Maximize |
| [kernel_builder](#kernel_builder) | Optimize VLIW SIMD kernel for tree traversal | Minimize |
| [kernel_engineering](#kernel_engineering) | Triton kernel for Triangle Multiplicative Update (AlphaFold3) | Maximize |
| [mnist](#mnist) | Handwritten digit classification (accuracy) | Maximize |
| [dlomix-pfly](#dlomix-pfly) | Pfly 4-class peptide detectability (accuracy) | Maximize |
| [spaceship_titanic](#spaceship_titanic) | Kaggle: predict passenger transportation (accuracy) | Maximize |
| [stanford_covid_vaccine](#stanford_covid_vaccine) | Predict mRNA degradation rates (MCRMSE) | Minimize |
| [math](#math) | 17 mathematical optimization problems | Maximize |
| [ADRS](#adrs) | 5 systems optimization problems (scheduling, placement, etc.) | Maximize |
| [frontier_cs_algo](#frontier_cs_algo) | 172 algorithmic competition problems (C++) | Maximize |
| [frontier_cs_research](#frontier_cs_research) | 127 research-level CS problems (Python) | Maximize |
| [frontier_eng](frontier_eng/README.md) | 74 generative-optimization benchmarks ported from EinsiaLab/Frontier-Engineering (KernelEngineering, JobShop, Optics, Cryptographic, ...) | Maximize |
| [frontier_swe](frontier_swe/README.md) | 4 long-horizon systems ports evaluated by pinned official Frontier-SWE tasks through Harbor | Maximize |
| [dna_design](#dna_design) | Design cell-type-specific DNA enhancer sequences (SAGA) | Maximize |
| [drug_design](#drug_design) | Design novel small-molecule antibiotics (SAGA) | Maximize |
| [swebench-verified](#swebench-verified) | Optimize a solver program across 500 SWE-bench instances | Maximize |
| [rsi_exam](#rsi_exam) | 35 separate research tasks, replayed in their original isolated verifiers | Maximize |
| [terminal-bench](#terminal-bench) | Optimize a solver agent for terminal/shell tasks | Maximize |

## Details

### circle_packing

Pack N=26 circles into a unit square to maximize the sum of radii. The benchmark is 2.635977 (AlphaEvolve result). Score = `sum_radii / 2.635977`.

- **Agents**: 1 (OpenCode runtime)
- **Timeout**: 600s

### erdos

Find a step function h: [0,2] -> [0,1] that minimizes the maximum overlap integral. State-of-the-art bound is 0.380871.

- **Agents**: 1
- **Timeout**: 1100s

### kernel_builder

Optimize instruction scheduling for a VLIW SIMD kernel running tree traversal. The simulator is frozen; only the instruction-building code can be changed. Baseline: 147,734 cycles, best known: 1,363 cycles.

- **Agents**: 1
- **Timeout**: 120s
- **Session**: Docker

### kernel_engineering

Implement a fused Triton kernel for the Triangle Multiplicative Update operation from AlphaFold3 (LayerNorm + projections + gating + einsum). Scored by `1000 / geometric_mean(runtime_us)` across 7 configurations.

- **Agents**: 1
- **Timeout**: 1200s

### mnist

Classify 28x28 handwritten digit images. 60k training / 10k test samples. Scored by accuracy.

- **Agents**: 4
- **Timeout**: 300s

### dlomix-pfly

Predict 4-class peptide detectability (Non-Flyer / Weak / Intermediate / Strong) from amino-acid sequence alone — the task introduced by the [Pfly model](https://pubs.acs.org/doi/10.1021/acs.jproteome.4c00973) (Abdul-Khalek et al., J. Proteome Res. 2025), packaged in [dlomix](https://github.com/wilhelm-lab/dlomix). Source dataset: [`Wilhelmlab/detectability-proteometools`](https://huggingface.co/datasets/Wilhelmlab/detectability-proteometools), frozen as parquet splits (236k train / 59k val / 33k test, class-balanced). Hidden test labels live inside the grader package.

- **Agents**: 1
- **Timeout**: 600s
- **Scoring**: Accuracy (macro-F1 + per-class F1 reported as feedback)
- **Baselines**: Random ≈ 0.25; AA-composition + LogReg seed ≈ 0.30–0.35; Pfly (BiGRU) ≈ 0.78
- **Packaged grader**: `dlomix_pfly_grader`

### spaceship_titanic

Kaggle competition: predict which passengers were transported to an alternate dimension. 13 features, binary classification. Top leaderboard accuracy: ~0.828.

- **Agents**: 4
- **Timeout**: 120s

### stanford_covid_vaccine

Predict mRNA degradation rates at each base position. Scored by Mean Columnwise RMSE over three degradation targets.

- **Agents**: 4
- **Timeout**: 300s

### math

17 mathematical optimization problems including circle packing variants, Heilbronn triangle problems, hexagon packing, autocorrelation inequalities, and point placement problems. Each runs 2 agents with a 600s timeout.

### ADRS

5 systems optimization problems:

- **Cloudcast** -- minimize data transfer cost across multi-cloud networks
- **Prism** -- LLM model placement on GPU clusters (minimize KV cache pressure)
- **EPLB** -- expert parallelism load balancing for Mixture-of-Experts
- **LLM-SQL** -- column reordering to maximize LLM prefix cache hits
- **Transaction Scheduling** -- minimize makespan for concurrent transactions

### frontier_cs_algo

172 algorithmic competition problems. Solutions are self-contained C++17 files compiled with `g++ -std=c++17 -O2`. Each is scored 0-100 across 70 test cases. The grader is shared from `examples/frontier_cs_algo/_grader/` (each task directory carries a copy) and depends on the external `frontier_cs` package, installed in `grader.setup` via `uv pip install git+https://github.com/FrontierCS/Frontier-CS.git`; evaluation additionally requires a local go-judge instance running in Docker.

### frontier_cs_research

127 research-level CS problems with multiple variants (e.g. scheduling under different availability/deadline/overhead configurations). Solutions in Python with 1800s timeouts. The grader is shared from `examples/frontier_cs_research/_grader/` (each task directory carries a copy) and depends on the external `frontier_cs` package, installed in `grader.setup` via `uv pip install git+https://github.com/FrontierCS/Frontier-CS.git`.

### frontier_eng

74 generative-optimization benchmarks ported from [EinsiaLab/Frontier-Engineering](https://github.com/EinsiaLab/Frontier-Engineering) — KernelEngineering, JobShop families, Optics (16 sub-tasks), Cryptographic, Robotics, EngDesign, etc. All share one packaged grader (`examples/frontier_eng/_grader/`) that reads each benchmark's upstream `frontier_eval/` metadata, runs the documented eval command in a sandbox, and parses `metrics.json`. See [examples/frontier_eng/README.md](frontier_eng/README.md) for the full task list, runtime-environment notes, and regeneration instructions.

- **Tasks**: 74 (one CORAL example per upstream leaf benchmark; EngDesign is one example with 7 sub-instances)
- **Direction**: Maximize (combined_score per upstream contract)
- **Grader**: `frontier_eng_grader.grader:Grader`
- **Generator**: `examples/frontier_eng/_scripts/generate_tasks.py` (re-runnable from a fresh upstream checkout)

### frontier_swe

Four long-horizon systems candidates adapted from Frontier-SWE: Git v2.47.0
in Zig, a Lua 5.4 native compiler, libexpat 2.6.4 in x86-64 assembly, and
dart_style 3.1.4 in Haskell. Each seed is a non-zero cumulative source bundle.
The packaged grader fetches a pinned upstream task into `.coral/private/` and
uses Harbor to materialize the bundle in a digest-pinned official image and
run its verifier; no upstream tests or task images are vendored in CORAL. See
[examples/frontier_swe/README.md](frontier_swe/README.md) for prerequisites,
seed scores, and validation commands.

- **Tasks**: 4
- **Direction**: Maximize (official Frontier-SWE reward)
- **Grader**: `frontier_swe_grader.grader:Grader`
- **Runtime**: Harbor with Docker (default) or Modal

### dna_design

Design 200bp DNA enhancer sequences highly active in HepG2 (liver) cells while minimizing off-target activity. Adapted from the [SAGA](https://github.com/btyu/SAGA) benchmark ([Du et al., 2025](https://arxiv.org/abs/2512.21782)).

- **Agents**: 1
- **Timeout**: 600s
- **Scoring**: GC content + diversity (always); Enformer-based expression prediction (optional)

### drug_design

Design novel small-molecule antibiotics against K. pneumoniae with drug-like properties. Adapted from the [SAGA](https://github.com/btyu/SAGA) benchmark ([Du et al., 2025](https://arxiv.org/abs/2512.21782)).

- **Agents**: 1
- **Timeout**: 600s
- **Scoring**: QED + novelty + PAINS filter (always); MiniMol activity + ChemProp toxicity (optional)

### swebench-verified

Meta-solver optimization: agents improve a `solve.py` that wraps a Terminus2-based Harbor agent for fixing real GitHub bugs. The grader calls `harbor run -d swe-bench-verified` with the solver as a custom agent, using tiered evaluation (5 → 30 → all). Harbor handles repo setup, patch application, and test execution.

- **Agents**: 1
- **Timeout**: 14400s (4h for full eval)
- **Scoring**: Fraction of instances solved (pass rate)
- **Setup**: `uv pip install anthropic`; requires Harbor CLI (`uvx harbor`)
- **Baseline**: Terminus2 agent architecture (tmux-based multi-turn interaction)
- **Note**: Harbor runs Docker containers, so CORAL must run on the host (no Docker-in-Docker)

### rsi_exam

RSI-Exam's 35 public research tasks, each with its own configuration, seed, and
verifier under `rsi_exam/`. See [rsi_exam/README.md](rsi_exam/README.md) for Docker
prerequisites, data preparation, and the distinction from the official protocol.

- **Agents**: 1 by default; stock return ranking uses 2 sandboxed Codex agents
- **Budget**: 1 real evaluation by default; stock return ranking allows 10
- **Timeout**: per-task upstream build and verifier budgets, plus orchestration overhead
- **Session**: local; Docker runs task and verifier containers

### terminal-bench

Meta-solver optimization: agents improve a `solve.py` that wraps a Terminus2-based Harbor agent for completing terminal/shell tasks in Docker containers. The grader calls `harbor run -d terminal-bench@2.0` with the solver as a custom agent, and parses the job results.

- **Agents**: 1
- **Timeout**: 7200s (2h)
- **Scoring**: Pass rate across terminal-bench tasks
- **Setup**: `uv pip install anthropic`; requires Harbor CLI (`uvx harbor`)
- **Baseline**: Terminus2 agent architecture (tmux-based multi-turn interaction)
- **Note**: Harbor runs Docker containers, so CORAL must run on the host (no Docker-in-Docker)

## Writing Your Own

The quickest way to scaffold a new task is `coral init my-task`. To do it manually, create the three pieces:

1. **`seed/`** -- starter code with the function signature your grader will call
2. **`grader/`** -- a Python package whose `grader.py` holds a `TaskGrader` subclass that imports and runs the agent's code, then returns a score
3. **`task.yaml`** -- wire them together with `grader.entrypoint` + `grader.setup` and the config above

Use `coral validate my-task` to test your grader before launching agents.
