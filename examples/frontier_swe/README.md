# Frontier-SWE tasks

Four long-horizon systems tasks from
[Proximal-Labs/Frontier-SWE](https://github.com/Proximal-Labs/frontier-swe),
adapted to CORAL without vendoring the upstream verifier or task images.

Each task starts from a non-zero candidate. The candidate is stored as a
cumulative text bundle so a CORAL agent can change a multi-file project in
one artifact:

```text
=== FILE: relative/path ===
complete file contents
=== END FILE ===
```

The seed candidates retain the source project's MIT terms; see
[`LICENSE.seed-candidates`](LICENSE.seed-candidates). The Dart-to-Haskell
bundle also carries its own BSD-3-Clause file as part of the materialized
candidate.

The packaged grader checks the bundle, fetches the upstream task at commit
`111464af7933002a9192240798cbcf65d2790296` into `.coral/private/`, pins its
official container image by manifest digest, and asks
[Harbor](https://harborframework.com/) to run the official verifier. The
grader's Harbor agent only materializes the candidate inside the official task
workspace; it does not use an LLM or alter the verifier.

[`_grader/`](_grader/) is the source-of-truth grader package. Each task carries
an identical `grader/` copy so the task remains self-contained when copied or
mounted on its own.

## Tasks

| Task | Candidate target | Validated seed score | Typical verifier time |
|---|---|---:|---:|
| [Git to Zig](git-to-zig/task.yaml) | `/app/zig-port` | 0.197833 | ~13 min |
| [Lua Native Compiler](lua-native-compiler/task.yaml) | `/app/lua-native-compiler` | 0.406593 | ~2 min |
| [libexpat to x86-64 assembly](libexpat-to-x86asm/task.yaml) | `/app/asm-port` | 1397.818394 | <1 min |
| [dart_style to Haskell](dart-haskell/task.yaml) | `/app/dart-style` | 0.0546 | ~2 min |

The seed scores above were observed with the pinned task and image. The
libexpat reward includes a performance component and therefore varies by
machine; runtime also varies with image pulls and available compute.

## Requirements

- `git` and `uv` on the CORAL host.
- One of:
  - Docker with enough capacity for the task's official image (the default), or
  - a configured Modal account. Set
    `CORAL_FRONTIER_SWE_HARBOR_ENVIRONMENT=modal` before validation or a run.
- Network access for the grader to fetch the pinned upstream task and for
  Harbor to obtain the official image. The task container itself keeps the
  upstream `allow_internet = false` policy.

Harbor is pinned and launched through `uvx --python 3.12`, so it remains
separate from both CORAL and the grader venv.

## Validate and run

Validate one baseline end to end:

```bash
CORAL_FRONTIER_SWE_HARBOR_ENVIRONMENT=modal \
  uv run coral validate examples/frontier_swe/git-to-zig
```

Start a normal CORAL run (Docker is used unless the environment variable above
selects Modal):

```bash
uv run coral start -c examples/frontier_swe/git-to-zig/task.yaml
```

Official evaluations are intentionally expensive. The grader uses one Harbor
trial at a time, stores Harbor output under the attempt's `eval_logs/`, and
never writes generated artifacts into the candidate checkout.

## Upstream assets and isolation

The CORAL repository contains only the task adapters and the contributed seed
candidates. It does not copy Frontier-SWE instructions, tests, reference trees,
or container files. Those assets are resolved from the pinned upstream commit
at evaluation time and cached only under `.coral/private/`, which agents cannot
read. This also keeps verifier changes from silently changing scores.
