# Frontmatter spec — the full structured-trace schema

Read this when you need the authoritative list of every field, its
vocabulary, and which subsystem reads it. The main SKILL.md keeps a
condensed view; this file is the source of truth.

The schema is **expected for every new note**. The parser leaves every
field optional only so legacy data from earlier runs still loads —
that's a backward-compat concession for old data, not a license to
skip fields when you write a new note. Per-variant required-vs-optional
mapping is in the table near the end of this file.

## Per-field reference

| Field | Type | Vocabulary / format | What it means | Consumed by |
|---|---|---|---|---|
| `creator` | string | `<agent_id>` (kebab-case) | Author of this note. Missing → surfaced as the sentinel `unknown` in list views; team-level filters skip the note. | `coral notes`, `notes_by`, librarian, consolidate roster, migration attribution |
| `created` | string | ISO-8601 (`2026-06-25T14:32:00Z`) | When the note was written. Missing → list/graph sort falls back to file mtime. | sort order in `coral notes`, knowledge-graph node date |
| `commit` | string | git hash, or `"n/a"` | The `coral eval` commit this note describes (Variant A). | UI links, attempt-to-note crosslinks |
| `generation` | int | ≥ 0 | Bump when a focus note's direction meaningfully shifts (Variant C). | consolidate roster — stable, high-generation focus notes are signals of committed specialization. |
| `type` | enum | `experiment \| hypothesis \| dead_end \| open_question \| synthesis` | What kind of claim this is. Picks the node category in the knowledge graph. | `notes_graph` node `type`, color/shape in the dashboard graph view |
| `claim` | string | one testable sentence | The thing this note asserts. Not a title — a falsifiable statement. | future search / aggregation; lint enforces presence per-variant |
| `status` | enum | `confirmed \| refuted \| untested` | Author-reported claim status, separate from system evidence checks. | knowledge-graph node color; consolidate filters |
| `confidence` | enum | `low \| medium \| high` | How confident the team should be in this claim. Discrete on purpose — LLM-written floats aren't calibrated, and a 3-bucket vocabulary is what agents agree on across runs. | knowledge-graph node size; sort order in synthesis views |
| `based_on` | string OR list[string] | attempt hash(es) | The graded artifact(s) this builds on. Use a YAML list when this work draws on more than one prior attempt — single-prior is the common case but not the only one. | knowledge-graph `based_on` edge (planned), attempt → note crosslink |
| `evidence` | dict | `{attempt, baseline, score_delta, verified}` | `attempt` and `baseline` identify the result and explicit comparison attempts. `score_delta` is result minus baseline, regardless of score direction. `verified` is the author's replication assertion. | `coral notes --audit` checks records and arithmetic independently; lint warns if a delta lacks a baseline or confirmed lacks verified |
| `supersedes` | list[string] | note paths or slugs | Prior notes this one replaces. Use this instead of overwriting — the graph then carries the lineage. | knowledge-graph `supersedes` edges (typed) |
| `refutes` | list[string] | note paths or slugs | Prior notes whose claim this one disproves. | knowledge-graph `refutes` edges (typed) |
| `touched` | list[string] | file paths in the repo | Code files this work modified. Helps future agents grep for related work. | future blame integration; search |
| `tags` | list[string] | free-form kebab-case | Topic tags for filtering. | future tag-based search |
| `next` | list[string] | one-line action items | Concrete next steps in descending expected payoff. Mirrors the body's "## Next" section. | future planner suggestions |

## Vocabularies in detail

### System evidence audit

`coral notes --audit` and the dashboard report `evidence`, `numeric`,
`comparability`, and `replication` checks separately from these author fields.
Each check is `passed`, `failed`, or `unchecked`. Numeric matching does not
verify causality. `based_on` and `parent_hash` are never inferred as baselines.
Quote all-numeric hashes in YAML; use full hashes or unique prefixes of at
least four hex characters. Old notes without structured evidence remain readable.

Reports are computed from fresh snapshots and contain note/attempt content
hashes. They are returned separately and never written into the source notes.
See the CLI reference for the optional attempt evaluation-context contract.

### `type:`

- **`experiment`** — single eval (or small set of related evals) reflection. Default for the reflect heartbeat. Use for both per-eval reflections and infra diagnostics.
- **`hypothesis`** — declared bet, not yet tested. Default for the pivot heartbeat's focus note. Should pair with `status: untested` until you have evidence.
- **`dead_end`** — an approach the team should not retry. Pair with `refutes:` pointing at the note that proposed it (if any).
- **`open_question`** — a knowledge gap or unresolved contradiction. Use when the entry is significant enough to live in its own file (sections in `_open-questions.md` don't need their own type — the file owns it).
- **`synthesis`** — distilled team belief across 3+ evidence points. Default for the consolidate heartbeat's `_synthesis/` notes.

### `status:`

- **`confirmed`** — verified evidence supports the claim. Requires `evidence.verified: true` to make sense (lint warns if missing).
- **`refuted`** — verified evidence contradicts the claim. Often paired with `refutes:` pointing at the note that proposed the (now-refuted) claim.
- **`untested`** — no evidence yet. The honest default for a hypothesis / focus note.

### `confidence:`

Three levels. Pick based on what you actually know, not on how excited you are:

- **`high`** — multiple verified evidence points agree, mechanism is understood, and you'd be surprised to see a counter-example. Synthesis notes with 3+ confirming attempts typically land here.
- **`medium`** — the default when you have some evidence but room for surprise. A single confirmed eval usually means `medium`, not `high`.
- **`low`** — you'd bet against the claim if pushed, or there's no evidence behind it yet but you still want the note on record (a hypothesis you're not committed to, a hunch that didn't pan out).

If you have *nothing* to base a confidence on, omit the field. Don't put a placeholder — lint will catch the empty `<placeholder>` and flag it as not in the vocabulary.

Why no numerics: LLM-written floats (`0.7`, `0.83`) aren't calibrated — the difference between 0.7 and 0.8 is noise, not signal. Discrete buckets are honest about the resolution actually achievable, and stay consistent across agents (your `0.7` may be another agent's `0.5`, but you'd both agree on `medium`).

## Rules of thumb

- **`claim:` is one testable sentence**, not a title — "matmul tile=32 improves score" beats "matmul experiments."
- **`status:` follows the evidence**: `confirmed` only with `evidence.verified: true`; downgrade to `untested` if the grader hasn't seen it yet.
- **`confidence:` is grounded in evidence, not enthusiasm** — a focus note declaring `high` on day one without evidence is a signal you haven't actually tested anything. Default to `medium` and earn `high` with verified results.
- **Use `supersedes:` instead of overwriting**: write a new synthesis note that points at the old one. The graph then shows the lineage; the old claim isn't silently lost.
- **Body links count too**: a free-text note that writes `Based on [eval-12](experiments/eval-12-simd.md)` already shows up as a `references` edge in the graph. The trace schema is for typed edges (`supersedes` / `refutes`); free-text links are for everything else.
- **Link every prior that informed the work, not just the immediately previous eval.** The default failure mode is a chain (`eval-N → eval-N-1 → eval-N-2`) — every note refers to one parent, the graph is just a line, and a reader who lands on eval-5 has no idea that eval-2's mechanism was also load-bearing. Ask: "what notes did I actually read while designing this experiment?" — list each one in the body `## References` section as a `[label](path.md)` link. For `based_on:` in the frontmatter, prefer a YAML list with every prior attempt this work draws on, even if one is dominant.

## Sentinel: `creator: unknown`

A note without a `creator:` field — or with a blank one — is rendered with
`creator: unknown` in `coral notes` and in the knowledge-graph node, and
appears in `scripts/unattributed.py` output. The sentinel is **not** an alias
for any real agent; `notes_by("unknown")` returns nothing because the lookup
re-parses raw frontmatter and only matches files that actually wrote a
`creator:` line. The visible tag is the signal — if you see your own note
marked `(unknown)`, add a `creator:` line to fix it.

## Per-variant required-vs-optional mapping

Bold = required for that variant. Everything else is add-when-applicable.

| Variant | `type:` | Required trace fields | Add when applicable |
|---|---|---|---|
| A — experiment (reflect heartbeat) | `experiment` | **`claim`**, **`status`**, **`confidence`**, **`evidence`** (`attempt` + `verified`) | `based_on`, `touched`, `next`, `tags` |
| B — infra (diagnostic) | `experiment` | **`claim`** (the workaround/fix), **`status`**, **`touched`** | `next`, `tags` |
| C — focus (pivot heartbeat) | `hypothesis` | **`claim`** (your bet), **`status: untested`**, **`confidence`** (your prior) | `based_on`, `tags` |
| D.1 — synthesis (consolidate heartbeat) | `synthesis` | **`claim`**, **`status`**, **`confidence`** | `supersedes` (prior synthesis you replace), `refutes`, `tags` |
| D.3 — open question (standalone file) | `open_question` | **`claim`** (the question) | `based_on`, `tags` |
| Dead end you want flagged for the team | `dead_end` | **`claim`** (what doesn't work), **`refutes`** (the note that proposed it) | `evidence`, `tags` |

For D.2 (entries appended to `_connections.md`) and D.3 entries that stay
inside `_open-questions.md`, the trace schema doesn't apply — those files
are aggregates, not per-claim notes.

## Verification

The skill bundles `scripts/lint.py` which mechanizes every check above —
required-field presence, vocabulary membership, the `status: confirmed`
without `evidence.verified: true` pairing, filename conventions, and the
`type:` vs path agreement. Run it before considering a note done; it's
advisory by default (`--strict` for exit-code enforcement).
