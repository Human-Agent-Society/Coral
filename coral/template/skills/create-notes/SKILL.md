---
name: create-notes
description: "Write a note to {shared_dir}/notes/ that future agents can actually act on. Use after every coral eval, when a heartbeat (reflect / consolidate / pivot) asks for a note, or when you discover a grader / build / runtime issue that future agents will hit. Covers 4 note variants (experiment / infra / focus / synthesis), the required frontmatter with the team-level `creator:` filter, the structured-trace schema (`type` / `claim` / `status` / `confidence` / `based_on` / `evidence` / `supersedes` / `refutes` / `touched`) that populates the dashboard knowledge graph, the self-audit checklist (backfilled predictions, abandoned paths, sourced magic numbers, cross-links), the bundled `scripts/{stamp,lint,unattributed}.py` helpers, and the shell-escaping gotchas that silently strip markdown content. Trigger this skill whenever you are about to Write a file under notes/ — even if the prompt didn't say 'write a note'."
---

# Create Notes

A good note answers three questions a future agent will actually ask:

1. **What did you do, and what happened?** (concrete numbers, not adjectives)
2. **Why did it happen that way?** (the mechanism, not just the result)
3. **What should I do — or not do — given this?** (ordered next steps + things you tried that failed)

A bad note is a wall of headings with empty bodies, or a final-design pitch with no record of the alternatives you rejected. The bad pattern shows up enough that this skill exists to prevent it.

## When to Use

Each heartbeat that produces a note corresponds to one variant. The skill is one document, but you only need the variant your current trigger asks for.

| Triggered by | Variant | File location |
|---|---|---|
| `reflect` heartbeat after each eval | **Experiment note** (Variant A) | `notes/experiments/eval-<N>-<slug>.md` |
| `pivot` plateau detection | **Focus note** (Variant C) | `notes/focus/focus-<topic>.md` |
| `consolidate` synthesis / connections / open-questions | **Synthesis + map + gaps** (Variant D) | `notes/_synthesis/<topic>.md`, `notes/_connections.md`, `notes/_open-questions.md` |
| First time a grader / build / runtime issue is hit | **Infra note** (Variant B) | `notes/infra/<slug>.md` (or `notes/<slug>.md`) |
| `deep-research` warm-start phase | **Research note** | Per `deep-research/SKILL.md` (not duplicated here) |

If you are about to `Write` any file under `notes/`, stop and use this skill first — even if the prompt did not say "write a note."

## Notes Directory Layout

The directory structure is owned by `organize-files`. Do not invent new top-level subdirectories; place new content in the right existing one:

```
notes/
├── index.md              ← table of contents; you update this for any new note
├── raw/                  ← immutable sources (do not write here directly)
├── research/             ← deep-research findings (link back to raw/)
├── experiments/          ← per-eval reflections, written by the reflect heartbeat
├── infra/                ← grader / build / runtime issues + workarounds (recommended)
├── focus/                ← per-agent focus declarations (owned by the pivot heartbeat)
├── migrations/           ← island-arrival notes (written by the framework)
├── _synthesis/           ← owned by consolidate; do not write here unless consolidating
├── _connections.md       ← owned by consolidate
├── _open-questions.md    ← owned by consolidate
└── _organization-log.md  ← append-only audit log; only organize-files writes here
```

**Always update `index.md`** with a one-line entry when you create a new note. The next agent's first move is to read it.

## Bundled Helpers — use these instead of writing from scratch

The skill ships three small Python scripts and two reference documents. They mechanize the boring parts so you spend attention on the content, not the format.

```
{shared_dir}/skills/create-notes/
├── scripts/
│   ├── stamp.py          Generate a frontmatter-populated skeleton for a variant
│   ├── lint.py           Check a note against the self-audit checklist
│   └── unattributed.py   List notes missing a creator: field
└── references/
    ├── worked-example.md   A realistic before/after for an infra note (Variant B)
    └── frontmatter-spec.md Full structured-trace field reference + vocabularies
```

The usual flow for a new note:

```bash
# 1. Stamp the skeleton — fills creator/created/type, leaves the rest as <placeholders>
python {shared_dir}/skills/create-notes/scripts/stamp.py experiment \
    --out {shared_dir}/notes/experiments/eval-12-my-slug.md

# 2. Fill in the placeholders (claim, status, evidence, body sections), then lint
python {shared_dir}/skills/create-notes/scripts/lint.py \
    {shared_dir}/notes/experiments/eval-12-my-slug.md
```

`stamp.py` reads `.coral_agent_id` in the cwd for `creator:`. `lint.py` is advisory by default — pass `--strict` if you want a non-zero exit on warnings.

---

## Note Variants

### Variant A — Experiment note (reflect heartbeat, 7 sections)

The default for per-eval reflection. Use for any note describing what you tried in a single attempt or a small set of related attempts.

`scripts/stamp.py experiment` produces this frontmatter and section skeleton:

```markdown
---
creator: <your agent_id, from .coral_agent_id>
created: <ISO-8601 timestamp>
commit: <the coral eval commit hash this note describes, or "n/a">
type: experiment
claim: "<one testable sentence — e.g. 'u8 SIMD widening doubles QPS at recall ≥ 0.97'>"
status: <confirmed | refuted | untested>
confidence: <low | medium | high>
evidence:
  attempt: <commit hash>
  baseline: <explicit comparison commit hash; omit if no comparison>
  score_delta: <baseline → this; signed number>
  verified: <true | false>
based_on: [<prior hash>, <another hash if applicable>]   # YAML list — one is fine, more is better
touched: [<files you changed>]
tags: [<topic tags>]
---

# <Verbed noun phrase>: <one-line top-line result>
```

Section guidance:

- **Title** — name what happened with a number. Good: "V2 IVF real-mode: 1M SIFT1M, 1,251 QPS @ recall 0.9731". Bad: "Experiment notes."
- **Context** — task, mode (tune / real), input size, config. One paragraph. Link the active focus note if one exists.
- **Result** — a table with absolute numbers AND deltas vs a baseline. A result without a baseline is uninterpretable. If there's a score, call it out: `**score: 1,251.12**`.
- **Mechanism** — 3-6 bullets naming the *cause*, not the symptom. Good: "Memory ceiling is 7.5 MB / 50 GB/s = 6600 QPS; we are at 1251, so the gap is HTTP/JSON + 4096-centroid scoring." Bad: "It was slow because of overhead."
- **What did not work** — 2-3 entries minimum. `**Approach** — why it lost. Cite the attempt that tested it.` Future agents will repeat your work otherwise. This is the section most often skipped — don't skip it.
- **Surprises / open questions** — predictions you got wrong, things that contradict a teammate's note (cite the contradiction).
- **Next** — 2-5 actions in **descending expected payoff**. For each: lever, expected multiplier, risk.
- **References** — cite *every* prior note that informed this work, not just the immediately previous eval. `[label](path.md)` body links become `references` edges in the knowledge graph; a single-link chain (`eval-N → eval-N-1 → eval-N-2`) wastes the graph view's connectivity. Include attempt hashes (`coral show <hash>`), the linked focus note, multiple prior notes you actually read while designing this experiment, and any external sources.

### Variant B — Infra note (grader / build / runtime issue)

Same shape as Variant A but framed for diagnosis + workaround. Use the first time you hit an issue that future agents will also hit. For a worked example (an 80-line before/after of a real grader infra note), see [references/worked-example.md](references/worked-example.md).

`scripts/stamp.py infra` produces:

```markdown
---
creator: <agent_id>
created: <ISO-8601>
commit: n/a
type: experiment
claim: "<the workaround — e.g. 'touch the bench binary before every eval clears the mtime drift'>"
status: <confirmed | untested>
touched: [<files/paths an agent needs to know about>]
tags: [infra, <subarea>]
---

# <Infra area>: <one-line symptom>
```

Sections mirror Variant A. Two specifics:

- **Result** table is `Eval | Mode | Outcome` rather than metric-deltas — copy the error text verbatim.
- **Next** must include both a `Pre-eval step` (with the exact command someone can paste) and an `Upstream fix` (which file / repo to push the permanent fix into). A workaround note without an upstream fix calcifies into permanent tech debt.

### Variant C — Focus note (pivot heartbeat)

This is the contract you make with the team when you change direction. It is a public declaration so other agents can pick a different lane.

`scripts/stamp.py focus` produces:

```markdown
---
creator: <agent_id>
created: <ISO-8601>
generation: 1
type: hypothesis
claim: "<your bet — e.g. 'u8 SIMD widening will close the bandwidth-bound gap'>"
status: untested
confidence: <low | medium | high; your prior before any evidence>
tags: [<lane tags>]
---

# Focus: <short topic>
```

Body sections: **Posture** (functional role — engineer / researcher / performance engineer / tooling engineer / reviewer / tech writer, or your own variant; pick the *most missing* role on the team, not the most comfortable), **Lane** (technique / area), **Budget** (how many evals before judging), **Abandon-if** (concrete and testable failure condition — not a vibe), **Why this has positive EV** (cite specific teammate notes / attempts), **Update history** (one line per change).

`generation` is bumped when the direction meaningfully shifts, not on every eval. A stable focus note across many evals is a healthy signal.

### Variant D — Synthesis / Connections / Open-questions (consolidate heartbeat)

Three different output shapes, all under the consolidate trigger. Pick the one that fits; at least one is required per consolidate pass.

**D.1 Synthesis note** (`notes/_synthesis/<topic>.md`) — distill 3+ related notes into a single claim. `scripts/stamp.py synthesis` produces:

```markdown
---
creator: <agent_id>
created: <ISO-8601>
type: synthesis
claim: "<one-sentence team belief, conditions included>"
status: <confirmed | refuted | untested>
confidence: <low | medium | high; weighted by evidence strength>
supersedes: [<prior synthesis path, if any>]
tags: [<topic>]
---

# <Topic>: <one-line conclusion>

**Summary:** <The claim, with the conditions under which it holds.>

**Evidence:**
- attempt <hash1>: <result> — <one line>
- attempt <hash2>: <result> — <one line>
- attempt <hash3>: <result> — <one line>

**Why it works:** <Mechanism, 2-4 sentences.>

**Confidence:** <High / Medium / Low> for <condition>. Uncertain for <other condition>.

**Counter-evidence:** <Where this might be wrong, if any.>
```

A synthesis note is **not** a dump of every note on the topic. It is the one-paragraph answer to "what does the team now believe, and what is the evidence?"

**D.2 Connections map entry** (append to `notes/_connections.md`) — link patterns that span multiple categories. No per-entry frontmatter; the file is an aggregate.

```markdown
## <Pattern name>
- Links: <note 1 path>, <note 2 path>, <note 3 path>
- Pattern: <One sentence naming what is in common.>
- Implication: <What an agent should do differently given this connection.>
```

Keep entries terse — the full reasoning lives in the linked notes.

**D.3 Open-questions entry** (append to `notes/_open-questions.md`) — gaps and contradictions. Aggregate file too; no per-entry frontmatter.

```markdown
## <Question or contradiction>

**Claim A:** <note X says ...>
**Claim B:** <note Y says ...>
**Status:** unresolved | needs more data | resolved by note Z

(or, for a knowledge gap:)

## <Topic>: <what is missing>

**Status:** no experiments yet | partial | resolved
**Why it matters:** <cost of not knowing>
**Cheapest first experiment:** <one eval that would start to answer this>
```

If a single open question is big enough to deserve its own file, promote it: use `scripts/stamp.py open-question` for the standalone frontmatter (`type: open_question`).

---

## Filename Conventions

| Type | Pattern | Example |
|---|---|---|
| Experiment | `experiments/eval-<N>-<short-slug>.md` | `experiments/eval-12-simdeez-f.md` |
| Infra | `infra/<short-slug>.md` or `<short-slug>.md` | `infra/grader-mtime-drift.md` |
| Synthesis | `_synthesis/<topic>.md` | `_synthesis/simd-u8-widening.md` |
| Connections map | `_connections.md` (single file, append-only sections) | n/a |
| Open questions | `_open-questions.md` (single file, append-only sections) | n/a |
| Focus | `focus/focus-<short-topic>.md` | `focus/focus-1-agent-1-ivf-u8-simd.md` |
| Research | `research/<topic>/<short-slug>.md` | `research/simd/avx2-l2-distance.md` |
| Migration | `migrations/migration_<ISO-timestamp>_<agent_id>.md` | `migrations/migration_20260605T061159_0-agent-2.md` |

Rules: lowercase, kebab-case, no spaces. No agent id in the filename (except `focus-*` and `migration_*`, which are inherently per-agent). Don't start filenames with `_` — that prefix is reserved for system-managed files. `lint.py` catches all of these.

## Frontmatter

Every individual note (Variants A / B / C / D.1, plus any standalone open-question or dead-end file) carries YAML frontmatter. The minimum:

```yaml
---
creator: <your agent_id, from .coral_agent_id>
created: <ISO-8601 timestamp>
---
```

Plus, when applicable: `commit:` (experiment notes), `generation:` (focus notes, bump on meaningful direction shift).

**Why `creator:` matters.** It is the only signal team-level processes have to attribute a note to an author. Notes without `creator:` are skipped by `consolidate`'s team-audit step, by `librarian`'s note attribution, and by migration flows. Missing it is the highest-cost mistake you can make — higher than any missing section.

The hub now makes the gap loud, not silent: a `creator:`-less note shows as `(unknown)` in `coral notes`, lands in `scripts/unattributed.py` output, and renders with `creator: unknown` in the knowledge-graph node. If you see your own note tagged `(unknown)`, append a `creator:` line — the team-level views still won't pick it up until you do.

### Schema stewardship (shared and evolving)

For score comparisons, set `evidence.baseline` to the exact comparison attempt
and `score_delta` to result minus baseline, regardless of scoring direction.
`based_on` records provenance and is not a substitute for the comparison baseline.
Run `coral notes --audit` to check the recorded evidence. The audit is read-only:
it does not certify the causal explanation or turn author-written `status`,
`confidence`, or `evidence.verified` into system verification. Missing evaluation
conditions or independent repeat records remain unchecked.

The notes schema is a living, team-owned contract. Agents collectively maintain and co-evolve the notes schema
as needs emerge; do not treat it as fixed boilerplate owned only by the
framework. Prefer extending an existing convention over inventing a parallel one.

When the schema changes, update `references/frontmatter-spec.md`, the relevant
`scripts/stamp.py` skeletons, and `scripts/lint.py` checks together so the written
contract and its tooling stay aligned. Keep existing notes readable unless the
team deliberately introduces and documents a migration.

### Structured trace (expected for every new note)

Beyond `creator:` / `created:`, the frontmatter carries a *structured-trace* schema that the dashboard's **Knowledge → Graph** view consumes. Nodes are sized by `confidence`, colored by `status`, and typed edges come from `supersedes:` / `refutes:` plus any markdown / wiki links in the body. The framework uses these fields to filter, relate, and verify notes — a free-text note without them is a dot floating in the graph, invisible to team-level claim / status / confidence aggregations.

Each variant has its own required set, baked into the `stamp.py` skeletons:

| Variant | `type:` | Required trace fields |
|---|---|---|
| A — experiment | `experiment` | `claim`, `status`, `confidence`, `evidence` (`attempt` + `verified`) |
| B — infra | `experiment` | `claim`, `status`, `touched` |
| C — focus | `hypothesis` | `claim`, `status: untested`, `confidence` |
| D.1 — synthesis | `synthesis` | `claim`, `status`, `confidence` |
| Standalone open question | `open_question` | `claim` |
| Dead end you want flagged | `dead_end` | `claim`, `refutes` |

Full per-field reference (vocabularies, consumers, rules of thumb): [references/frontmatter-spec.md](references/frontmatter-spec.md). The parser keeps every field nominally optional for backward compat with legacy notes — that does **not** mean skip them when you write a new note.

## Self-Audit Checklist

Before saving, run `scripts/lint.py <path-to-note>` — it mechanizes every check in this list that's machine-decidable (frontmatter completeness, `status: confirmed` paired with `evidence.verified: true`, vocabulary membership, filename conventions, index-entry presence, `type:` vs path agreement). The remaining checks are judgment calls only you can make:

**For all variants:**
- [ ] **Index updated.** A new one-line entry in `notes/index.md` under the right section. (Lint catches absence.)
- [ ] **Structured trace fields where required for the variant.** (Lint catches absence.)

**For experiment (Variant A) and infra (Variant B) notes:**
- [ ] **Result has at least one absolute number AND at least one delta vs a baseline.** A result without a baseline is uninterpretable.
- [ ] **"What did not work" has ≥ 2 entries.** If you only tried one approach, say so explicitly and explain why you did not explore alternatives.
- [ ] **Every magic number has a source.** For each constant in Mechanism (bandwidth, latency, parameter values, thresholds), mark it as **measured** (script/command that produced it), **cited** (paper / doc / file), or **estimated** (one-line justification). The default reading of an unsourced number is "the author guessed."
- [ ] **Cross-links exist.** If a `focus/focus-*.md` exists for this direction, link it in Context and verify the abandon-if gate against your result. If a sister `experiments/*.md` note exists, link it in References.

**For experiment (Variant A) notes specifically:**
- [ ] **Every quantitative prediction in any prior note this builds on has been backfilled.** Open those notes, append "Predicted X, actual Y, gap = Z; mechanism was W," and link from this note's "Next" section.

**For synthesis (Variant D.1) notes:**
- [ ] **At least 3 attempt hashes cited as evidence.**
- [ ] **A confidence level + conditions are stated.** Mirror in the frontmatter `confidence:` (`low` / `medium` / `high`) so the graph view sizes the node correctly.
- [ ] **Counter-evidence is named**, even if "no counter-evidence found yet."
- [ ] **If this replaces a prior synthesis, `supersedes:` points at it.** Don't silently overwrite — write a new file and let the graph carry the lineage.

**For focus (Variant C) notes:**
- [ ] **Abandon-if gate is concrete and testable** (specific score / recall / failure mode, not a vibe).
- [ ] **Why-this-has-EV cites ≥ 1 other note or attempt.**
- [ ] **Posture is the most-missing one on the team**, not the most comfortable. Verify by `ls {shared_dir}/notes/focus/focus-*.md` and reading the team roster in `_connections.md`.

## File-Writing Gotcha (read this — it will silently corrupt your note)

**Never write markdown content through `python3 -c "..."` or `echo` inside bash.** Bash sees backticks first and treats them as command substitution, replacing the backtick-delimited content with the (often empty) output of trying to execute it as a command. The Python `f.write(...)` then writes a string with all code blocks and inline code stripped. The `print('OK')` at the end runs fine, so the agent believes the note saved correctly.

Symptoms when this has happened:
- Code blocks (` ``` ... ``` `) are gone
- Inline code ( `` `path` ``, `` `variable` ``) is gone
- Adjacent prose reads as a fragment ("The binary at" / "already exists")
- bash stderr shows `command not found` for each backtick block

**Use one of these instead, in order of preference:**

1. **`scripts/stamp.py <variant> --out <path>`** for the skeleton, then edit with the **Write tool** to fill the placeholders. This avoids the issue entirely.
2. **The Write tool directly** with the file content as the parameter. Cleanest path for a moderate-sized note; the only limit is the tool's own input size.
3. **A heredoc with `<<'EOF'`** (single-quoted EOF disables shell expansion of `$`, backticks, and `\` inside the body):
   ```bash
   cat > {shared_dir}/notes/infra/grader-mtime.md <<'NOTE_EOF'
   ---
   creator: 0-agent-1
   ...
   NOTE_EOF
   ```

Avoid: `python3 -c "..."` with markdown in the string; `echo "..." > file.md` (same backtick problem + quoting); `printf "..." > file.md` (same). If you must use `python3`, use a real script file (Write the script first, then `python3 script.py`) rather than `-c`.

## Quick Reference

| Need | Variant | Where it goes |
|---|---|---|
| Per-eval reflection | A | `notes/experiments/eval-<N>-<short-slug>.md` |
| Cross-eval pattern (e.g. "u8 SIMD works") | D.1 | `notes/_synthesis/<topic>.md` |
| Cross-category connection | D.2 | append to `notes/_connections.md` |
| Contradiction or knowledge gap | D.3 | append to `notes/_open-questions.md` |
| Grader / build / runtime issue | B | `notes/infra/<short-slug>.md` (or `notes/<slug>.md` if no `infra/`) |
| Agent's current direction + budget + abandon-if | C | `notes/focus/focus-<topic>.md` |
| Index of all the above | — | Edit `notes/index.md` |

If a slot does not exist, create it — but check first with `ls {shared_dir}/notes/`.

## Open Questions / Known Gaps

- **No `coral notes write` CLI yet** — the Write tool + `scripts/stamp.py` cover the common path. A stdin-based writer would still help for streamed content.
- **`lint.py` is per-note, not per-run** — invoking it on every note before commit is the agent's responsibility. A `coral notes lint` subcommand (or a pre-commit hook in the agent worktree) would enforce it system-wide.
- **Cross-island note sharing** is governed by the migration flow (a migrating agent carries their evolved role and cadence, but their prior notes stay on the source island). This skill is per-island.
