# AGENTS.md — AI-assisted contributions to CORAL

CORAL is itself infrastructure for running autonomous coding agents, so we
fully expect contributors to use Claude Code, Codex, Cursor, Kiro, OpenCode,
or other agents while developing changes. This document sets the ground rules.

These rules apply to **any** PR where an AI coding agent meaningfully wrote,
refactored, or generated code, tests, or docs. If you are not sure whether
your workflow counts, it does — read on.

> **TL;DR**
> - All PRs target the `dev` branch, never `main` (see CONTRIBUTING.md).
> - Release tags point to the promoted `dev` commit, not the `main` merge commit;
>   do not add main-only release changes or routine main-to-dev sync PRs.
> - A human author must read every changed line and be able to defend the design end-to-end.
> - No drive-by mechanical PRs (lone typo fixes, single-line style nits, batch reformat across the repo).
> - Don't open duplicate PRs against the same issue. Check first.
> - Keep AGENTS.md in sync with CLAUDE.md when project guidance changes.
> - Don't dump raw agent transcripts into the PR description. Summarize.

## 1. Human accountability

- **Pure agent-authored PRs are not accepted.** A human must understand the
  change, review every line, and respond to review comments themselves —
  not by piping reviewer questions back into an agent and posting the reply.
- The submitting human is responsible for:
  - Correctness of the diff,
  - Running tests + lint locally before pushing,
  - Following up on review feedback,
  - Any regressions introduced by the change.
- "An agent wrote it" is not an explanation for design choices in review.

## 2. Scope discipline

- **Bundle agent output with substantive work.** Don't open standalone PRs
  for things like:
  - a single typo fix,
  - one isolated `ruff`-style nit,
  - reformatting a file the agent happened to touch,
  - changing a comment without changing behavior.
  Roll these into a larger PR that actually does something, or batch many
  of them into one explicit cleanup PR with a clear scope statement.
- **One concern per PR.** If your agent generated a refactor and a bug fix
  in the same run, split them.
- **Don't reformat the world.** Repo-wide formatting / import-sorting /
  rename-the-variable PRs are a "no" unless coordinated with maintainers
  first (and then they go on `.git-blame-ignore-revs`).

## 3. Duplicate-work checks

Before opening a PR, verify nobody is already on it:

```bash
gh issue view <issue_number> --repo Human-Agent-Society/CORAL --comments
gh pr list --repo Human-Agent-Society/CORAL --state open --search "<issue_number> in:body"
gh pr list --repo Human-Agent-Society/CORAL --state open --search "<short keywords>"
```

If an open PR is already in flight:

- If it looks stalled, **comment there** to check in. Don't open a parallel PR.
- If your approach is materially different, file an issue (or comment on the
  existing one) explaining the difference before submitting.

## 4. Keep agent guidance synchronized

`CLAUDE.md` is the detailed project guide used by Claude Code, and
`AGENTS.md` is the cross-agent contribution guide. Keep them consistent:

- If you update `CLAUDE.md` with project structure, workflow, command,
  testing, PR, or agent-specific guidance, check whether `AGENTS.md` needs the
  same rule or a tool-agnostic version of it.
- If you update `AGENTS.md` with contribution rules that should affect Claude
  Code sessions, check whether `CLAUDE.md` needs the corresponding detail.
- If only one file should change, say why in the PR description.

## 5. Disclosure in PR descriptions

If an agent did the bulk of the writing, say so plainly. We don't penalize
agent-assisted PRs — we just want the review framed correctly.

Use `.github/PULL_REQUEST_TEMPLATE.md` as-is. Do not replace it with a custom
short summary. In particular:

- Keep the `Motivation`, `Changes`, `Test plan`, `Affected areas`, and
  `Checklist` sections.
- Tick every affected area that applies.
- Tick checklist items only when they are true. If the full-suite template
  commands were not run, leave those boxes unchecked and list the narrower
  commands you did run in `Test plan`.
- Reference issues in the body (`Closes #123`) when applicable.

A minimal disclosure looks like:

> *Authored with assistance from \<tool\>. I read every changed line, ran the
> tests below locally, and verified the behavior described in the test plan.*

What **not** to do:

- Don't paste raw agent transcripts, multi-thousand-line reasoning traces,
  or hidden "system prompt" blocks. Summarize the design instead.
- Don't fabricate test results. If you didn't run it, don't claim you did.
- Don't link to private agent sessions. Reviewers can't follow those.

## 6. CORAL-specific gotchas for agents

These tend to bite AI-generated PRs in this repo specifically:

- **`.coral/private/` is hidden from agents at runtime.** Don't write code
  that assumes agents can read grader internals (answer keys, hidden test
  inputs, grader source). See `coral/workspace/project.py`.
- **The grader runs in its own venv** (`.coral/private/grader_venv/`).
  Adding a dep to `pyproject.toml` does not put it on the grader's path —
  use `grader.setup` in `task.yaml` instead.
- **Grader output must not write to `codebase_path`.** The daemon force-
  removes that directory after grading. Anything the grader wants to
  persist goes through `ScoreBundle` or `self.private_dir`.
- **Score direction matters.** `Task.maximize` controls which way is
  better — agents often flip this. Double-check against existing examples.
- **Don't commit hidden answers under `seed/`.** Anything in `seed/` ends
  up in the agent's worktree, visible to the running agent.
- **CORAL.md is generated** by `coral/template/coral_md.py` — don't hand-
  edit the rendered file in a worktree, edit the template instead.
- **Audit note evidence with `coral notes --audit`** (optionally `--status STATUS`,
  `--search KW`, or `-n N`). This read-only command checks recorded observations separately
  from author status. A score comparison needs `evidence.baseline`; missing
  conditions remain unchecked, and matching arithmetic does not verify causality.

If you're adding a new task, the [`coral-new-task` skill](.claude/skills/coral-new-task/SKILL.md)
walks through these in detail. For framework-level changes, see
[`coral-extend`](.claude/skills/coral-extend/SKILL.md); for debugging,
[`coral-debug`](.claude/skills/coral-debug/SKILL.md).

## 7. Reviewer guidance

Reviewers should:

- Flag PRs that look like raw agent output without human review (telltale
  signs: unrelated drive-by changes, invented APIs, broken imports, fake
  citations, "test plan" sections that don't match the diff).
- Ask the human author — not the agent — to explain non-obvious choices.
- Close low-value mechanical PRs with a pointer to this document.

---

By submitting a PR you confirm you've read CONTRIBUTING.md and, if
applicable, this document.
