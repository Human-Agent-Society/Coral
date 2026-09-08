# Contributing to CORAL

Thanks for your interest in CORAL! This guide covers how to file issues, set up a
development environment, and submit changes. CORAL is released under the
[Apache 2.0 License](LICENSE) — by contributing, you agree that your contributions
are licensed under the same terms.

## Ways to contribute

- **Report a bug** — open a [GitHub issue](https://github.com/Human-Agent-Society/CORAL/issues)
  with a minimal reproduction, the command you ran, the relevant output from
  `.coral/public/logs/`, and your `coral --version` / OS / Python version.
- **Propose a feature or design change** — open a *Discussion* (or an issue
  labelled `proposal`) before writing code, so we can align on scope.
- **Add a new task** under `examples/` — see the [`coral-new-task` skill](.claude/skills/coral-new-task/SKILL.md)
  for the end-to-end recipe (`task.yaml` + `seed/` + grader package).
- **Extend the framework** — new runtime, CLI command, hook, bundled
  skill/subagent, or config field. See the [`coral-extend` skill](.claude/skills/coral-extend/SKILL.md).
- **Improve documentation** — both this repo's READMEs and the docs site
  under `docs/`.
- **Triage and review** — comments on open issues and PRs are welcome.

If you are unsure whether a change belongs in CORAL, open an issue first.
Larger features almost always benefit from a short design discussion before
implementation.

## Development setup

Prerequisites: **Python 3.11+** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Human-Agent-Society/CORAL.git
cd CORAL
uv sync --extra dev        # core + pytest + ruff + mypy
# uv sync --all-extras     # also installs the UI deps for `coral ui`
```

Run the CLI from your checkout with `uv run coral ...`.

### Tests and lint

Before opening a PR, make sure these pass locally:

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run ruff format --check .
```

`ruff` is configured in `pyproject.toml` (`target-version = "py311"`,
`line-length = 100`, rules `E, F, I, N, W, UP`). Type-aware work is checked
with `uv run mypy coral` where strict mode applies.

When you change code under `coral/`, the [`coral-debug` skill](.claude/skills/coral-debug/SKILL.md)
documents the smallest reproduce loop per subsystem (grader, daemon, CLI,
hooks, manager, workspace, hub, template, config, web).

## Project layout

CORAL's architecture, directory layout, and core types are summarised in
[CLAUDE.md](CLAUDE.md) at the repo root. Skim it before making non-trivial
changes — it covers the eval loop, `.coral/{public,private}/` split, the
grader daemon, heartbeats, and the runtime registry.

## Branches and commits

- CORAL uses a two-branch model: **`dev` is the integration branch** where all
  day-to-day development lands, and `main` tracks releases. Maintainers merge
  `dev` into `main` as part of the release process — **all PRs must target
  `dev`**, never `main` directly.
- Promotion PRs from `dev` to `main` use **Create a merge commit**. Once one is
  merged, the promotion workflow verifies that the promoted `dev` commit and
  the resulting `main` commit have identical trees, then tags the `dev` commit
  and publishes that exact source. Because the tag is in both branches'
  ancestry, `main` does not need to be merged back into `dev` after a release.
- Promotions default to a patch release. Apply `release:minor` or
  `release:major` to the promotion PR to select a larger bump. Direct changes
  on `main` are unsupported: they make the tree-equality release check fail.
- Create a topic branch off `dev`. Suggested naming: `feat/<short-desc>`,
  `fix/<short-desc>`, `docs/<short-desc>`, `refactor/<short-desc>`.
- Keep commits focused. Prefer several small, reviewable commits over one
  large one.
- We follow [Conventional Commits](https://www.conventionalcommits.org/) for
  commit subjects. Examples from the recent history:
  - `feat(agent): allow custom runtime via module.path:ClassName entrypoint`
  - `fix(web): migrate to Starlette 1.0 lifespan handlers`
  - `docs: trim READMEs, move runtime/gateway details into docs site`
  - `refactor(prompts): rename identity→role, distinguish objective from task`
- Reference issues in the body (`Closes #123`) when applicable.

## Pull requests

1. Fork the repo (external contributors) or push your branch (maintainers).
2. Make sure tests, `ruff check`, and `ruff format` all pass.
3. Open a PR against `dev` (PRs against `main` will be retargeted or closed)
   with:
   - a short **summary** of what changed and **why**,
   - a **test plan** (commands you ran, manual checks, screenshots if UI),
   - links to any related issues or discussions.
4. PR title should follow Conventional Commits (it usually becomes the squash
   commit subject).
5. Keep the PR scoped — unrelated cleanups belong in a separate PR.
6. A maintainer will review. Expect comments; please respond and push fixups
   rather than force-pushing rewritten history while review is in progress.
7. We typically **squash-merge ordinary topic PRs** once CI is green and at
   least one maintainer has approved. The `dev` to `main` release promotion is
   the exception: use **Create a merge commit** so the workflow can verify and
   tag the exact promoted `dev` parent.
8. **Vercel previews for fork PRs**: deployments of the `web/` dashboard and
   the `docs/` site are blocked by Vercel's fork protection until a maintainer
   authorizes them. Instead of clicking through the Vercel dashboard, a
   maintainer can comment `/ci run` on the PR — a workflow deploys both
   previews and posts the URLs back as commit statuses and a PR comment.
   Comments from users without write access are ignored.

## Coding guidelines

- **Stay in scope.** Bug fixes shouldn't carry along refactors of unrelated
  code. One-off operations don't need new abstractions.
- **Don't add backwards-compat shims** unless an existing public surface is
  changing in a way that would break downstream users.
- **Comments explain *why*, not *what*.** Well-named identifiers cover the
  what. Comment hidden constraints, subtle invariants, or workarounds.
- **Public APIs and config fields** — if you add or change a field in
  `coral/config.py`, a CLI flag, a hook, or a runtime contract, update the
  docs (both `docs/content` and any affected READMEs).
- **New tasks** must include a working `coral validate <task>` and a smoke
  run that produces at least one finalized score. Don't ship hidden answer
  keys in `seed/`.
- **Tests** — add or update tests under `tests/` for any behavior change.
  Async tests use `pytest-asyncio` (auto mode is already configured).

## Reporting security issues

Please **do not** open public issues for security vulnerabilities. Email the
maintainers at the address listed in the project's GitHub org profile, or use
GitHub's *Report a vulnerability* button on the repo. We will acknowledge
within a few business days.

## Code of conduct

Be respectful, assume good intent, and keep technical discussions focused on
the work. Harassment or personal attacks aren't tolerated in issues, PRs, or
discussions. Maintainers may moderate or remove comments that violate this.

## Recognition

Contributors are credited via the Git history and the GitHub *Contributors*
page. If a contribution materially changes the project (new runtime, new
benchmark family, major refactor), feel free to add yourself to the
acknowledgements section of `README.md` in the same PR.

---

Thanks again — CORAL improves fastest when the community pulls together.
