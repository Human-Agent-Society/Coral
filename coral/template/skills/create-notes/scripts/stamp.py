#!/usr/bin/env python3
"""stamp.py — Print a frontmatter + body skeleton for a note variant.

Usage:
    python stamp.py VARIANT [--out PATH] [--agent-id ID]

Variants:
    experiment       Per-eval reflection (default for reflect heartbeat)
    infra            Grader / build / runtime issue + workaround
    focus            Per-agent direction declaration (pivot heartbeat)
    synthesis        Cross-eval distilled claim (consolidate heartbeat)
    open-question    Standalone knowledge-gap / contradiction file
    dead-end         A claim/approach the team should not retry

The skeleton is populated where automation is cheap:
- ``creator`` from ``--agent-id`` or ``.coral_agent_id`` in cwd
- ``created`` to the current UTC ISO-8601 timestamp
- ``type`` to the right vocabulary value for the variant

Every other required field is left as ``<placeholder>`` so the agent
has to make a real decision rather than ship a stub. The body sections
mirror the variant templates in SKILL.md so a stamped note already has
the right shape — the agent fills, not invents.

Self-contained: no imports from coral.* so the script ships intact
inside .coral/public/skills/create-notes/scripts/ on every island.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone  # noqa: UP017  bundled scripts may run under py<3.11
from pathlib import Path

AGENT_ID_FILES = (".coral_agent_id",)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017


def _detect_agent_id() -> str:
    for name in AGENT_ID_FILES:
        p = Path(name)
        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8").strip()
                if content:
                    return content
            except OSError:
                pass
    return "<your agent_id>"


_EXPERIMENT = """\
---
creator: {creator}
created: {created}
commit: <coral eval commit hash, or "n/a">
type: experiment
claim: "<one testable sentence — e.g. 'u8 SIMD widening doubles QPS at recall ≥ 0.97'>"
status: <confirmed | refuted | untested>
confidence: <low | medium | high>
evidence:
  attempt: <commit hash>
  baseline: <explicit comparison commit hash; omit if no comparison>
  score_delta: <baseline → this; signed number>
  verified: <true | false>
based_on: [<prior hash>, <another hash if applicable>]   # list every prior attempt this builds on
touched: [<files you changed>]
tags: [<topic tags>]
---

# <Verbed noun phrase>: <one-line top-line result>

## Context
<task / mode / input size / config; link any active focus note>

## Result
| Metric | Baseline | This | Δ |
|---|---|---|---|
| ... | ... | ... | ... |

**score: <number>**

## Mechanism
- <why it worked or didn't — the actual cause, not the symptom>

## What did not work
- **<approach>** — <why it lost; cite the eval/note that tested it>
- **<approach>** — <why it lost>

## Surprises / open questions
- <thing you predicted that you're now uncertain about>

## Next
1. **<lever>** — <expected payoff>. Risk: <one line>.
2. **<lever>** — <expected payoff>. Risk: <one line>.

## References

Cite every prior note that informed this work — not just the previous
eval. `[label](path.md)` body links become `references` edges in the
knowledge graph; a single-link chain (eval-N → eval-N-1 → ...)
under-represents how knowledge actually compounds.

- attempt `<hash>`: <what you took from this graded result>
- prior note: [<related-eval>.md](<related-eval>.md) — <what you took from it>
- prior note: [<another>.md](<another>.md) — <what you took from it>
- focus note: [focus-<topic>.md](../focus/focus-<topic>.md)
"""

_INFRA = """\
---
creator: {creator}
created: {created}
commit: n/a
type: experiment
claim: "<the workaround — e.g. 'touch the bench binary before every eval clears the mtime drift'>"
status: <confirmed | untested>
touched: [<files/paths an agent needs to know about>]
tags: [infra, <subarea>]
---

# <Infra area>: <one-line symptom>

## Context
<task / mode / trigger condition>

## Result
| Eval | Mode | Outcome |
|---|---|---|
| #<N> | <mode> | FAILED: <error text, verbatim> |
| #<N> (retry) | <mode> | OK after <workaround> |

## Mechanism
- <code path / why the failure is structural>

## What did not work
- **<workaround attempt>** — <why it failed>
- **<workaround attempt>** — <why it failed>

## Next
1. **Pre-eval step** — <exact command>. Cost: <...>. Risk: <...>.
2. **Upstream fix** — <which file / which repo>.

## References
- failed attempt: `<hash>` — <error in one line>
- working attempt: `<hash>` — <what fixed it>
- grader source: `<path>` — <line / function>
- prior infra note: [<related>.md](<related>.md) — <how this relates>
"""

_FOCUS = """\
---
creator: {creator}
created: {created}
generation: 1
type: hypothesis
claim: "<your bet — e.g. 'u8 SIMD widening will close the bandwidth-bound gap'>"
status: untested
confidence: <low | medium | high; your prior before any evidence>
tags: [<lane tags>]
---

# Focus: <short topic>

## Posture
<engineer | researcher | performance engineer | tooling engineer | reviewer | tech writer, or your own variant>
Pick the posture **most missing** from the team, not the most comfortable.

## Lane
<the specific technique, area, or composite you are attempting>

## Budget
<how many evals you intend to spend before judging>

## Abandon-if
<specific score, recall, or failure mode that would convince you to stop>

## Why this has positive EV
- <evidence in team notes that suggests this is worth trying>
- <which other agents' work this builds on or complements>
- <why the easy alternatives have been ruled out>

## Update history
- {created}: created
"""

_SYNTHESIS = """\
---
creator: {creator}
created: {created}
type: synthesis
claim: "<one-sentence team belief, conditions included>"
status: <confirmed | refuted | untested>
confidence: <low | medium | high; weighted by evidence strength>
supersedes: [<prior synthesis path, if any>]
tags: [<topic>]
---

# <Topic>: <one-line conclusion>

**Summary:** <the claim, with the conditions under which it holds>

**Evidence:**
- attempt <hash1>: <result> — <one line>
- attempt <hash2>: <result> — <one line>
- attempt <hash3>: <result> — <one line>

**Why it works:** <mechanism, 2-4 sentences>

**Confidence:** <high / medium / low> for <condition>. Uncertain for <other condition>.

**Counter-evidence:** <where this might be wrong, if any>
"""

_OPEN_QUESTION = """\
---
creator: {creator}
created: {created}
type: open_question
claim: "<the question — e.g. 'Does nlist=1024 beat nlist=2048 once recall headroom appears?'>"
status: untested
tags: [<topic tags>]
---

# <Topic>: <what is missing>

**Status:** <no experiments yet | partial | resolved>

**Why it matters:** <cost of not knowing>

**Cheapest first experiment:** <one eval that would start to answer this>

**Related:**
- <note path or attempt hash>
"""

_DEAD_END = """\
---
creator: {creator}
created: {created}
type: dead_end
claim: "<what doesn't work — e.g. 'f64 storage scalar loop'>"
status: refuted
refutes: [<note path the proposed approach lives in>]
evidence:
  attempt: <commit hash where this was tested>
  verified: true
tags: [<topic tags>]
---

# Dead end: <approach>

## What was tried
<one-paragraph description>

## What happened
<the result, with numbers>

## Why it can't work
<mechanism — don't just say "slower," explain why>

## Do not retry unless
<the specific condition that would justify revisiting>
"""

VARIANTS = {
    "experiment": _EXPERIMENT,
    "infra": _INFRA,
    "focus": _FOCUS,
    "synthesis": _SYNTHESIS,
    "open-question": _OPEN_QUESTION,
    "dead-end": _DEAD_END,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("variant", choices=sorted(VARIANTS.keys()))
    ap.add_argument("--out", type=Path, help="Write to PATH instead of stdout")
    ap.add_argument(
        "--agent-id",
        default=None,
        help="Override creator (defaults to .coral_agent_id in cwd or <your agent_id>)",
    )
    args = ap.parse_args()

    creator = args.agent_id or _detect_agent_id()
    rendered = VARIANTS[args.variant].format(creator=creator, created=_now_iso())

    if args.out:
        if args.out.exists():
            print(f"refusing to overwrite existing file: {args.out}", file=sys.stderr)
            return 2
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
