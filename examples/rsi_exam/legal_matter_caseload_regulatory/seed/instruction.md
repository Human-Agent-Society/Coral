# Answer legal questions from the underlying case files

You are the analyst covering six unrelated legal matters. Each has its own folder
of case documents under `tools/world_corpus/`:

| folder | matter |
|---|---|
| `w421_telemarketing` | Senior Living Lending — TCPA / TSR telemarketing compliance, mini-TCPA state laws, class actions |
| `w434_spinoff` | Harborview — spinoff transaction, cross-border data transfer, SPA indemnification |
| `w_jv_antislapp` | BlueAnchor / BlueLNG JV — JV agreement terms, anti-SLAPP, shipyard fire and delay |
| `w423_gdpr` | Northstar — GDPR compliance, international transfers, breach notification |
| `w420_druglabel` | Livyra — FDA drug labelling changes, adverse-event reporting |
| `w416_tariff` | TAC — tariffs, Master Supply Agreement, force majeure under the UCC |

1,000+ documents in total. Task names are prefixed with their matter, e.g.
`w421_telemarketing/World421_AP_02`.

There are two sets of questions, and **you are expected to work both of them in
full**.

**18 visible questions — 3 per matter.** Answer every one of them and submit
through `matter`, which replies with a bare score like `4/6`. It never tells you
*which* criteria you met, and never shows you the criteria themselves. You have
**90 submissions total — five per visible question** — so you can revise and
resubmit. **Your job on these is to push every one of them as high as you can.**

**18 held-out questions — 3 per matter.** You answer these blind and submit them
**once**, with `matter final`, which ends the engagement — no feedback, no second
attempt. Their prompts are in `tools/heldout_prompts.json` from the start.

The two halves come from the same six matters and are graded by the same kind of
criteria. So whatever you learn on the visible half — how specific to be, how to
enumerate, which documents settle which point — is exactly what the held-out half
will ask of you. **Working the visible questions properly is how you do well on
the held-out ones.**

## The `matter` tool

```
matter status                    budget left, best score per question
matter tasks                     the visible questions
matter show <task>               one question in full
matter submit <task> <file>      grade an answer  (costs 1 of your 90)
matter close                     end the visible phase early
matter final [answers.json]      submit the held-out answers — ONCE, terminal
```

The rubrics live in a separate grading service you cannot read or reach except
through `matter submit`. There is no copy anywhere in this container.

You have 90 submissions in total across the visible questions.  Reading the documents costs nothing;
only `matter submit` spends budget.

## What to submit

Write your held-out answers into **`methods/main/answers.json`** as you go — it
is a plain file, fill it in question by question rather than leaving it to the
end. It maps each held-out `task_name` to your answer text:

```json
{
  "w419_oilspill/World419_DM_02": "Neither Star Tankers ... $56,709,300 ...",
  "w433_seniorliving/World433_JS_08": "..."
}
```

Then run `matter final`. An unanswered question scores zero on every one of its
criteria, so answer all 18 even where you are unsure.

## How answers are judged

**A question counts only if you satisfy EVERY one of its criteria.** Getting 9
of 10 scores the same as getting 0 of 10 — the criteria are all must-haves. The
`x/y` score you get back during the visible phase is there so you can tell
whether a revision helped; it is not partial credit.

Each question has binary criteria of the form *"States that &lt;some specific
finding&gt;"*. A criterion is satisfied only if your answer actually asserts that
finding. So:

- **State conclusions directly, then support them.** No greeting, no preamble,
  no "here is what I found" — none of that satisfies any criterion.
- **Be specific.** Exact figures, statutory subsections, policy names, article
  numbers, regulation citations. A vague characterisation satisfies nothing; the figure or
  citation the criterion names does.
- **Enumerate completely.** Many criteria are one-per-item: every policy that
  covers a claim, every article that applies, every step of a liability
  computation. Missing one item costs one criterion. Length is not coverage — a
  long answer that names three of seven articles scores three.
- **Follow each question's own instructions on form and length.** Some want a
  yes/no per item, some one or two sentences, some several paragraphs. A few ask
  for a memo or a schedule; give its full text as your answer.
- **Do not hedge.** A criterion asks whether you *stated* something. A conclusion
  buried in qualifications may not read as stated at all.

## What you have

- `tools/world_corpus/<matter>/` — that matter's documents as text, plus
  `_index.json` listing every path. Some entries are marked
  `"extractable": false` (audio, images) — those have no text and cannot be read.
- `tools/heldout_prompts.json` — the 18 held-out questions.
- `methods/main/answers.json` — where your held-out answers go.
- CPU only, Python 3. Do not fetch anything external.
