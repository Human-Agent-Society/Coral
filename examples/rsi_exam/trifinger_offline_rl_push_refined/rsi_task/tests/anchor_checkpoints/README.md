# Anchor checkpoint archive

These three checkpoints are the ones the four numbers in `tests/anchors.json` were actually
measured from. They are archived so the anchor gate can cover this task: grading here is
ARTIFACT-EVAL (the graded object is the checkpoint produced by `train()`, not `solver.py`), so
a gate that copies only the method directory would always see "no checkpoint" and score 0.

These files enter no image: `tests/Dockerfile` copies file by file (never `COPY . /tests`), and
`environment/Dockerfile` does not touch `tests/` at all. They are bind-mounted only when the
gate runs.

## What the three are

| Directory | Tier | Reward landing | Metric on the sealed split (mean return) | Size | sha256 (first 16) |
|---|---|---|---|---|---|
| `baseline/model.pt` | baseline | 0.0 | 169.06890972595647 | 27 B | `6304baf7d4806e5d` |
| `reference/model.pt` | reference | 0.3 | 491.3456237488426 | 368 KiB | `6219da093333f89a` |
| `sota/model.pt` | sota | 0.6 | 653.8302272409492 | 22 MiB | `c9c0f22379e4f99f` |

- **baseline** is not a training product but a 27-byte placeholder JSON
  (`{"arch":"random_baseline"}`). The baseline policy is uniform-random and
  `solution/methods/baseline/solver.py` never reads weights; the checkpoint directory exists
  only to satisfy grade.py's "directory present and non-empty" gate.
- **reference** is behavior cloning on the shipped mixed dataset, following the recipe in
  `solution/methods/reference/solver.py`.
- **sota** is the hidden advantage-weighted BC 5-model ensemble from
  `solution/methods/sota/solver.py` (5 members, 22 MiB).

## Which sealed split

sealed-v3: `sha256("trifinger-sealed-v3-{k}") mod (2**31-1)` for k = 0..31, as defined in
`tests/heldout_seeds.py`. The three metrics above are the mean return over those 32 episodes,
all measured through a real `bash /tests/test.sh`.

**If the split changes, all three metrics are void.** (`UPPER_BOUND = 750.0` is a theoretical
bound, independent of the split, and is unaffected.) Anyone modifying `heldout_seeds.py` must
re-measure and replace these three files.

## What this coverage catches, and what it does not

**Catches:** missing or wrong-version packages in the verifier image, mis-wired mount points or
permissions (the class of silent failure the readability gate guards against), drift or damage
in `anchors.json`, changes to the reward mapping, and outright breakage of the scoring chain.

**Does not catch:** whether the training recipe still reproduces the checkpoint. The gate runs
already-trained weights; `train()` in `solution/methods/*/solver.py` is never executed. Failures
that surface only during training -- sklearn/torch version drift, a dead dataset URL, broken
training hyperparameters -- are invisible to it. Covering that layer requires real training (the
sota tier is a 5-model ensemble, costing hours).
