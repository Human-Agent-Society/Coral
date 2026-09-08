# 2048 Strategy

You inherit a weak Python policy that plays deterministic seeded 2048 games. Improve
the policy using repeated experiments on the public seed suite; your submitted policy
is replayed from scratch on sealed seeds from the same generator for scoring.

## Hard Constraints

- Edit `methods/main/policy.py`; it must define `choose_move(board)` and return one of
  `"UP"`, `"DOWN"`, `"LEFT"`, or `"RIGHT"`.
- `board` is a tuple of four tuples containing tile values, with zero for an empty cell.
- The policy may use only Python's standard library and must be deterministic for a
  given board.
- `methods/main/policy.py` must not exceed **10 MB**. Learned weights are allowed within
  that budget — embed them in the file, since nothing outside `methods/main/` is available at
  grading time. `selfcheck.py` enforces the cap, and so does the grader; an oversized file
  scores zero.
- Do not modify `game2048.py`, `evaluate.py`, `selfcheck.py`, or the public seed file.
- **CPU budget: 225 seconds of CPU per game**, enforced as one pooled limit across the whole
  sealed suite by a single policy process. Spend it unevenly if you like, but exhausting the
  pool kills the process, and every game not yet played then scores zero. `selfcheck.py`
  applies the same per-game budget to the public suite and reports `cpu_seconds_per_game`.
- A single move must return within 5 seconds.
- A callback error, illegal return value, or per-move timeout ends that game.
- The final policy must be self-contained in `methods/main/` and must not use files,
  subprocesses, network access, clocks, environment variables, or randomness.

## What You Have

- `methods/main/policy.py`: the weak starting policy.
- `game2048.py`: the exact game rules, move order, tile spawning, and raw score logic.
- `evaluate.py`: the public evaluation harness.
- `visible_seeds.json`: the fixed public games used for comparable development scores.
- `selfcheck.py`: runs the current policy on every public game and reports mean raw
  game score, median score, maximum tile, and per-seed results.
- `methods/experiment_log.md`: record each comparable attempt and whether it was kept
  or reverted. Save evaluated policy snapshots and results under
  `methods/versions/vN/`.

## What You Submit

Submit the complete `methods/` directory. The selected policy must be located at
`methods/main/policy.py`; experiment history and notes may remain under `methods/`.

## How It Is Judged

Each game starts with two spawned tiles and follows standard 4-by-4 2048 merge rules.
After every legal move, a 2 tile is spawned with probability 0.9 and a 4 tile with
probability 0.1. The raw metric is the arithmetic mean of the standard merge score
over the seed suite, and higher is better. The normalized score is a monotonic function of sealed
game performance and is not shown to you; optimize raw score and cross-seed generalization.

The hidden seeds are disjoint from the public seeds but use the identical simulator,
spawn distribution, board size, move budget, and scoring code.
