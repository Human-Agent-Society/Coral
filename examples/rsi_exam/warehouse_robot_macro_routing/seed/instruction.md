# Program the cleanup robot to sort every ball into its basket

The office floor is a mess. Balls of every colour are scattered across an
`N x N` warehouse grid criss-crossed by shelving walls, and each colour has one
matching basket somewhere on the floor. You have exactly one robot and one
controller. The robot understands four buttons — **forward, turn right, turn
left, swap** — and the controller has one more trick: a **macro** you can record
once and replay as many times as you like. Every button you press costs you,
including each macro replay, so a clumsy plan that walks the robot back and forth
racks up a huge tab. Your job is to press the *fewest* buttons that still lands
every ball in its own basket before the move budget runs out.

You control a machine through a macro language; the objective is defined below.
The entire difficulty is macro compression: a naive plan is a long string of
`F/R/L/S`, but the tours the robot repeats (walk-to-cell, drop, walk-back) are
full of structure, and a well-designed macro turns hundreds of moves into a
handful of pressed buttons. You are scored *relative to a reproduced contest
rank-1 solver*, so matching a strong human is the bar.

## The machine

There is an `N x N` grid. Cell `(0,0)` is the top-left; `(i,j)` is `i` cells
down and `j` cells right. The outer boundary is walled, and there may be walls
between adjacent interior cells. Every cell is reachable from every other cell
without crossing a wall.

On the grid are `M` **balls** and `M` **baskets**. For each type
`k` (`0 <= k < M`) there is exactly one ball of type `k` and one basket of type
`k`. Initially each cell holds at most one ball or basket.

The robot starts at `(0,0)` **facing right**, holding nothing. You control it
with a sequence of buttons:

**Basic buttons**

- `F` (forward): move one cell in the current facing. If a wall blocks the
  destination, the robot stays put (the button is still spent).
- `R` (turn right): rotate 90° clockwise in place.
- `L` (turn left): rotate 90° counter-clockwise in place.
- `S` (swap): exchange the ball in hand with the ball on the current cell.
  - empty hand + ball here -> pick it up (cell becomes empty);
  - ball in hand + empty cell -> drop it (hand becomes empty);
  - ball in hand + ball here -> swap the two;
  - empty hand + empty cell -> nothing happens.
  - A ball sitting on a basket cell is swapped just like any other ball.

**Controller buttons**

- `M` (macro): if not currently recording, start recording. If currently
  recording, stop and **register** the recorded sequence as *the* macro
  (replacing any previous one).
- `P` (play): replay the **most recently registered** macro. If none is
  registered yet, nothing happens.

While recording, any basic `F/R/L/S` you press is both executed *and* appended
to the macro being recorded. A `P` pressed while recording replays the
previously registered macro, and the basic operations it expands to are executed
*and* appended to the macro being recorded (you cannot replay the macro you are
currently recording — only the last completed one).

> Example: with `RFF` already registered, running `MFPM` records a new macro.
> `M` starts recording; `F` executes+records; `P` replays the registered `RFF`
> (executed and appended); `M` stops. The basic ops executed are `FRFF`, and the
> newly registered macro is `FRFF`.

Initially no macro is registered and nothing is being recorded.

## Budget and scoring

You are given a basic-operation cap `T`. **After macro expansion**, at most `T`
basic operations are executed; the `T+1`-th basic operation is not executed and
the run is cut off there. So `P` is cheap to *press* but its expansion is
charged against `T`.

Let `A` be the length of the button sequence you output — **`M` and `P` each
count as one button**. Let `V` be the number of balls sitting on their matching
basket at the end of the simulation. The absolute score for a case is:

- `A` if `V == M` (all balls delivered) — **lower is better**;
- `T * (M - V)` if `V < M` (a big penalty).

You are scored on the **absolute score of each case — lower is better**. A reproduced
contest rank-1 solver's per-case scores ship with the task so you can gauge how strong
your solution is, but **how the raw scores map to the final reward is deliberately not
disclosed**. Optimise the raw score itself.

Closing the gap to a top contest solver is almost entirely about better macro synthesis.

## Input format (stdin, one instance)

```
N M T
v_0
...
v_{N-1}        # N lines: v_i is a length-(N-1) 01 string; v_i[j]=1 <=> wall between (i,j) and (i,j+1)
h_0
...
h_{N-2}        # N-1 lines: h_i is a length-N 01 string; h_i[j]=1 <=> wall between (i,j) and (i+1,j)
b_0 c_0 d_0 e_0
...
b_{M-1} c_{M-1} d_{M-1} e_{M-1}   # ball k starts at (b_k,c_k); basket k is at (d_k,e_k)
```

Constraints: `10 <= N <= 20`, `N/2 <= M <= 2N`, `1 <= T <= 2 N^2 M`. All ball
and basket cells are distinct.

## Output format (stdout)

Print the button sequence, one character per line **or** all on lines with no
spaces — any whitespace layout is fine; only the characters `F R L S M P` are
read, in order. The output length `A` must be `<= T`.

## What you submit

Your solver lives in `methods/main/` (the graded directory), containing:

- **`run.sh`** (required): run once per test case as
  `bash run.sh < instance.txt > out.txt`. It must read one instance on stdin and
  write the button sequence on stdout. Any language.
- **`build.sh`** (optional): if present, the grader runs it **once** before
  grading (e.g. to compile a C++ solver). Do your compilation here and have
  `run.sh` exec the built binary.

The starter `methods/main/run.sh` runs the shipped greedy baseline
(`methods/main/solution.py`). Replace it with your solver.

## Local dev bench

- `tools/in/` — the 100 visible instances (seeds 0-99).
- `tools/gen` — the **official generator**. Make more instances with
  `./tools/gen seeds.txt --dir=OUTDIR` where `seeds.txt` is one unsigned-64-bit
  seed per line. **For local testing use seeds in `0..10000` only** — the sealed
  grading seeds live far outside that range, so staying inside it keeps your
  practice set from colliding with the hidden set.
- `tools/vis` — the **official visualiser/judge**: `./tools/vis in.txt out.txt`
  prints `Score = <absolute score>` and writes a `vis.html` you can open.
- `python3 selfcheck.py [N]` — runs your `run.sh` on the first `N` visible cases
  (default 100), prints each case's raw score alongside the reference solver's raw
  score on the same case. Free and unlimited.

The generator, the distribution, and the visualiser are exactly the contest's.
The sealed evaluation reruns your `run.sh` on 200 fresh sealed instances from the
same generator and scores them with the same visualiser; there is no feedback
loop — whatever sits in `methods/main/` at the end is what is graded.

## Notes

- CPU only, no network. Python 3 and `g++` are available in both the workbench
  and the grader.
- Each sealed case runs your `run.sh` under a wall-clock cap (20 s/case); a
  crash, timeout, malformed output, or a case that fails to deliver all balls
  scores that case's `rel` at ~0. Complete every case first, then optimise
  length.
- The reference contest was a 2-second-per-case time limit; you have more slack
  here, but the aggregate is dominated by macro *quality*, not raw search time.
