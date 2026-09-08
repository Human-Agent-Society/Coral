# City-Wide Signal Coordination

You inherit a full-city traffic microsimulation — a real metropolitan road network with
~600 signalised junctions and a synthetic morning-peak demand — running on its original,
un-optimised static signal timing. Improve the signal control so the whole
city moves better under load, minimising this frozen cost (the exact definition lives in
`sim/eval_core.py`):

```
Cost =     1 · Σ timeLoss     seconds of delay over arrived vehicles
     +   160 · teleports      SUMO teleports, charged the time-to-teleport
     +    20 · Σ stops        number of stops (energy / emission proxy)
     +  3600 · unfinished     vehicles still in-network or pending at END
```

Note the last term: `3600 · unfinished` is the *largest* component at every load in this
band, so serving vehicles at all matters more than shaving delay off the ones that
already get through. You submit a signal program and/or an online controller; it is
re-run on sealed hidden load and seed settings and scored by the same frozen cost, lower
is better.

## Hard Constraints

- You may change **signals only** — phase selection, phase durations, and offsets. You
  may not modify the network, the demand, or vehicle models.
- Your submission is the directory `methods/main/`. It may contain:
  - `controller.py` exposing `class Controller` with `setup(conn)` (called once with the
    live libsumo connection) and `step(t)` (called every simulation step), and/or
  - `tls.add.xml`, a static `tlLogic` program loaded as a SUMO additional file. It may
    **only** contain `tlLogic`/`phase`/`param` elements. The verifier snapshots the file
    and checks the snapshot against that whitelist before the simulation starts, so any
    other additional-file element (`rerouter`, `calibrator`, ...) — which would change the
    demand or the network — makes the run invalid (baseline score for every case).
- Signal programs are handed to SUMO as-is: it is the simulator, not a separate rule
  checker, that decides what a program does. There is **no** min-green / yellow-transition
  / conflict-matrix / cycle-length validator — a program SUMO accepts will run, and you
  are scored on the traffic it produces, not on whether it looks conventional. A solution
  that crashes, or whose additional-file is rejected, scores the baseline for that case.
- An online controller must be light: per-step decisions run inside the sim loop.

## What You Have

- `sim/eval_core.py` — the exact, frozen scoring core (the cost formula, demand window,
  and SUMO flags). Read it to understand precisely how you are scored.
- `methods/main/controller.py` — the weak baseline you edit (a no-op: the city runs its
  original static timing). `methods/main/tls_info.json` — per-junction phase/link
  structure you can use to build programs or controllers.
- `selfcheck.py` — runs your current solution on the **visible** load settings
  (scale 0.25 and 0.37) and prints the raw cost. Iterate against it.
- The full city network and demand are in the container, under `$SCENARIO_DATA`.

## What You Submit

Your edited `methods/main/` (a `controller.py` and/or a `tls.add.xml`, plus any helper
files it reads). The verifier re-runs it on the hidden settings using only the contract
above.

## How It Is Judged

The sealed verifier runs your solution on several hidden (load-scale, seed) cases drawn
from the same load band as the visible ones — some lighter, some heavier — and recomputes
the frozen cost per case. Each case's
cost is then **divided by that case's own no-op baseline cost** (the inherited static
timing on the identical settings) before averaging — so no single heavy case carries
hidden extra weight, and the no-op baseline scores exactly 1.0 by construction. Lower
mean normalised cost is better; your score rises monotonically as it drops. Beating the
inherited static timing is the goal, and the headroom is real: the inherited timing is a
fixed plan that never looks at a queue, on a network where queues build very unevenly.
It is also a genuine local optimum — a change that helps one corridor routinely costs
more somewhere downstream, so a method has to be judged city-wide, not junction by
junction.

**Time budget.** The budget is **simulation time, not wall clock**: every run simulates
exactly the fixed window in `sim/eval_core.py` (`BEGIN=25200` to `END=28200`, demand
until `DEMAND_END=27000`), then stops. Vehicles not served by `END` are charged
`W_UNFINISHED = 3600` s each — a smooth penalty, not a disqualification — so the cost of
a given submission is bit-identical on any machine. Separately, each hidden case has a
**1200 s wall-clock backstop**: a pathologically slow controller that overruns it scores
the baseline (normalised 1.0) for that case. Honest controllers with light per-step work
finish well inside it.
