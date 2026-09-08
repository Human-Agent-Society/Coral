"""Weak baseline you inherit: a no-op controller — the city runs on its original 2013
static signal timing, untouched. Under load the raw scenario gridlocks, so its Cost is
large; your job is to bring it down.

Improve the control policy by editing `setup()`/`step()` below (and/or adding a
`tls.add.xml` static program next to this file). The verifier re-runs your solution on
sealed hidden load/seed settings and scores the same frozen Cost.

Contract (the sealed evaluator calls exactly this — there is no `solve()`):
    class Controller:
        def setup(self, conn): ...   # conn is the live libsumo module; cache TLS info here
        def step(self, t): ...       # called every sim step at time t; issue signal commands
You may ONLY change signals (phase selection / durations / offsets). You may not touch
the network, the demand, or vehicle models.

There is NO min-green / yellow-transition / conflict-matrix / cycle-length validator:
SUMO is the only arbiter, so any program it accepts will run and you are scored purely on
the traffic it produces. The only checks that exist are (a) SUMO must accept the run at
all, and (b) a `tls.add.xml` must contain nothing but `tlLogic`/`phase`/`param` elements.
A solution that crashes, or whose additional file is rejected, scores the baseline for
that case. See `instruction.md` "Hard Constraints".
"""


class Controller:
    def setup(self, conn):
        # No-op baseline: do not intervene — SUMO runs the built-in static programs.
        self.conn = conn

    def step(self, t):
        return
