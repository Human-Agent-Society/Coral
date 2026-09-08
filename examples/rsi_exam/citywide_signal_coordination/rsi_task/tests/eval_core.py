"""Shared SUMO evaluation core for the city-wide signal-coordination task.

This is the PUBLIC rule set — the frozen scoring protocol. The visible selfcheck and
the sealed grader both call it. Nothing here is secret; what stays sealed is the hidden
load/seed protocol baked into the verifier image.

Cost (lower is better):
    Cost = 1.0   * sum(timeLoss)      seconds of delay over arrived vehicles
         + 160   * teleports          SUMO teleports, charged the time-to-teleport
         + 20    * sum(waitingCount)  number of stops (energy/emission proxy)
         + 3600  * unfinished         vehicles still in-network / pending at end

W_TELEPORT was 10000 until the 2026-08-05 re-anchor and that value INVERTED the
metric.  A teleport is the simulator removing a vehicle that has been stuck for
`time-to-teleport` = 160 s and re-inserting it downstream; charging it 10 000 s
priced one such event at 62x the delay it certifies, so Cost effectively counted
teleports.  Measured consequence: a standard acyclic max-pressure controller cuts
sum(timeLoss) by 15-18% and stops by 4% at every load tested, and the old weight
scored it WORSE than doing nothing at scale 0.35 / 0.40 / 0.50 / 0.70.  At 160 the
same measurements rank it correctly (0.972 / 0.976 at 0.35 / 0.40).  Gridlock is
still charged -- a teleported vehicle keeps its timeLoss, and if it never arrives
it still costs W_UNFINISHED -- it is simply no longer charged 62x over.
(Ratios at W_TELEPORT = 0 differ from those at 160 by < 0.001, so the metric is not
sensitive to the exact value; 160 is chosen because it is the one value the
simulation itself certifies.)

Time budget is SIMULATION time, not wall clock: every run simulates exactly the fixed
window 25200-27000 (injection, the 07:00-07:30 morning peak) + a 1200 s drain to 28200
(no new vehicles), then stops.  Vehicles not served by END are charged W_UNFINISHED
each — a smooth penalty, not a disqualification.  Because truncation is by simulation
time, the Cost of a given (solution, scale, seed) is bit-identical on any machine.
(v2 window, 2026-08 re-anchor.  The v1 window 21600-28800+drain-to-30600 is retired;
all v1 anchor measurements are invalid for v2.)

Fixed time-to-teleport=160, max-depart-delay=900. A solution directory may contain:
    tls.add.xml    a static signal program, loaded as an additional file.  It may ONLY
                   contain <tlLogic>/<phase>/<param> elements: anything else a SUMO
                   additional file can carry (rerouter, calibrator, vaporizer, ...)
                   would change the demand or the network, which the task forbids.
                   A file that fails this check makes the run invalid (baseline score).
    controller.py  an online controller exposing `class Controller: setup(conn); step(t)`

Where the whitelist runs (trust boundary).  `freeze_and_validate_additional()` snapshots
tls.add.xml to a private path and validates that snapshot; `run_sim()` then hands SUMO
only the snapshot.  When grading, the snapshot and its validation happen in a trusted process that never
imports the submission, and only the resulting path is handed to the untrusted one,
which does not re-validate; monkeypatching anything in this module from submitted code
therefore cannot weaken the whitelist.
Only the agent-visible selfcheck lets run_sim() do its own freeze+validate — on that
side of the boundary the caller and the submission are the same party anyway.

Where the submitted controller runs (trust boundary, part 2).
SUMO runs HERE, in the process that also writes the tripinfo and reads the teleport /
unfinished counters.  The submitted `controller.py` runs in a SEPARATE process
(unprivileged when grading) and reaches SUMO only through the filtered proxy in
`sim_rpc.py`: read-only libsumo calls plus `trafficlight` signal commands.  Everything
else — `vehicle.remove`, `simulationStep`, `close`, route/calibrator writers — raises
PermissionError in the controller process.  Until this merge `run_sim` imported
controller.py into its own process and handed `controller.setup()` the bare `libsumo`
module, so "you may change signals only" was stated but not enforced, and the tripinfo
path travelled in the argv of the process that imported the submission.  Same rule on
both sides of the image, so a controller that works under `selfcheck.py` works under the
sealed grader.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET

from sim_rpc import ControllerHost  # same directory; see its module docstring

BEGIN, DEMAND_END, END = 25200, 27000, 28200
# W_TELEPORT = the simulator's own time-to-teleport (160 s below): the delay a
# teleported vehicle demonstrably suffered.  See the module docstring for the
# measurements that retired the previous 10000.0.
W_TIMELOSS, W_TELEPORT, W_STOP, W_UNFINISHED = 1.0, 160.0, 20.0, 3600.0

# tls.add.xml whitelist: signal programs only (see docstring).  frozenset, not set: the
# validator runs inside the trusted grader process, and a mutable container would be one
# `ADD_ALLOWED_TAGS.add("rerouter")` away from disabling the whitelist.
ADD_ROOT_TAGS = frozenset({"additional", "additionals", "add"})
ADD_ALLOWED_TAGS = frozenset({"tlLogic", "phase", "param"})
# A signal program for ~600 junctions is well under a megabyte; past this limit a file is
# either a parser-DoS payload or not a signal program at all.
ADDITIONAL_MAX_BYTES = 64 << 20
_ADD_HEAD_BYTES = 1 << 16


# Where the scenario lives differs by image -- the verifier bakes it at /data/scenario,
# the agent environment at /app/data/scenario -- so this module has to resolve one of two
# fixed locations.  It does NOT read the environment: eval_core runs
# inside the TRUSTED grader process, so an env-settable data root would let whoever can set
# the container environment choose which network the grader simulates, i.e. what the
# submission is scored against.  Resolution is by existence over a hard-coded tuple.
_SCENARIO_ROOTS = ("/data/scenario", "/app/data/scenario")


def _paths():
    for data in _SCENARIO_ROOTS:
        net = os.path.join(data, "net.xml")
        if os.path.exists(net):
            return net, os.path.join(data, "demand.rou.xml")
    raise RuntimeError(
        "scenario data not found in any of " + ", ".join(_SCENARIO_ROOTS))


def validate_tls_additional(path):
    """Reject any tls.add.xml element that is not a signal program.  Raises ValueError.

    This runs inside the TRUSTED grader process, so it is itself an attack surface: it
    refuses oversized files and any DTD / internal-entity declaration before handing the
    bytes to ElementTree, which expands internal entities and would otherwise inflate a
    billion-laughs payload inside the grader.
    """
    size = os.path.getsize(path)
    if size > ADDITIONAL_MAX_BYTES:
        raise ValueError(f"tls.add.xml: {size} bytes exceeds the "
                         f"{ADDITIONAL_MAX_BYTES}-byte limit for a signal program")
    with open(path, "rb") as fh:
        head = fh.read(_ADD_HEAD_BYTES)
    for marker in (b"<!DOCTYPE", b"<!ENTITY"):
        if marker in head:
            raise ValueError(f"tls.add.xml: {marker.decode()} declarations are not "
                             "allowed (entity expansion)")
    root = ET.parse(path).getroot()
    if root.tag not in ADD_ROOT_TAGS:
        raise ValueError(f"tls.add.xml: root element <{root.tag}> is not allowed "
                         f"(expected one of {sorted(ADD_ROOT_TAGS)})")
    for el in root.iter():
        if el is root or not isinstance(el.tag, str):
            continue
        if el.tag not in ADD_ALLOWED_TAGS:
            raise ValueError(f"tls.add.xml: element <{el.tag}> is not allowed — "
                             f"signal programs only ({sorted(ADD_ALLOWED_TAGS)})")


def freeze_and_validate_additional(solution_dir):
    """Snapshot <solution_dir>/tls.add.xml into a private directory, validate the
    SNAPSHOT, and return its path — or None when the solution ships no additional file.

    Validating the copy rather than the original closes the time-of-check/time-of-use
    window: whatever the submission does to its own tls.add.xml afterwards no longer
    changes the bytes SUMO parses.  The caller owns the returned path and should remove
    its parent directory when done.  Raises ValueError if the snapshot is not a pure
    signal program.
    """
    if not solution_dir:
        return None
    src = os.path.join(solution_dir, "tls.add.xml")
    if not os.path.exists(src):
        return None
    dst_dir = tempfile.mkdtemp(prefix="tls_checked_")
    dst = os.path.join(dst_dir, "tls.add.xml")
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst)
    validate_tls_additional(dst)
    return dst


def run_sim(solution_dir, scale, seed, tripinfo_out, add_frozen=None,
            controller_uid=None, controller_gid=None, controller_scratch=None):
    """Run one SUMO simulation with the given solution. Returns raw counters; the
    caller computes Cost (so the trusted parent owns the scoring, not the child).

    The counters below and the tripinfo file belong to THIS process.  The submitted
    controller runs in a separate process started by `sim_rpc.ControllerHost` and can
    reach neither: it never learns `tripinfo_out`, and every libsumo call it makes is
    executed here after a policy check.  `controller_uid`/`controller_gid` drop that
    process to an unprivileged uid (the sealed grader passes 65534; the agent-visible
    selfcheck passes None, where caller and submission are the same party anyway), and
    `controller_scratch` is the only directory it is given to write in.

    `add_frozen`: a path produced by `freeze_and_validate_additional()` in a process that
    does NOT import the submission.  When given it is used verbatim and NOTHING here
    re-checks it — the check already happened on the trusted side of the boundary, and
    re-checking would only re-expose the whitelist to the submission's module-level code.
    This is the path the grader takes.  The empty string means "the trusted side looked
    and there is no additional file": also use nothing, also validate nothing.

    Only when `add_frozen` is None (the agent-visible selfcheck, no trust boundary) does
    run_sim freeze and validate the solution's tls.add.xml itself — before importing the
    controller — and delete the snapshot afterwards.
    """
    import libsumo as ls

    net, rou = _paths()
    cmd = [
        "sumo", "-n", net, "-r", rou,
        "--begin", str(BEGIN), "--end", str(END),
        "--route-steps", "200", "--no-internal-links",
        "--scale", str(scale), "--time-to-teleport", "160",
        "--max-depart-delay", "900", "--seed", str(seed),
        "--no-warnings", "--no-step-log",
        "--tripinfo-output", tripinfo_out,
    ]

    owned_add = None
    if add_frozen is None:
        add_frozen = owned_add = freeze_and_validate_additional(solution_dir)
    if add_frozen:
        cmd += ["--additional-files", add_frozen]

    # The submission is NOT imported here.  ControllerHost forks a separate process for
    # it and serves its libsumo calls through the sim_rpc policy filter.
    host = None
    if solution_dir and os.path.exists(os.path.join(solution_dir, "controller.py")):
        host = ControllerHost(solution_dir, controller_uid, controller_gid,
                              controller_scratch)

    t0 = time.time()
    try:
        ls.start(cmd)
        if host:
            host.setup()
        teleports = 0
        t = BEGIN
        # Deterministic termination guard: the loop is bounded by SIMULATION time (END),
        # never by wall clock; max_steps only protects against a stuck clock.
        max_steps = (END - BEGIN) + 100
        steps = 0
        while t < END and steps < max_steps:
            ls.simulationStep()
            steps += 1
            teleports += ls.simulation.getStartingTeleportNumber()
            t = ls.simulation.getTime()
            if host:
                host.step(t)
            if ls.simulation.getMinExpectedNumber() == 0 and t > DEMAND_END:
                break
        running_at_end = ls.vehicle.getIDCount()
        pending = ls.simulation.getPendingVehicles()
        ls.close()
    finally:
        if host:
            host.close()
    if owned_add:
        shutil.rmtree(os.path.dirname(owned_add), ignore_errors=True)
    return {"teleports": teleports,
            "unfinished": running_at_end + len(pending),
            "wall_s": round(time.time() - t0, 1)}


def cost_from_tripinfo(tripinfo_out, counters):
    """Trusted scoring: parse SUMO's own tripinfo output + the raw counters into Cost."""
    total_timeloss = total_stops = arrived = 0
    for _, el in ET.iterparse(tripinfo_out):
        if el.tag == "tripinfo":
            total_timeloss += float(el.get("timeLoss"))
            total_stops += int(el.get("waitingCount"))
            arrived += 1
            el.clear()
    cost = (W_TIMELOSS * total_timeloss + W_TELEPORT * counters["teleports"]
            + W_STOP * total_stops + W_UNFINISHED * counters["unfinished"])
    return {"cost": round(cost), "timeLoss": round(total_timeloss),
            "teleports": counters["teleports"], "stops": total_stops,
            "arrived": arrived, "unfinished": counters["unfinished"]}


def evaluate(solution_dir, scale, seed, tripinfo_out, add_frozen=None, **kw):
    counters = run_sim(solution_dir, scale, seed, tripinfo_out, add_frozen, **kw)
    res = cost_from_tripinfo(tripinfo_out, counters)
    res.update(scale=scale, seed=seed, wall_s=counters["wall_s"])
    return res
