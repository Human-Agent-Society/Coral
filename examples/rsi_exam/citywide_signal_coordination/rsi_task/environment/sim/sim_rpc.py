"""Filtered libsumo bridge: SUMO lives in the trusted process, the submitted
controller lives in a separate, unprivileged one.

Why this file exists
--------------------
The scoring inputs (SUMO's tripinfo output and the teleport / unfinished counters)
must be produced by the party that grades, not by the party that is graded. Before
this bridge the submitted `controller.py` was imported *into* the process that ran
SUMO, so it could stub out the libsumo counters and rewrite the tripinfo file. The
grader then "recomputed" the Cost from numbers the submission had authored.

Now:

    trusted process                       untrusted process (uid `solver`)
    ---------------                       -------------------------------
    libsumo.start()/simulationStep()
    reads the real counters
    writes the tripinfo it owns
    ControllerHost.setup()/step(t)  <-->  Controller.setup(conn)/step(t)
    serves + FILTERS every conn call      `conn` is a proxy, not libsumo

The controller never sees the tripinfo path, cannot write the directory it lives
in, and cannot reach the counters: every one of its libsumo calls is executed by
the trusted process and is checked against a policy first.

Call policy (this is the `conn` contract)
-----------------------------------------
* any read-only call on any libsumo domain -- method name starting with `get`,
  `subscribe` or `unsubscribe` (subscriptions are read-only and are the cheap way
  to poll many junctions per step);
* signal commands on the `trafficlight` domain only: setPhase, setPhaseDuration,
  setProgram, setRedYellowGreenState, setProgramLogic,
  setCompleteRedYellowGreenDefinition;
* `conn.trafficlight.Phase(...)` / `conn.trafficlight.Logic(...)` are constructed
  locally in the controller process and marshalled across.

Anything else (vehicle/lane/edge writers, simulationStep, close, load, ...) raises
`PermissionError` in the controller process. That is the task's own hard constraint
-- "you may change signals only" -- finally enforced instead of merely stated.

Wire format: 4-byte little-endian length + payload.
  untrusted -> trusted : JSON  (never unpickled: the parent must not execute
                                anything the child chooses)
  trusted   -> untrusted: pickle (trusted direction; keeps libsumo return types
                                  exact -- tuples stay tuples)
"""
from __future__ import annotations

import json
import os
import pickle
import signal
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_FRAME = struct.Struct("<I")
_MAX_FRAME = 32 << 20

# Wall-clock budgets for one controller turn. A controller that blows these is
# treated exactly like a crashing one (the case scores the baseline), which is the
# same outcome it had before when it made the case exceed case_timeout_s.
SETUP_TIMEOUT_S = 900.0
STEP_TIMEOUT_S = 120.0
CLOSE_TIMEOUT_S = 30.0

# --- policy -----------------------------------------------------------------
DOMAINS = frozenset((
    "busstop", "calibrator", "chargingstation", "edge", "gui", "inductionloop",
    "junction", "lane", "lanearea", "meandata", "multientryexit", "overheadwire",
    "parkingarea", "person", "poi", "polygon", "rerouter", "route", "routeprobe",
    "simulation", "trafficlight", "variablespeedsign", "vehicle", "vehicletype",
))
READ_PREFIXES = ("get", "subscribe", "unsubscribe")
SIGNAL_SETTERS = {
    "trafficlight": frozenset((
        "setPhase", "setPhaseDuration", "setProgram", "setRedYellowGreenState",
        "setProgramLogic", "setCompleteRedYellowGreenDefinition",
    )),
}
LOCAL_FACTORIES = {"trafficlight": frozenset(("Phase", "Logic"))}


class ControllerFault(RuntimeError):
    """The untrusted controller crashed, hung, or broke the protocol."""


def _allowed(domain, method):
    if domain not in DOMAINS or not method or method.startswith("_"):
        return False
    if method.startswith(READ_PREFIXES):
        return True
    return method in SIGNAL_SETTERS.get(domain, ())


# --- marshalling ------------------------------------------------------------
class RemotePhase:
    """Picklable stand-in for libsumo's TraCIPhase (SWIG objects cannot pickle)."""
    __slots__ = ("duration", "state", "minDur", "maxDur", "next", "name")

    def __init__(self, duration, state, minDur=None, maxDur=None, next=(), name=""):
        self.duration, self.state = duration, state
        self.minDur, self.maxDur = minDur, maxDur
        self.next, self.name = next, name

    def __repr__(self):
        return f"Phase(duration={self.duration!r}, state={self.state!r})"


class RemoteLogic:
    """Picklable stand-in for libsumo's TraCILogic."""
    __slots__ = ("programID", "type", "currentPhaseIndex", "phases", "subParameter")

    def __init__(self, programID, type=0, currentPhaseIndex=0, phases=(),
                 subParameter=None):
        self.programID, self.type = programID, type
        self.currentPhaseIndex = currentPhaseIndex
        self.phases = list(phases)
        self.subParameter = dict(subParameter or {})

    def getPhases(self):
        return list(self.phases)

    def __repr__(self):
        return f"Logic(programID={self.programID!r}, phases={len(self.phases)})"


def _is_swig(obj):
    return type(obj).__module__.split(".")[0] == "libsumo"


def _plain(obj):
    """Trusted -> untrusted: strip un-picklable SWIG wrappers, keep shapes exact."""
    if obj is None or isinstance(obj, (str, bytes, bool, int, float)):
        return obj
    if isinstance(obj, tuple):
        return tuple(_plain(o) for o in obj)
    if isinstance(obj, list):
        return [_plain(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if _is_swig(obj):
        if hasattr(obj, "phases") and hasattr(obj, "programID"):
            return RemoteLogic(obj.programID, obj.type, obj.currentPhaseIndex,
                               [_plain(p) for p in obj.getPhases()],
                               dict(getattr(obj, "subParameter", {}) or {}))
        if hasattr(obj, "state") and hasattr(obj, "duration"):
            return RemotePhase(obj.duration, obj.state, obj.minDur, obj.maxDur,
                               tuple(obj.next), obj.name)
        return repr(obj)
    try:
        pickle.dumps(obj, protocol=4)
    except Exception:  # noqa: BLE001
        return repr(obj)
    return obj


def _enc_arg(obj):
    """Untrusted -> trusted: JSON-safe encoding of controller call arguments."""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_enc_arg(o) for o in obj]
    if isinstance(obj, dict):
        return {str(k): _enc_arg(v) for k, v in obj.items()}
    if isinstance(obj, RemotePhase) or (_is_swig(obj) and hasattr(obj, "state")
                                        and hasattr(obj, "duration")):
        return {"__phase__": [obj.duration, obj.state, obj.minDur, obj.maxDur,
                              list(obj.next or ()), obj.name]}
    if isinstance(obj, RemoteLogic) or (_is_swig(obj) and hasattr(obj, "phases")
                                        and hasattr(obj, "programID")):
        phases = obj.getPhases() if hasattr(obj, "getPhases") else obj.phases
        return {"__logic__": [obj.programID, obj.type, obj.currentPhaseIndex,
                              [_enc_arg(p) for p in phases],
                              dict(getattr(obj, "subParameter", {}) or {})]}
    raise TypeError(f"argument of type {type(obj).__name__} cannot be sent to SUMO")


def _dec_arg(obj, ls):
    """Trusted side: rebuild real libsumo objects from the JSON the child sent."""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, list):
        return [_dec_arg(o, ls) for o in obj]
    if isinstance(obj, dict):
        if "__phase__" in obj and len(obj) == 1:
            d, st, mn, mx, nxt, nm = obj["__phase__"]
            return ls.trafficlight.Phase(float(d), str(st),
                                         float(mn if mn is not None else d),
                                         float(mx if mx is not None else d),
                                         tuple(int(i) for i in (nxt or ())), str(nm))
        if "__logic__" in obj and len(obj) == 1:
            pid, typ, cur, phases, sub = obj["__logic__"]
            lg = ls.trafficlight.Logic(str(pid), int(typ), int(cur),
                                       [_dec_arg(p, ls) for p in phases])
            for k, v in (sub or {}).items():
                try:
                    lg.subParameter[str(k)] = str(v)
                except Exception:  # noqa: BLE001
                    pass
            return lg
        return {str(k): _dec_arg(v, ls) for k, v in obj.items()}
    raise TypeError(f"unsupported argument encoding: {type(obj).__name__}")


# --- framed channel ---------------------------------------------------------
# ~5e6 round trips per case (625 junctions polled every other second for 9000 s),
# so this is a hot path: one write syscall per frame, no per-read select -- the
# turn deadline is an interval timer installed once per setup()/step() instead.
class _Timeout(BaseException):
    """Raised from SIGALRM. BaseException on purpose: it must not be swallowed by
    the `except Exception` that turns a failed libsumo call into an error reply."""


class _Channel:
    """Length-prefixed frames over a pair of raw fds."""

    def __init__(self, rfd, wfd):
        self._r, self._w = rfd, wfd
        self._buf = bytearray()

    def close(self):
        for fd in (self._r, self._w):
            try:
                os.close(fd)
            except OSError:
                pass

    def send(self, payload: bytes):
        buf = _FRAME.pack(len(payload)) + payload
        n, total = 0, len(buf)
        while n < total:
            n += os.write(self._w, buf[n:] if n else buf)

    def _fill(self, n):
        buf, rfd = self._buf, self._r
        while len(buf) < n:
            chunk = os.read(rfd, 262144)
            if not chunk:
                raise ControllerFault("controller process closed the channel")
            buf += chunk

    def recv(self) -> bytes:
        buf = self._buf
        if len(buf) < 4:
            self._fill(4)
        ln = _FRAME.unpack_from(buf)[0]
        if ln > _MAX_FRAME:
            raise ControllerFault(f"oversized frame ({ln} bytes)")
        end = 4 + ln
        if len(buf) < end:
            self._fill(end)
        out = bytes(buf[4:end])
        del buf[:end]
        return out


class _TurnDeadline:
    """Wall-clock bound on one controller turn, as an interval timer.

    Falls back to no timer if signals are unavailable (not the main thread); the
    grader's own case_timeout_s is the backstop in that case.
    """

    def __init__(self, seconds):
        self.seconds, self._prev, self._armed = seconds, None, False

    def __enter__(self):
        try:
            self._prev = signal.signal(signal.SIGALRM, _raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
            self._armed = True          # NOT `_prev is not None`: signal.signal()
        except (ValueError, OSError, AttributeError):   # returns None when the old
            self._armed = False         # handler was not set from Python, and the
        return self                     # timer would then never be disarmed

    def __exit__(self, *exc):
        if self._armed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                if self._prev is not None:
                    signal.signal(signal.SIGALRM, self._prev)
            except (ValueError, OSError):
                pass
        return False


def _raise_timeout(signum, frame):
    raise _Timeout()


# --- trusted side -----------------------------------------------------------
class ControllerHost:
    """Runs the submitted controller out-of-process and serves its libsumo calls.

    `controller_uid` is None on the agent side (self-check: same user, the boundary
    is only about fidelity there) and the dedicated unprivileged uid in the sealed
    verifier.
    """

    def __init__(self, solution_dir, controller_uid=None, controller_gid=None,
                 scratch=None, extra_env=None):
        self._closed = False
        self._fns = {}
        c2p_r, c2p_w = os.pipe()
        p2c_r, p2c_w = os.pipe()
        for fd in (c2p_w, p2c_r):
            os.set_inheritable(fd, True)
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": scratch or "/tmp",
            "TMPDIR": scratch or "/tmp",
            "XDG_CACHE_HOME": scratch or "/tmp",
            # BLAS/OpenMP pin for the untrusted controller process.
            # Additive constants only -- the trust-boundary logic around this dict is
            # unchanged. Constants, not os.environ lookups, on purpose.
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        # The SUMO_HOME passthrough is REMOVED. It let the grading
        # host point the untrusted controller's SUMO/libsumo installation somewhere else
        # (a simulator swap under the scorer), and it was dead weight anyway: neither
        # Dockerfile sets SUMO_HOME -- eclipse-sumo/libsumo come from pip and locate
        # themselves. Nothing here reads the environment any more.
        env.update(extra_env or {})
        kw = {}
        if controller_uid is not None:
            kw.update(user=controller_uid, group=controller_gid, extra_groups=[])
        self._proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "sim_rpc.py"), "--controller",
             solution_dir, str(p2c_r), str(c2p_w)],
            pass_fds=(p2c_r, c2p_w),
            stdin=subprocess.DEVNULL,
            stdout=2,          # controller prints land on stderr, never on the
            stderr=2,          # channel that carries the trusted counters
            cwd=scratch or None, env=env,
            start_new_session=True,   # its own process group; cannot signal us
            **kw)
        os.close(p2c_r)
        os.close(c2p_w)
        self._ch = _Channel(c2p_r, p2c_w)

    # -- protocol
    def _death_note(self):
        """Spell out HOW the controller process died, if it is gone.

        Without this, every way a controller can die -- a segfault, the OOM killer, or
        a SIGBUS from touching a shared-memory buffer bigger than the container's
        64 MiB /dev/shm -- is reported by the identical text "controller process closed
        the channel", which is also what a controller that quietly exited produces.
        The /dev/shm case is the nastiest of the three because the allocation SUCCEEDS
        and only the first write faults, so there is no traceback anywhere.
        """
        try:
            try:
                self._proc.wait(timeout=2)
            except Exception:  # noqa: BLE001  (still running / not waitable)
                pass
            rc = self._proc.poll()
        except Exception:  # noqa: BLE001
            return ""
        if rc is None:
            return ""
        if rc >= 0:
            return " -- controller process exited with returncode %d" % rc
        sig = -rc
        note = " -- controller process was killed by signal %d" % sig
        if sig == 7:
            note += (" (SIGBUS: typically a shared-memory/mmap buffer larger than the"
                     " container's 64 MiB /dev/shm; it is created successfully and"
                     " faults on first write)")
        elif sig == 9:
            note += " (SIGKILL: typically the OOM killer)"
        elif sig == 11:
            note += " (SIGSEGV)"
        return note

    def _cmd(self, obj, timeout):
        try:
            self._ch.send(pickle.dumps(obj, protocol=4))
            with _TurnDeadline(timeout):
                try:
                    self._serve()
                except _Timeout:
                    raise ControllerFault(
                        f"controller did not finish its turn within {timeout}s") from None
        except ControllerFault as exc:
            raise ControllerFault("%s%s" % (exc, self._death_note())) from None
        except BrokenPipeError as exc:
            raise ControllerFault("controller process closed the channel (%s)%s"
                                  % (exc, self._death_note())) from None

    def setup(self):
        self._cmd(("setup",), SETUP_TIMEOUT_S)

    def step(self, t):
        self._cmd(("step", t), STEP_TIMEOUT_S)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._cmd(("close",), CLOSE_TIMEOUT_S)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._ch.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            self._proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            pass

    def _bound(self, domain, method):
        """Resolve + policy-check once, then reuse (the same handful of methods is
        called millions of times)."""
        fn = self._fns.get((domain, method))
        if fn is None:
            if not _allowed(domain, method):
                raise PermissionError(
                    f"conn.{domain}.{method} is not permitted: this task allows "
                    f"read-only libsumo calls plus trafficlight signal commands only")
            import libsumo as ls
            fn = getattr(getattr(ls, domain), method)
            self._fns[(domain, method)] = fn
        return fn

    def _serve(self):
        """Answer the controller's libsumo calls until it reports the turn is done."""
        import libsumo as ls
        recv, send, dumps, loads = (self._ch.recv, self._ch.send,
                                    pickle.dumps, json.loads)
        while True:
            msg = loads(recv())
            kind = msg[0]
            if kind == "call":
                try:
                    _, domain, method, args = msg
                    fn = self._bound(domain, method)
                    if not (len(args) == 1 and type(args[0]) is str):
                        args = [_dec_arg(a, ls) for a in args]
                    reply = ("ok", _plain(fn(*args)))
                except Exception as exc:  # noqa: BLE001  (_Timeout is a BaseException)
                    reply = ("err", type(exc).__name__, str(exc)[:500])
                send(dumps(reply, protocol=4))
                continue
            if kind == "done":
                return
            if kind == "fail":
                raise ControllerFault(f"controller raised: {str(msg[1])[:500]}")
            raise ControllerFault(f"bad frame from controller: {str(kind)[:80]}")


# --- untrusted side ---------------------------------------------------------
class _Node:
    """`conn.<domain>` / `conn.<domain>.<method>` -- resolved on the trusted side.

    Nodes are memoised: `conn.trafficlight.getPhase` is looked up once per junction
    per step by a typical controller, and allocating two objects each time shows up.
    """
    __slots__ = ("_c", "_dom", "_meth", "_kids")

    def __init__(self, client, dom, meth=None):
        self._c, self._dom, self._meth = client, dom, meth
        self._kids = {}

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if self._meth is not None:
            raise AttributeError(f"conn.{self._dom}.{self._meth}.{name}")
        node = self._kids.get(name)
        if node is None:
            if name in LOCAL_FACTORIES.get(self._dom, ()):  # Phase / Logic ctors
                node = _local_factory(self._dom, name)
            else:
                node = _Node(self._c, self._dom, name)
            self._kids[name] = node
        return node

    def __call__(self, *args):
        if self._meth is None:  # e.g. conn.simulationStep() -- module-level call
            return self._c.call("", self._dom, args)
        return self._c.call(self._dom, self._meth, args)

    def __repr__(self):
        return f"<conn.{self._dom}{'.' + self._meth if self._meth else ''}>"


_BUILTIN_EXC = {e.__name__: e for e in (PermissionError, TypeError, ValueError,
                                        KeyError, AttributeError, IndexError,
                                        NotImplementedError)}


def _exc_class(name):
    """Re-raise the trusted side's failure as the same class the controller would
    have seen in-process -- including libsumo's own TraCIException."""
    if name in _BUILTIN_EXC:
        return _BUILTIN_EXC[name]
    try:
        import libsumo as _ls
        cls = getattr(_ls, name, None) or getattr(_ls.exceptions, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            return cls
    except Exception:  # noqa: BLE001
        pass
    return RuntimeError


def _local_factory(domain, name):
    try:
        import libsumo as _ls
        return getattr(getattr(_ls, domain), name)
    except Exception:  # noqa: BLE001
        return RemotePhase if name == "Phase" else RemoteLogic


class RemoteConn:
    """What the submitted `Controller.setup()` receives instead of `libsumo`."""

    def __init__(self, channel):
        self._ch = channel
        self._doms = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        node = self._doms.get(name)
        if node is None:
            node = self._doms[name] = _Node(self, name)
        return node

    def call(self, domain, method, args):
        if not (len(args) == 1 and type(args[0]) is str):
            args = [_enc_arg(a) for a in args]
        ch = self._ch
        ch.send(json.dumps(["call", domain, method, args],
                           separators=(",", ":")).encode())
        rep = pickle.loads(ch.recv())
        if rep[0] == "ok":
            return rep[1]
        raise _exc_class(rep[1])(rep[2])


def _controller_main(solution_dir, rfd, wfd):
    import importlib
    import traceback

    try:    # die with the simulator: an orphaned controller stuck in step() would
            # otherwise spin until the container exits (PR_SET_PDEATHSIG, SIGKILL)
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 9, 0, 0, 0)
    except Exception:  # noqa: BLE001
        pass

    ch = _Channel(rfd, wfd)
    conn = RemoteConn(ch)
    controller = None
    while True:
        try:
            cmd = pickle.loads(ch.recv())
        except ControllerFault:
            return 0
        try:
            if cmd[0] == "setup":
                sys.path.insert(0, solution_dir)
                mod = importlib.import_module("controller")
                importlib.reload(mod)
                controller = mod.Controller()
                controller.setup(conn)
            elif cmd[0] == "step":
                if controller is not None:
                    controller.step(cmd[1])
            elif cmd[0] == "close":
                ch.send(json.dumps(["done"]).encode())
                return 0
            else:
                raise RuntimeError(f"unknown command {cmd[0]!r}")
        except BaseException as exc:  # noqa: BLE001
            traceback.print_exc()
            try:
                ch.send(json.dumps(
                    ["fail", f"{type(exc).__name__}: {str(exc)[:400]}"]).encode())
            except Exception:  # noqa: BLE001
                pass
            return 1
        ch.send(json.dumps(["done"]).encode())


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--controller":
        sys.exit(_controller_main(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))
    sys.stderr.write("sim_rpc.py is not meant to be run directly\n")
    sys.exit(2)
