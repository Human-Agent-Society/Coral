"""Trusted grading parent for the held-out memory-QA task.

Flow (two-phase, per docs/anti-cheat.md):
  1. load the sealed holdout (6 unseen conversations, 1,174 QA) into memory,
     write a stripped copy (sessions + questions, no answers), delete the
     sealed file from disk;
  2. start a local proxy that pins every LLM request to the fixed base model
     and holds the real API key (the child never sees it), then run the
     untrusted child, which imports the agent's memory system, ingests all
     sessions once, and answers every question;
  3. score token-level F1 in this parent and map the mean through the
     anchors -> reward.json + score_details.json.
"""

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# A-only: the upstream has no AAAA record and the dual-stack lookup fails behind the
# egress sidecar. lockdown_network() below pins whatever this returns into /etc/hosts.
_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_v4(host, port, family=0, *args, **kwargs):
    if family in (0, socket.AF_UNSPEC):
        family = socket.AF_INET
    return _getaddrinfo(host, port, family, *args, **kwargs)


if os.environ.get("MEMORY_LLM_DUAL_STACK", "") != "1":
    socket.getaddrinfo = _getaddrinfo_v4

# Hardcoded: a host that can set these with -e can move the score without
# leaving a trace. Author-side re-measurement edits a copy of this tree.
PIN_MODEL = "gpt-4o-mini"
CHILD_TIMEOUT = 9600
OUT = "/logs/verifier/reward.json"
HERE = os.path.dirname(os.path.abspath(__file__))
SEALED = os.path.join(HERE, "heldout", "holdout_sealed.json")
ANCHORS = os.path.join(HERE, "anchors.json")

# Filled by load_anchors() before the child starts. No env fallback: a missing
# file must fail loudly, not silently reprice the band.
BASELINE = None
UPPER_BOUND = None


def load_anchors():
    """Read the sealed band (root-owned 0400) while still root."""
    global BASELINE, UPPER_BOUND
    with open(ANCHORS) as f:
        a = json.load(f)
    BASELINE, UPPER_BOUND = float(a["baseline"]), float(a["upper_bound"])

UPSTREAM_BASE = os.environ.get("MEMORY_LLM_API_BASE",
                               os.environ.get("OPENAI_API_BASE",
                                              "https://api.openai.com/v1")).rstrip("/")
UPSTREAM_KEY = os.environ.get("MEMORY_LLM_API_KEY",
                              os.environ.get("OPENAI_API_KEY", ""))

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
    "once": "1", "twice": "2", "thrice": "3", "single": "1", "double": "2",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
}


def _tokenize(s):
    return [_NUMBER_WORDS.get(t, t)
            for t in _PUNCT_RE.sub(" ", str(s).lower()).split()]


def token_f1(prediction, reference):
    p, r = _tokenize(prediction), _tokenize(reference)
    if not p or not r:
        return 0.0
    rc = list(r)
    c = 0
    for t in p:
        if t in rc:
            c += 1
            rc.remove(t)
    if c == 0:
        return 0.0
    pr, rec = c / len(p), c / len(r)
    return 2 * pr * rec / (pr + rec)


class PinnedModelProxy(BaseHTTPRequestHandler):
    """Forward chat-completion requests upstream with the model forced to
    PIN_MODEL and the real key attached. The child only ever talks to this."""

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            body["model"] = PIN_MODEL
            body.pop("stream", None)
            req = urllib.request.Request(
                UPSTREAM_BASE + self.path.removeprefix("/v1"),
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {UPSTREAM_KEY}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = resp.read()
                self.send_response(resp.status)
        except urllib.error.HTTPError as e:
            payload = e.read()
            self.send_response(e.code)
        except Exception as e:
            payload = json.dumps({"error": str(e)}).encode()
            self.send_response(502)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def reward_of(metric):
    # two-point linear: BASELINE -> 0, UPPER_BOUND -> 1 (clamped to [0, 1])
    r = (metric - BASELINE) / max(1e-12, UPPER_BOUND - BASELINE)
    return float(min(1.0, max(0.0, r)))


def write_out(reward, extra, note=""):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    payload = {"reward": round(float(reward), 6),
               **{k: round(float(v), 6) for k, v in extra.items()}}
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    if note:
        print(note)


def fail_closed(msg):
    """Grader-side failure: reward 0 with a marker the summariser can tell apart
    from a submission that simply scored 0."""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"reward": 0.0, "error": f"grader failed: {msg}"}, f, indent=2)
    print(f"grader failed: {msg}")


def lockdown_network():
    """Inside the verifier container only: pin the model-API host in
    /etc/hosts, kill DNS, and return the unprivileged user for the child.
    The verifier needs the net solely for the pinned model endpoint; the
    child (which runs the submitted code) must not be able to resolve
    anything else — the holdout's source data is public online."""
    if not os.path.exists("/.dockerenv") or os.geteuid() != 0:
        return None  # local anchor-measurement run: leave the host alone
    import socket
    from urllib.parse import urlparse
    host = urlparse(UPSTREAM_BASE).hostname
    # Fail closed: without the pin there is no drop to `nobody`, and the child
    # would run as root with the holdout's public source one lookup away.
    ip = socket.getaddrinfo(host, 443)[0][4][0]
    with open("/etc/hosts", "w") as f:
        f.write(f"127.0.0.1 localhost\n{ip} {host}\n")
    with open("/etc/resolv.conf", "w") as f:
        f.write("nameserver 127.0.0.1\n")
    subprocess.run(["chmod", "-R", "a+rX", "/app/methods", "/opt/hf"],
                   check=False)
    print(f"lockdown: DNS disabled, {host} pinned to {ip}, child runs as nobody")
    return "nobody"


# Opens each path as whoever runs it. Kept as source so it can be handed to a
# subprocess that has actually dropped privileges.
_READ_PROBE = """
import os, sys
for p in sys.argv[1:]:
    if os.path.isfile(p):
        open(p, "rb").read(1)
        continue
    for dirpath, _d, files in os.walk(p):
        if files:
            open(os.path.join(dirpath, files[0]), "rb").read(1)
            break
    else:
        raise SystemExit("nothing readable under " + p)
"""


def assert_readable_as(user, *paths):
    """The child reads the submission and the cached embedders after dropping
    to `user`. os.access() here would answer for root, and a stat() check misses
    parent-directory x bits, ACLs and mount options -- so actually drop and open.
    Unreadable artifacts otherwise surface as 'no predictions', i.e. a silent 0
    indistinguishable from a bad submission."""
    r = subprocess.run([sys.executable, "-c", _READ_PROBE, *paths],
                       user=user, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{user} cannot read {list(paths)}: "
                           f"{(r.stderr or r.stdout).strip()[-300:]}")


def load_holdout():
    """Sealed file -> merged sessions + QA list (same loader logic as the
    visible eval; sessions namespaced per conversation)."""
    with open(SEALED) as f:
        raw = json.load(f)
    sessions, qa_pairs = [], []
    for si, s in enumerate(raw):
        conv = s["conversation"]
        sample_id = str(s.get("sample_id", si))
        session_keys = sorted(
            [k for k in conv
             if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda x: int(x.split("_")[1]),
        )
        for sk in session_keys:
            turns_raw = conv[sk]
            if isinstance(turns_raw, str):
                try:
                    turns_raw = json.loads(turns_raw)
                except json.JSONDecodeError:
                    turns_raw = []
            turns = [{"speaker": t.get("speaker", "?"), "text": t.get("text", "")}
                     for t in (turns_raw or [])]
            sessions.append([f"{sample_id}::{sk}", conv.get(f"{sk}_date_time", ""),
                             turns])
        for qa in s.get("qa", []):
            ref = qa.get("answer") or qa.get("adversarial_answer", "")
            qa_pairs.append({"conversation": sample_id,
                             "question": qa["question"],
                             "answer": str(ref),
                             "category": int(qa.get("category", 0))})
    return sessions, qa_pairs


def main():
    try:
        load_anchors()
    except Exception as e:
        return fail_closed(f"anchors unreadable ({ANCHORS}): {e}")

    # phase 1: gold into memory, stripped questions to disk, sealed off disk
    sessions, qa_pairs = load_holdout()
    os.remove(SEALED)
    questions = [{"qid": i, "question": qa["question"]}
                 for i, qa in enumerate(qa_pairs)]
    stripped_path, preds_path = "/tmp/holdout_questions.json", "/tmp/predictions.json"
    with open(stripped_path, "w") as f:
        json.dump({"sessions": sessions, "questions": questions}, f)

    # phase 2: pinned-model proxy + untrusted child
    if not UPSTREAM_KEY:
        return write_out(0.0, {"valid": 0},
                         note="MEMORY_LLM_API_KEY missing in verifier env")
    try:
        child_user = lockdown_network()
        if child_user:
            assert_readable_as(child_user, "/app/methods", "/opt/hf")
    except Exception as e:
        return fail_closed(f"sandbox setup: {e}")
    server = ThreadingHTTPServer(("127.0.0.1", 0), PinnedModelProxy)
    proxy_port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    child_env = {k: v for k, v in os.environ.items()
                 if k not in ("OPENAI_API_KEY", "OPENAI_BASE_URL",
                              "MEMORY_LLM_API_KEY",
                              "HTTP_PROXY", "HTTPS_PROXY",
                              "http_proxy", "https_proxy")}
    cache_dir = tempfile.mkdtemp(prefix="memsys_cache_")
    os.chmod(cache_dir, 0o777)
    child_env.update({
        "OPENAI_API_BASE": f"http://127.0.0.1:{proxy_port}/v1",
        "OPENAI_API_KEY": "sealed-proxy",
        "LLM_MODEL": PIN_MODEL,
        "MEMSYS_CACHE_DIR": cache_dir,
    })
    child_kwargs = {"user": child_user} if child_user else {}
    try:
        subprocess.run(
            [sys.executable, os.path.join(HERE, "run_holdout.py"),
             stripped_path, preds_path],
            check=True, timeout=CHILD_TIMEOUT, env=child_env, **child_kwargs,
        )
        with open(preds_path) as f:
            preds = json.load(f)
    except Exception as e:
        return write_out(0.0, {"valid": 0}, note=f"memory-system child failed: {e}")
    finally:
        server.shutdown()

    # phase 3: score in the trusted parent
    details, per_cat, per_conv = [], {}, {}
    for i, qa in enumerate(qa_pairs):
        pred = preds.get(str(i), "")
        s = token_f1(pred, qa["answer"])
        details.append({"qid": i, "conversation": qa["conversation"],
                        "category": qa["category"],
                        "prediction": pred[:120], "f1": round(s, 4)})
        per_cat.setdefault(qa["category"], []).append(s)
        per_conv.setdefault(qa["conversation"], []).append(s)

    metric = sum(d["f1"] for d in details) / len(details)
    det_path = os.path.join(os.path.dirname(OUT), "score_details.json")
    os.makedirs(os.path.dirname(det_path), exist_ok=True)
    with open(det_path, "w") as f:
        json.dump({"instances": details,
                   "anchors": {"baseline": BASELINE, "upper_bound": UPPER_BOUND},
                   "metric": metric,
                   "per_conversation": {c: round(sum(v) / len(v), 4)
                                        for c, v in sorted(per_conv.items())}},
                  f, indent=1)

    write_out(reward_of(metric), {
        "valid": 1, "metric": round(metric, 4),
        "n_questions": len(details),
        **{f"f1_cat{c}": round(sum(v) / len(v), 4)
           for c, v in sorted(per_cat.items())},
    })


if __name__ == "__main__":
    main()
