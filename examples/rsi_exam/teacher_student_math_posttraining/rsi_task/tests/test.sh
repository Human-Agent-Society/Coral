#!/usr/bin/env bash
set -uo pipefail
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.txt /logs/verifier/reward.json
python3 /tests/grade.py
RC=$?
if [ "$RC" -ne 0 ] || [ ! -s /logs/verifier/reward.json ]; then
  # Preserve any grader-failure marker, otherwise synthesize one.
  ERR=""
  if [ -s /logs/verifier/grade_debug.json ]; then
    ERR=$(python3 -c "import json,sys; d=json.load(open('/logs/verifier/grade_debug.json')); e=d.get('error') or d.get('errors') or ''; print(e if isinstance(e,str) else json.dumps(e))" 2>/dev/null || true)
  fi
  case "$ERR" in
    grader\ failed:*) ;;
    *) ERR="grader failed: grade.py exit $RC" ;;
  esac
  # reward.json must be dict[str, float|int]: a string value makes harbor reject
  # the whole trial (ValidationError), and any key sorting before "reward" is what
  # the hub displays instead of the reward. The message goes to grade_debug.json.
  printf '{"reward":0.0}\n' >/logs/verifier/reward.json
  python3 -c "import json,sys,pathlib
p=pathlib.Path('/logs/verifier/grade_debug.json')
d={}
try: d=json.loads(p.read_text())
except Exception: pass
if not isinstance(d,dict): d={'previous':d}
d['error']=sys.argv[1]; d['grader_failed']=1
p.write_text(json.dumps(d,indent=2))" "$ERR"
fi
# reward.json is authoritative; keep reward.txt in sync for any reader that only checks it.
printf '%s\n' "$(python3 -c "import json; print(json.load(open('/logs/verifier/reward.json'))['reward'])")" >/logs/verifier/reward.txt
# Harbor 0.21 parses these files through the host side of a Docker bind mount
# before its container-stop ownership repair runs. Keep both success and fallback
# outputs host-readable while remaining non-writable to unprivileged users.
chmod 0644 \
  /logs/verifier/reward.json \
  /logs/verifier/reward.txt \
  /logs/verifier/grade_debug.json \
  /logs/verifier/score_details.json \
  2>/dev/null || true
cat /logs/verifier/reward.txt
