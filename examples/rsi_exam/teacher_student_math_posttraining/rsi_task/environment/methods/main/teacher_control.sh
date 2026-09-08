#!/usr/bin/env bash
# Start/stop only this task's teacher process group. Never searches or kills by GPU.
set -euo pipefail
ACTION=${1:-status}
MODE=${2:-dp2}
PIDFILE=/app/run/teacher.pid
LOG=/app/results/teacher.log
PORT=${TEACHER_PORT:-8000}

alive() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "$ACTION" in
  start)
    if alive; then echo "teacher already running pid=$(cat "$PIDFILE")"; exit 1; fi
    rm -f "$PIDFILE"
    if [ "$MODE" = dp2-short ]; then
      export CUDA_VISIBLE_DEVICES=0,1
      max_model_len=4096
      gpu_util=0.82
      extra=(--data-parallel-size 2 --max-num-seqs 64)
    elif [ "$MODE" = dp2 ] || [ "$MODE" = dp2-long ]; then
      export CUDA_VISIBLE_DEVICES=0,1
      max_model_len=16384
      gpu_util=0.90
      extra=(--data-parallel-size 2 --max-num-seqs 8)
    elif [ "$MODE" = gpu1 ]; then
      export CUDA_VISIBLE_DEVICES=1
      max_model_len=4096
      gpu_util=0.82
      extra=(--max-num-seqs 32)
    else
      echo "mode must be dp2-short, dp2-long, dp2, or gpu1" >&2; exit 2
    fi
    mkdir -p /app/run /app/results
    nohup setsid /opt/venvs/vllm/bin/vllm serve /app/models/teacher \
      --served-model-name teacher --host 127.0.0.1 --port "$PORT" \
      --max-model-len "$max_model_len" --gpu-memory-utilization "$gpu_util" "${extra[@]}" >"$LOG" 2>&1 &
    pid=$!; echo "$pid" >"$PIDFILE"
    for _ in $(seq 1 300); do
      if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "teacher ready mode=$MODE pid=$pid port=$PORT"; exit 0
      fi
      if ! kill -0 "$pid" 2>/dev/null; then tail -100 "$LOG"; rm -f "$PIDFILE"; exit 1; fi
      sleep 2
    done
    echo "teacher startup timed out" >&2; exit 1
    ;;
  stop)
    if ! alive; then rm -f "$PIDFILE"; echo "teacher not running"; exit 0; fi
    pid=$(cat "$PIDFILE")
    cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
    case "$cmd" in *vllm*serve*/app/models/teacher*) ;; *) echo "PID validation failed; refusing to kill $pid" >&2; exit 1;; esac
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$pgid" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "teacher process group stopped"
    ;;
  status)
    if alive; then echo "teacher running pid=$(cat "$PIDFILE")"; else echo "teacher stopped"; fi
    ;;
  *) echo "usage: $0 {start|stop|status} [dp2-short|dp2-long|dp2|gpu1]" >&2; exit 2;;
esac
