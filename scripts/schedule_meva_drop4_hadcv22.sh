#!/usr/bin/env bash
# Schedule run_meva_drop4_hadcv22.py after a delay (default 2h). Uses sleep+nohup (no `at` required).
set -euo pipefail

REPO="/home/ssandree/Motion-guided-ROI/motion-detector"
DELAY="${1:-2h}"
LOG="${REPO}/outputs/meva_drop4_hadcv22_run.log"
PIDFILE="${REPO}/outputs/meva_drop4_hadcv22_scheduled.pid"
META="${REPO}/outputs/meva_drop4_hadcv22_scheduled.meta"

case "$DELAY" in
  *h) SEC=$(( ${DELAY%h} * 3600 )) ;;
  *m) SEC=$(( ${DELAY%m} * 60 )) ;;
  *s) SEC=$(( ${DELAY%s} )) ;;
  *) SEC="$DELAY" ;;
esac

if ! [[ "$SEC" =~ ^[0-9]+$ ]] || (( SEC < 1 )); then
  echo "usage: $0 [DELAY]   e.g. 2h | 90m | 7200" >&2
  exit 1
fi

mkdir -p "${REPO}/outputs"
START_AT="$(date -d "+${SEC} seconds" '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || date)"

if [[ -f "$PIDFILE" ]]; then
  old_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Already scheduled (pid $old_pid). Cancel: kill $old_pid" >&2
    cat "$META" 2>/dev/null || true
    exit 1
  fi
fi

nohup bash -c "
  sleep ${SEC}
  cd \"${REPO}\"
  source \"\$HOME/.local/opencv-cuda/env.sh\"
  echo \"===== MEVA run started \$(date '+%Y-%m-%d %H:%M:%S %Z') =====\"
  exec python scripts/run_meva_drop4_hadcv22.py
" >>"$LOG" 2>&1 &

sched_pid=$!
echo "$sched_pid" >"$PIDFILE"
{
  echo "scheduled_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "run_starts_at=${START_AT}"
  echo "delay_sec=${SEC}"
  echo "scheduler_pid=${sched_pid}"
  echo "log=${LOG}"
} >"$META"

echo "Scheduled in ${SEC}s (~${DELAY}). Pipeline starts about: ${START_AT}"
echo "Scheduler pid: ${sched_pid}  (cancel: kill ${sched_pid})"
echo "Log (appended): ${LOG}"
