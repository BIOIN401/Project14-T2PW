#!/usr/bin/env bash
# Retry a bounded_run job until the shared heavy lock is actually acquired.
# Exit 95 (BOUNDED_RUN_HEAVY_LOCK_HELD) means the child NEVER STARTED and is not
# a result, so it is retried rather than reported. The lock is never cleared,
# broken or stolen -- this only waits for its holder to release it.
set -u
LABEL="$1"; JSON="$2"; LOG="$3"; shift 3
VENV="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
cd /c/t/c110 || exit 1

for attempt in $(seq 1 40); do
  PYTHONPATH=/c/t/c110/src T2PW_OFFLINE_CURATOR=1 PYTHONIOENCODING=utf-8 \
    "$VENV" docs/pwml_recovery_sprint/evidence/bounded_run.py \
    --timeout 1800 --label "$LABEL" --heavy-lock C-110 --json "$JSON" \
    -- "$@" > "$LOG" 2>&1
  code=$?
  if [ "$code" -ne 95 ]; then
    echo "ACQUIRED on attempt $attempt; child exit=$code"
    exit "$code"
  fi
  echo "attempt $attempt: exit 95, lock held by $(grep -o '\"holder\": \"[^\"]*\"' /c/t/heavylock/holder.json 2>/dev/null | head -1); waiting"
  sleep 15
done
echo "GAVE UP after 40 attempts -- lock never free"
exit 95
