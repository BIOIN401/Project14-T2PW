#!/bin/bash
# usage: runpy.sh LABEL OUTDIR TIP_TREE script.py args...   (bounded_run + heavy lock; PYTHONPATH=TIP_TREE/src)
LABEL=$1; OUT=$2; TIP=$3; shift 3
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
mkdir -p "$OUT"
cd "$TIP"
export PYTHONPATH="$TIP/src" T2PW_OFFLINE_CURATOR=1
for attempt in $(seq 1 45); do
  $PY "$TIP/docs/pwml_recovery_sprint/evidence/bounded_run.py" --label "$LABEL" --timeout 300 --heavy-lock R4b --json "$OUT/$LABEL.json" -- \
    $PY -u "$@" > "$OUT/$LABEL.log" 2>&1
  rc=$?
  [ "$rc" = "95" ] || break
  echo "lock busy (attempt $attempt), waiting 8s"; sleep 8
done
echo "wrapper exit: $rc"
