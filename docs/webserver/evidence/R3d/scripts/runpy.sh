#!/bin/bash
# usage: runpy.sh LABEL OUTDIR script.py args...   (bounded_run + heavy lock; PYTHONPATH=tip src)
LABEL=$1; OUT=$2; shift 2
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
mkdir -p "$OUT"
cd C:/t/ws-R3d
export PYTHONPATH=C:/t/ws-R3d/src T2PW_OFFLINE_CURATOR=1
$PY C:/t/wsR3dtmp/waitfree.py "$LABEL" || { echo BUSY; exit 3; }
$PY C:/t/ws-R3d/docs/pwml_recovery_sprint/evidence/bounded_run.py --label "$LABEL" --timeout 300 --heavy-lock R3d --json "$OUT/$LABEL.json" -- \
  $PY -u "$@" > "$OUT/$LABEL.log" 2>&1
echo "wrapper exit: $?"
