#!/bin/bash
LABEL=$1; TO=$2; shift 2
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
TREE=${TREE:-C:/t/ws-R3d}
export PYTHONPATH="$TREE/src" T2PW_OFFLINE_CURATOR=1
cd "$TREE"
$PY "$TREE/docs/pwml_recovery_sprint/evidence/bounded_run.py" --label "rev-$LABEL" --timeout "$TO" --heavy-lock rev-R3d --json "C:/t/rev-R3d/out/$LABEL.json" -- "$PY" "$@" > "C:/t/rev-R3d/out/$LABEL.log" 2>&1
echo "wrapper exit: $?"
