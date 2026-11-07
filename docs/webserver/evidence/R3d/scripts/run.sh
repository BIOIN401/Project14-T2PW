#!/bin/bash
# usage: run.sh TREE EVDIR SEQ LABEL TIMEOUT LOCK(0|1) tests...
TREE=$1; EV=$2; SEQ=$3; LABEL=$4; TO=$5; LOCK=$6; shift 6
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
mkdir -p "$EV/pin" C:/t/wsR3dtmp
cd "$TREE"
export PYTHONPATH="$TREE/src" T2PW_OFFLINE_CURATOR=1
LOCKARGS=()
if [ "$LOCK" = "1" ]; then LOCKARGS=(--heavy-lock R3d); fi
$PY "$TREE/docs/pwml_recovery_sprint/evidence/bounded_run.py" --label "$LABEL" --timeout "$TO" "${LOCKARGS[@]}" --json "$EV/$SEQ-$LABEL.json" -- \
  $PY -u "$TREE/docs/pwml_recovery_sprint/evidence/pinned_pytest.py" --expect-tree "$TREE" --pin-verdict "$EV/pin/$SEQ-$LABEL.pin.json" \
  -q --basetemp=C:/t/wsR3dtmp/$SEQ-$LABEL "$@" > "$EV/$SEQ-$LABEL.log" 2>&1
echo "wrapper exit: $?"
grep -E "passed|failed|error|FAILED|ERROR|FINAL SURV|survivor|exit code|exit reason|wrapper vs HEAD|dirty|heavy" "$EV/$SEQ-$LABEL.log" | cut -c1-220 | head -60
