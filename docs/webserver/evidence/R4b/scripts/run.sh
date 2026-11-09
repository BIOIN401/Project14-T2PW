#!/bin/bash
# usage: run.sh TREE EVDIR SEQ LABEL TIMEOUT tests...   (always takes the heavy lock R4b; retries exit 95)
TREE=$1; EV=$2; SEQ=$3; LABEL=$4; TO=$5; shift 5
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
mkdir -p "$EV/pin" C:/t/wsR4btmp/bt
cd "$TREE"
export PYTHONPATH="$TREE/src" T2PW_OFFLINE_CURATOR=1
BT=C:/t/wsR4btmp/bt/$SEQ-$LABEL
for attempt in $(seq 1 45); do
  rm -rf "$BT"
  $PY "$TREE/docs/pwml_recovery_sprint/evidence/bounded_run.py" --label "$LABEL" --timeout "$TO" --heavy-lock R4b --json "$EV/$SEQ-$LABEL.json" -- \
    $PY -u "$TREE/docs/pwml_recovery_sprint/evidence/pinned_pytest.py" --expect-tree "$TREE" --pin-verdict "$EV/pin/$SEQ-$LABEL.pin.json" \
    -q --basetemp="$BT" "$@" > "$EV/$SEQ-$LABEL.log" 2>&1
  rc=$?
  if [ "$rc" != "95" ]; then break; fi
  echo "lock busy (attempt $attempt), waiting 8s"
  sleep 8
done
echo "wrapper exit: $rc"
grep -E "passed|failed|error|FAILED|ERROR|FINAL SURV|survivor|exit code|exit reason|wrapper vs HEAD|dirty|heavy" "$EV/$SEQ-$LABEL.log" | cut -c1-220 | head -60
