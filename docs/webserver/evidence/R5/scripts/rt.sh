#!/bin/bash
# usage: rt.sh TREE LABEL EVID_DIR TIMEOUT [pytest args...]   (one bounded, foreground pytest run)
TREE="$1"; LABEL="$2"; EVID="$3"; TO="$4"; shift 4
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
EV=docs/pwml_recovery_sprint/evidence
BT="C:/t/ws-R5-tmp/bt/$LABEL"
mkdir -p "$EVID" "C:/t/ws-R5-tmp/bt"
cd "$TREE" || exit 2
export PYTHONPATH="$TREE/src" T2PW_OFFLINE_CURATOR=1 PYTHONIOENCODING=utf-8
for try in 1 2 3 4 5 6 7 8; do
  "$PY" "$EV/bounded_run.py" --heavy-lock R5 --timeout "$TO" --label "R5-$LABEL" --cwd "$TREE" \
    --json "$EVID/$LABEL.json" "$PY" "$EV/pinned_pytest.py" --expect-tree "$TREE" -q \
    "--basetemp=$BT-$try" "$@" > "$EVID/$LABEL.out" 2>&1 < /dev/null
  rc=$?
  [ "$rc" -ne 95 ] && break
  sleep 45
done
grep -E "^[0-9]+ (passed|failed)|^=+ .*(passed|failed|error)|passed|failed in" "$EVID/$LABEL.out" | tail -n 2 | cut -c1-200
echo "[$LABEL] exit=$rc try=$try survivors=$(grep -o '"survivors[^,]*' "$EVID/$LABEL.json" | head -1)"
exit $rc
