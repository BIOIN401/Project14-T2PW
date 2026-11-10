#!/bin/bash
# usage: run.sh LABEL tests...   one attempt; exit 95 = heavy lock held (caller waits and retries)
LABEL=$1; shift
T=C:/t/ws-RT1; EV=C:/t/ev-RT1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
cd $T
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
$PY $T/docs/pwml_recovery_sprint/evidence/bounded_run.py --label RT1-$LABEL --timeout 540 --heavy-lock RT1 --json $EV/$LABEL.json -- \
 $PY -u $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=C:/t/bt/RT1/$LABEL "$@" > $EV/$LABEL.out 2>&1
rc=$?
echo "$LABEL exit=$rc"; grep -E "[0-9]+ (passed|failed)|FAILED|ERROR " $EV/$LABEL.out | tail -6 | cut -c1-200
grep -E '"final_surviving_count"|"exit_code"' $EV/$LABEL.json | tr -d '\n'; echo
