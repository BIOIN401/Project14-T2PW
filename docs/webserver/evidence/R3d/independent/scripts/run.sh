#!/bin/bash
# usage: run.sh LABEL tests...
LABEL=$1; shift
T=C:/t/ws-R3d-trial; EV=C:/t/ev-R3d/independent
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
s=$(date +%s)
while ! powershell.exe -NoProfile -File C:/t/ev-R3d/scripts/gate.ps1 >> $EV/gate-$LABEL.txt 2>&1; do sleep 15; done
echo "$LABEL gate wait $(( $(date +%s)-s ))s" >> $EV/waits.log
cd $T
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
$PY $T/docs/pwml_recovery_sprint/evidence/bounded_run.py --label R3d-ind-$LABEL --timeout 540 --heavy-lock ws-R3d-ind --json $EV/$LABEL.json -- \
 $PY -u $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=C:/t/bt/R3dind/$LABEL "$@" > $EV/$LABEL.out 2>&1
rc=$?
echo "$LABEL exit=$rc"; grep -E "[0-9]+ (passed|failed)|FAILED|ERROR " $EV/$LABEL.out | tail -6 | cut -c1-200
grep -E '"final_surviving_count"|"exit_code"' $EV/$LABEL.json | tr -d '\n'; echo
