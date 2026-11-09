#!/bin/bash
# usage: run.sh LABEL tests...   (R4c copy of batch2 run.sh; gate wait bounded to 3 tries)
LABEL=$1; shift
T=C:/t/ws-R5; EV=C:/t/ev-R5/independent
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
s=$(date +%s)
g=0
until powershell.exe -NoProfile -File C:/t/ev-R5/scripts/gate.ps1 >> $EV/gate-$LABEL.txt 2>&1; do
  g=$((g+1)); if [ $g -ge 3 ]; then echo "$LABEL GATE_BLOCKED (live pytest/bounded_run present); not run" | tee -a $EV/waits.log; exit 99; fi
  sleep 15
done
echo "$LABEL gate wait $(( $(date +%s)-s ))s" >> $EV/waits.log
cd $T
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
$PY $T/docs/pwml_recovery_sprint/evidence/bounded_run.py --label R5-ind-$LABEL --timeout 540 --heavy-lock R5-ind --json $EV/$LABEL.json -- \
 $PY -u $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=C:/t/bt/R5ind/$LABEL "$@" > $EV/$LABEL.out 2>&1
rc=$?
echo "$LABEL exit=$rc"; grep -E "[0-9]+ (passed|failed)|FAILED|ERROR " $EV/$LABEL.out | tail -6 | cut -c1-200
grep -E '"final_surviving_count"|"exit_code"' $EV/$LABEL.json | tr -d '\n'; echo
