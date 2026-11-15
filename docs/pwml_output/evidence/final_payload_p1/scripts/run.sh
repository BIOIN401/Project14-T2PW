#!/bin/bash
# usage: run.sh LABEL tests...   waits (<= ~60 min, 5-min sleeps) for heavy lock absent + RAM >= 6 GB
LABEL=$1; shift
T=C:/t/fp; EV=C:/t/ev-fp/out
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
mkdir -p C:/t/bt/fp
n=0
until powershell.exe -NoProfile -File C:/t/ev-fp/ram.ps1 >> $EV/gate-$LABEL.txt 2>&1; do
  n=$((n+1)); if [ $n -ge 12 ]; then echo "$LABEL GATE_BLOCKED after 12 waits"; exit 99; fi
  sleep 290
done
cd $T
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
BT=C:/t/bt/fp/$LABEL-$(date +%s)
$PY $T/docs/pwml_recovery_sprint/evidence/bounded_run.py --label fp-$LABEL --timeout 540 --heavy-lock fp --json $EV/$LABEL.json -- \
 $PY -u $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=$BT "$@" > $EV/$LABEL.out 2>&1
rc=$?
echo "$LABEL exit=$rc"; grep -E "[0-9]+ (passed|failed)|FAILED|ERROR " $EV/$LABEL.out | tail -6 | cut -c1-200
grep -E '"final_surviving_count"|"exit_code"' $EV/$LABEL.json | tr -d '\n'; echo
