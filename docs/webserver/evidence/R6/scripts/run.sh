#!/bin/bash
# usage: run.sh LABEL tests...   bounded wrapper, heavy lock R6, tree C:/t/ws-R6
LABEL=$1; shift
T=${T:-C:/t/ws-R6}; EV=${EV:-C:/t/ev-R6/out}
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
mkdir -p $EV C:/t/bt/R6
cd $T
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
$PY $T/docs/pwml_recovery_sprint/evidence/bounded_run.py --label R6-$LABEL --timeout 540 --heavy-lock R6 --json $EV/$LABEL.json -- \
 $PY -u $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=C:/t/bt/R6/$LABEL-$(date +%s) "$@" > $EV/$LABEL.out 2>&1
rc=$?
echo "$LABEL exit=$rc"; grep -E "[0-9]+ (passed|failed)|FAILED|ERROR " $EV/$LABEL.out | tail -6 | cut -c1-200
grep -E '"final_surviving_count"|"exit_code"' $EV/$LABEL.json | tr -d '\n'; echo
