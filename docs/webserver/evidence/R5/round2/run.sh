#!/bin/bash
# usage: run.sh LABEL TREE OUTDIR tests-and-args...   (bounded, heavy lock R5fix2, tree pinned)
LABEL=$1; T=$2; EV=$3; shift 3
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
mkdir -p C:/t/bt/R5fix2 "$EV"
cd "$T"
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
$PY $T/docs/pwml_recovery_sprint/evidence/bounded_run.py --label R5-fix2-$LABEL --timeout 540 --heavy-lock R5fix2 --json $EV/$LABEL.json -- \
 $PY -u $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=C:/t/bt/R5fix2/$LABEL "$@" > $EV/$LABEL.out 2>&1
rc=$?
echo "$LABEL exit=$rc"; grep -E "[0-9]+ (passed|failed)|^FAILED|^ERROR " $EV/$LABEL.out | tail -8 | cut -c1-200
grep -E '"final_surviving_count"|"exit_code"' $EV/$LABEL.json | tr -d '\n'; echo
