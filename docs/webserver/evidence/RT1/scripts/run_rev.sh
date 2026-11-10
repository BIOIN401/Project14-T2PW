#!/bin/bash
# usage: run_rev.sh LABEL tests...   runs against the scratch tree C:/t/ev-RT1/rev/base (revert proofs)
LABEL=$1; shift
T=C:/t/ev-RT1/rev/base; EV=C:/t/ev-RT1/rev
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
cd $T
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
for try in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18; do
  $PY $T/docs/pwml_recovery_sprint/evidence/bounded_run.py --label RT1rev-$LABEL --timeout 300 --heavy-lock RT1 --json $EV/$LABEL.json -- \
   $PY -u $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=C:/t/bt/RT1/rev-$LABEL "$@" > $EV/$LABEL.out 2>&1
  rc=$?
  [ $rc -ne 95 ] && break
  sleep 20
done
echo "$LABEL exit=$rc"; grep -E "[0-9]+ (passed|failed)" $EV/$LABEL.out | tail -1; grep -E "^FAILED" $EV/$LABEL.out | cut -c1-140
