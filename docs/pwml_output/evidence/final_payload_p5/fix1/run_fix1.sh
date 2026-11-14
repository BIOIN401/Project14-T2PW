#!/bin/bash
# retry the bounded wrapper's own heavy-lock acquire every 2 s (pattern of nucD/run_nucd3.sh)
NAME=$1; shift
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
E=C:/t/fp/docs/pwml_output/evidence/final_payload_p5/tests
mkdir -p $E C:/t/tmpbase
cd C:/t/fp
export PYTHONPATH=C:/t/fp/src T2PW_OFFLINE_CURATOR=1
n=0
while [ $n -lt 5400 ]; do
  n=$((n+1))
  $PY docs/pwml_recovery_sprint/evidence/bounded_run.py --label fp-$NAME --timeout 900 --heavy-lock fp5-$NAME --json $E/$NAME.json -- \
    $PY docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree C:/t/fp -q --basetemp=C:/t/tmpbase/$NAME-$$ "$@" > $E/$NAME.stdout.txt 2>&1
  rc=$?
  if [ $rc -ne 95 ]; then break; fi
  sleep 2
done
echo exit=$rc tries=$n
grep -E "passed|failed|error" $E/$NAME.stdout.txt | head -8; grep -iE "surviving" $E/$NAME.stdout.txt | head -3
