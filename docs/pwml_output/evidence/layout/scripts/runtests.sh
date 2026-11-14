#!/bin/bash
# usage: runtests.sh NAME tests...   (bounded wrapper, heavy lock 'lay', pinned_pytest, fresh basetemp)
# Report: docs/pwml_output/evidence/layout/tests/NAME.json and NAME.stdout.txt
NAME=$1; shift
T=C:/t/lay
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
E=$T/docs/pwml_output/evidence/layout/tests; mkdir -p $E C:/t/layx/tmpbase
n=0
until [ ! -e C:/t/heavylock/holder.json ] && [ "$(powershell.exe -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)' | tr -d '\r')" -ge 4 ]; do
  n=$((n+1)); [ $n -ge 1440 ] && { echo GATE_BLOCKED; exit 99; }; sleep 5; done
cd $T
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
unset T2PW_PWML_LAYOUT T2PW_FINAL_PAYLOAD
$PY C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/pwml_recovery_sprint/evidence/bounded_run.py --label lay-$NAME --timeout 1500 --heavy-lock lay --json $E/$NAME.json -- \
  $PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=C:/t/layx/tmpbase/$NAME-$$ "$@" > $E/$NAME.stdout.txt 2>&1
echo "$NAME exit=$?"
grep -E "passed|failed|error" $E/$NAME.stdout.txt | tail -3; grep -E "FINAL SURVIVING" $E/$NAME.stdout.txt
