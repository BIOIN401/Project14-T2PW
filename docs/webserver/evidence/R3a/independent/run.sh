#!/bin/bash
# usage: run.sh label timeout files...
D=C:/t/evidence/R3a/independent
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
label=$1; to=$2; shift 2
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
  o=$(powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -and (\$_.CommandLine -match 'pytest|bounded_run') -and \$_.CommandLine -notmatch 'get_pytest_options' -and \$_.CommandLine -notmatch 'ws-R3a-ind' } | ForEach-Object { \$_.ProcessId }" | tr -d '\r' | tr '\n' ' ')
  echo "GATE $label check$i $(date +%T) others=[$o]" >> $D/waits.log
  [ -z "$(echo $o | tr -d ' ')" ] && break
  sleep 20
done
mkdir -p C:/t/bt/R3aind
cd C:/t/ws-R3a-ind
T2PW_OFFLINE_CURATOR=1 PYTHONPATH=C:/t/ws-R3a-ind/src "$PY" docs/pwml_recovery_sprint/evidence/bounded_run.py --timeout 580 --heavy-lock R3a-ind-$label --label R3a-ind-$label --json $D/$label.json -- "$PY" docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree C:/t/ws-R3a-ind -q --basetemp=C:/t/bt/R3aind/$label "$@" 2>&1 | tee $D/$label.log | grep -E 'passed|failed|error|FAILED|ERROR|survivor|SURVIV|exit' | tail -15
echo "EXIT ${PIPESTATUS[0]}"
