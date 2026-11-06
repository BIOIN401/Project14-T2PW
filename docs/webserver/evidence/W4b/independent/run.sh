#!/bin/bash
EV=C:/t/evidence/W4b/independent
T=C:/t/ws-W4b-trial
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
label=$1; tmo=$2; shift 2
gate() {
  powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.CommandLine -match 'pytest|bounded_run') -and (\$_.CommandLine -notmatch 'ws-W4b-trial') -and (\$_.CommandLine -notmatch 'Pylance|Get-CimInstance|grep\.exe') } | ForEach-Object { \$_.ProcessId }" | tr -d '\r' | grep -c '[0-9]'
}
start=$(date +%s)
while true; do
  n=$(gate)
  echo "$(date +%T) gate $label foreign=$n" >> $EV/waits.log
  [ "$n" = "0" ] && break
  sleep 15
done
while true; do
  mkdir -p C:/t/bt/W4bind
  bt=C:/t/bt/W4bind/$label-$(date +%s)
  cd $T
  T2PW_OFFLINE_CURATOR=1 PYTHONPATH=$T/src $PY docs/pwml_recovery_sprint/evidence/bounded_run.py --label W4b-$label --timeout $tmo --json $EV/$label.json --heavy-lock W4b-$label -- $PY -u docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree C:/t/ws-W4b-trial -q --basetemp=$bt "$@" > $EV/$label.out 2>&1
  rc=$?
  if [ $rc = 95 ]; then echo "$(date +%T) $label lock refused, retry" >> $EV/waits.log; sleep 20; continue; fi
  break
done
echo "$label exit=$rc waited_total_s=$(( $(date +%s)-start ))" | tee -a $EV/summary.log
grep -E "passed|failed|error|SURVIV|survivor" $EV/$label.out | tail -4
