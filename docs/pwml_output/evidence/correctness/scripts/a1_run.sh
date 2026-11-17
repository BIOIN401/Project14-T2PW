#!/bin/bash
# usage: a1_run.sh TIMEOUT JSON -- cmd...   gate (heavy lock absent + >=4GB free) then bounded run holding lock corr-a1
TO=$1; JS=$2; shift 3
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
n=0
until [ ! -e C:/t/heavylock/holder.json ] && [ "$(powershell.exe -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)' | tr -d '\r')" -ge 4 ]; do
  n=$((n+1)); if [ $n -ge 120 ]; then echo "GATE_BLOCKED"; exit 99; fi
  sleep 30
done
export PYTHONPATH=C:/t/integ/src T2PW_OFFLINE_CURATOR=1
$PY C:/t/integ/docs/pwml_recovery_sprint/evidence/bounded_run.py --label corr-a1 --timeout $TO --heavy-lock corr-a1 --json $JS -- "$@"
echo "BW_EXIT=$?"
