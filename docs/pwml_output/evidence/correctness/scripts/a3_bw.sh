#!/bin/bash
# usage: bw.sh LABEL TIMEOUT JSON -- cmd...   (waits for heavy lock absent + >=4GB free, then bounded run holding lock 'lay')
LABEL=$1; TO=$2; JS=$3; shift 4
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
n=0
until [ ! -e C:/t/heavylock/holder.json ] && [ "$(powershell.exe -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)' | tr -d '\r')" -ge 4 ]; do
  n=$((n+1)); if [ $n -ge 60 ]; then echo "GATE_BLOCKED"; exit 99; fi
  sleep 30
done
$PY C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/pwml_recovery_sprint/evidence/bounded_run.py --label $LABEL --timeout $TO --heavy-lock corr-a3 --json $JS -- "$@"
echo "BW_EXIT=$?"
