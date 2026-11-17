#!/bin/bash
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
n=0
until [ ! -e C:/t/heavylock/holder.json ] && [ "$(powershell.exe -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)' | tr -d '\r')" -ge 4 ]; do
  n=$((n+1)); if [ $n -ge 240 ]; then echo "GATE_BLOCKED"; exit 99; fi
  sleep 30
done
echo "GATE_CLEAR $(date)"
export PYTHONPATH=C:/t/integ/src T2PW_OFFLINE_CURATOR=1
cd C:/t/integ
$PY C:/t/integ/docs/pwml_recovery_sprint/evidence/bounded_run.py --heavy-lock corr-a2 --timeout 1800 --label CORR-A2-ir --json C:/t/corr-assess/a2/ir_run.json -- $PY -X utf8 C:/t/corr-assess/scripts/a2_ir_replay.py --src C:/t/integ/src --flag 1 --label a2_ir --out C:/t/corr-assess/a2/replay --layout v2 --idx 7,128,9,116,5,134,135,78,125,129,138,120,148,19,144,126,122,121,142,114,14,10,58,12,150,68
echo "BW_EXIT=$?"
