#!/bin/bash
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
n=0
until [ ! -e C:/t/heavylock/holder.json ] && [ "$(powershell.exe -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)' | tr -d '\r')" -ge 4 ]; do
  n=$((n+1)); if [ $n -ge 120 ]; then echo "GATE_BLOCKED"; exit 99; fi
  sleep 30
done
export PYTHONPATH=C:/t/integ/src T2PW_OFFLINE_CURATOR=1
cd C:/t/integ
$PY C:/t/integ/docs/pwml_recovery_sprint/evidence/bounded_run.py --heavy-lock corr-a0 --timeout 3600 --label CORR-A0-dump --json C:/t/corr-assess/replay_run.json -- $PY -X utf8 C:/t/corr-assess/scripts/dump_replay.py --src C:/t/integ/src --flag 1 --label on_v2 --start 0 --count 152 --out C:/t/corr-assess/replay --layout v2
echo "BW_EXIT=$?"
