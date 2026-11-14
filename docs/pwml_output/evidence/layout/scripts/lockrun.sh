#!/bin/bash
# usage: lockrun.sh LABEL TIMEOUT JSON -- cmd...
# Waits until C:/t/heavylock/holder.json is absent (and >= 4 GB RAM free), then runs cmd under bounded_run holding the
# heavy lock as 'lay'. A lost race (bounded_run exit 95: lock taken first, nothing ran) is retried. Up to ~3 h.
LABEL=$1; TO=$2; JS=$3; shift 4
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
for n in $(seq 1 50000); do
  if [ -e C:/t/heavylock/holder.json ]; then sleep 0.2; continue; fi
  if [ $((n % 50)) -eq 1 ] && [ "$(powershell.exe -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)' | tr -d '\r')" -lt 4 ]; then sleep 5; continue; fi
  $PY C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/pwml_recovery_sprint/evidence/bounded_run.py --label $LABEL --timeout $TO --heavy-lock lay --json $JS -- "$@"
  rc=$?
  if [ $rc -ne 95 ]; then echo "LOCKRUN_EXIT=$rc attempt=$n"; exit $rc; fi
done
echo GATE_BLOCKED; exit 99
