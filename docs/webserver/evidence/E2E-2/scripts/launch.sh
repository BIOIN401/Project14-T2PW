#!/bin/bash
cd C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW
unset T2PW_OFFLINE_CURATOR
export T2PW_E2E_STACKS=1 PYTHONUTF8=1 PYTHONPATH="C:/t/e2e2/site;C:/t/ws-e2e/src"
while true; do
  # gate: no pytest/bounded_run of another worktree alive
  n=$(powershell.exe -NoProfile -File C:/t/e2e2/gate.ps1)
  n=$(echo $n | tr -d '\r ')
  if [ "$n" != "0" ]; then echo "GATE waiting, $n pytest/bounded_run procs $(date +%H:%M:%S)"; sleep 10; continue; fi
  echo "LAUNCH $(date +%H:%M:%S)"
  .venv/Scripts/python.exe -u docs/pwml_recovery_sprint/evidence/bounded_run.py --timeout 7200 --label E2E-2-live --heavy-lock E2E-2 --cwd C:/t/ws-e2e --json C:/t/e2e2/report/E2E-2-wrapper.json -- C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe -u C:/t/e2e2/drive.py
  rc=$?
  echo "WRAPPER_EXIT $rc $(date +%H:%M:%S)"
  if [ "$rc" = "95" ]; then sleep 10; continue; fi
  break
done
