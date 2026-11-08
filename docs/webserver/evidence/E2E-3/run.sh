#!/bin/bash
# Foreground; retries only while the shared heavy lock is held (exit 95). Loop is bounded.
cd C:/t/ws-E2E3
for i in $(seq 1 60); do
  C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe -u docs/pwml_recovery_sprint/evidence/bounded_run.py \
    --timeout 530 --label E2E-3 --heavy-lock E2E3 --cwd C:/t/ws-E2E3 \
    --json docs/webserver/evidence/E2E-3/E2E-3-wrapper.json -- \
    C:/t/e2e3-venv/Scripts/python.exe -u docs/webserver/evidence/E2E-3/drive.py
  rc=$?
  echo "WRAPPER_EXIT $rc $(date +%H:%M:%S)"
  if [ "$rc" = "95" ]; then sleep 20; continue; fi
  break
done
exit $rc
