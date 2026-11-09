#!/bin/bash
# usage: run.sh <label> <json-name> <basetemp-subdir> <pytest args...>   (foreground; retries only on exit 95)
label=$1; js=$2; bt=$3; shift 3
cd C:/t/ws-W6c
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
export PYTHONPATH=C:/t/ws-W6c/src T2PW_OFFLINE_CURATOR=1
mkdir -p C:/t/w6c_tmp
for i in $(seq 1 25); do
  $PY -u docs/pwml_recovery_sprint/evidence/bounded_run.py --timeout 500 --label "$label" --heavy-lock W6c \
    --cwd C:/t/ws-W6c --json "docs/webserver/evidence/W6c/$js" -- \
    $PY -u docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree C:/t/ws-W6c -q --basetemp="C:/t/w6c_tmp/$bt" "$@"
  rc=$?
  echo "WRAPPER_EXIT $rc $(date +%H:%M:%S)"
  if [ "$rc" = "95" ]; then sleep 20; continue; fi
  break
done
exit $rc
