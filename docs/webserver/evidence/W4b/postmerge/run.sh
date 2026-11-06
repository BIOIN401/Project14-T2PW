#!/bin/bash
# usage: run.sh name files...
R=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW
OUT=C:/t/evidence/W4b/postmerge
name=$1; shift
while true; do
  n=$(powershell -NoProfile -File C:/t/evidence/W4b/postmerge/gate.ps1)
  n=$(echo $n | tr -d '\r ')
  echo "$(date +%T) gate $name other=$n" >> $OUT/waits.log
  [ "$n" = "0" ] && break
  sleep 20
done
mkdir -p C:/t/bt/W4bpm
export PYTHONPATH=C:/t/ws-int/src T2PW_OFFLINE_CURATOR=1
cd C:/t/ws-int
$R/.venv/Scripts/python.exe $R/docs/pwml_recovery_sprint/evidence/bounded_run.py --label W4b-pm-$name --timeout 900 --heavy-lock W4b-pm --json $OUT/$name.json -- $R/.venv/Scripts/python.exe $R/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree C:/t/ws-int -q --basetemp=C:/t/bt/W4bpm/$name "$@" 2>&1 | grep -E "passed|failed|error|survivor|SURVIV" | tail -5
echo "exit ${PIPESTATUS[0]}"
