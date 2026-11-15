#!/bin/bash
# usage: runboth.sh LABEL SRC WORK OUT ; retries until the heavy lock is acquired (exit 95 = lock unavailable, child not run)
LABEL=$1; SRC=$2; WORK=$3; OUT=$4
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
mkdir -p $OUT
cd C:/t/pwi/t2pw
while true; do
  $PY C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/pwml_recovery_sprint/evidence/bounded_run.py --label pwpatch2-$LABEL --timeout 5400 --heavy-lock pwi --json $OUT.bounded.json -- \
    $PY -X utf8 C:/t/pwi/t2pw/tools/pathwhiz_import/run_import.py --list C:/t/p5scratch/pwpatch2/list.txt --smpdb-src $SRC --work $WORK --out $OUT --resume > $OUT.stdout.txt 2>&1
  rc=$?
  [ $rc -ne 95 ] && break
  sleep 4
done
echo $LABEL exit=$rc
