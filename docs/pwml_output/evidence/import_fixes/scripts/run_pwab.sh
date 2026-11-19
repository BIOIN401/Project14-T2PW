#!/bin/bash
# PathWhiz A/B import (project t2pw-fix, ports 3337 db / 3034 app, PathWhiz t2pw/latest 63959982 clone, MolDB -> closed port).
# Both arms in one lock hold, then the stack goes down (volumes kept).
W=C:/t/pwml-fix-work
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
docker info > /dev/null 2>&1 || { echo "DOCKER_NOT_RUNNING"; exit 3; }
for arm in base after; do
  $PY -X utf8 $W/harness/run_import.py --list $W/pwab/${arm}_list.txt --out $W/pw/$arm --smpdb-src C:/t/pwpr/latest --work $W/smpdb-run --resume
  echo "EXIT_import_$arm=$?"
done
cd $W/harness && SMPDB_SRC=$W/smpdb-run OUT_DIR=$W/pw/after PWML_DIR=$W/harness/pwml_in docker compose -p t2pw-fix down
echo "STACK_DOWN rc=$?"
