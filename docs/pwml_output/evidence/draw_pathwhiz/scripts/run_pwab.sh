#!/bin/bash
# PathWhiz A/B import (project t2pw-pwdraw, ports 3357 db / 3054 app, PathWhiz t2pw/latest 63959982 clone under
# C:/t/pwpr/pwdraw-run, MolDB -> closed port). Both arms in one lock hold; screenshots of the 8 showcase files.
# Then the stack goes down (volumes kept).
W=C:/t/lane-pwdraw-work
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
docker info > /dev/null 2>&1 || { echo "DOCKER_NOT_RUNNING"; exit 3; }
for arm in base after; do
  $PY -X utf8 $W/harness/run_import.py --list $W/pwab/${arm}_list.txt --screenshots $(cat $W/pwab/${arm}_shots.txt) \
     --out $W/pw/$arm --smpdb-src C:/t/pwpr/latest --work C:/t/pwpr/pwdraw-run --resume
  echo "EXIT_import_$arm=$?"
done
cd $W/harness && SMPDB_SRC=C:/t/pwpr/pwdraw-run OUT_DIR=$W/pw/after PWML_DIR=$W/harness/pwml_in docker compose -p t2pw-pwdraw down
echo "STACK_DOWN rc=$?"
