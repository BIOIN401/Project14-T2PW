#!/bin/bash
# usage: pwab.sh AFTER_LABEL -- PathWhiz import A/B (project t2pw-pwdraw, ports 3357/3054, PathWhiz t2pw/latest clone at
# C:/t/pwpr/pwdraw-run), both arms imported fresh, screenshots of 10 files; then the stack goes down (volumes kept).
L=$1
W=C:/t/vis-pw1-work
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
$PY -I -X utf8 $W/scripts/prep_pwab.py base $L
docker info > /dev/null 2>&1 || { echo "DOCKER_NOT_RUNNING"; exit 3; }
for arm in base after; do
  $PY -I -X utf8 $W/harness/run_import.py --list $W/pwab/${arm}_list.txt --screenshots $(cat $W/pwab/${arm}_shots.txt) \
     --out $W/pw/${arm}_$L --smpdb-src C:/t/pwpr/latest --work C:/t/pwpr/pwdraw-run
  echo "EXIT_import_$arm=$?"
done
cd $W/harness && SMPDB_SRC=C:/t/pwpr/pwdraw-run OUT_DIR=$W/pw/after_$L PWML_DIR=$W/harness/pwml_in docker compose -p t2pw-pwdraw down
echo "STACK_DOWN rc=$?"
$PY -I -X utf8 $W/scripts/pw_ab.py $W/pw/base_$L $W/pw/after_$L $W/metrics/pathwhiz_ab_$L.json
echo "PWAB_DONE"
