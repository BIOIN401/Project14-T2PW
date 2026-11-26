#!/bin/bash
# usage: pwab.sh LABEL -- VIS-2 PathWhiz import of the after arm (project t2pw-pwdraw, PathWhiz t2pw/latest clone at
# C:/t/pwpr/pwdraw-run), 4 screenshots; stack down after (volumes kept); A/B against VIS-1's after arm (= our base).
L=$1
W=C:/t/vis-pw2-work
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
$PY -I -X utf8 $W/scripts/prep_pwab.py $L
docker info > /dev/null 2>&1 || { echo "DOCKER_NOT_RUNNING"; exit 3; }
$PY -I -X utf8 $W/harness/run_import.py --list $W/pwab/after_${L}_list.txt --screenshots $(cat $W/pwab/after_${L}_shots.txt) \
   --out $W/pw/after_$L --smpdb-src C:/t/pwpr/latest --work C:/t/pwpr/pwdraw-run
echo "EXIT_import_after=$?"
cd $W/harness && SMPDB_SRC=C:/t/pwpr/pwdraw-run OUT_DIR=$W/pw/after_$L PWML_DIR=$W/harness/pwml_in docker compose -p t2pw-pwdraw down
echo "STACK_DOWN rc=$?"
$PY -I -X utf8 $W/scripts/pw_ab.py C:/t/vis-pw1-work/pw/after_final2 $W/pw/after_$L $W/metrics/pathwhiz_ab_$L.json
echo "PWAB_DONE"
