#!/bin/bash
# Review round 1 PathWhiz A/B (second hold, Docker started again): base arm into pw/base2 with --resume (same base
# files as before, so only the new F3 files import), after arm (replay after5 = HEAD) fresh into pw/after5,
# then the stack goes down (volumes kept).
W=C:/t/lane-pwdraw-work
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
L=after5
echo "HEAD $(git -C C:/t/lane-pwdraw rev-parse HEAD)"
$PY -I -X utf8 $W/scripts/prep_pwab.py base $L
docker info > /dev/null 2>&1 || { echo "DOCKER_NOT_RUNNING"; exit 3; }
for pair in base:base2 after:after5; do
  arm=${pair%%:*}; out=${pair##*:}
  $PY -I -X utf8 $W/harness/run_import.py --list $W/pwab/${arm}_list.txt --screenshots $(cat $W/pwab/${arm}_shots.txt) \
     --out $W/pw/$out --smpdb-src C:/t/pwpr/latest --work C:/t/pwpr/pwdraw-run --resume
  echo "EXIT_import_$out=$?"
done
cd $W/harness && SMPDB_SRC=C:/t/pwpr/pwdraw-run OUT_DIR=$W/pw/after5 PWML_DIR=$W/harness/pwml_in docker compose -p t2pw-pwdraw down
echo "STACK_DOWN rc=$?"
echo "FINAL4_DONE"
