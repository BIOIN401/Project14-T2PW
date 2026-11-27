#!/bin/bash
# usage: pwab3.sh LABEL OUTLABEL slug... : PathWhiz checkpoint of the named showcases of replay LABEL (screenshot retry); ent-item6 as slug ent-item6
L=$1; O=$2; shift 2
W=C:/t/vis-pw8-work
H=C:/t/vis-pw2-work/harness
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
D=$W/pwab/$O; mkdir -p $D
for s in "$@"; do
  if [ "$s" = ent-item6 ]; then cp $W/cc_item6/$L/ent-item6.pwml $D/v5_ent-item6.pwml; else cp $W/cc_show/$L/$s.pwml $D/v5_$s.pwml; fi
done
ls $D/*.pwml > $W/pwab/${O}_list.txt
docker info > /dev/null 2>&1 || { echo "DOCKER_NOT_RUNNING"; exit 3; }
$PY -I -X utf8 $H/run_import.py --list $W/pwab/${O}_list.txt --screenshots $(cat $W/pwab/${O}_list.txt) \
   --out $W/pw/$O --smpdb-src C:/t/pwpr/latest --work C:/t/pwpr/pwdraw-run
echo "EXIT_import=$?"
cd $H && SMPDB_SRC=C:/t/pwpr/pwdraw-run OUT_DIR=$W/pw/$O PWML_DIR=$H/pwml_in docker compose -p t2pw-pwdraw down
echo "STACK_DOWN rc=$?"
echo "PWAB_DONE"
