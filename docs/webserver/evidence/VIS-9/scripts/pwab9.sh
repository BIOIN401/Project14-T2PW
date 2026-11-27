#!/bin/bash
# usage: pwab9.sh LABEL OUTLABEL [slug...] -- import the showcase PWMLs of replay LABEL (all 9 when no slug given) plus PathBank's
# PW000779 into the t2pw-pwdraw stack (VIS-2 harness, patched importer t2pw/latest via C:/t/pwpr/pwdraw-run), screenshot, stack down.
L=$1; O=$2; shift 2
W=C:/t/vis-pw9-work; H=C:/t/vis-pw2-work/harness
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
REF=C:/t/vis-ref/super_pathway_visualizations/super_pathway_visualizations/pwml_markups
D=$W/pwab/$O; mkdir -p $D
if [ $# -eq 0 ]; then cp $W/out/$L/show9/*.pwml $D/; cp $REF/PW000779.pwml $D/ref_PW000779.pwml; else for s in "$@"; do cp $W/out/$L/show9/$s.pwml $D/; done; fi
ls $D/*.pwml > $W/pwab/${O}_list.txt
docker info > /dev/null 2>&1 || { echo "DOCKER_NOT_RUNNING"; exit 3; }
$PY -I -X utf8 $H/run_import.py --list $W/pwab/${O}_list.txt --screenshots $(cat $W/pwab/${O}_list.txt) \
   --out $W/pw/$O --smpdb-src C:/t/pwpr/latest --work C:/t/pwpr/pwdraw-run
echo "EXIT_import=$?"
cd $H && SMPDB_SRC=C:/t/pwpr/pwdraw-run OUT_DIR=$W/pw/$O PWML_DIR=$H/pwml_in docker compose -p t2pw-pwdraw down
echo "STACK_DOWN rc=$?"
echo "PWAB_DONE"
