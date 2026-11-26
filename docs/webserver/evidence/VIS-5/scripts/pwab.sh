#!/bin/bash
# usage: pwab.sh LABEL -- VIS-5 PathWhiz checkpoint: import ent-item6, lipid-a, ceramide, chlorophyll of replay LABEL plus
# PathBank's PW000779 (reference, imported the same way) into the t2pw-pwdraw stack (VIS-2 harness, PathWhiz clone at
# C:/t/pwpr/pwdraw-run), screenshot all five, stack down afterwards (volumes kept).
L=$1
W=C:/t/vis-pw5-work
H=C:/t/vis-pw2-work/harness
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
REF=C:/t/vis-ref/super_pathway_visualizations/super_pathway_visualizations/pwml_markups
D=$W/pwab/$L; mkdir -p $D
cp $W/cc_item6/$L/ent-item6.pwml $D/v5_ent-item6.pwml
for s in lipid-a ceramide-sphingolipid chlorophyll; do cp $W/cc_show/$L/$s.pwml $D/v5_$s.pwml; done
cp $REF/PW000779.pwml $D/ref_PW000779.pwml
ls $D/*.pwml > $W/pwab/${L}_list.txt
docker info > /dev/null 2>&1 || { echo "DOCKER_NOT_RUNNING"; exit 3; }
$PY -I -X utf8 $H/run_import.py --list $W/pwab/${L}_list.txt --screenshots $(cat $W/pwab/${L}_list.txt) \
   --out $W/pw/$L --smpdb-src C:/t/pwpr/latest --work C:/t/pwpr/pwdraw-run
echo "EXIT_import=$?"
cd $H && SMPDB_SRC=C:/t/pwpr/pwdraw-run OUT_DIR=$W/pw/$L PWML_DIR=$H/pwml_in docker compose -p t2pw-pwdraw down
echo "STACK_DOWN rc=$?"
echo "PWAB_DONE"
