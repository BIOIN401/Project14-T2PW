#!/bin/bash
# usage: pwab8.sh LABEL OUTLABEL [slug...] -- PathWhiz checkpoint (VIS-2 harness, project t2pw-pwdraw) of the 9 VIS-8 PWMLs (8 showcases +
# ent-item6; or the named slugs) plus PathBank's PW000779 reference; imports and screenshots all; stack down afterwards.
L=$1; O=$2; shift 2
W=C:/t/vis-pw8-work
H=C:/t/vis-pw2-work/harness
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
REF=C:/t/vis-ref/super_pathway_visualizations/super_pathway_visualizations/pwml_markups
D=$W/pwab/$O; rm -rf $D; mkdir -p $D
SL="$@"; [ -z "$SL" ] && SL="$(ls $W/cc_show/$L/*.pwml | xargs -n1 basename | sed 's/.pwml//') ent-item6"
for s in $SL; do
  if [ "$s" = ent-item6 ]; then cp $W/cc_item6/$L/ent-item6.pwml $D/v5_ent-item6.pwml; else cp $W/cc_show/$L/$s.pwml $D/v5_$s.pwml; fi
done
cp $REF/PW000779.pwml $D/ref_PW000779.pwml
ls $D/*.pwml > $W/pwab/${O}_list.txt
docker info > /dev/null 2>&1 || { echo "DOCKER_NOT_RUNNING"; exit 3; }
$PY -I -X utf8 $H/run_import.py --list $W/pwab/${O}_list.txt --screenshots $(cat $W/pwab/${O}_list.txt) \
   --out $W/pw/$O --smpdb-src C:/t/pwpr/latest --work C:/t/pwpr/pwdraw-run
echo "EXIT_import=$?"
cd $H && SMPDB_SRC=C:/t/pwpr/pwdraw-run OUT_DIR=$W/pw/$O PWML_DIR=$H/pwml_in docker compose -p t2pw-pwdraw down
echo "STACK_DOWN rc=$?"
echo "PWAB_DONE"
