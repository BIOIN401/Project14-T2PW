#!/bin/bash
# One bounded job (heavy lock 'side'): the PathWhiz import harness on the fixed replays.
#  1. the 59 flag-on harness files (same list as layout v2.1: on_v2 idx 6.. of the layout package), fixed writer
#  2. side_db.py on the 27 files with a side mismatch (on_v1 pwml), fixed writer
#  3. SBML export of one affected pathway (37), before (layout-package replay) and after (fixed writer)
# Harness: C:/t/pwi/t2pw/tools/pathwhiz_import (project t2pw-pwi, default work copy), unchanged.
E=C:/t/side/docs/pwml_output/evidence; S=$E/side_fix; L=$E/layout/scripts
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
PWPY=C:/t/e2e3-venv/Scripts/python.exe
H=C:/t/pwi/t2pw/tools/pathwhiz_import/run_import.py
O=C:/t/sidex/pwi; R=C:/t/sidex/replay; B=C:/t/layx/replay
mkdir -p $O $S/logs $S/side_db $S/sbml
sed "s#C:/t/layx/replay/#$R/#" C:/t/layx/pwi/list59_v2.txt > $O/list59_after.txt
sed "s#C:/t/layx/replay/#$R/#" C:/t/layx/pwi/side_list.txt > $O/side_list_after.txt
cp $O/list59_after.txt $O/side_list_after.txt $S/
sha256sum C:/t/pwi/t2pw/tools/pathwhiz_import/work/smpdb/lib/pwml_parser.rb
$PY -X utf8 $H --list $O/list59_after.txt --out $O/after_59 --chunk 10 > $O/after_59.out 2>&1
echo "import59 exit=$?"; tail -1 $O/after_59.out
$PY -X utf8 $L/side_db.py $O/side_list_after.txt $S/side_db > $S/side_db/side_db.stdout.txt 2>&1
echo "side_db exit=$?"; tail -1 $S/side_db/side_db.stdout.txt
for v in before after; do
  if [ $v = before ]; then F=$B/on_v1/pwml/37.pwml; else F=$R/on_v1/pwml/37.pwml; fi
  $PY -X utf8 $H $F --screenshots $F --out $O/sbml_$v > $O/sbml_$v.out 2>&1
  echo "import37 $v exit=$?"
  $PWPY $S/scripts/fetch_sbml.py $S/sbml/37_$v.sbml
done
