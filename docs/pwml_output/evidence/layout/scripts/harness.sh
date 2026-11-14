#!/bin/bash
# One bounded job holding the heavy lock: import the 59 flag-on harness files as v1 and as v2 (same importer state),
# then screenshot 8 v2 files and the PW012926 control. Harness: C:/t/pwi/t2pw/tools/pathwhiz_import (project t2pw-pwi,
# default work copy), unchanged. Output: C:/t/layx/pwi/{v1_59,v2_59,v2shots}.
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
H=C:/t/pwi/t2pw/tools/pathwhiz_import/run_import.py
O=C:/t/layx/pwi
sha256sum C:/t/pwi/t2pw/tools/pathwhiz_import/work/smpdb/lib/pwml_parser.rb
for v in v1 v2; do
  $PY -X utf8 $H --list $O/list59_$v.txt --out $O/${v}_59 --chunk 10 --resume > $O/${v}_59.out 2>&1
  echo "import $v exit=$?"; tail -1 $O/${v}_59.out
done
S=""
for i in 7 128 9 116 134 126 58 10; do S="$S C:/t/layx/replay/on_v2/pwml/$i.pwml"; done
$PY -X utf8 $H --screenshots C:/t/lay/reference/PW012926.pwml $S --out $O/v2shots --resume > $O/v2shots.out 2>&1
echo "shots exit=$?"; tail -1 $O/v2shots.out
