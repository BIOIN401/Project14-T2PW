#!/bin/bash
# v2.1 final pass (copy of harness_v21.sh with an output TAG): import the 59 flag-on harness files as v2.1 (on_v2 replays at the v2.1 code), then screenshot 8.
# Same shared harness (project t2pw-pwi, default work copy), unchanged. Output: C:/t/layx/pwi/{TAG_59,TAGshots}, TAG default v21.
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
H=C:/t/pwi/t2pw/tools/pathwhiz_import/run_import.py
O=C:/t/layx/pwi
sha256sum C:/t/pwi/t2pw/tools/pathwhiz_import/work/smpdb/lib/pwml_parser.rb
$PY -X utf8 $H --list $O/list59_v2.txt --out $O/${TAG:-v21}_59 --chunk 10 --resume > $O/${TAG:-v21}_59.out 2>&1
echo "import v21 exit=$?"; tail -1 $O/${TAG:-v21}_59.out
S=""
for i in 7 128 9 116 134 126 58 10; do S="$S C:/t/layx/replay/on_v2/pwml/$i.pwml"; done
$PY -X utf8 $H --screenshots $S --out $O/${TAG:-v21}shots --resume > $O/${TAG:-v21}shots.out 2>&1
echo "shots exit=$?"; tail -1 $O/${TAG:-v21}shots.out
