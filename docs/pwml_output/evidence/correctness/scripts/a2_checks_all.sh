#!/bin/bash
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
C="$PY -B -X utf8 C:/t/corr-assess/scripts/a2_check.py --proto C:/t/corr-assess/a2/proto"
O=C:/t/corr-assess/a2/checks
mkdir -p $O
$C --out $O/before.json
$C --out $O/reg.json --opts '{"hide_reg":1}'
$C --out $O/enz.json --opts '{"enz_clone":1}'
$C --out $O/merge.json --opts '{"merge_dups":1,"plain_tags":1}'
$C --out $O/all.json --opts '{"hide_reg":1,"enz_clone":1,"merge_dups":1,"plain_tags":1,"legend":1}'
