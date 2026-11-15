#!/bin/bash
# A2 mock-up renders: <idx>_<idea>_after.png ; before = the unmodified-option render
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
R="$PY -B -X utf8 C:/t/corr-assess/scripts/a2_render.py C:/t/corr-assess/a2/proto C:/t/corr-assess/a2/mock --maxw 1400"
$PY -B -X utf8 C:/t/corr-assess/scripts/a2_equiv.py
$R --idx 7,12,134 --suffix _before
$R --idx 7 --suffix _reg_after --opts '{"hide_reg":1,"legend":1}'
$R --idx 7,12,134 --suffix _enz_after --opts '{"enz_clone":1}'
$R --idx 7,12,134 --suffix _merge_after --opts '{"merge_dups":1,"plain_tags":1}'
$R --idx 7,12,134 --suffix _all_after --opts '{"hide_reg":1,"enz_clone":1,"merge_dups":1,"plain_tags":1,"legend":1}'
