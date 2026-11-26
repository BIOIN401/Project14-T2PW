#!/bin/bash
# usage: rend_set.sh LABEL IDX... -- render replay runs (base and LABEL) with the VIS-3 PathBank-style renderer
L=$1; shift
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/vis-pw2-work
A=(); B=()
for i in "$@"; do A+=($W/replay/$L/pwml/$i.pwml); [ -s $W/renders/base/$i.png ] || B+=($W/replay/base/pwml/$i.pwml); done
$PY -I -X utf8 C:/t/vis-iter/rend.py C:/t/merge-vis3 $W/renders/$L "${A[@]}"
echo "EXIT_render=$?"
if [ ${#B[@]} -gt 0 ]; then $PY -I -X utf8 C:/t/vis-iter/rend.py C:/t/merge-vis3 $W/renders/base "${B[@]}"; echo "EXIT_render_base=$?"; fi
echo "REND_DONE"
