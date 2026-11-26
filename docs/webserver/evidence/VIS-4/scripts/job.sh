#!/bin/bash
# usage: job.sh ROUND... [+focused TAG] [+timing]  -- in one heavy-lock hold: render each round (r0 = merged VIS-3
# baseline tree), optionally the focused tests and the render timing.
W=C:/t/vis-pw4-work
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
S=C:/t/vis-pw1-work/cc_show/final2
while [ $# -gt 0 ]; do
  case "$1" in
    +focused) shift; "C:/Program Files/Git/bin/bash.exe" $W/focused.sh "$1" ;;
    +svg) echo "== svg"; $PY -I $W/rend.py C:/t/vis-pw4 $W/svg svg C:/t/vis-pw1-work/cc_item6/final2/ent-item6.pwml \
               C:/t/vis-ref/super_pathway_visualizations/super_pathway_visualizations/pwml_markups/PW000001.pwml ;;
    +timing) echo "== timing"; $PY -I $W/timing.py C:/t/vis-pw4 $W/timing \
               C:/t/vis-pw1-work/cc_item6/final2/ent-item6.pwml $S/lipid-a.pwml $S/chlorophyll.pwml ;;
    *) echo "== round $1"; "C:/Program Files/Git/bin/bash.exe" $W/round.sh "$1" ;;
  esac
  shift
done
echo JOB_DONE
