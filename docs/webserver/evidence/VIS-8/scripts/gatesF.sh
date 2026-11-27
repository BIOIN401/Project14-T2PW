#!/bin/bash
# VIS-8 gates through the queue (clean tree, HEAD stamped): focused pwml/render/fpw/corpus-test run, then suite2 (SMOKE 511, golden seams, equivalence, vision, web)
Q="C:/Program Files/Git/bin/bash.exe"
W=C:/t/vis-pw8-work
T=C:/t/vis-pw8
sed "s#R=vis7-focused-\$1#R=vis8-focused-\$1#" C:/t/vis-pw8-work/scripts/focused.sh > $W/scripts/focused8.sh
"$Q" C:/t/vis-runs/queue.sh VIS-8 vis8-focused 3000 $W/logs/vis8_focused.json 4 $W/logs/vis8_focused.log -- "$Q" $W/scripts/focused8.sh f
"$Q" C:/t/vis-runs/queue.sh VIS-8 vis8-suite 6000 $W/logs/vis8_suite.json 4 $W/logs/vis8_suite.log -- "$Q" C:/t/vis-runs/suite2.sh $T vis8
echo GATESF_DONE
