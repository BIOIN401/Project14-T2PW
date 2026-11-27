#!/bin/bash
Q="C:/Program Files/Git/bin/bash.exe"
S=C:/t/vis-pw8-work/scripts
W=C:/t/vis-pw8-work
"$Q" C:/t/vis-runs/queue.sh VIS-8 vis8-corpus 3000 $W/logs/m1_corpus.json 4 $W/logs/m1_corpus.log -- "$Q" $S/corpus.sh m1 C:/t/vis-pw8/src
"$Q" C:/t/vis-runs/queue.sh VIS-8 vis8-pwab 3000 $W/logs/m1_pwab.json 4 $W/logs/m1_pwab.log -- "$Q" $S/pwab8.sh m1 p1
echo CHAINF_DONE
