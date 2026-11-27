#!/bin/bash
# corpus replay then PathWhiz A/B, each through the queue (holder VIS-7)
Q="C:/Program Files/Git/bin/bash.exe"
GB="C:/Program Files/Git/bin/bash.exe"
W=C:/t/vis-pw7-work
"$GB" C:/t/vis-runs/queue.sh VIS-7 vis7-corpus-c1 3000 $W/logs/vis7-corpus-c1.json 4 $W/logs/vis7-corpus-c1.log -- "$GB" $W/scripts/corpus.sh c1 C:/t/vis-pw7/src
echo CORPUS_DONE
EXPORT_EXTRA="coenzyme-a enterobactin furochromone menaquinone sour-taste"
EXTRA="$EXPORT_EXTRA" "$GB" C:/t/vis-runs/queue.sh VIS-7 vis7-pwab-r30 3600 $W/logs/vis7-pwab.json 4 $W/logs/vis7-pwab.log -- "$GB" $W/scripts/pwab.sh a3
echo PWAB_QUEUED_DONE
