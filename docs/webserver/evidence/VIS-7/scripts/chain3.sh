#!/bin/bash
GB="C:/Program Files/Git/bin/bash.exe"
W=C:/t/vis-pw7-work
"$GB" C:/t/vis-runs/queue.sh VIS-7 vis7-pwab-retry 2400 $W/logs/vis7-pwab-retry.json 4 $W/logs/vis7-pwab-retry.log -- "$GB" $W/scripts/pwab3.sh a3 a3b coenzyme-a ent-item6 sour-taste
echo RETRY_DONE
