#!/bin/bash
GB="C:/Program Files/Git/bin/bash.exe"
W=C:/t/vis-pw7-work
"$GB" C:/t/vis-runs/queue.sh VIS-7 vis7-focused-final 3000 $W/logs/vis7-focused-final.json 4 $W/logs/vis7-focused-final.log -- "$GB" $W/scripts/focused.sh final
echo FOCUSED_QUEUED_DONE
"$GB" C:/t/vis-runs/queue.sh VIS-7 vis7-suite2 7200 $W/logs/vis7-suite2.json 4 $W/logs/vis7-suite2.log -- "$GB" C:/t/vis-runs/suite2.sh C:/t/vis-pw7 vis7
echo SUITE_QUEUED_DONE
