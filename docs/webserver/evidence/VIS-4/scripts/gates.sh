#!/bin/bash
# Queue, one after the other: round r4 + focused tests + timing + SVG, then suite2 (SMOKE, golden, equivalence, vision, web).
Q="C:/Program Files/Git/bin/bash.exe"
"$Q" C:/t/vis-runs/queue.sh vis4 vis4-r4 1800 C:/t/vis-runs/json/vis4-r4.json 4 C:/t/vis-runs/logs/vis4-r4.log -- \
  "$Q" C:/t/vis-pw4-work/job.sh r4 +focused c +svg +timing
"$Q" C:/t/vis-runs/queue.sh vis4 vis4-suite2 3600 C:/t/vis-runs/json/vis4-suite2.json 4 C:/t/vis-runs/logs/vis4-suite2.log -- \
  "$Q" C:/t/vis-runs/suite2.sh C:/t/vis-pw4 vis4
echo GATES_QUEUED_DONE
