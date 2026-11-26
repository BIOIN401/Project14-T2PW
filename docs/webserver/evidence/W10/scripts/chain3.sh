#!/bin/bash
# W10 review round: focused tests, then the merge-gate suite, on the new head, in one heavy-lock hold.
G="C:/Program Files/Git/bin/bash.exe"
"$G" C:/t/ws-search-work/focused.sh
"$G" C:/t/vis-runs/suite2.sh C:/t/ws-search w10r
echo "CHAIN3_DONE"
