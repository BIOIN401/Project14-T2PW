#!/bin/bash
# W10: live check, then the merge-gate suite, in one heavy-lock hold.
G="C:/Program Files/Git/bin/bash.exe"
echo "== LIVE"
"$G" C:/t/ws-search-work/live.sh
echo "== SUITE"
"$G" C:/t/vis-runs/suite2.sh C:/t/ws-search w10
echo "CHAIN_DONE"
