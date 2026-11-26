#!/bin/bash
# usage: focused.sh LABEL TESTFILE...  -- focused pytest on the VIS-1 worktree (pinned_pytest, own basetemp)
T=C:/t/vis-pw1; R=vis1-$1; shift
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/$R; mkdir -p C:/t/vis-runs/bt
cd $T
echo "HEAD $(git -C $T rev-parse HEAD) dirty=$(git -C $T status --porcelain | tr '\n' ' ')"
$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=$B "$@" 2>&1 | tail -40
echo "EXIT_pytest=${PIPESTATUS[0]}"
git -C $T checkout -- data/enrichment_cache.json data/id_mapping_cache.json 2>/dev/null
echo "FOCUSED_DONE"
