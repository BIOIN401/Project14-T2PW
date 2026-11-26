#!/bin/bash
# VIS-4 review-fix focused tests: render tests + service run_pathway, then one render round (r5). usage: focused2.sh RUNID
T=C:/t/vis-pw4; R=vis4-focused-$1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
PP="$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q"
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/$R; mkdir -p $B
cd $T
echo "HEAD $(git -C $T rev-parse HEAD)"; echo "STATUS_BEGIN"; git -C $T status --porcelain; echo "STATUS_END"
run() { label=$1; shift; echo "== $label"; $PP --basetemp=$B/$label "$@" 2>&1 | tail -8; echo "EXIT_$label=${PIPESTATUS[0]}"; }
run render tests/test_render_pathbank.py tests/test_render_structures.py
run service tests/test_service_run_pathway.py
unset PYTHONPATH T2PW_OFFLINE_CURATOR
"C:/Program Files/Git/bin/bash.exe" C:/t/vis-pw4-work/round.sh r5
echo "FOCUSED2_DONE"
