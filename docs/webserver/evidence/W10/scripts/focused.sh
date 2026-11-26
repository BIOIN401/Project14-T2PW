#!/bin/bash
# W10 focused: the new smart-search tests plus every existing web test file.
T=C:/t/ws-search; R=w10-focused
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
PP="$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q"
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/$R; mkdir -p $B
cd $T
echo "HEAD $(git -C $T rev-parse HEAD)"
run() { label=$1; shift; echo "== $label"; $PP --basetemp=$B/$label "$@" 2>&1 | tail -15; echo "EXIT_$label=${PIPESTATUS[0]}"; }
run search tests/test_web_smart_search.py
run web tests/test_web_*.py tests/test_vision_web.py
echo "FOCUSED_DONE"
