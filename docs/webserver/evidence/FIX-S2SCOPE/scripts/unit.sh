#!/bin/bash
# FIX-S2SCOPE: the new test file plus the C-060a AST wiring test, one process.
T=C:/t/fix-s2scope; R=s2scope-unit-$(date +%s)
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
PP="$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q"
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/$R; mkdir -p $B
cd $T
run() { label=$1; shift; echo "== $label"; timeout 1200 $PP --basetemp=$B/$label "$@" 2>&1 | tail -120; echo "EXIT_$label=${PIPESTATUS[0]}"; }
run unit tests/test_fix_s2scope.py tests/test_c060a_requested_pathway_wiring.py
echo UNIT_DONE
