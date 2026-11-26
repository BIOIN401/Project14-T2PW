#!/bin/bash
# VIS-5 targeted test run with full output. usage: t2.sh RUNID test-ids...
T=C:/t/vis-pw5; R=vis5-t2-$1; shift
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/$R; mkdir -p $B
cd $T
$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=$B/t "$@" 2>&1 | tail -150
echo "EXIT_t2=${PIPESTATUS[0]}"
