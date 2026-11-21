#!/bin/bash
# usage: run_focused.sh RUNID TESTFILE...   -- focused pytest on the lane tree
R=$1; shift
T=C:/t/lane-pwdraw
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
mkdir -p C:/t/lane-pwdraw-work/bt/$R
cd $T
$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=C:/t/lane-pwdraw-work/bt/$R/t "$@" 2>&1 | tail -40
echo "EXIT_focused=${PIPESTATUS[0]}"
