#!/bin/bash
# usage: pt.sh LABEL files...   light pytest through bounded_run (no heavy lock), for the focused dev loop
L=$1; shift
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
BR=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/pwml_recovery_sprint/evidence/bounded_run.py
mkdir -p C:/t/vis-runs/bt/vis7
export PYTHONPATH=C:/t/vis-pw7/src T2PW_OFFLINE_CURATOR=1
cd C:/t/vis-pw7
$PY $BR --label vis7-$L --timeout 900 --json C:/t/vis-pw7-work/logs/pt_$L.json -- $PY -I -m pytest -q -p no:cacheprovider --basetemp=C:/t/vis-runs/bt/vis7/$L-$RANDOM "$@" 2>&1 | grep -E "passed|failed|error|Error|FAIL|assert" | head -30; echo
