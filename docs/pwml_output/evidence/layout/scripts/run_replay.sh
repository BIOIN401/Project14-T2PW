#!/bin/bash
# usage: run_replay.sh LABEL FLAG LAYOUT   (LAYOUT empty string = T2PW_PWML_LAYOUT unset)
# Replays the 152 stored runs of final_payload_p4/d3 through run_pwml_export with this tree's code.
# Output: C:/t/layx/replay/LABEL/ (pwml/*.pwml, results_0.jsonl); waits for the heavy lock, holds it as 'lay'.
LABEL=$1; FLAG=$2; LAYOUT=$3
T=C:/t/lay
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
mkdir -p C:/t/layx/replay/$LABEL
bash C:/t/layx/bw.sh lay-replay-$LABEL 1800 C:/t/layx/replay/$LABEL/bounded.json -- \
  $PY -X utf8 $T/docs/pwml_output/evidence/layout/scripts/replay.py --src $T/src --flag $FLAG --label $LABEL \
  --start 0 --count 152 --out C:/t/layx/replay --layout "$LAYOUT" > C:/t/layx/replay/$LABEL/chunk.out 2>&1
echo "$LABEL exit=$?"; grep -E "FINAL SURVIVING|exit reason" C:/t/layx/replay/$LABEL/chunk.out; wc -l < C:/t/layx/replay/$LABEL/results_0.jsonl
