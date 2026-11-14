#!/bin/bash
# One bounded job (heavy lock 'side'): replay the 4 modes with the fixed writer, side_pwml.py on before and after, side-only check.
T=C:/t/side; E=$T/docs/pwml_output/evidence; S=$E/side_fix; L=$E/layout/scripts
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
B=C:/t/layx/replay; R=C:/t/sidex/replay; mkdir -p $S/metrics $R
for spec in "off_v1 0 " "on_v1 1 " "off_v2 0 v2" "on_v2 1 v2"; do
  set -- $spec; LB=$1; F=$2; LAY=${3:-}
  rm -rf $R/$LB
  $PY -X utf8 $L/replay.py --src $T/src --flag $F --label $LB --start 0 --count 152 --out $R --layout "$LAY" > $R/$LB.out 2>&1
  echo "replay $LB exit=$? written=$(ls $R/$LB/pwml | wc -l)"
  cp $R/$LB/results_0.jsonl $S/metrics/replay_results_$LB.jsonl
  $PY -X utf8 $S/scripts/side_only_check.py $B/$LB/pwml $R/$LB/pwml $S/metrics/side_only_$LB.json; echo "side_only $LB exit=$?"
done
$PY -X utf8 $L/side_pwml.py $S/metrics/side_pwml_before.json $B/off_v1/pwml $B/on_v1/pwml $B/off_v2/pwml $B/on_v2/pwml
$PY -X utf8 $L/side_pwml.py $S/metrics/side_pwml_after.json $R/off_v1/pwml $R/on_v1/pwml $R/off_v2/pwml $R/on_v2/pwml
