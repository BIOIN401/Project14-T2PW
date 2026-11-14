#!/bin/bash
# One bounded job: the four replays (flag off/on x layout unset/v2), v1 byte identity against the P4 d3 outputs,
# v2 content identity against v1, layout metrics (stored and PathWhiz template sizes), v1/v2 renders of 8 files.
T=C:/t/lay; E=$T/docs/pwml_output/evidence/layout; D3=$T/docs/pwml_output/evidence/final_payload_p4/d3/replay
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
R=C:/t/layx/replay; M=$E/metrics; mkdir -p $M
for spec in "off_v1 0 " "on_v1 1 " "off_v2 0 v2" "on_v2 1 v2"; do
  set -- $spec; L=$1; F=$2; LAY=${3:-}
  rm -rf $R/$L
  $PY -X utf8 $E/scripts/replay.py --src $T/src --flag $F --label $L --start 0 --count 152 --out $R --layout "$LAY" > $R.$L.out 2>&1
  echo "replay $L exit=$? written=$(ls $R/$L/pwml | wc -l)"
  cp $R/$L/results_0.jsonl $M/replay_results_$L.jsonl
done
for m in off on; do
  $PY -X utf8 $E/scripts/sha_compare.py $D3/$m/results_0.jsonl $R/${m}_v1/pwml $M/sha_${m}_v1_vs_d3.json; echo "sha $m exit=$?"
  $PY -X utf8 $E/scripts/content_check.py $R/${m}_v1/pwml $R/${m}_v2/pwml $M/content_${m}_v2_vs_v1.json; echo "content $m exit=$?"
  for v in v1 v2; do
    $PY -X utf8 $E/scripts/layout_metrics.py $M/metrics_${m}_$v.json $R/${m}_$v/pwml > /dev/null
    $PY -X utf8 $E/scripts/layout_metrics.py --pathwhiz-sizes $M/metrics_${m}_${v}_pwsizes.json $R/${m}_$v/pwml > /dev/null
  done
done
$PY -X utf8 $E/scripts/render_pairs.py $T/src $R/on_v1/pwml $R/on_v2/pwml $E/render 7,128,9,116,134,126,58,10
