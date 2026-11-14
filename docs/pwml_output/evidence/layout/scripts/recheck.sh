#!/bin/bash
# Identity checks only (after final_measure.sh): v1 against the recorded P4 hashes, v2 content against v1.
T=C:/t/lay; E=$T/docs/pwml_output/evidence/layout; D3=$T/docs/pwml_output/evidence/final_payload_p4/d3/replay
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
R=C:/t/layx/replay; M=$E/metrics
for m in off on; do
  $PY -X utf8 $E/scripts/sha_compare.py $D3/$m/results_0.jsonl $R/${m}_v1/pwml $M/sha_${m}_v1_vs_d3.json; echo "sha $m exit=$?"
  $PY -X utf8 $E/scripts/content_check.py $R/${m}_v1/pwml $R/${m}_v2/pwml $M/content_${m}_v2_vs_v1.json; echo "content $m exit=$?"
done
