#!/bin/bash
# usage: measure.sh LABEL REF_DIR   -> content check of C:/t/layx/replay/LABEL/pwml against REF_DIR, then layout metrics
# Reports land in docs/pwml_output/evidence/layout/metrics/.
LABEL=$1; REF=$2
T=C:/t/lay; E=$T/docs/pwml_output/evidence/layout
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
$PY -X utf8 $E/scripts/content_check.py $REF C:/t/layx/replay/$LABEL/pwml $E/metrics/content_$LABEL.json
echo "content_check exit=$?"
$PY -X utf8 $E/scripts/layout_metrics.py $E/metrics/$LABEL.json C:/t/layx/replay/$LABEL/pwml
