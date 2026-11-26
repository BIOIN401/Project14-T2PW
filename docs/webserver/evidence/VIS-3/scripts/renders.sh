#!/bin/bash
# VIS-3 render batch: 16 PathBank references + 6 of our PWMLs, then the side-by-side pages.
W=C:/t/vis-pw3-work
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
export PYTHONPATH=C:/t/vis-pw3/src
REF=C:/t/vis-ref/super_pathway_visualizations/super_pathway_visualizations/pwml_markups
$PY $W/render_many.py $W/r1 png $REF/*.pwml
$PY $W/render_many.py $W/r2 png $W/ours/*.pwml
$PY $W/render_many.py $W/r2svg svg $W/ours/ent-item6.pwml $REF/PW000001.pwml
$PY $W/sbs.py $W/sbs $W/ref $W/r1 "PathBank SVG (Edge)" "VIS-3 render of the PWML"
$PY $W/sbs.py $W/sbs-ours $W/old $W/r2 "current result image (card grid)" "VIS-3 render of the same run's PWML"
echo RENDERS_DONE
