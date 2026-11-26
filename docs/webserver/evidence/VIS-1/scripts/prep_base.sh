#!/bin/bash
# Base arm (7bf8588e) derived files: showcase copies, layout metrics, renders with arrowheads.
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/vis-pw1-work
$PY -I -X utf8 $W/scripts/cc_copy.py base
export PYTHONPATH=C:/t/vis-pw1-work/base/src
$PY -X utf8 $W/scripts/layout_metrics.py --pathwhiz-sizes $W/metrics/lm97_base.json $W/replay/base/pwml > /dev/null
$PY -X utf8 $W/scripts/layout_metrics.py --pathwhiz-sizes $W/metrics/lm8_base.json $W/cc_show/base > /dev/null
$PY -X utf8 $W/scripts/render_arrows.py $W/renders/base $W/cc_show/base/enterobactin.pwml $W/cc_show/base/chlorophyll.pwml $W/cc_show/base/lipid-a.pwml $W/cc_show/base/menaquinone.pwml $W/cc_show/base/coenzyme-a.pwml $W/cc_show/base/ceramide-sphingolipid.pwml
echo "PREP_BASE_DONE"
