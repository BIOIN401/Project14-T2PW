#!/bin/bash
# usage: dev.sh LABEL -- replay HEAD worktree (97 + 8), content check against base, layout + edge metrics, renders.
L=$1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/vis-pw1-work
echo "HEAD $(git -C C:/t/vis-pw1 rev-parse HEAD) dirty=$(git -C C:/t/vis-pw1 status --porcelain | wc -l)"
"C:/Program Files/Git/bin/bash.exe" $W/scripts/run_replays.sh $L C:/t/vis-pw1/src
$PY -I -X utf8 $W/scripts/cc_copy.py $L
$PY -I -X utf8 $W/scripts/content_check.py $W/replay/base/pwml $W/replay/$L/pwml $W/metrics/cc97_$L.json
echo "EXIT_cc97=$?"
$PY -I -X utf8 $W/scripts/content_check.py $W/cc_show/base $W/cc_show/$L $W/metrics/cc8_$L.json
echo "EXIT_cc8=$?"
if [ ! -s $W/metrics/lm97_base.json ]; then
  PYTHONPATH=C:/t/vis-pw1-work/base/src $PY -X utf8 $W/scripts/layout_metrics.py --pathwhiz-sizes $W/metrics/lm97_base.json $W/replay/base/pwml > /dev/null
  PYTHONPATH=C:/t/vis-pw1-work/base/src $PY -X utf8 $W/scripts/layout_metrics.py --pathwhiz-sizes $W/metrics/lm8_base.json $W/cc_show/base > /dev/null
fi
export PYTHONPATH=C:/t/vis-pw1/src
$PY -X utf8 $W/scripts/layout_metrics.py --pathwhiz-sizes $W/metrics/lm97_$L.json $W/replay/$L/pwml > /dev/null
$PY -X utf8 $W/scripts/layout_metrics.py --pathwhiz-sizes $W/metrics/lm8_$L.json $W/cc_show/$L > /dev/null
$PY -I -X utf8 $W/scripts/sum_metrics.py $W/metrics/lm97_base.json $W/metrics/lm97_$L.json $W/metrics/sum97_$L.json
$PY -I -X utf8 $W/scripts/sum_metrics.py $W/metrics/lm8_base.json $W/metrics/lm8_$L.json $W/metrics/sum8_$L.json
$PY -X utf8 $W/scripts/edge_metrics.py $W/metrics/em_$L.json base97=$W/replay/base/pwml $L-97=$W/replay/$L/pwml base8=$W/cc_show/base $L-8=$W/cc_show/$L
$PY -X utf8 $W/scripts/render_arrows.py $W/renders/$L $W/cc_show/$L/enterobactin.pwml $W/cc_show/$L/chlorophyll.pwml $W/cc_show/$L/lipid-a.pwml $W/cc_show/$L/menaquinone.pwml $W/cc_show/$L/coenzyme-a.pwml $W/cc_show/$L/ceramide-sphingolipid.pwml
git -C C:/t/vis-pw1 status --porcelain
echo "DEV_DONE $L"
