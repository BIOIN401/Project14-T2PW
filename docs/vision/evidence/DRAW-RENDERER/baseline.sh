#!/bin/bash
# usage: baseline.sh TREE TAG   -- metrics on the 97 stored runs (official CLI) + showcase 8 + dumped IRs
T=$1; TAG=$2
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/lane-draw-work
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
cd $T
if [ ! -e $W/ir97/DONE ]; then
  $PY -X utf8 $W/scripts/dump97.py $T $W/ir97 > $W/out/dump97.log 2>&1 && touch $W/ir97/DONE
  tail -1 $W/out/dump97.log
fi
$PY -X utf8 -m t2pw.vision.checks.layout_metrics --runlist C:/t/ev-cm/runlist.json --idx C:/t/vis-runs/scratch/generate/ok_idx.txt --scope default --out $W/out/metrics97_$TAG.json --totals-out $W/out/totals97_$TAG.json > $W/out/metrics97_$TAG.log 2>&1
echo "METRICS97_EXIT=$?"
$PY -X utf8 $W/scripts/showcase.py $T $W/out/showcase_$TAG > $W/out/showcase_$TAG.log 2>&1
echo "SHOWCASE_EXIT=$?"
tail -1 $W/out/showcase_$TAG.log
echo BASELINE_DONE
