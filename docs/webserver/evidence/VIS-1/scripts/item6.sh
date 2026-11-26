#!/bin/bash
# usage: item6.sh LABEL -- replay the server's item-6 enterobactin payload through base and the worktree; render both.
L=$1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/vis-pw1-work
if [ ! -s $W/item6/base/ent-item6/pathway.pwml ]; then
  PYTHONPATH=$W/base/src T2PW_OFFLINE_CURATOR=1 $PY -B -X utf8 $W/scripts/show_replay.py --src $W/base/src --spec $W/spec_item6.json --out $W/item6/base
  fi
PYTHONPATH=C:/t/vis-pw1/src T2PW_OFFLINE_CURATOR=1 $PY -B -X utf8 $W/scripts/show_replay.py --src C:/t/vis-pw1/src --spec $W/spec_item6.json --out $W/item6/$L
mkdir -p $W/cc_item6/base $W/cc_item6/$L
cp $W/item6/base/ent-item6/pathway.pwml $W/cc_item6/base/ent-item6.pwml
cp $W/item6/$L/ent-item6/pathway.pwml $W/cc_item6/$L/ent-item6.pwml
$PY -I -X utf8 $W/scripts/content_check.py $W/cc_item6/base $W/cc_item6/$L $W/metrics/cc_item6_$L.json
echo "EXIT_cc_item6=$?"
PYTHONPATH=C:/t/vis-pw1/src $PY -X utf8 $W/scripts/render_arrows.py $W/renders/$L $W/cc_item6/$L/ent-item6.pwml
[ -s $W/renders/base/base_ent-item6.png ] || PYTHONPATH=$W/base/src $PY -X utf8 $W/scripts/render_arrows.py $W/renders/base $W/cc_item6/base/ent-item6.pwml
echo "ITEM6_DONE"
