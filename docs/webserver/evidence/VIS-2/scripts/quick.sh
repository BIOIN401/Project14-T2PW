#!/bin/bash
# usage: quick.sh LABEL -- VIS-2 dev loop: replay the 8 showcases + item-6 from the worktree, content check against
# VIS-1 final2 (49ab4da8), render with the VIS-3 PathBank-style renderer (merge-vis3 tree).
L=$1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/vis-pw2-work
T=C:/t/vis-pw2
echo "HEAD $(git -C $T rev-parse HEAD) dirty=$(git -C $T status --porcelain | tr '\n' ' ')"
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
$PY -B -X utf8 $W/scripts/show_replay.py --src $T/src --spec $W/spec.json --out $W/show/$L > $W/logs/show_$L.txt 2>&1
echo "EXIT_show=$?"
$PY -B -X utf8 $W/scripts/show_replay.py --src $T/src --spec $W/spec_item6.json --out $W/item6/$L > $W/logs/item6_$L.txt 2>&1
echo "EXIT_item6=$?"
$PY -I -X utf8 $W/scripts/cc_copy.py $L
mkdir -p $W/cc_item6/$L
cp $W/item6/$L/ent-item6/pathway.pwml $W/cc_item6/$L/ent-item6.pwml
$PY -I -X utf8 $W/scripts/content_check.py $W/cc_show/base $W/cc_show/$L $W/metrics/cc8_$L.json
echo "EXIT_cc8=$?"
$PY -I -X utf8 $W/scripts/content_check.py $W/cc_item6/base $W/cc_item6/$L $W/metrics/cc_item6_$L.json
echo "EXIT_cc_item6=$?"
git -C $T checkout -- data/enrichment_cache.json data/id_mapping_cache.json 2>/dev/null
unset PYTHONPATH
$PY -I -X utf8 C:/t/vis-iter/rend.py C:/t/merge-vis3 $W/renders/$L $W/cc_item6/$L/ent-item6.pwml $W/cc_show/$L/chlorophyll.pwml $W/cc_show/$L/ceramide-sphingolipid.pwml $W/cc_show/$L/lipid-a.pwml $W/cc_show/$L/enterobactin.pwml $W/cc_show/$L/menaquinone.pwml $W/cc_show/$L/coenzyme-a.pwml $W/cc_show/$L/sour-taste.pwml
echo "EXIT_render=$?"
if [ ! -s $W/renders/base/ent-item6.png ]; then
  $PY -I -X utf8 C:/t/vis-iter/rend.py C:/t/merge-vis3 $W/renders/base $W/cc_item6/base/ent-item6.pwml $W/cc_show/base/chlorophyll.pwml $W/cc_show/base/ceramide-sphingolipid.pwml $W/cc_show/base/lipid-a.pwml $W/cc_show/base/enterobactin.pwml $W/cc_show/base/menaquinone.pwml $W/cc_show/base/coenzyme-a.pwml $W/cc_show/base/sour-taste.pwml
  echo "EXIT_render_base=$?"
fi
echo "QUICK_DONE $L"
