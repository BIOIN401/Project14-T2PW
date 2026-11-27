#!/bin/bash
# usage: quick.sh LABEL -- VIS-5 dev loop: replay 8 showcases + item-6 through C:/t/vis-pw8 PWML export (no LLM, no network),
# content check against base, render with the tree's own renderer; also renders 4 PathBank reference PWMLs.
L=$1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/vis-pw8-work
T=C:/t/vis-pw8
REF=C:/t/vis-ref/super_pathway_visualizations/super_pathway_visualizations/pwml_markups
mkdir -p $W/logs $W/metrics
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
$PY -I -X utf8 $W/scripts/content_check.py $W/cc_show/base $W/cc_show/$L $W/metrics/cc8s_$L.json strict | head -c 200
$PY -I -X utf8 $W/scripts/content_check.py $W/cc_item6/base $W/cc_item6/$L $W/metrics/cc_item6s_$L.json strict | head -c 200
git -C $T checkout -- data/enrichment_cache.json data/id_mapping_cache.json 2>/dev/null
unset PYTHONPATH
export T2PW_STRUCTURE_CACHE=$W/structcache T2PW_STRUCTURE_BUDGET_S=${SB:-90}
$PY -I -X utf8 C:/t/vis-iter/rend.py $T $W/renders/$L $W/cc_item6/$L/ent-item6.pwml $W/cc_show/$L/*.pwml
echo "EXIT_render=$?"
$PY -I -X utf8 C:/t/vis-iter/rend.py $T $W/renders/$L/ref $REF/PW000001.pwml $REF/PW000779.pwml $REF/PW012911.pwml $REF/PW000017.pwml
echo "EXIT_render_ref=$?"
echo "QUICK_DONE $L"
