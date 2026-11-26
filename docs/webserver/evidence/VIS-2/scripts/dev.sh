#!/bin/bash
# usage: dev.sh LABEL [focused] -- VIS-2: (optional focused tests), 97-run replay + 8 showcases + item-6 of the worktree,
# content check against VIS-1 final2 (49ab4da8), art summary.
L=$1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/vis-pw2-work
T=C:/t/vis-pw2
B="C:/Program Files/Git/bin/bash.exe"
echo "HEAD $(git -C $T rev-parse HEAD)"; echo "STATUS_BEGIN"; git -C $T status --porcelain; echo "STATUS_END"
if [ "$2" = "focused" ]; then
  cd $T
  "$B" $W/scripts/focused.sh $L $(ls tests/test_pwml_*.py) tests/test_final_payload_writer_fixes.py tests/test_orch768_replay_corpus.py
fi
"$B" $W/scripts/run_replays.sh $L $T/src
$PY -I -X utf8 $W/scripts/cc_copy.py $L
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
$PY -B -X utf8 $W/scripts/show_replay.py --src $T/src --spec $W/spec_item6.json --out $W/item6/$L > $W/logs/item6_$L.txt 2>&1
echo "EXIT_item6=$?"
unset PYTHONPATH
mkdir -p $W/cc_item6/$L
cp $W/item6/$L/ent-item6/pathway.pwml $W/cc_item6/$L/ent-item6.pwml
$PY -I -X utf8 $W/scripts/content_check.py $W/replay/base/pwml $W/replay/$L/pwml $W/metrics/cc97_$L.json
echo "EXIT_cc97=$?"
$PY -I -X utf8 $W/scripts/content_check.py $W/cc_show/base $W/cc_show/$L $W/metrics/cc8_$L.json
echo "EXIT_cc8=$?"
$PY -I -X utf8 $W/scripts/content_check.py $W/cc_item6/base $W/cc_item6/$L $W/metrics/cc_item6_$L.json
echo "EXIT_cc_item6=$?"
$PY -I -X utf8 $W/scripts/art_summary.py $W/metrics/art_$L.json $W/replay/$L/pwml $W/cc_show/$L $W/cc_item6/$L
git -C $T checkout -- data/enrichment_cache.json data/id_mapping_cache.json 2>/dev/null
$PY -I -X utf8 C:/t/vis-iter/rend.py C:/t/merge-vis3 $W/renders/$L $W/cc_item6/$L/ent-item6.pwml $W/cc_show/$L/chlorophyll.pwml $W/cc_show/$L/ceramide-sphingolipid.pwml $W/cc_show/$L/lipid-a.pwml $W/cc_show/$L/enterobactin.pwml $W/cc_show/$L/menaquinone.pwml $W/cc_show/$L/coenzyme-a.pwml > /dev/null
echo "EXIT_render_show=$?"
"$B" $W/scripts/rend_set.sh $L 79 132 119 134 7 46 114 11
echo "STATUS_AFTER"; git -C $T status --porcelain
echo "DEV_DONE $L"
