#!/bin/bash
# usage: final.sh LABEL -- VIS-1 gates on the committed HEAD in one heavy-lock hold: focused tests, the 97 + 8 replays
# (content check, layout + edge metrics, renders), the item-6 replay, then suite2 (SMOKE, golden, equivalence, vision, web).
L=$1
B="C:/Program Files/Git/bin/bash.exe"
W=C:/t/vis-pw1-work
T=C:/t/vis-pw1
echo "HEAD $(git -C $T rev-parse HEAD)"; echo "STATUS_BEGIN"; git -C $T status --porcelain; echo "STATUS_END"
cd $T
"$B" $W/scripts/focused.sh $L $(ls tests/test_pwml_*.py) tests/test_final_payload_writer_fixes.py tests/test_orch768_replay_corpus.py
"$B" $W/scripts/dev.sh $L
"$B" $W/scripts/item6.sh $L
"$B" C:/t/vis-runs/suite2.sh $T vis1-$L
git -C $T checkout -- data/enrichment_cache.json data/id_mapping_cache.json 2>/dev/null
echo "STATUS_AFTER"; git -C $T status --porcelain
echo "FINAL_DONE"
