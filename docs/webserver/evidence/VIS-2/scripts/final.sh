#!/bin/bash
# usage: final.sh LABEL -- VIS-2 gates on the committed HEAD in one heavy-lock hold: focused tests, 97 + 8 + item-6 replays
# (content check against VIS-1 final2 = 49ab4da8, art summary, renders), then suite2 (SMOKE, golden, equivalence, vision, web).
L=$1
B="C:/Program Files/Git/bin/bash.exe"
T=C:/t/vis-pw2
"$B" C:/t/vis-pw2-work/scripts/dev.sh $L focused
"$B" C:/t/vis-runs/suite2.sh $T vis2-$L
git -C $T checkout -- data/enrichment_cache.json data/id_mapping_cache.json 2>/dev/null
echo "STATUS_FINAL"; git -C $T status --porcelain
echo "FINAL_DONE"
