#!/bin/bash
# FIX-S2SCOPE live leg (from acc-runs/scripts/leg5g.sh). usage: leg.sh LEG [KEY=VAL ...]
# Gemma, verifier on (server config) in both arms; the arm flag comes in as KEY=VAL. 9600 s wrapper > 9000 s batch limit.
LEG=$1; shift
TREE=C:/t/fix-s2scope; NAME=PMC12096016
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
A=C:/t/acc-runs
LOGD=$A/logs-s2scope; OUT=$A/legs-s2scope/$LEG
mkdir -p $LOGD $A/legs-s2scope
rm -f $LOGD/$LEG.usage.jsonl
unset T2PW_REACTION_VERIFIER T2PW_INTERACTION_TYPING T2PW_VERIFIER_GRANULARITY T2PW_VD18_SPAN_TEMPLATES T2PW_STAGE2_RESPECT_SCOPE T2PW_TRANSPORT_SIDES
export T2PW_LLM_MODEL=google/gemma-4-31b-it T2PW_REACTION_VERIFIER=1
for kv in "$@"; do export "$kv"; done
while [ -e $A/PAUSE ]; do sleep 30; done
cd $TREE
git checkout -q -- data/enrichment_cache.json data/id_mapping_cache.json
date +%s > $LOGD/$LEG.start
export T2PW_AB_USAGE_LOG=$LOGD/$LEG.usage.jsonl T2PW_EXTRACTION_DRAWS=3 PYTHONPATH="$TREE/src;C:/t/llm-ab/abtools" PYTHONIOENCODING=utf-8
"C:/Program Files/Git/bin/bash.exe" C:/t/vis-runs/queue.sh s2scope s2scope-$LEG 9600 $LOGD/$LEG.json 4 $LOGD/$LEG.qlog -- \
  $PY -u -X utf8 C:/t/corr-assess/scripts/live_launch.py --script $TREE/scripts/batch_run.py --topics $A/topics/$NAME.txt --modes strict --out $OUT --fresh --timeout 9000 --deadline 2.6
echo "LEG_DONE $LEG $(grep -o 'BW_EXIT=[0-9]*' $LOGD/$LEG.qlog | tail -1)"
date +%s > $LOGD/$LEG.end
cd $TREE && git checkout -q -- data/enrichment_cache.json data/id_mapping_cache.json
