#!/bin/bash
# usage: legtrna.sh LEG PMC [KEY=VAL ...]   one live strict leg on C:/t/fix-trna (adapted from acc-runs/scripts/leg5g.sh)
LEG=$1; NAME=$2; shift 2
TREE=C:/t/fix-trna
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
A=C:/t/acc-runs
LOGD=$A/logs-trna; OUT=$A/legs-trna/$LEG
mkdir -p $LOGD $A/legs-trna
rm -f $LOGD/$LEG.usage.jsonl
unset T2PW_REACTION_VERIFIER T2PW_INTERACTION_TYPING T2PW_VERIFIER_GRANULARITY T2PW_VD18_SPAN_TEMPLATES T2PW_AUDIT_NA_DECLARED
export T2PW_LLM_MODEL=google/gemma-4-31b-it
for kv in "$@"; do export "$kv"; done
while [ -e $A/PAUSE ]; do sleep 30; done
cd $TREE
git checkout -q -- data/enrichment_cache.json data/id_mapping_cache.json
date +%s > $LOGD/$LEG.start
export T2PW_AB_USAGE_LOG=$LOGD/$LEG.usage.jsonl T2PW_EXTRACTION_DRAWS=3 PYTHONPATH="$TREE/src;C:/t/llm-ab/abtools" PYTHONIOENCODING=utf-8
"C:/Program Files/Git/bin/bash.exe" C:/t/vis-runs/queue.sh fixtrna trna-$LEG 9600 $LOGD/$LEG.json 4 $LOGD/$LEG.qlog -- \
  $PY -u -X utf8 C:/t/corr-assess/scripts/live_launch.py --script $TREE/scripts/batch_run.py --topics $A/topics/$NAME.txt --modes strict --out $OUT --fresh --timeout 9000 --deadline 2.6
echo "LEG_DONE $LEG $(grep -o 'BW_EXIT=[0-9]*' $LOGD/$LEG.qlog | tail -1)"
date +%s > $LOGD/$LEG.end
cd $TREE && git checkout -q -- data/enrichment_cache.json data/id_mapping_cache.json
