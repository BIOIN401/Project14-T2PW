#!/bin/bash
# usage: run.sh LABEL TREE -- replay 8 showcases + item-6 + the 97-PWML corpus through TREE/src export (no LLM, no network).
L=$1; T=$2
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/vis-pw9-work
mkdir -p $W/out/$L $W/logs
echo "TREE $T HEAD $(git -C $T rev-parse HEAD 2>/dev/null) dirty=$(git -C $T status --porcelain 2>/dev/null | tr '\n' ' ')"
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
$PY -B -X utf8 $W/scripts/show_replay.py --src $T/src --spec $W/spec.json --out $W/out/$L/show > $W/logs/show_$L.txt 2>&1
echo "EXIT_show=$?"
$PY -B -X utf8 $W/scripts/show_replay.py --src $T/src --spec $W/spec_item6.json --out $W/out/$L/show > $W/logs/item6_$L.txt 2>&1
echo "EXIT_item6=$?"
$PY -B -X utf8 $W/scripts/replay97.py --src $T/src --label corpus --out $W/out/$L > $W/logs/corpus_$L.txt 2>&1
echo "EXIT_corpus=$?"; tail -1 $W/logs/corpus_$L.txt
mkdir -p $W/out/$L/show9
for d in $W/out/$L/show/*/; do s=$(basename $d); cp $d/pathway.pwml $W/out/$L/show9/$s.pwml; done
git -C $T checkout -- data/enrichment_cache.json data/id_mapping_cache.json 2>/dev/null
echo "RUN_DONE $L"
