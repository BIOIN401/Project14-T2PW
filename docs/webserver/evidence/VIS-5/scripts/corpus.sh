#!/bin/bash
# usage: corpus.sh LABEL SRC -- VIS-5 robustness: the 97-PWML replay corpus (VIS-2 replay97.py) through SRC's export.
L=$1; SRC=$2
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/vis-pw5-work
export PYTHONPATH=$SRC T2PW_OFFLINE_CURATOR=1
$PY -B -X utf8 $W/scripts/replay97.py --src $SRC --label $L --out $W/replay > $W/logs/corpus_$L.txt 2>&1
echo "EXIT_replay97=$?"
git -C C:/t/vis-pw5 checkout -- data/enrichment_cache.json data/id_mapping_cache.json 2>/dev/null
echo "CORPUS_DONE $L"
