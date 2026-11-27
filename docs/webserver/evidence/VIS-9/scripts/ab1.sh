#!/bin/bash
# VIS-9 A/B replay: base tree (b9534f68 archive) and head worktree, then content checks.
GB="C:/Program Files/Git/bin/bash.exe"; W=C:/t/vis-pw9-work; PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
"$GB" $W/scripts/run.sh base $W/base_tree
"$GB" $W/scripts/run.sh $1 C:/t/vis-pw9
mkdir -p $W/metrics
$PY -I -X utf8 $W/scripts/cc9.py $W/out/base/show9 $W/out/$1/show9 $W/metrics/cc9_show_$1.json; echo "EXIT_cc_show=$?"
$PY -I -X utf8 $W/scripts/cc9.py $W/out/base/corpus/pwml $W/out/$1/corpus/pwml $W/metrics/cc9_corpus_$1.json; echo "EXIT_cc_corpus=$?"
echo "AB1_DONE"
