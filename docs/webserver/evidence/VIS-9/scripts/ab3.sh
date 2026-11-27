#!/bin/bash
# VIS-9 review round: head replay h2 (base 'out/base' from ab1 is reused: same base tree b9534f68), content checks,
# focused tests, then suite2.
GB="C:/Program Files/Git/bin/bash.exe"; W=C:/t/vis-pw9-work; PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
"$GB" $W/scripts/run.sh h3 C:/t/vis-pw9
$PY -I -X utf8 $W/scripts/cc9.py $W/out/base/show9 $W/out/h3/show9 $W/metrics/cc9_show_h3.json; echo "EXIT_cc_show=$?"
$PY -I -X utf8 $W/scripts/cc9.py $W/out/base/corpus/pwml $W/out/h3/corpus/pwml $W/metrics/cc9_corpus_h3.json; echo "EXIT_cc_corpus=$?"
for m in "" strict; do
  $PY -I -X utf8 C:/t/vis-pw7-work/scripts/content_check.py $W/out/base/show9 $W/out/h3/show9 $W/metrics/v7cc_show_h3$m.json $m | head -c 120; echo
  $PY -I -X utf8 C:/t/vis-pw7-work/scripts/content_check.py $W/out/base/corpus/pwml $W/out/h3/corpus/pwml $W/metrics/v7cc_corpus_h3$m.json $m | head -c 120; echo
done
"$GB" $W/scripts/focused2.sh r4
"$GB" C:/t/vis-runs/suite2.sh C:/t/vis-pw9 vis9-suite2-s5
echo "AB2_DONE"
