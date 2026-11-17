#!/bin/bash
# runs the remaining idx in chunks of 5, each under its own bounded wrapper (lock corr-a1), --resume style
cd C:/t/corr-assess
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
ALL="7 128 116 5 134 135 78 125 129 138 120 148 19 144 126 122 121 142 114 14 10 58 12 150 68"
set -- $ALL
k=1
while [ $# -gt 0 ]; do
  chunk=""
  for j in 1 2 3 4 5; do
    [ $# -gt 0 ] && { chunk="$chunk $1"; shift; }
  done
  bash scripts/a1_run.sh 3400 a1/logs/chunk$k.json -- $PY C:/t/corr-assess/scripts/a1_import_all.py --resume --idx $chunk > a1/logs/chunk$k.log 2>&1
  echo "chunk$k done:$chunk exit=$?"
  k=$((k+1))
done
echo ALL_CHUNKS_DONE
