#!/bin/bash
# usage: run_replays.sh LABEL SRC   -- 152-run replay (97 PWML) + the 8 showcase replays of one code tree, in one lock hold.
L=$1; SRC=$2
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/pwml-fix-work
export PYTHONPATH=$SRC T2PW_OFFLINE_CURATOR=1
$PY -X utf8 $W/replay97.py --src $SRC --label $L --out $W/replay
echo "EXIT_replay97=$?"
$PY -X utf8 $W/show_replay.py --src $SRC --spec $W/spec.json --out $W/show/$L
echo "EXIT_show=$?"
$PY -X utf8 $W/show_replay.py --src $SRC --spec $W/b2spec.json --out $W/b2/$L
echo "EXIT_b2=$?"
echo "REPLAYS_DONE $L"
