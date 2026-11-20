#!/bin/bash
# usage: run_replays.sh LABEL SRC   -- 152-run replay (97 PWML) + the 8 showcase replays of one code tree, in one lock hold.
L=$1; SRC=$2
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/lane-pwdraw-work
export PYTHONPATH=$SRC T2PW_OFFLINE_CURATOR=1
$PY -B -X utf8 $W/scripts/replay97.py --src $SRC --label $L --out $W/replay
echo "EXIT_replay97=$?"
$PY -B -X utf8 $W/scripts/show_replay.py --src $SRC --spec $W/spec.json --out $W/show/$L
echo "EXIT_show=$?"
echo "REPLAYS_DONE $L"
