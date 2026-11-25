#!/bin/bash
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
export PYTHONPATH=C:/t/fix-s2scope/src PYTHONIOENCODING=utf-8
cd C:/t/vis-runs/scratch/s2scope
timeout 1800 $PY -X utf8 replay.py C:/t/fix-s2scope C:/t/fix-s2scope/docs/webserver/evidence/FIX-S2SCOPE/replay.json
echo "EXIT_replay=$?"
