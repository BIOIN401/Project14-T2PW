#!/bin/bash
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
export PYTHONPATH=C:/t/fix-s2scope/src PYTHONIOENCODING=utf-8
E=C:/t/fix-s2scope/docs/webserver/evidence/FIX-S2SCOPE
cd C:/t/vis-runs/scratch/s2scope
timeout 1500 $PY -X utf8 replay.py C:/t/fix-s2scope $E/replay_heme.json; echo "EXIT_replay=$?"
timeout 900 $PY -X utf8 oosdump.py C:/t/fix-s2scope $E/oos_rows.json; echo "EXIT_oosdump=$?"
