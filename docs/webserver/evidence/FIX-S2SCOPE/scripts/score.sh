#!/bin/bash
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
export PYTHONPATH=C:/t/fix-s2scope/src PYTHONIOENCODING=utf-8
cd C:/t/vis-runs/scratch/s2scope
timeout 1200 $PY -X utf8 score_ab.py C:/t/fix-s2scope C:/t/fix-s2scope/docs/webserver/evidence/FIX-S2SCOPE/ab_score.json
echo "EXIT_score=$?"
