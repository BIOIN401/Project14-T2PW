#!/bin/bash
# usage: q.sh LABEL  (light bounded run of quick.sh)
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
BR=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/pwml_recovery_sprint/evidence/bounded_run.py
$PY $BR --label vis7-$1 --timeout 1500 --json C:/t/vis-pw7-work/logs/q_$1.json -- "C:/Program Files/Git/bin/bash.exe" C:/t/vis-pw7-work/scripts/quick.sh $1 > C:/t/vis-pw7-work/logs/q_$1.log 2>&1
echo "Q_EXIT=$?"; grep -E "EXIT_|FINAL SURV|HEAD|QUICK_DONE" C:/t/vis-pw7-work/logs/q_$1.log
