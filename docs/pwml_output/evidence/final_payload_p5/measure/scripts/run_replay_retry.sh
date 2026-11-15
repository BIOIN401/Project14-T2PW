#!/bin/bash
# usage: run_replay_retry.sh LABEL FLAG ; retries the wrapper's own heavy-lock acquire on exit 95
LABEL=$1; FLAG=$2
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
R=C:/t/p5scratch/measure/replay; mkdir -p $R/$LABEL
S=C:/t/fp/docs/pwml_output/evidence/final_payload_p5/measure/scripts
cd $R
n=0
while [ $n -lt 9000 ]; do
  n=$((n+1))
  $PY C:/t/fp/docs/pwml_recovery_sprint/evidence/bounded_run.py --label fp-p5m-$LABEL --timeout 2400 --heavy-lock fp-p5m --json $R/$LABEL/bounded.json -- \
    $PY -X utf8 $S/replay.py --src C:/t/p5scratch/measure/code/src --flag $FLAG --label $LABEL --start 0 --count 152 > $R/$LABEL/chunk.out 2>&1
  rc=$?
  if [ $rc -ne 95 ]; then break; fi
  sleep 2
done
echo "$LABEL exit=$rc tries=$n"; tail -2 $R/$LABEL/chunk.out | cut -c1-200; wc -l $R/$LABEL/results_0.jsonl
