#!/bin/bash
# usage: chunk.sh LABEL SRC FLAG START COUNT [extra replay.py args]
# waits <= ~55 min for heavy lock absent + free RAM >= 4 GB, then runs replay.py under bounded_run
LABEL=$1; SRC=$2; FLAG=$3; START=$4; COUNT=$5; shift 5
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
R=C:/t/fp/docs/pwml_output/evidence/final_payload_p4/replay; mkdir -p $R/$LABEL
n=0
until [ ! -e C:/t/heavylock ] && [ "$(powershell.exe -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)' | tr -d '\r')" -ge 4 ]; do
  n=$((n+1)); if [ $n -ge 11 ]; then echo "$LABEL GATE_BLOCKED after 11 waits"; exit 99; fi
  sleep 290
done
cd $R
$PY C:/t/fp/docs/pwml_recovery_sprint/evidence/bounded_run.py --label fp-replay-$LABEL-$START --timeout 2400 --heavy-lock fp --json $R/$LABEL/bounded_$START.json -- \
  $PY -X utf8 C:/t/fp/docs/pwml_output/evidence/final_payload_p4/scripts/replay.py --src $SRC --flag $FLAG --label $LABEL --start $START --count $COUNT "$@" > $R/$LABEL/chunk_$START.out 2>&1
echo "$LABEL $START exit=$?"; tail -3 $R/$LABEL/chunk_$START.out | cut -c1-200; wc -l $R/$LABEL/results_$START.jsonl
