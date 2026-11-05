#!/bin/bash
# usage: seg3.sh LABEL -- one gated record segment of the third recording (work dir rec3)
LABEL=$1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
cd /c/t/ws-R25 || exit 1
if ! $PY C:/t/wsR25tmp/gate.py check; then echo "GATE NOT CLEAR; segment not started"; tail -1 C:/t/wsR25tmp/gate.log | cut -c1-300; exit 7; fi
export PYTHONPATH=C:/t/ws-R25/src
$PY docs/pwml_recovery_sprint/evidence/bounded_run.py --label "$LABEL" --timeout 575 --heavy-lock R2.5 \
  --json "C:/t/wsR25tmp/ev3/$LABEL.json" -- \
  $PY -u C:/t/ws-R25/tests/equivalence/harness.py record \
  --env-file C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.env --work C:/t/wsR25tmp/rec3 \
  > "C:/t/wsR25tmp/ev3/$LABEL.log" 2>&1
echo "wrapper exit $?" | tee -a C:/t/wsR25tmp/gate.log
grep -aE "^segment|^case |exit reason|duration  |FINAL SURV|NOT WRITING|wrote |HEAVY_LOCK_HELD|Traceback|Error" "C:/t/wsR25tmp/ev3/$LABEL.log" | cut -c1-220 | tail -15
echo "live answers so far: $(cat C:/t/wsR25tmp/rec3/segments/*.jsonl 2>/dev/null | wc -l) lines across segment files; last: $(wc -l < $(ls C:/t/wsR25tmp/rec3/segments/seg-*.jsonl 2>/dev/null | tail -1) 2>/dev/null)"
