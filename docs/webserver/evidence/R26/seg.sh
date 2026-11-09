#!/bin/bash
# usage: seg.sh ATTEMPT LABEL   -- one gated record segment (work dir recN, evidence evN).
A=$1; LABEL=$2
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
$PY C:/t/wsR26tmp/gate.py wait 240 || { echo "GATE NOT CLEAR"; tail -1 C:/t/wsR26tmp/gate.log | cut -c1-300; exit 7; }
cd /c/t/ws-R26 || exit 1
export PYTHONPATH=C:/t/ws-R26/src PYTHONHASHSEED=0
$PY docs/pwml_recovery_sprint/evidence/bounded_run.py --label "$LABEL" --timeout 340 --heavy-lock R26 \
  --json "C:/t/wsR26tmp/ev$A/$LABEL.json" -- \
  $PY -u C:/t/ws-R26/tests/equivalence/harness.py record --case farnesol_research \
  --env-file C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.env --work "C:/t/wsR26tmp/rec$A" \
  > "C:/t/wsR26tmp/ev$A/$LABEL.log" 2>&1
echo "wrapper exit $?"
grep -aE "^segment|^case |exit reason|duration  |FINAL SURV|NOT WRITING|wrote |HEAVY_LOCK_HELD|Traceback|Error" "C:/t/wsR26tmp/ev$A/$LABEL.log" | cut -c1-220 | tail -15
