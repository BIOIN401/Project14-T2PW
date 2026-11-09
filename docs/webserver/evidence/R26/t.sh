#!/bin/bash
# usage: t.sh LABEL TREE selection...   -- one gated pytest run through the bounded wrapper, heavy lock R26.
# TREE is the absolute worktree/copy to measure; its evidence goes to C:/t/wsR26tmp/ev.
LABEL=$1; TREE=$2; shift 2
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
mkdir -p C:/t/wsR26tmp/ev C:/t/wsR26tmp/bt
$PY C:/t/wsR26tmp/gate.py wait 240 || { echo "GATE NOT CLEAR"; tail -1 C:/t/wsR26tmp/gate.log | cut -c1-300; exit 7; }
cd "$TREE" || exit 1
export PYTHONPATH="$TREE/src" T2PW_OFFLINE_CURATOR=1 T2PW_SPECIES_LLM=0 T2PW_SPECIES_NCBI=0
$PY "$TREE/docs/pwml_recovery_sprint/evidence/bounded_run.py" --label "$LABEL" --timeout 340 --heavy-lock R26 \
  --json "C:/t/wsR26tmp/ev/$LABEL.json" -- \
  $PY -u "$TREE/docs/pwml_recovery_sprint/evidence/pinned_pytest.py" --expect-tree "$TREE" -q -rs --durations=5 \
  --basetemp="C:/t/wsR26tmp/bt/$LABEL" "$@" > "C:/t/wsR26tmp/ev/$LABEL.log" 2>&1
echo "wrapper exit $?"
grep -aE "passed|failed|error|exit reason|duration  |FINAL SURV|HEAVY_LOCK_HELD|wrapper vs HEAD|REFUSED" "C:/t/wsR26tmp/ev/$LABEL.log" | cut -c1-220 | tail -12
