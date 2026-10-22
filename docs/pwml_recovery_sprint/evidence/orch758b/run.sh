#!/bin/bash
# usage: run.sh <label> <timeout> <test-file-list-file>
cd C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch758b || exit 1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe; EV=docs/pwml_recovery_sprint/evidence; L=$1; T=$2
P=$($PY $EV/g11/g11_evidence.py next --task ORCH-758b --label $L 2>&1 | tail -1)
case "$P" in *rror*|"") echo INVALID; exit 1;; esac; case "$P" in *"$L.json") : ;; *) echo "INVALID $P"; exit 1;; esac
N=$(basename "$P" .json); mkdir -p $EV/orch758b; echo "$P"
FILES=$(tr -d '\r' < "$3" | tr '\n' ' ')
PYTHONPATH=$PWD/src PYTHONIOENCODING=utf-8 T2PW_OFFLINE_CURATOR=1 $PY $EV/bounded_run.py --label $L --timeout $T --json "$P" -- \
  $PY -u $EV/pinned_pytest.py --pin-verdict $EV/g11/pin/ORCH-758b/$N.pin.json -q --basetemp=C:/t/bt-orch758b/$N-$RANDOM $FILES \
  > $EV/orch758b/$N.log 2>&1; echo exit=$?; grep -E "passed|failed|error|FINAL SURVIVING" $EV/orch758b/$N.log | tail -5
