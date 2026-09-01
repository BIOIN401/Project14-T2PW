#!/usr/bin/env bash
# REV-108 job wrapper. usage: rev108_job.sh <label> <timeout> <logfile> <tree> -- <cmd...>
set -u
PY="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
G11="C:/t/rev108/docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py"
BR="C:/t/rev108/docs/pwml_recovery_sprint/evidence/bounded_run.py"
LABEL="$1"; TIMEOUT="$2"; LOG="$3"; TREE="$4"; shift 5   # shift past '--'
P=$("$PY" "$G11" next --task REV-108 --label "$LABEL" 2>/dev/null | tail -1)
if [ -z "$P" ]; then echo "G11 EMPTY for label=$LABEL"; exit 1; fi
case "$P" in *REV-108*"$LABEL".json) : ;; *) echo "G11 INVALID: $P"; exit 1;; esac
echo "G11PATH=$P"
PYTHONPATH="$TREE/src" T2PW_OFFLINE_CURATOR=1 PYTHONIOENCODING=utf-8 \
  "$PY" "$BR" --timeout "$TIMEOUT" --label "$LABEL" --heavy-lock REV-108 --json "$P" \
  -- "$PY" "$@" > "$LOG" 2>&1
RC=$?
echo "RC=$RC"
grep -E "FINAL SURVIVING COUNT|cleanup" "$LOG" | tail -5
