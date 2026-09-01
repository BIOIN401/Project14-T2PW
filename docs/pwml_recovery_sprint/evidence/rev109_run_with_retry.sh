#!/usr/bin/env bash
# REV-109 reviewer: acquire the SHARED heavy lock by RETRYING, never by breaking it.
# bounded_run exit 95 (BOUNDED_RUN_HEAVY_LOCK_HELD) means the child was NEVER
# started -- it is not a result, so we simply wait and try again.
# usage: run_with_retry.sh <label> <timeout-s> <logfile> -- <cmd...>
set -u
VP="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
TREE=C:/t/rev109
LABEL="$1"; TMO="$2"; LOG="$3"; shift 4   # drop the literal --

P=$("$VP" "$TREE/docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py" \
      next --task REV-109 --label "$LABEL" 2>/dev/null | tail -1)
if [ -z "$P" ]; then echo "G11 PATH EMPTY -- refusing to run"; exit 90; fi
case "$P" in
  *REV-109*"$LABEL".json) echo "G11: $P" ;;
  *) echo "G11 PATH INVALID: $P"; exit 91 ;;
esac

cd "$TREE" || exit 92
ATTEMPT=0
while [ $ATTEMPT -lt 40 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  PYTHONPATH="$TREE/src" T2PW_OFFLINE_CURATOR=1 PYTHONIOENCODING=utf-8 \
    "$VP" docs/pwml_recovery_sprint/evidence/bounded_run.py \
      --timeout "$TMO" --label "$LABEL" --heavy-lock REV-109 --json "$P" \
      -- "$@" > "$LOG" 2>&1
  RC=$?
  if [ $RC -ne 95 ]; then
    echo "attempt $ATTEMPT: wrapper exit $RC"
    exit $RC
  fi
  echo "attempt $ATTEMPT: heavy lock held by another card; child NOT started; waiting"
  sleep 15
done
echo "gave up after $ATTEMPT attempts -- lock never free"
exit 95
