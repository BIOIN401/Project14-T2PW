#!/usr/bin/env bash
# REV-111 -- retry-until-acquired driver for the two remaining reviewer jobs.
#
# Three agents contend for C:/t/heavylock and exit 95 means the child NEVER
# STARTED -- an infrastructure event, not a result. This retries the SAME
# unmodified bounded_run.py invocation until it acquires, and NEVER touches,
# clears or routes around the mutex.
#
# D-026 shape: same approved wrapper, task id and output paths recorded before
# launch, one heavy job at a time enforced by the lock itself, polled rather
# than duplicated, and each canonical JSON report inspected afterwards.
set -u

TREE=C:/t/rev111
PY=c:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
EV=$TREE/docs/pwml_recovery_sprint/evidence
cd "$TREE" || exit 1

export PYTHONPATH=C:/t/rev111/src
export T2PW_OFFLINE_CURATOR=1
export PYTHONIOENCODING=utf-8

run_until_acquired() {
  local label=$1 json=$2 log=$3 timeout=$4
  shift 4
  local attempt=0
  while [ $attempt -lt 240 ]; do
    attempt=$((attempt + 1))
    "$PY" "$EV/bounded_run.py" --timeout "$timeout" --label "$label" \
      --heavy-lock REV-111 --json "$json" -- "$@" > "$log" 2>&1
    local rc=$?
    if [ $rc -ne 95 ]; then
      echo "=== $label acquired on attempt $attempt, exit $rc"
      return $rc
    fi
    sleep 3
  done
  echo "=== $label NEVER ACQUIRED after $attempt attempts"
  return 95
}

run_until_acquired rev111-adversarial-probe-r2 \
  "$EV/g11/REV-111/03-rev111-adversarial-probe-r2.json" \
  "$EV/rev111_reviewer_probe.log" 900 \
  "$PY" -u "$EV/rev111_reviewer_probe.py"
PROBE_RC=$?

run_until_acquired rev111-reviewer-mutations \
  "$EV/g11/REV-111/04-rev111-reviewer-mutations.json" \
  "$EV/rev111_reviewer_mutations.log" 1800 \
  "$PY" -u "$EV/rev111_reviewer_mutations.py"
MUT_RC=$?

echo "PROBE_RC=$PROBE_RC MUTATIONS_RC=$MUT_RC"
echo "ALL REV-111 REVIEWER JOBS DONE"
