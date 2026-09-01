#!/usr/bin/env bash
set -u
TREE=C:/t/rev111
PY=c:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
EV=$TREE/docs/pwml_recovery_sprint/evidence
cd "$TREE" || exit 1
export PYTHONPATH=C:/t/rev111/src T2PW_OFFLINE_CURATOR=1 PYTHONIOENCODING=utf-8
a=0
while [ $a -lt 240 ]; do
  a=$((a+1))
  "$PY" "$EV/bounded_run.py" --timeout 900 --label rev111-fsync-threat-model \
    --heavy-lock REV-111 --json "$EV/g11/REV-111/05-rev111-fsync-threat-model.json" \
    -- "$PY" -u "$EV/rev111_fsync_threat_model.py" > "$EV/rev111_fsync_threat_model.log" 2>&1
  rc=$?
  [ $rc -ne 95 ] && { echo "acquired on attempt $a, exit $rc"; break; }
  sleep 3
done
echo "FSYNC ADDENDUM DONE rc=$rc"
