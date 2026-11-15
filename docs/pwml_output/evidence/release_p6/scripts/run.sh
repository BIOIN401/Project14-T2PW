#!/bin/bash
# usage: run.sh LABEL TIMEOUT [ENV=VAL...] -- tests...
L=$1; TO=$2; shift 2
while [ -e C:/t/heavylock/holder.json ]; do sleep 10; done
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
export PYTHONPATH=C:/t/rel/src T2PW_OFFLINE_CURATOR=1
while [ "$1" != "--" ]; do export "$1"; shift; done; shift
cd C:/t/rel
mkdir -p C:/t/rel-reports/tmp/$L
$PY C:/t/rel/docs/pwml_recovery_sprint/evidence/bounded_run.py --heavy-lock rel-runner --timeout $TO --label REL-$L --json C:/t/rel-reports/$L.json -- $PY C:/t/rel/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree C:/t/rel -q --basetemp=C:/t/rel-reports/tmp/$L/bt "$@" > C:/t/rel-reports/$L.log 2>&1
echo "exit=$?" >> C:/t/rel-reports/$L.log
