#!/usr/bin/env bash
# W9-local: stop what start.sh started. Kills ONLY the two bounded_run PIDs start.sh recorded
# (with their trees: uvicorn, the worker and any running job). Never by image name.
set -u
D=${T2PW_DATA:-C:/t/t2pw-server-data}
[ -s "$D/pids" ] || { echo "nothing recorded in $D/pids"; exit 0; }
for p in $(cat "$D/pids"); do
  taskkill //PID "$p" //T //F || echo "pid $p was not running"
done
rm -f "$D/pids"

# A job killed mid-run strands the heavy lock. Clear it only if it names THIS deployment's
# job holder and that holder's process is gone; anything else is left for a human.
L=C:/t/heavylock/holder.json
if [ -f "$L" ] && grep -q '"holder": "W9-local-job"' "$L"; then
  pid=$(grep -o '"pid": [0-9]*' "$L" | grep -o '[0-9]*$')
  if ! tasklist //FI "PID eq $pid" //NH | grep -q " $pid "; then
    rm -f "$L" && rmdir C:/t/heavylock && echo "cleared W9-local's stranded heavy lock (pid $pid)"
  fi
fi
