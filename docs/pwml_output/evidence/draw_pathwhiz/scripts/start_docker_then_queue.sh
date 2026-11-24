#!/bin/bash
# Outside any lock: wait until no heavy job holds the lock and free RAM >= 7 GB, start Docker Desktop (it must not be a
# child of the bounded wrapper), then queue final4.sh (the PathWhiz A/B) through the shared queue.
free_gb() { powershell.exe -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)' | tr -d '\r'; }
n=0
until [ ! -e C:/t/heavylock/holder.json ] && [ "$(free_gb)" -ge 7 ]; do
  n=$((n+1)); [ $n -ge 480 ] && { echo "NEVER_ROOM"; exit 1; }
  sleep 30
done
echo "room at $(date) free=$(free_gb)GB"
docker desktop start 2>&1 | tail -1
until docker info > /dev/null 2>&1; do sleep 5; done
echo "docker up, free=$(free_gb)GB"
"C:/Program Files/Git/bin/bash.exe" C:/t/vis-runs/queue.sh pwdraw pwdraw-final4 7200 C:/t/lane-pwdraw-work/logs/final4.json 4 \
  C:/t/lane-pwdraw-work/logs/final4.log -- "C:/Program Files/Git/bin/bash.exe" C:/t/lane-pwdraw-work/scripts/final4.sh
echo "QUEUE_RC=$?"
