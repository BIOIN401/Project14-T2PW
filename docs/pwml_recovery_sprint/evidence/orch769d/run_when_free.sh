#!/usr/bin/env bash
# usage: run_when_free.sh <outfile> <cmd...>
# Waits for the heavy lock to clear and >= 6 GB free RAM, runs, retries while the wrapper exits 95.
out="$1"; shift
while true; do
  while [ -f C:/t/heavylock/holder.json ]; do sleep 10; done
  free=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" | tr -d '\r ')
  if [ "${free:-0}" -lt 6291456 ]; then echo "low RAM ${free}KB, waiting" >&2; sleep 30; continue; fi
  "$@" > "$out" 2>&1
  rc=$?
  if [ $rc -ne 95 ]; then echo "RC=$rc freeKB_before=$free" >> "$out"; echo "RC=$rc"; exit 0; fi
  sleep 5
done
