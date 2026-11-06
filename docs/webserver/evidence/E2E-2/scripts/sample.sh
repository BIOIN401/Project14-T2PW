#!/bin/bash
while true; do
  echo "=== SAMPLE $(date +%H:%M:%S)"
  find C:/t/ws-e2e/tmp C:/t/ws-e2e/outputs -type f -mmin -400 -printf '%TH:%TM:%TS %s %p\n' 2>/dev/null | sort -r | head -8
  ls -l C:/t/e2e2/stacks | awk '{print $5, $9}' | tail -n +2
  if grep -q "^WRAPPER_EXIT" C:/t/e2e2/report/wrapper_stdout.log; then echo "WRAPPER_EXIT seen"; break; fi
  sleep 600
done
