#!/bin/bash
# usage: compare.sh EVDIR   -> per chunk: tip summary | base summary | failing-node sets identical? | survivors
EV=$1
for t in "$EV"/t-*.log; do
  id=$(basename "$t" .log); id=${id#t-}; b="$EV/b-$id.log"
  ts=$(grep -E "^[0-9]+ (passed|failed)|^=+ .*(passed|failed)" "$t" | tail -1 | sed 's/ in [0-9.]*s.*//; s/=//g')
  tsv=$(grep -o "FINAL SURVIVING COUNT *: *[0-9]*" "$t" | grep -o "[0-9]*$")
  tf=$(grep -E "^FAILED|^ERROR" "$t" | sed 's/ - .*//' | sort)
  if [ -f "$b" ]; then
    bs=$(grep -E "^[0-9]+ (passed|failed)|^=+ .*(passed|failed)" "$b" | tail -1 | sed 's/ in [0-9.]*s.*//; s/=//g')
    bsv=$(grep -o "FINAL SURVIVING COUNT *: *[0-9]*" "$b" | grep -o "[0-9]*$")
    bf=$(grep -E "^FAILED|^ERROR" "$b" | sed 's/ - .*//' | sort)
    same=$([ "$tf" = "$bf" ] && echo identical || echo DIFFERENT)
  else bs="n/a"; bsv="-"; same="-"; fi
  printf "%-14s | tip: %-40s surv=%s | base: %-40s surv=%s | failing sets: %s\n" "$id" "$ts" "$tsv" "$bs" "$bsv" "$same"
  [ -n "$tf" ] && echo "$tf" | sed 's/^/      tip  /'
  [ "$same" = "DIFFERENT" ] && echo "$bf" | sed 's/^/      base /'
done
