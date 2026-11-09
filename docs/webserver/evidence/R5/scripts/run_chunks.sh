#!/bin/bash
# usage: run_chunks.sh TREE PREFIX name-regex
TREE="$1"; P="$2"; RE="$3"
EV="C:/t/ws-R5-tmp/ev/$P"
while IFS='|' read -r name args; do
  [[ "$name" =~ $RE ]] || continue
  bash C:/t/ws-R5-tmp/rt.sh "$TREE" "$P-$name" "$EV" 280 $args --no-header -p no:cacheprovider
done < C:/t/ws-R5-tmp/chunks.txt
