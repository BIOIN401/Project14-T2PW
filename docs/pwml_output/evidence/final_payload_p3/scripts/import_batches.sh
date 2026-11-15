#!/bin/bash
# pass 2: batches of 10 files, each one bounded invocation that finishes well inside 540 s (no mid-chunk kill)
D=C:/t/fp/docs/pwml_output/evidence/final_payload_p3/import
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
for N in controls on; do
  mkdir -p $D/$N/lists
  split -l 10 -d $D/list_$N.txt $D/$N/lists/b_
  for L in $D/$N/lists/b_*; do
    B=$(basename $L)
    until [ ! -e C:/t/heavylock ]; do sleep 20; done
    $PY C:/t/fp/docs/pwml_recovery_sprint/evidence/bounded_run.py --label pwi-$N-$B --timeout 540 --heavy-lock pwi --json $D/$N/$B.bounded.json -- \
      $PY -X utf8 C:/t/pwi/t2pw/tools/pathwhiz_import/run_import.py --list $L --out $D/$N/$B --chunk 10 > $D/$N/$B.out 2>&1
    echo "$N $B exit=$?"
  done
  cat $D/$N/b_*/results.jsonl > $D/$N/results.jsonl
done
echo ALLDONE
