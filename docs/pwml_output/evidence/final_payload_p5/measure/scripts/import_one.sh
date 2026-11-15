#!/bin/bash
# usage: import_one.sh NAME SMPDB_SRC LISTFILE  (chunks of 12 files, --resume, one bounded run per chunk, lock-acquire retried on 95)
NAME=$1; SRC=$2; LIST=$3
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/pwi/p5w/$NAME-work; O=C:/t/p5scratch/measure/import/$NAME-out; L=C:/t/p5scratch/measure/import/$NAME-lists
mkdir -p $L $O; rm -f $L/c_*
split -l 12 -d $LIST $L/c_
for CL in $L/c_*; do
  B=$(basename $CL); tries=0; rc=1
  while [ $tries -lt 3 ] && [ $rc -ne 0 ]; do
    n=0
    while [ $n -lt 9000 ]; do
      n=$((n+1))
      $PY C:/t/fp/docs/pwml_recovery_sprint/evidence/bounded_run.py --label pwi-p5-$NAME-$B --timeout 1500 --heavy-lock pwi-$NAME --json $O/$B.bounded.json -- \
        $PY -X utf8 C:/t/pwi/t2pw/tools/pathwhiz_import/run_import.py --list $CL --out $O --resume --chunk 12 --smpdb-src $SRC --work $W > $O/$B.out 2>&1
      rc=$?
      if [ $rc -ne 95 ]; then break; fi
      sleep 2
    done
    tries=$((tries+1))
    echo "$NAME $B exit=$rc lockretries=$n try=$tries $(date +%H:%M:%S)"
  done
done
echo "$NAME DONE results=$(wc -l < $O/results.jsonl)"
