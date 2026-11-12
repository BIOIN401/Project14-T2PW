#!/bin/bash
# usage: import_loop.sh NAME   ; repeats bounded invocations with --resume until results.jsonl has all lines in list
N=$1; D=C:/t/fp/docs/pwml_output/evidence/final_payload_p2/import
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
want=$(grep -c . $D/list_$N.txt)
for i in $(seq 1 12); do
  have=0; [ -f $D/$N/results.jsonl ] && have=$(grep -c . $D/$N/results.jsonl)
  [ "$have" -ge "$want" ] && break
  until [ ! -e C:/t/heavylock ]; do sleep 20; done
  $PY C:/t/fp/docs/pwml_recovery_sprint/evidence/bounded_run.py --label pwi-$N-$i --timeout 540 --heavy-lock pwi --json $D/$N.bounded_$i.json -- \
    $PY -X utf8 C:/t/pwi/t2pw/tools/pathwhiz_import/run_import.py --list $D/list_$N.txt --out $D/$N --resume --chunk 10 > $D/$N.invoke_$i.out 2>&1
  echo "$N invocation $i exit=$? have=$have want=$want"
done
