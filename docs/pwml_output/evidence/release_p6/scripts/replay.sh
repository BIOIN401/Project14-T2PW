#!/bin/bash
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
E=C:/t/rel/docs/pwml_output/evidence/layout/scripts
B=C:/t/rel-reports
R=$B/replay
while [ -e C:/t/heavylock/holder.json ]; do sleep 10; done
export PYTHONPATH=C:/t/rel/src T2PW_OFFLINE_CURATOR=1
cd C:/t/rel
run() { # label flag
  $PY C:/t/rel/docs/pwml_recovery_sprint/evidence/bounded_run.py --heavy-lock rel-runner --timeout 3000 --label REL-replay-$1 --json $B/replay_$1.json -- $PY -X utf8 $E/replay.py --src C:/t/rel/src --flag $2 --label $1 --start 0 --count 152 --out $R --layout v2 > $B/replay_$1.log 2>&1
  echo "replay $1 exit=$?" >> $B/replay_all.log
  cp $R/$1/results_0.jsonl $B/replay_results_$1.jsonl
  $PY -X utf8 $E/layout_metrics.py $B/metrics_$1.json $R/$1/pwml > $B/metrics_$1.out 2>&1
}
run on_v2 1
$PY -X utf8 $E/side_pwml.py $B/side_pwml.json $R/on_v2/pwml $R/on_v2/pwml > $B/side_pwml.out 2>&1
echo done >> $B/replay_all.log
