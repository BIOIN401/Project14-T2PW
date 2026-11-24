#!/bin/bash
# usage: gates2.sh TREE RUNID -- final: after-metrics, PWML proof on base tree and branch (+ sha compare), suite2, judge.
T=$1; R=$2
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/lane-draw-work
BASE=C:/t/lane-draw-base
"C:/Program Files/Git/bin/bash.exe" $W/scripts/baseline.sh $T final
export T2PW_OFFLINE_CURATOR=1
if [ ! -s $W/out/proof_base.log ] || ! grep -q NO_REGRESSION $W/out/proof_base.log; then
  PYTHONPATH=$BASE/src $PY -X utf8 C:/t/vis-runs/scratch/generate/proof_generate.py $BASE C:/t/vis-runs/scratch/generate/replay_out/base_on $W/proof_base on > $W/out/proof_base.log 2>&1
  echo "PROOF_BASE_EXIT=$?"
fi
grep NO_REGRESSION $W/out/proof_base.log
PYTHONPATH=$T/src $PY -X utf8 C:/t/vis-runs/scratch/generate/proof_generate.py $T C:/t/vis-runs/scratch/generate/replay_out/base_on $W/proof_$R on > $W/out/proof_$R.log 2>&1
echo "PROOF_EXIT=$?"
grep NO_REGRESSION $W/out/proof_$R.log
python $W/scripts/proofcmp.py $W/out/proof_base.log $W/out/proof_$R.log
"C:/Program Files/Git/bin/bash.exe" C:/t/vis-runs/suite2.sh $T $R
PYTHONPATH=$T/src $PY -X utf8 $W/scripts/judge.py $T $W/judge $W/pairs.json > $W/out/judge.log 2>&1
echo "JUDGE_EXIT=$?"
tail -1 $W/out/judge.log
echo GATES2_DONE
