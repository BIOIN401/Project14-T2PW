#!/bin/bash
# usage: gates3.sh TREE RUNID -- review round 1: tree state, 97-run + showcase metrics, selfcheck on the 97 chosen drawings,
# PWML proof (branch vs base tree sha), suite2. One heavy hold.
T=$1; R=$2
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/lane-draw-work
echo "HEAD $(git -C $T rev-parse HEAD)"; echo "STATUS_BEGIN"; git -C $T status --porcelain; echo "STATUS_END"
"C:/Program Files/Git/bin/bash.exe" $W/scripts/baseline.sh $T $R
NOSHOW=1 PYTHONPATH=$T/src $PY -X utf8 $W/scripts/showcase.py $T $W/out/sc97_$R "$W/ir97/*.json" > $W/out/sc97_$R.log 2>&1
echo "SC97_EXIT=$?"; tail -1 $W/out/sc97_$R.log
export T2PW_OFFLINE_CURATOR=1
grep NO_REGRESSION $W/out/proof_base.log
PYTHONPATH=$T/src $PY -X utf8 C:/t/vis-runs/scratch/generate/proof_generate.py $T C:/t/vis-runs/scratch/generate/replay_out/base_on $W/proof_$R on > $W/out/proof_$R.log 2>&1
echo "PROOF_EXIT=$?"
grep NO_REGRESSION $W/out/proof_$R.log
python $W/scripts/proofcmp.py $W/out/proof_base.log $W/out/proof_$R.log
"C:/Program Files/Git/bin/bash.exe" C:/t/vis-runs/suite2.sh $T $R
echo GATES3_DONE
