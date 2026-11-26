#!/bin/bash
# usage: rend.sh LABEL IDX... -- render replay runs of base and LABEL with arrowheads
L=$1; shift
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
W=C:/t/vis-pw1-work
for i in "$@"; do
  mkdir -p $W/rr/base $W/rr/$L
  cp $W/replay/base/pwml/$i.pwml $W/rr/base/r$i.pwml; cp $W/replay/$L/pwml/$i.pwml $W/rr/$L/r$i.pwml
  PYTHONPATH=$W/base/src $PY -X utf8 $W/scripts/render_arrows.py $W/renders/base $W/rr/base/r$i.pwml
  PYTHONPATH=C:/t/vis-pw1/src $PY -X utf8 $W/scripts/render_arrows.py $W/renders/$L $W/rr/$L/r$i.pwml
done
echo REND_DONE
