#!/bin/bash
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
for n in "$@"; do
  $PY C:/t/ev-RT1/mutate.py apply $n || { echo "$n: APPLY FAILED"; continue; }
  bash C:/t/ev-RT1/run_rev.sh $n tests/test_llm_call_cap.py tests/test_llm_call_logging.py
  $PY C:/t/ev-RT1/mutate.py restore $n
done
