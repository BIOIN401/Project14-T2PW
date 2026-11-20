#!/bin/bash
# usage: run_after.sh LABEL RUNID  -- focused layout tests, then the 97 + 8 replays of the lane tree, in one hold.
L=$1; R=$2
"C:/Program Files/Git/bin/bash.exe" C:/t/lane-pwdraw-work/scripts/run_focused.sh $R tests/test_pwml_layout.py tests/test_pwml_l3_chain_layout.py tests/test_pwml_l2_label_fit.py
"C:/Program Files/Git/bin/bash.exe" C:/t/lane-pwdraw-work/scripts/run_replays.sh $L C:/t/lane-pwdraw/src
