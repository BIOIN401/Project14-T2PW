#!/bin/bash
# One heavy-lock hold: focused tests, the 97 + 8 replays (after3), suite2, then the PathWhiz A/B (base vs after3).
B="C:/Program Files/Git/bin/bash.exe"
W=C:/t/lane-pwdraw-work
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
cd C:/t/lane-pwdraw
"$B" $W/scripts/run_focused.sh f4 $(ls tests/test_pwml_*.py) tests/test_final_payload_writer_fixes.py tests/test_orch768_replay_corpus.py
"$B" $W/scripts/run_replays.sh after3 C:/t/lane-pwdraw/src
$PY -X utf8 $W/scripts/content_check.py $W/replay/base/pwml $W/replay/after3/pwml $W/metrics/cc97_after3.json
echo "EXIT_cc97=$?"
"$B" C:/t/vis-runs/suite2.sh C:/t/lane-pwdraw pwdraw-s1
$PY -X utf8 $W/scripts/prep_pwab.py base after3
"$B" $W/scripts/run_pwab.sh
echo "FINAL_DONE"
