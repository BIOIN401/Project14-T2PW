#!/bin/bash
# Review round 1 re-gate, one heavy-lock hold on the committed HEAD of draw/pathwhiz:
# focused tests, the 97 + 8 replays (after5), content check (97 and 8), suite2. The PathWhiz A/B is final4.sh
# (Docker had to be stopped: free RAM 3 GB, under the 4 GB gate floor, with it up).
B="C:/Program Files/Git/bin/bash.exe"
W=C:/t/lane-pwdraw-work
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
cd C:/t/lane-pwdraw
echo "HEAD $(git rev-parse HEAD)"
echo "STATUS_BEGIN"; git status --porcelain; echo "STATUS_END"
"$B" $W/scripts/run_focused.sh f6 $(ls tests/test_pwml_*.py) tests/test_final_payload_writer_fixes.py tests/test_orch768_replay_corpus.py
L=after5
export T2PW_OFFLINE_CURATOR=1
$PY -I -B -X utf8 $W/scripts/replay97.py --src C:/t/lane-pwdraw/src --label $L --out $W/replay
echo "EXIT_replay97=$?"
$PY -I -B -X utf8 $W/scripts/show_replay.py --src C:/t/lane-pwdraw/src --spec $W/spec.json --out $W/show/$L
echo "EXIT_show=$?"
$PY -I -X utf8 $W/scripts/cc_copy.py $L
$PY -I -X utf8 $W/scripts/content_check.py $W/replay/base/pwml $W/replay/$L/pwml $W/metrics/cc97_$L.json
echo "EXIT_cc97=$?"
$PY -I -X utf8 $W/scripts/content_check.py $W/cc_show/base $W/cc_show/$L $W/metrics/cc8_$L.json
echo "EXIT_cc8=$?"
"$B" C:/t/vis-runs/suite2.sh C:/t/lane-pwdraw pwdraw-s3
echo "FINAL3_DONE"
