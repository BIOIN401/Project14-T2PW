#!/bin/bash
# One heavy-lock hold on HEAD 4b701677: focused tests, the 97 + 8 replays (after4), content check, suite2, then the
# PathWhiz A/B with both arms imported fresh (base into pw/base2, after into pw/after2; final.sh's run lost its
# showcase imports to CRLF paths), then the stack goes down (volumes kept).
B="C:/Program Files/Git/bin/bash.exe"
W=C:/t/lane-pwdraw-work
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
cd C:/t/lane-pwdraw
git log --oneline -1
"$B" $W/scripts/run_focused.sh f5 $(ls tests/test_pwml_*.py) tests/test_final_payload_writer_fixes.py tests/test_orch768_replay_corpus.py
"$B" $W/scripts/run_replays.sh after4 C:/t/lane-pwdraw/src
$PY -X utf8 $W/scripts/content_check.py $W/replay/base/pwml $W/replay/after4/pwml $W/metrics/cc97_after4.json
echo "EXIT_cc97=$?"
"$B" C:/t/vis-runs/suite2.sh C:/t/lane-pwdraw pwdraw-s2
$PY -X utf8 $W/scripts/prep_pwab.py base after4
docker info > /dev/null 2>&1 || { echo "DOCKER_NOT_RUNNING"; exit 3; }
for arm in base after; do
  $PY -X utf8 $W/harness/run_import.py --list $W/pwab/${arm}_list.txt --screenshots $(cat $W/pwab/${arm}_shots.txt)      --out $W/pw/${arm}2 --smpdb-src C:/t/pwpr/latest --work C:/t/pwpr/pwdraw-run --resume
  echo "EXIT_import_${arm}2=$?"
done
cd $W/harness && SMPDB_SRC=C:/t/pwpr/pwdraw-run OUT_DIR=$W/pw/after2 PWML_DIR=$W/harness/pwml_in docker compose -p t2pw-pwdraw down
echo "STACK_DOWN rc=$?"
echo "FINAL2_DONE"
