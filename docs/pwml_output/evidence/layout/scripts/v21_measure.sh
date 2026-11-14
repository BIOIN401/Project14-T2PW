#!/bin/bash
# v2.1 pass, one bounded job: final_measure.sh into metrics/v2_1 and render/v2_1, then the focused tests.
T=C:/t/lay; E=$T/docs/pwml_output/evidence/layout
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
M_DIR=$E/metrics/v2_1 RENDER_DIR=$E/render/v2_1 "C:/Program Files/Git/usr/bin/bash.exe" $E/scripts/final_measure.sh
$PY $E/scripts/side_pwml.py $E/metrics/v2_1/side_pwml.json C:/t/layx/replay/off_v1/pwml C:/t/layx/replay/on_v1/pwml C:/t/layx/replay/off_v2/pwml C:/t/layx/replay/on_v2/pwml
cd $T
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
unset T2PW_PWML_LAYOUT T2PW_FINAL_PAYLOAD
mkdir -p $E/tests/v2_1 C:/t/layx/tmpbase
$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=C:/t/layx/tmpbase/focused21-$$ \
  tests/test_pwml_layout.py tests/test_pwml_writer.py tests/test_pwml_ir.py tests/test_pwml_ir_duplicate_row_refusal.py \
  tests/test_final_payload_writer_fixes.py > $E/tests/v2_1/focused.stdout.txt 2>&1
echo "focused exit=$?"; tail -1 $E/tests/v2_1/focused.stdout.txt
