#!/bin/bash
# Retry the bounded run until the heavy lock is acquired (a failed acquire exits 95 and runs nothing).
T=C:/t/lay; PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
E=$T/docs/pwml_output/evidence/layout/tests
cd $T
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
unset T2PW_PWML_LAYOUT T2PW_FINAL_PAYLOAD
for n in $(seq 1 6000); do
  if [ -e C:/t/heavylock/holder.json ]; then sleep 0.2; continue; fi
  $PY C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/pwml_recovery_sprint/evidence/bounded_run.py --label lay-golden --timeout 1500 --heavy-lock lay --json $E/golden.json -- \
    $PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=C:/t/layx/tmpbase/golden-$$-$n \
    tests/test_batch_driver_seam_golden.py tests/test_c011_freeze_seam_golden_equivalence.py tests/test_streamlit_stage8_export_contract.py tests/test_streamlit_quarantine_boundary.py > $E/golden.stdout.txt 2>&1
  rc=$?
  if [ $rc -ne 95 ]; then echo "golden exit=$rc attempt=$n"; grep -E "passed|failed|error" $E/golden.stdout.txt | tail -2; grep "FINAL SURVIVING" $E/golden.stdout.txt; exit 0; fi
done
echo GATE_BLOCKED
