#!/bin/sh
# Reviewer independent runs on the committed tip worktree, through bounded_run + pinned_pytest.
T=C:/t/ws-R3c
PY=/c/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
mkdir -p C:/t/rev-R3c/bt-tip C:/t/rev-R3c/out-tip
run() {
  name=$1; shift
  for i in $(seq 1 30); do
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1 "$PY" $T/docs/pwml_recovery_sprint/evidence/bounded_run.py \
      --label rev-R3c-$name --timeout 500 --heavy-lock rev-R3c --json C:/t/rev-R3c/out-tip/$name.json --cwd $T -- \
      "$PY" -u $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q -p no:cacheprovider \
      --basetemp=C:/t/rev-R3c/bt-tip/$name "$@" > C:/t/rev-R3c/out-tip/$name.log 2>&1
    rc=$?
    [ $rc -ne 95 ] && break
    sleep 20
  done
  echo "$name rc=$rc $(grep -E ' in [0-9.]+s' C:/t/rev-R3c/out-tip/$name.log | tail -1) $(grep -oE 'FINAL SURVIVING COUNT\s*:\s*[0-9]+' C:/t/rev-R3c/out-tip/$name.log | tail -1)" | tee -a C:/t/rev-R3c/out-tip/summary.txt
}
case "$1" in
  a)
    run g1 tests/test_batch_driver_seam_golden.py
    run g2 tests/test_c011_freeze_seam_golden_equivalence.py
    run g3 tests/test_streamlit_stage8_export_contract.py
    run g4 tests/test_streamlit_quarantine_boundary.py ;;
  b)
    run eq tests/test_equivalence_app_vs_service.py
    run focus tests/test_streamlit_stage2_orchestration.py tests/test_service_extraction_stage1.py tests/test_c060a_requested_pathway_wiring.py tests/test_service_extraction.py tests/test_service_extraction_stage0.py tests/test_service_reporter.py
    run c055 tests/test_c055_rag_loop_wiring.py
    run triage tests/test_rag_triage_orchestration.py ;;
esac
