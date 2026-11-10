#!/bin/bash
# sequential runs, each retried while the shared heavy lock is held (exit 95)
one() {
  local label=$1; shift
  for try in $(seq 1 40); do
    bash C:/t/ev-RT1/run.sh "$label" "$@" > C:/t/ev-RT1/$label.summary 2>&1
    if grep -q "$label exit=95" C:/t/ev-RT1/$label.summary; then sleep 30; continue; fi
    break
  done
  cat C:/t/ev-RT1/$label.summary
}
one g1 tests/test_batch_driver_seam_golden.py
one g2 tests/test_c011_freeze_seam_golden_equivalence.py
one g3 tests/test_streamlit_stage8_export_contract.py
one g4 tests/test_streamlit_quarantine_boundary.py
one q1 tests/test_equivalence_app_vs_service.py
one q2 tests/test_equivalence_second_fixture.py
one q3 tests/test_equivalence_run_pathway.py
one n1 tests/test_llm_call_cap.py tests/test_llm_call_logging.py
one n2 tests/test_service_run_pathway.py
one n3 tests/test_batch_driver_run_pathway.py
one n4 tests/test_batch_preflight.py tests/test_c111_timeout_observability.py tests/test_llm_client_request_timeout.py
one n5 tests/test_llm_client_empty_completion_retry.py
one sm $(cat C:/t/ev-R5/scripts/smoke.lst)
echo ALLDONE
