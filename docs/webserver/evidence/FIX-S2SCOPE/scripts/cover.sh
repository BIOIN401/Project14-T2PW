#!/bin/bash
# FIX-S2SCOPE: every test file importing service/extraction.py or batch/driver.py (plus the C-060a AST test),
# minus the files suite2 runs (SMOKE, golden-seam, equivalence). AppTest files one per process, 1200 s each.
T=C:/t/fix-s2scope; R=s2scope-cover
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
PP="$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q"
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/$R; mkdir -p $B
cd $T
echo "HEAD $(git -C $T rev-parse HEAD)"; echo "STATUS_BEGIN"; git status --porcelain; echo "STATUS_END"
run() { label=$1; shift; echo "== $label"; timeout 1200 $PP --basetemp=$B/$label "$@" 2>&1 | tail -4; echo "EXIT_$label=${PIPESTATUS[0]}"; }
run plain tests/test_fix_s2scope.py tests/test_c060a_requested_pathway_wiring.py tests/test_final_payload_default_seam.py tests/test_reaction_verifier.py tests/test_service_run_pathway.py tests/test_stage0_case_c_retry_clobber.py tests/test_transport_sides.py tests/test_batch_driver_quarantine_artifacts.py tests/test_batch_pwml_artifact_naming.py tests/test_batch_run.py tests/test_c056d_gate_failure_semantic_carry.py tests/test_c077_stage0_conflict_disposition.py tests/test_c087_prefreeze_declination_demotes_release_status.py tests/test_c088_extracted_not_serialized_disposition.py tests/test_c110_negative_control_status.py tests/test_c111_timeout_observability.py tests/test_c119_g9_base_failure_proof.py tests/test_c119_superseded_intermediate_report.py tests/test_c125_scope_specialization.py tests/test_c126_final_contract_authority.py tests/test_c126_g9_base_failure_proof.py tests/test_deadline_budget.py tests/test_deadline_leg_timeout.py tests/test_release_status_classification.py tests/test_service_pathway_interface.py
for f in test_batch_driver_run_pathway test_service_extraction test_service_extraction_stage0 test_service_extraction_stage1 test_service_extraction_stage2 test_service_post_run test_batch_preflight test_batch_research_gate_fail_open test_c052_prefreeze_report_at_the_streamlit_seams test_c082_post_pipeline_seam test_c083_inner_timeout_row test_orch759a_fold_app_path test_orch759b_recorder_reattach test_orch759d_app_timeout test_service_driver_bridge; do
  run $f tests/$f.py
done
echo "COVER_DONE"
