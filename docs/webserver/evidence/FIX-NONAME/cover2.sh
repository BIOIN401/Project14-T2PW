#!/bin/bash
# FIX-NONAME covering tests, about 8 files per pytest process, each under its own timeout.
# usage: cover2.sh TREE RUNID
T=$1; R=$2
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
PP="$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q"
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/$R; mkdir -p $B
cd $T
echo "HEAD $(git -C $T rev-parse HEAD)"
run() { label=$1; shift; echo "== $label $*"; timeout 600 $PP --basetemp=$B/$label "$@" 2>&1 | tail -3; echo "EXIT_$label=${PIPESTATUS[0]}"; }
run g1 tests/test_apply_audit_patch_lineage.py tests/test_apply_audit_patch_lock_policy.py tests/test_apply_audit_patch_referential_integrity.py tests/test_audit_json_llm_payload.py tests/test_curation_patch_transaction.py tests/test_orch769d_audit_patch_rebase.py tests/test_fix_noname_post_audit.py tests/test_c105_actor_role_evidence.py
run g2 tests/test_c107_actor_cue_calibration.py tests/test_c108_f155_class.py tests/test_c120_identity.py tests/test_c124_species_creation.py tests/test_c099_wrapper_species_preservation.py tests/test_c075_source_support_armed.py tests/test_c079_interactive_prompt_allowlist.py tests/test_c055_rag_loop_wiring.py
run g3 tests/test_gap_resolver.py tests/test_gap_resolver_agent_tools.py tests/test_gap_resolver_biological_state_resolution.py tests/test_gap_resolver_lineage.py tests/test_gap_resolver_stage3_issues.py tests/test_db_resolver_primitives.py tests/test_rag_typed_gap_resolution.py tests/test_orch762_protein_identity.py
run g4 tests/test_final_payload_default_seam.py tests/test_final_payload_fill.py tests/test_final_payload_meta.py tests/test_orch757_llm_token_budgets.py tests/test_orch758b_empty_chunk.py tests/test_orch759a_duplicate_protein_fold.py tests/test_orch759b_recorder_reattach.py tests/test_extraction_diagnostics.py
run g5 tests/test_rag_ingest.py tests/test_rag_provenance_gates.py tests/test_rag_retrieve.py tests/test_rag_synthesize.py tests/test_rag_triage_orchestration.py tests/test_adversarial_diagnostics_isolation.py tests/test_stage_one_boundary.py
run g6 tests/test_service_extraction_stage1.py tests/test_service_extraction_stage2.py tests/test_service_post_run.py tests/test_curator_offline_mode.py tests/test_interactive_curator.py
run c052 tests/test_c052_prefreeze_report_at_the_streamlit_seams.py
run fold-app tests/test_orch759a_fold_app_path.py
run st-stage2 tests/test_streamlit_stage2_orchestration.py
echo "COVER_DONE"
