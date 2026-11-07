#!/bin/bash
# usage: chunks.sh TREE PREFIX ID...    (R3c: runs the named chunks, waitfree before each heavy one; exits 3 if busy)
TREE=$1; PFX=$2; shift 2
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
EV=${EV:-C:/t/wsR3ctmp/ev}
declare -A C L
C[g1]="tests/test_batch_driver_seam_golden.py"; L[g1]=1
C[g2]="tests/test_c011_freeze_seam_golden_equivalence.py"; L[g2]=1
C[g3]="tests/test_streamlit_stage8_export_contract.py"; L[g3]=1
C[g4]="tests/test_streamlit_quarantine_boundary.py"; L[g4]=1
C[2a]="tests/test_c030a_object_sharing_at_the_freeze_seam.py tests/test_c030_canonical_identity_fallback.py tests/test_c052_prefreeze_report_at_the_streamlit_seams.py"; L[2a]=1
C[2b]="tests/test_c060a_requested_pathway_wiring.py tests/test_research_mode_orchestration.py tests/test_streamlit_stage2_orchestration.py tests/test_stage_one_paper_dictionary.py"; L[2b]=1
C[2b-c055]="tests/test_c055_rag_loop_wiring.py"; L[2b-c055]=1
C[2c]="tests/test_c082_species_rename_declination.py tests/test_f179_reaction_support.py tests/test_prefreeze_compound_resolution.py tests/test_prefreeze_third_export_seam.py"; L[2c]=1
C[2c-c126]="tests/test_c126_g9_base_failure_proof.py"; L[2c-c126]=1
C[2d]="tests/test_strict_quarantine.py tests/test_strict_quarantine_real_artifact_replay.py tests/test_stage0_case_c_retry_clobber.py tests/test_rag_seed_entity_reimport.py tests/test_reaction_evidence.py tests/test_rag_admission_production_path.py tests/test_rag_triage_orchestration.py tests/test_rag_differential_gate.py tests/test_stage0_scope_fallback.py tests/test_orch757_llm_token_budgets.py tests/test_orch759a_duplicate_protein_fold.py tests/test_entity_admission.py"; L[2d]=1
C[2d-stage3]="tests/test_stage3_gate_report_lifecycle.py"; L[2d-stage3]=1
C[2d-c119g9]="tests/test_c119_g9_base_failure_proof.py"; L[2d-c119g9]=1
C[2e]="tests/test_preprocessor.py"; L[2e]=1
C[2e-759b]="tests/test_orch759b_recorder_reattach.py"; L[2e-759b]=1
C[2e-hook]="tests/test_reaction_evidence_app_hook.py"; L[2e-hook]=1
C[2e-759afold]="tests/test_orch759a_fold_app_path.py"; L[2e-759afold]=1
C[2f]="tests/test_batch_preflight.py"; L[2f]=1
C[2f-driver]="tests/test_batch_driver.py"; L[2f-driver]=1
C[2f-gate]="tests/test_batch_research_gate_fail_open.py"; L[2f-gate]=1
C[2g]="tests/test_orch759c_quarantine_bucket_closure.py tests/test_orch766_unreferenced_species.py tests/test_orch768_replay_corpus.py tests/test_orch769b_actor_protein_registry.py tests/test_orch773_omit_per_row.py tests/test_orch787_duplicate_rows_consolidated.py tests/test_strict_failure_replay.py tests/test_c083_inner_timeout_row.py"; L[2g]=1
C[2g-c082pp]="tests/test_c082_post_pipeline_seam.py"; L[2g-c082pp]=1
C[2g-c056d]="tests/test_c056d_gate_failure_semantic_carry.py"; L[2g-c056d]=1
C[2g-c119sup]="tests/test_c119_superseded_intermediate_report.py"; L[2g-c119sup]=1
C[2g-759d]="tests/test_orch759d_app_timeout.py"; L[2g-759d]=1
C[2g-revview]="tests/test_reaction_evidence_view.py"; L[2g-revview]=1
C[web]="tests/test_service_driver_bridge.py tests/test_service_pathway_interface.py tests/test_service_inputs.py tests/test_service_organisms.py tests/test_web_jobs.py tests/test_web_worker.py tests/test_web_retry.py tests/test_web_limits.py tests/test_web_notify.py tests/test_web_design_system.py tests/test_web_curator_queue.py tests/test_web_postrun.py tests/test_web_app.py"; L[web]=0
C[ext]="tests/test_service_extraction.py tests/test_service_reporter.py"; L[ext]=1
C[solo-clobber]="tests/test_stage0_case_c_retry_clobber.py"; L[solo-clobber]=1
C[solo-scope]="tests/test_stage0_scope_fallback.py"; L[solo-scope]=1
C[solo-c060a]="tests/test_c060a_requested_pathway_wiring.py"; L[solo-c060a]=1
C[solo-759a]="tests/test_orch759a_duplicate_protein_fold.py"; L[solo-759a]=1
C[new]="tests/test_service_extraction_stage1.py"; L[new]=1
C[svc3]="tests/test_service_extraction.py tests/test_service_extraction_stage0.py tests/test_service_extraction_stage1.py tests/test_service_reporter.py"; L[svc3]=1
C[svc]="tests/test_service_extraction.py tests/test_service_extraction_stage0.py tests/test_service_reporter.py"; L[svc]=1
C[solo-triage]="tests/test_rag_triage_orchestration.py"; L[solo-triage]=1
C[solo-admission]="tests/test_rag_admission_production_path.py"; L[solo-admission]=1
C[solo-ragseed]="tests/test_rag_seed_entity_reimport.py"; L[solo-ragseed]=1
C[solo-s2orch]="tests/test_streamlit_stage2_orchestration.py"; L[solo-s2orch]=1
C[tripwire]="tests/test_service_extraction.py::test_real_app_warns_and_stops_before_stage_zero"; L[tripwire]=1
C[eq]="tests/test_equivalence_app_vs_service.py"; L[eq]=1
C[at-c052]="tests/test_c052_prefreeze_report_at_the_streamlit_seams.py"; L[at-c052]=1
C[at-c082sr]="tests/test_c082_species_rename_declination.py"; L[at-c082sr]=1
C[at-c083]="tests/test_c083_inner_timeout_row.py"; L[at-c083]=1
C[at-pf3]="tests/test_prefreeze_third_export_seam.py"; L[at-pf3]=1
C[at-bridge]="tests/test_service_driver_bridge.py"; L[at-bridge]=1
for id in "$@"; do
  echo "=== $PFX-$id"
  if [ "${L[$id]}" = "1" ]; then $PY C:/t/wsR3ctmp/waitfree.py "$PFX-$id" || { echo "STILL BUSY, stopping at $id"; exit 3; }; fi
  bash C:/t/wsR3ctmp/run.sh "$TREE" "$EV" "$PFX" "$id" 560 "${L[$id]}" ${C[$id]} | grep -E "wrapper exit|passed|failed|error|wrapper vs HEAD|exit reason|FINAL SURV|heavy lock|timed out"
done
