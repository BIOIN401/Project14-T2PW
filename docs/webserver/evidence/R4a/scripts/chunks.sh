#!/bin/bash
# usage: chunks.sh TREE PREFIX ID...
# Runs the named chunks through run.sh (bounded_run + pinned_pytest + heavy lock). A chunk whose log
# already has a cleanup report is skipped, so the same list can be re-invoked. Stops starting new
# chunks after 100 s, and on a held lock, so each call stays short; re-invoke to continue.
TREE=$1; PFX=$2; shift 2
EV=${EV:-C:/t/wsR4atmp/ev}
SMOKE="tests/test_reference_repair.py tests/test_strict_quarantine.py tests/test_strict_quarantine_contract_alignment.py tests/test_strict_quarantine_locks_and_scope.py tests/test_strict_quarantine_versioning.py tests/test_empty_extraction_payload.py tests/test_bench_goldset_and_semantic.py tests/test_bench_acquisition_and_artifacts.py tests/test_bench_controls.py tests/test_completeness_audit.py tests/test_batch_driver.py tests/test_stage3_gate_report_lifecycle.py tests/test_rag_admission_production_path.py tests/test_rag_gap_admission.py tests/test_rag_triage_orchestration.py tests/test_rag_provenance_gates.py tests/test_pipeline_reaction_rag_provenance.py tests/test_research_mode_orchestration.py tests/test_map_ids_name_gate.py tests/test_db_candidate_species_evidence.py tests/test_c102_coverage_denominator.py tests/test_c106_mutation_harness_executable.py"
declare -A C
C[g1]="tests/test_batch_driver_seam_golden.py"
C[g2]="tests/test_c011_freeze_seam_golden_equivalence.py"
C[g3]="tests/test_streamlit_stage8_export_contract.py"
C[g4]="tests/test_streamlit_quarantine_boundary.py"
C[eq]="tests/test_equivalence_app_vs_service.py"
C[new]="tests/test_service_export.py"
C[2a]="tests/test_c030a_object_sharing_at_the_freeze_seam.py tests/test_c030_canonical_identity_fallback.py tests/test_c052_prefreeze_report_at_the_streamlit_seams.py"
C[2b]="tests/test_c060a_requested_pathway_wiring.py tests/test_research_mode_orchestration.py tests/test_streamlit_stage2_orchestration.py tests/test_stage_one_paper_dictionary.py"
C[2b-c055]="tests/test_c055_rag_loop_wiring.py"
C[2c]="tests/test_c082_species_rename_declination.py tests/test_f179_reaction_support.py tests/test_prefreeze_compound_resolution.py tests/test_prefreeze_third_export_seam.py"
C[2c-c126]="tests/test_c126_g9_base_failure_proof.py"
C[2d]="tests/test_strict_quarantine.py tests/test_strict_quarantine_real_artifact_replay.py tests/test_stage0_case_c_retry_clobber.py tests/test_rag_seed_entity_reimport.py tests/test_reaction_evidence.py tests/test_rag_admission_production_path.py tests/test_rag_triage_orchestration.py tests/test_rag_differential_gate.py tests/test_stage0_scope_fallback.py tests/test_orch757_llm_token_budgets.py tests/test_orch759a_duplicate_protein_fold.py tests/test_entity_admission.py"
C[2d-stage3]="tests/test_stage3_gate_report_lifecycle.py"
C[2d-c119g9]="tests/test_c119_g9_base_failure_proof.py"
C[2e]="tests/test_preprocessor.py"
C[2e-759b]="tests/test_orch759b_recorder_reattach.py"
C[2e-hook]="tests/test_reaction_evidence_app_hook.py"
C[2e-759afold]="tests/test_orch759a_fold_app_path.py"
C[2f]="tests/test_batch_preflight.py"
C[2f-driver]="tests/test_batch_driver.py"
C[2f-gate]="tests/test_batch_research_gate_fail_open.py"
C[2g]="tests/test_orch759c_quarantine_bucket_closure.py tests/test_orch766_unreferenced_species.py tests/test_orch768_replay_corpus.py tests/test_orch769b_actor_protein_registry.py tests/test_orch773_omit_per_row.py tests/test_orch787_duplicate_rows_consolidated.py tests/test_strict_failure_replay.py tests/test_c083_inner_timeout_row.py"
C[2g-c082pp]="tests/test_c082_post_pipeline_seam.py"
C[2g-c056d]="tests/test_c056d_gate_failure_semantic_carry.py"
C[2g-c119sup]="tests/test_c119_superseded_intermediate_report.py"
C[2g-759d]="tests/test_orch759d_app_timeout.py"
C[2g-revview]="tests/test_reaction_evidence_view.py"
C[at-c052]="tests/test_c052_prefreeze_report_at_the_streamlit_seams.py"
C[at-c082sr]="tests/test_c082_species_rename_declination.py"
C[at-c083]="tests/test_c083_inner_timeout_row.py"
C[at-pf3]="tests/test_prefreeze_third_export_seam.py"
C[at-bridge]="tests/test_service_driver_bridge.py"
C[web]="tests/test_service_driver_bridge.py tests/test_service_pathway_interface.py tests/test_service_inputs.py tests/test_service_organisms.py tests/test_web_jobs.py tests/test_web_worker.py tests/test_web_retry.py tests/test_web_limits.py tests/test_web_notify.py tests/test_web_design_system.py tests/test_web_curator_queue.py tests/test_web_postrun.py tests/test_web_app.py"
C[svc]="tests/test_service_extraction.py tests/test_service_extraction_stage0.py tests/test_service_reporter.py"
C[stage1]="tests/test_service_extraction_stage1.py"
C[stage2]="tests/test_service_extraction_stage2.py"
C[solo-c060a]="tests/test_c060a_requested_pathway_wiring.py"
C[solo-s2orch]="tests/test_streamlit_stage2_orchestration.py"
C[solo-ragseed]="tests/test_rag_seed_entity_reimport.py"
C[scans]="tests/test_curator_offline_mode.py tests/test_dependency_declarations.py tests/test_identity_evidence_hydration.py tests/test_c056c_semantic_evaluability.py tests/test_eval_v2_metrics.py"
C[smoke]="$SMOKE"
START=$SECONDS
for id in "$@"; do
  if [ -f "$EV/$PFX-$id.log" ] && grep -q "FINAL SURVIVING COUNT" "$EV/$PFX-$id.log"; then continue; fi
  if [ $((SECONDS-START)) -gt 100 ]; then echo "PAUSED before $id; re-invoke"; exit 0; fi
  echo "=== $PFX-$id"
  OUT=$(bash C:/t/wsR4atmp/run.sh "$TREE" "$EV" "$PFX-$id" 400 ${C[$id]})
  echo "$OUT" | grep -E "wrapper exit|^[0-9]+ (passed|failed)|FINAL SURV|timed out|STILL BUSY" | cut -c1-160
  if echo "$OUT" | grep -q "STILL BUSY"; then rm -f "$EV/$PFX-$id.log"; echo "LOCK BUSY at $id; re-invoke"; exit 3; fi
done
echo "ALL DONE ($PFX)"
