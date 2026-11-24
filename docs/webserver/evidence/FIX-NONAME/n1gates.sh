#!/bin/bash
# FIX-NONAME N1 re-checks: new + patch-module tests, replay on the fix tree, SMOKE. Each step under its own timeout.
T=C:/t/fix-noname; R=fixnoname-n1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
PP="$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q"
S=C:/t/vis-runs/scratch/fixnoname
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/$R; mkdir -p $B
cd $T
echo "HEAD $(git rev-parse HEAD) src-diff-vs-HEAD: $(git diff --stat HEAD -- src | tail -1)"
run() { label=$1; shift; echo "== $label"; timeout 900 $PP --basetemp=$B/$label "$@" 2>&1 | tail -4; echo "EXIT_$label=${PIPESTATUS[0]}"; }
run unit tests/test_fix_noname_post_audit.py tests/test_apply_audit_patch_lineage.py tests/test_apply_audit_patch_lock_policy.py tests/test_apply_audit_patch_referential_integrity.py tests/test_curation_patch_transaction.py tests/test_orch769d_audit_patch_rebase.py
echo "== replay"
timeout 900 $PY -I $S/replay.py $T/src $S/out_fix_n1.json $S/ppdirs.txt; echo "EXIT_replay=$?"
$PY -I $S/compare.py $S/out_base.json $S/out_fix_n1.json; $PY -I $S/stats.py $S/out_fix_n1.json
echo "== smoke"
run smoke tests/test_reference_repair.py tests/test_strict_quarantine.py tests/test_strict_quarantine_contract_alignment.py tests/test_strict_quarantine_locks_and_scope.py tests/test_strict_quarantine_versioning.py tests/test_empty_extraction_payload.py tests/test_bench_goldset_and_semantic.py tests/test_bench_acquisition_and_artifacts.py tests/test_bench_controls.py tests/test_completeness_audit.py tests/test_batch_driver.py tests/test_stage3_gate_report_lifecycle.py tests/test_rag_admission_production_path.py tests/test_rag_gap_admission.py tests/test_rag_triage_orchestration.py tests/test_rag_provenance_gates.py tests/test_pipeline_reaction_rag_provenance.py tests/test_research_mode_orchestration.py tests/test_map_ids_name_gate.py tests/test_db_candidate_species_evidence.py tests/test_c102_coverage_denominator.py tests/test_c106_mutation_harness_executable.py
echo "N1_DONE"
