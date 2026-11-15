#!/bin/bash
# v2.1: SMOKE (TEST_MATRIX command, 22 files) and the golden seam (4 files) at the tree's HEAD, in one bounded job.
T=C:/t/lay; E=$T/docs/pwml_output/evidence/layout/tests/v2_1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
mkdir -p $E C:/t/layx/tmpbase
cd $T
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
unset T2PW_PWML_LAYOUT T2PW_FINAL_PAYLOAD
git -C $T log --oneline -1
$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=C:/t/layx/tmpbase/smoke21-$$ \
  tests/test_reference_repair.py tests/test_strict_quarantine.py tests/test_strict_quarantine_contract_alignment.py \
  tests/test_strict_quarantine_locks_and_scope.py tests/test_strict_quarantine_versioning.py tests/test_empty_extraction_payload.py \
  tests/test_bench_goldset_and_semantic.py tests/test_bench_acquisition_and_artifacts.py tests/test_bench_controls.py \
  tests/test_completeness_audit.py tests/test_batch_driver.py tests/test_stage3_gate_report_lifecycle.py \
  tests/test_rag_admission_production_path.py tests/test_rag_gap_admission.py tests/test_rag_triage_orchestration.py \
  tests/test_rag_provenance_gates.py tests/test_pipeline_reaction_rag_provenance.py tests/test_research_mode_orchestration.py \
  tests/test_map_ids_name_gate.py tests/test_db_candidate_species_evidence.py tests/test_c102_coverage_denominator.py \
  tests/test_c106_mutation_harness_executable.py > $E/smoke.stdout.txt 2>&1
echo "smoke exit=$?"; tail -1 $E/smoke.stdout.txt
$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=C:/t/layx/tmpbase/golden21-$$ \
  tests/test_batch_driver_seam_golden.py tests/test_c011_freeze_seam_golden_equivalence.py \
  tests/test_streamlit_stage8_export_contract.py tests/test_streamlit_quarantine_boundary.py > $E/golden.stdout.txt 2>&1
echo "golden exit=$?"; tail -1 $E/golden.stdout.txt
