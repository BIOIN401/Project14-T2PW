#!/bin/bash
# One bounded job (heavy lock 'side'): focused, SMOKE, golden seam at the tree's HEAD; then the new test against the OLD writer
# (07d23e9c, exported to C:/t/sidex/base with: git archive 07d23e9c src reference tests/fixtures/pwml_layout).
T=C:/t/side; E=$T/docs/pwml_output/evidence/side_fix/tests
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
PP=$T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py
mkdir -p $E C:/t/sidex/tmpbase
cd $T
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
unset T2PW_PWML_LAYOUT T2PW_FINAL_PAYLOAD
git -C $T log --oneline -1
$PY $PP --expect-tree $T -q --basetemp=C:/t/sidex/tmpbase/focused-$$ tests/test_pwml*.py tests/test_final_payload_writer_fixes.py > $E/focused.stdout.txt 2>&1
echo "focused exit=$?"; tail -1 $E/focused.stdout.txt
$PY $PP --expect-tree $T -q --basetemp=C:/t/sidex/tmpbase/smoke-$$ \
  tests/test_reference_repair.py tests/test_strict_quarantine.py tests/test_strict_quarantine_contract_alignment.py \
  tests/test_strict_quarantine_locks_and_scope.py tests/test_strict_quarantine_versioning.py tests/test_empty_extraction_payload.py \
  tests/test_bench_goldset_and_semantic.py tests/test_bench_acquisition_and_artifacts.py tests/test_bench_controls.py \
  tests/test_completeness_audit.py tests/test_batch_driver.py tests/test_stage3_gate_report_lifecycle.py \
  tests/test_rag_admission_production_path.py tests/test_rag_gap_admission.py tests/test_rag_triage_orchestration.py \
  tests/test_rag_provenance_gates.py tests/test_pipeline_reaction_rag_provenance.py tests/test_research_mode_orchestration.py \
  tests/test_map_ids_name_gate.py tests/test_db_candidate_species_evidence.py tests/test_c102_coverage_denominator.py \
  tests/test_c106_mutation_harness_executable.py > $E/smoke.stdout.txt 2>&1
echo "smoke exit=$?"; tail -1 $E/smoke.stdout.txt
$PY $PP --expect-tree $T -q --basetemp=C:/t/sidex/tmpbase/golden-$$ \
  tests/test_batch_driver_seam_golden.py tests/test_c011_freeze_seam_golden_equivalence.py \
  tests/test_streamlit_stage8_export_contract.py tests/test_streamlit_quarantine_boundary.py > $E/golden.stdout.txt 2>&1
echo "golden exit=$?"; tail -1 $E/golden.stdout.txt
# the new test against the old writer: must FAIL on the side check (not at collection)
cd C:/t/sidex/base
export PYTHONPATH=C:/t/sidex/base/src
$PY -m pytest -q -p no:cacheprovider --basetemp=C:/t/sidex/tmpbase/old-$$ tests/test_pwml_member_side.py > $E/new_test_on_old_writer.stdout.txt 2>&1
echo "old-writer exit=$? (expect 1)"; tail -3 $E/new_test_on_old_writer.stdout.txt
