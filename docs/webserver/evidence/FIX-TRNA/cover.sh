#!/bin/bash
# FIX-TRNA gate: new tests + every test file importing audit_json_llm. AppTest files not included (none import it).
T=C:/t/fix-trna
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
PP="$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q"
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/fixtrna-cover; mkdir -p $B
cd $T
echo "HEAD $(git -C $T rev-parse HEAD)"
run() { label=$1; shift; echo "== $label"; timeout 1200 $PP --basetemp=$B/$label "$@" 2>&1 | tail -6; echo "EXIT_$label=${PIPESTATUS[0]}"; }
run new tests/test_fix_trna_audit_na_declared.py
run importers tests/test_audit_json_llm_payload.py tests/test_apply_audit_patch_referential_integrity.py tests/test_c075_source_support_armed.py tests/test_c079_interactive_prompt_allowlist.py tests/test_curation_patch_transaction.py tests/test_orch757_llm_token_budgets.py tests/test_orch758b_empty_chunk.py tests/test_orch762_protein_identity.py tests/test_rag_ingest.py tests/test_rag_provenance_gates.py tests/test_rag_retrieve.py tests/test_rag_synthesize.py tests/test_rag_triage_orchestration.py tests/test_fix_noname_post_audit.py
echo "COVER_DONE"
