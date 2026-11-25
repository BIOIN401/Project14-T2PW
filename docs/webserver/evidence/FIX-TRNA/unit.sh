#!/bin/bash
T=C:/t/fix-trna
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/fixtrna-unit; mkdir -p $B
cd $T
echo "HEAD $(git rev-parse HEAD)"; git status --porcelain tests src
timeout 600 $PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --basetemp=$B/new tests/test_fix_trna_audit_na_declared.py tests/test_audit_json_llm_payload.py 2>&1 | tail -8
echo "EXIT=${PIPESTATUS[0]}"
