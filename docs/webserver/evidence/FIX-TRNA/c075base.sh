#!/bin/bash
# FIX-TRNA: does test_c075 corpus test fail without the change? (src change stashed, restored after)
T=C:/t/fix-trna
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
PP="$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q"
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/fixtrna-c075; mkdir -p $B
cd $T
git stash push -q -- src/t2pw/curation/audit_json_llm.py
git status --porcelain
timeout 900 $PP --basetemp=$B/base tests/test_c075_source_support_armed.py 2>&1 | tail -8
echo "EXIT_base=${PIPESTATUS[0]}"
git stash pop -q
git status --porcelain
