#!/bin/bash
# The 3 failing covering tests, one per process, short tracebacks. usage: three.sh RUNID
T=C:/t/fix-noname; R=$1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
PP="$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q --tb=line"
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/$R; mkdir -p $B
cd $T
echo "HEAD $(git rev-parse HEAD) src-diff-vs-HEAD: $(git diff --stat HEAD -- src | tail -1)"
run() { label=$1; shift; echo "== $label"; timeout 600 $PP --basetemp=$B/$label "$@" 2>&1 | grep -E "^E |Error|assert|passed|failed" | head -8; echo "EXIT_$label=${PIPESTATUS[0]}"; }
run c075 "tests/test_c075_source_support_armed.py::test_corpus_the_armed_pass_over_every_committed_artifact_and_its_own_paper"
run orch762 "tests/test_orch762_protein_identity.py::test_new_the_flag_never_reaches_an_llm_input"
run a7 "tests/test_curator_offline_mode.py::test_a7_flag_is_read_in_exactly_one_module"
echo "THREE_DONE"
