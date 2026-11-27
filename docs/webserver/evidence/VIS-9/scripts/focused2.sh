#!/bin/bash
# VIS-9 after the re-record: the re-recorded pins, every other test that reads the orch768 legs or an equivalence
# fixture, and the PWML tests. usage: focused2.sh RUNID
T=C:/t/vis-pw9; R=vis9-focused2-$1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
PP="$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q"
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/$R; mkdir -p $B
cd $T
echo "HEAD $(git -C $T rev-parse HEAD)"; echo "STATUS_BEGIN"; git -C $T status --porcelain; echo "STATUS_END"
run() { label=$1; shift; echo "== $label"; $PP --basetemp=$B/$label "$@" 2>&1 | tail -8; echo "EXIT_$label=${PIPESTATUS[0]}"; }
run orch768 tests/test_orch768_replay_corpus.py
run orch772 tests/test_orch772_pwml_coverage.py tests/test_orch772b_no_pwml_cause.py tests/test_orch774s_morning_summary.py
run eq2 tests/test_equivalence_second_fixture.py tests/test_canonical_biological_equivalence.py
run pwml $(ls tests/test_pwml_*.py)
git -C $T checkout -- data/enrichment_cache.json data/id_mapping_cache.json 2>/dev/null
echo "FOCUSED2_DONE"
