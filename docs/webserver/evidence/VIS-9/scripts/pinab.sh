#!/bin/bash
# VIS-9: the two byte-pin tests that failed at HEAD, re-run with the VIS-9 fallback switched off (plugin vis9_off).
T=C:/t/vis-pw9; R=vis9-pinab-$1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
PP="$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q"
export PYTHONPATH="$T/src;C:/t/vis-pw9-work/plug" T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/$R; mkdir -p $B
cd $T
echo "HEAD $(git -C $T rev-parse HEAD)"; echo "STATUS_BEGIN"; git -C $T status --porcelain; echo "STATUS_END"
run() { label=$1; shift; echo "== $label"; $PP --basetemp=$B/$label "$@" 2>&1 | tail -6; echo "EXIT_$label=${PIPESTATUS[0]}"; }
run off-corpus -p vis9_off tests/test_orch768_replay_corpus.py
run off-eqapp -p vis9_off tests/test_equivalence_app_vs_service.py
run on-corpus tests/test_orch768_replay_corpus.py
git -C $T checkout -- data/enrichment_cache.json data/id_mapping_cache.json 2>/dev/null
echo "PINAB_DONE"
