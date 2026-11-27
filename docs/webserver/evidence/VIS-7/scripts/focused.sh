#!/bin/bash
# VIS-7 focused gate. usage: focused.sh RUNID
T=C:/t/vis-pw7; R=vis7-focused-$1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
PP="$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q"
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/$R; mkdir -p $B
cd $T
echo "HEAD $(git -C $T rev-parse HEAD)"; echo "STATUS_BEGIN"; git -C $T status --porcelain; echo "STATUS_END"
run() { label=$1; shift; echo "== $label"; $PP --basetemp=$B/$label "$@" 2>&1 | tail -25; echo "EXIT_$label=${PIPESTATUS[0]}"; }
run pwml $(ls tests/test_pwml_*.py)
run render $(ls tests/test_render_*.py)
run fpw tests/test_final_payload_writer_fixes.py
run corpus tests/test_orch768_replay_corpus.py
git -C $T checkout -- data/enrichment_cache.json data/id_mapping_cache.json 2>/dev/null
echo "FOCUSED_DONE"
