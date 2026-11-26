#!/bin/bash
# VIS-3 focused tests: the renderer, and the tests that run the service tail that writes pathway.png.
T=C:/t/vis-pw3; R=vis3-focused-$1
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
PP="$PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q"
export PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1
B=C:/t/vis-runs/bt/$R; mkdir -p $B
cd $T
echo "HEAD $(git -C $T rev-parse HEAD)"
run() { label=$1; shift; echo "== $label"; $PP --basetemp=$B/$label "$@" 2>&1 | tail -4; echo "EXIT_$label=${PIPESTATUS[0]}"; }
run render tests/test_render_pathbank.py
run service tests/test_service_pathway_interface.py tests/test_service_run_pathway.py tests/test_service_driver_bridge.py tests/test_batch_driver_run_pathway.py tests/test_web_jobs.py tests/test_web_worker.py tests/test_web_postrun.py tests/test_web_retry.py
echo "FOCUSED_DONE"
