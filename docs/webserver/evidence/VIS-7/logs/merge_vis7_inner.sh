#!/bin/bash
# Gates on webserver/refactor fd07d5e6 + vis/layout-polish 875a0ad2 (C:/t/merge-vis7). Lane flags unset.
unset T2PW_REACTION_VERIFIER T2PW_INTERACTION_TYPING T2PW_VERIFIER_GRANULARITY T2PW_VD18_SPAN_TEMPLATES T2PW_KEEP_RAW_LLM T2PW_AUDIT_NA_DECLARED T2PW_STAGE2_RESPECT_SCOPE T2PW_RESULT_RENDER
T=C:/t/merge-vis7
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
B=C:/t/vis-runs/bt/merge-vis7; mkdir -p $B
echo "HEAD $(git -C $T rev-parse HEAD)"; echo STATUS_BEGIN; git -C $T status --porcelain; echo STATUS_END
"C:/Program Files/Git/bin/bash.exe" C:/t/vis-runs/suite2.sh $T merge-vis7
cd $T
run() { label=$1; shift; echo "== $label"; timeout 1200 env PYTHONPATH=$T/src T2PW_OFFLINE_CURATOR=1 $PY $T/docs/pwml_recovery_sprint/evidence/pinned_pytest.py --expect-tree $T -q -rf --basetemp=$B/$label "$@" 2>&1 | grep -E "^FAILED|passed|failed|error"; echo "EXIT_$label=${PIPESTATUS[0]}"; }
run pwml $(ls tests/test_pwml_*.py) tests/test_final_payload_writer_fixes.py tests/test_orch768_replay_corpus.py
run render $(ls tests/test_render_*.py)
run service tests/test_service_run_pathway.py tests/test_service_result_render.py $(ls tests/test_web_*.py 2>/dev/null)
echo MERGE_VIS7_DONE
