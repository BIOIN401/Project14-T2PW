"""R3d mutation check: each single mutation of the moved code (service/extraction.py) or of
the app's call site must make the named tests fail. Every pytest run goes through
bounded_run + pinned_pytest with the heavy lock. The file is restored after every mutation
(and in a finally); the caller checks ``git status``.

usage: mutate.py OUTDIR [LO:HI]   (run against C:/t/ws-R3d, committed clean tree)
"""
import os
import re
import subprocess
import sys
from pathlib import Path

TREE = "C:/t/ws-R3d"
PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
SVC = Path(TREE, "src/t2pw/service/extraction.py")
APP = Path(TREE, "src/t2pw/app/streamlit_app.py")
EV = Path(sys.argv[1])
EV.mkdir(parents=True, exist_ok=True)
Path("C:/t/wsR3dtmp/mut-tmp").mkdir(parents=True, exist_ok=True)  # the --basetemp parent must exist
NEW = ["tests/test_service_extraction_stage2.py"]
EQ = ["tests/test_equivalence_app_vs_service.py"]
TOP = "import t2pw.paths as paths  # PROJECT_ROOT read at call time: the harness patches it\n"


def bound(module, name, call_old, call_new):
    """Revert one call-time lookup to an import-time binding at module level."""
    return [TOP, call_old], [TOP + f"from {module} import {name} as _bound\n", call_new]


MUTATIONS = {  # name: (file, old, new, tests)
    # -- call-time lookups reverted to module-level bound imports (the R3b review F1 rule)
    "bound-run_stage_two_with_feedback_loop": (SVC, *bound(
        "t2pw.pipeline.pipeline", "run_stage_two_with_feedback_loop",
        "stage_two_rounds = _pipeline.run_stage_two_with_feedback_loop(", "stage_two_rounds = _bound("), NEW),
    "bound-merge_additions": (SVC, [TOP, "    from t2pw.pipeline.pipeline import merge_additions\n"],
                              [TOP + "from t2pw.pipeline.pipeline import merge_additions\n", ""], NEW),
    "bound-build_and_save_draft_graph": (SVC, *bound(
        "t2pw.pipeline.pipeline", "build_and_save_draft_graph",
        "reaction_summary = _pipeline.build_and_save_draft_graph(", "reaction_summary = _bound("), NEW),
    "bound-write_reaction_preservation_report": (SVC, *bound(
        "t2pw.service.postpipeline", "_write_reaction_preservation_report",
        "= _postpipeline._write_reaction_preservation_report(", "= _bound("), NEW),
    "bound-run_rag_rounds": (SVC, *bound(
        "t2pw.service.rag_orchestration", "run_rag_rounds",
        "rag_loop_record = _rag_orchestration.run_rag_rounds(", "rag_loop_record = _bound("), NEW),
    "bound-diag-current-publish": (SVC, *bound(
        "t2pw.pipeline.extraction_diagnostics", "current",
        'state["extraction_diagnostics_recorder"] = _diag.current()',
        'state["extraction_diagnostics_recorder"] = _bound()'), NEW),
    # -- behaviour of the moved Stage-2 code
    "spinner-text": (SVC, 'reporter.spinner("Running Stage 2 inference...")', 'reporter.spinner("Stage 2")', NEW),
    "qa-rounds-not-int": (SVC, "qa_rounds=int(infer_rounds),", "qa_rounds=infer_rounds,", NEW),
    "stage-two-raw-payload": (SVC, "                    text,\n                    stage_one_in_scope,\n",
                              "                    text,\n                    stage_one,\n", NEW),
    "stage-two-unfolded-focus": (SVC, "    user_task_context = stage_one_result.user_task_context\n",
                                 "    user_task_context = None\n", NEW),
    "failure-message": (SVC, 'message = f"Inference stage failed: {failure}"', 'message = f"Stage 2 failed: {failure}"', NEW),
    "failure-kind": (SVC, "failure=InputFailure(KIND_STAGE_TWO_FAILED, message)",
                     "failure=InputFailure(KIND_STAGE_ONE_FAILED, message)", NEW),
    "failure-falls-through": (SVC, "            return ExtractionResult(failure=InputFailure(KIND_STAGE_TWO_FAILED, message))\n",
                              "            raise\n", NEW),
    "attempts-label-fallback": (SVC, 'else "Stage 2 attempts"', 'else "attempts"', NEW),
    "attempts-not-rendered": (SVC, "            render_attempts(label, failure.attempts, reporter=reporter)\n", "", NEW),
    "qa-hints-default": (SVC, 'qa_hints = stage_two.get("qa_hints", {}) if isinstance(stage_two, dict) else {}',
                         'qa_hints = stage_two.get("qa_hints", {}) if isinstance(stage_two, dict) else None', NEW),
    "merge-base-raw": (SVC, "        final_payload = merge_additions(\n            stage_one_in_scope,\n",
                       "        final_payload = merge_additions(\n            stage_one,\n", NEW),
    "merge-no-source-text": (SVC, "            source_text=text,\n        )\n", "        )\n", NEW),
    "merge-dict-context": (SVC, "    pathway_context=pathway_context_from_stage_zero(pathway_context),\n            source_text=text,",
                           "    pathway_context=pathway_context,\n            source_text=text,", NEW),
    "rag-merge-no-context": (SVC, "            rag_envelope,\n            pathway_context=pathway_context_from_stage_zero(pathway_context),\n",
                             "            rag_envelope,\n", NEW),
    "rag-strict-db-default": (SVC, 'strict_db=bool(state.get("pwml_strict_db", True))',
                              'strict_db=bool(state.get("pwml_strict_db", False))', NEW),
    "rag-user-flag-not-bool": (SVC, "            user_flag=bool(rag_incomplete_flag),\n            seed_text=text,\n",
                               "            user_flag=rag_incomplete_flag,\n            seed_text=text,\n", NEW),
    "rag-messages-dropped": (SVC, '        emit_rag_loop_messages(rag_loop_record["messages"], reporter=reporter)\n', "", NEW),
    "rag-payload-not-taken": (SVC, '        final_payload = rag_loop_record["payload"]\n', "", NEW),
    "rag-loop-attach-unguarded": (SVC, "        try:\n            rag_result.loop = state[\"rag_loop\"]\n        except Exception:  # noqa: BLE001 - a diagnostic attachment is never fatal\n            pass\n",
                                  "        rag_result.loop = state[\"rag_loop\"]\n", NEW),
    "session-record-keeps-payload": (SVC, 'if key not in {"payload", "messages"}', 'if key not in {"messages"}', NEW),
    "emit-json-dropped": (SVC, '        elif kind == "json":\n            reporter.json(body)\n', "", NEW),
    "publish-order-swapped": (SVC, '    state["qa_report"] = qa_report\n    state["reaction_summary"] = reaction_summary\n',
                              '    state["reaction_summary"] = reaction_summary\n    state["qa_report"] = qa_report\n', NEW),
    "publish-pop-dropped": (SVC, '    state.pop("post_pipeline_artifacts", None)\n\n', "\n", NEW),
    "publish-inference-flag-raw": (SVC, 'state["run_inference_enabled"] = bool(run_inference)',
                                   'state["run_inference_enabled"] = run_inference', NEW),
    "publish-raw-stage-one-as-final": (SVC, 'state["final_payload_snapshot"] = final_payload',
                                       'state["final_payload_snapshot"] = stage_one', NEW),
    "preservation-stage-name": (SVC, '_write_reaction_preservation_report("after_stage2", final_payload)',
                                '_write_reaction_preservation_report("after_stage_2", final_payload)', NEW),
    "defaults-on-state-reporter": (SVC, "    export_mode: Any,\n    state: MutableMapping[str, Any],\n    reporter: Reporter,\n) -> ExtractionResult:\n    \"\"\"Stage 2",
                                   "    export_mode: Any,\n    state: MutableMapping[str, Any] = None,\n    reporter: Reporter = None,\n) -> ExtractionResult:\n    \"\"\"Stage 2", NEW),
    # -- run_extraction
    "extraction-focus-not-stripped": (SVC, '    user_task_context = (request.user_task_context or "").strip() or None\n',
                                      "    user_task_context = request.user_task_context\n", NEW),
    "extraction-mode-not-in-state": (SVC, '    state["export_mode"] = request.export_mode\n', "", NEW),
    "extraction-stage-one-failure-ignored": (SVC, "    if stage_one.failure is not None:\n        return ExtractionResult(failure=stage_one.failure)\n",
                                             "", NEW),
    "extraction-input-failure-ignored": (SVC, "    if assembled.failure is not None:\n        return ExtractionResult(failure=assembled.failure)\n",
                                         "", NEW),
    # -- the app's call site (real-app AppTest tests)
    "app-no-stop-on-failure": (APP, "    if _stage_two.failure is not None:\n        st.stop()\n",
                               "    if _stage_two.failure is not None:\n        pass\n", NEW),
    "app-reset-dropped": (APP, '    reset_quarantine_state(PROJECT_ROOT / "outputs")\n    st.session_state["token_stats"]',
                          '    st.session_state["token_stats"]', NEW),
    "app-token-stats-dropped": (APP, '    st.session_state["token_stats"] = llm_client_module.get_token_stats()\n\nif st.session_state.get("pipeline_ready"):',
                                '\nif st.session_state.get("pipeline_ready"):', NEW),
    "app-state-not-session": (APP, "        export_mode=export_mode,\n        state=st.session_state,\n",
                              "        export_mode=export_mode,\n        state={},\n", NEW),
    # -- the equivalence test is non-vacuous for the new service arm
    "eq-extraction-one-qa-round": (SVC, "        infer_rounds=request.infer_rounds,\n", "        infer_rounds=1,\n", EQ),
    "eq-extraction-no-inference": (SVC, "        run_inference=request.run_inference,\n", "        run_inference=False,\n", EQ),
    "eq-merge-no-source-text": (SVC, "            source_text=text,\n        )\n", "        )\n", EQ),
}

if len(sys.argv) > 2:  # optional subset by index range, e.g. 0:11
    lo, hi = (int(x) for x in sys.argv[2].split(":"))
    MUTATIONS = dict(list(MUTATIONS.items())[lo:hi])
originals = {p: p.read_bytes() for p in (SVC, APP)}
results = {}
try:
    for name, (target, old, new, tests) in MUTATIONS.items():
        raw = originals[target].decode("utf-8")
        crlf = "\r\n" in raw
        src = raw.replace("\r\n", "\n")
        for o, n in (zip(old, new) if isinstance(old, list) else [(old, new)]):
            assert src.count(o) == 1, (name, o[:60], src.count(o))
            src = src.replace(o, n)
        target.write_bytes((src.replace("\n", "\r\n") if crlf else src).encode("utf-8"))
        env = dict(os.environ, PYTHONPATH=f"{TREE}/src", T2PW_OFFLINE_CURATOR="1")
        cmd = [PY, f"{TREE}/docs/pwml_recovery_sprint/evidence/bounded_run.py", "--label", f"mut-{name}",
               "--timeout", "300", "--heavy-lock", "R3d", "--json", str(EV / f"{name}.json"), "--",
               PY, "-u", f"{TREE}/docs/pwml_recovery_sprint/evidence/pinned_pytest.py", "--expect-tree", TREE,
               "-q", "-p", "no:cacheprovider", f"--basetemp=C:/t/wsR3dtmp/mut-tmp/{name}", *tests]
        try:
            out = subprocess.run(cmd, cwd=TREE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, timeout=400).stdout
        finally:
            target.write_bytes(originals[target])
        (EV / f"{name}.log").write_text(out, encoding="utf-8")
        summary = re.findall(r"^(?:\d+ (?:failed|passed)|ERROR|=+ .*error).*$", out, re.M)
        surv = re.findall(r"FINAL SURVIVING COUNT\s*:\s*(\d+)", out)
        results[name] = (summary[-1] if summary else "?", surv[-1] if surv else "?")
        line = f"{name:40s} {results[name][0][:70]}  survivors={results[name][1]}"
        print(line, flush=True)
        with open(EV / "summary.txt", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
finally:
    for p, b in originals.items():
        p.write_bytes(b)
killed = [n for n, (s, _) in results.items() if "failed" in s or "error" in s.lower()]
tail = f"killed {len(killed)}/{len(MUTATIONS)}; survived: {sorted(set(MUTATIONS) - set(killed))}"
print(tail)
with open(EV / "summary.txt", "a", encoding="utf-8") as fh:
    fh.write(tail + "\n")
sys.exit(0 if len(killed) == len(MUTATIONS) else 1)
