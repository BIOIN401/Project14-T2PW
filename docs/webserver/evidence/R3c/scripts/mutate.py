"""R3c mutation check: each single mutation of the moved code (service/extraction.py), of
the app's call site / shim, or of the reporter must make the named tests fail. Every
pytest run goes through bounded_run + pinned_pytest with the heavy lock. The file is
restored after every mutation (and in a finally); the caller checks ``git status``.

Line endings are normalised to LF to apply a mutation and the file's own bytes are
restored afterwards.

usage: mutate.py [LO:HI]   (run from C:/t/ws-R3c, committed clean tree)
"""
import os
import re
import subprocess
import sys
from pathlib import Path

TREE = "C:/t/ws-R3c"
PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
SVC = Path(TREE, "src/t2pw/service/extraction.py")
APP = Path(TREE, "src/t2pw/app/streamlit_app.py")
REP = Path(TREE, "src/t2pw/service/reporter.py")
EV = Path(TREE, "docs/webserver/evidence/R3c/mutation")
EV.mkdir(parents=True, exist_ok=True)
Path("C:/t/wsR3ctmp/mut-tmp").mkdir(parents=True, exist_ok=True)  # the --basetemp parent must exist
NEW = ["tests/test_service_extraction_stage1.py"]
TOP = "import t2pw.paths as paths  # PROJECT_ROOT read at call time: the harness patches it\n"


def bound(module, name, call_old, call_new):
    """Revert one call-time lookup to an import-time binding at module level."""
    return [TOP, call_old], [TOP + f"from {module} import {name} as _bound\n", call_new]


MUTATIONS = {  # name: (file, old, new, tests)
    # -- patch-target conversions reverted to bound imports (the R3b review F1 rule)
    "bound-run_stage_one_with_chunking": (SVC, *bound(
        "t2pw.pipeline.pipeline", "run_stage_one_with_chunking",
        "stage_one, chunk_details = _pipeline.run_stage_one_with_chunking(", "stage_one, chunk_details = _bound("), NEW),
    "bound-settle_stage_one": (SVC, *bound(
        "t2pw.pipeline.stage_one_boundary", "settle_stage_one",
        "_boundary = _stage_one_boundary.settle_stage_one(", "_boundary = _bound("), NEW),
    "bound-filter_out_of_scope_reactions": (SVC, *bound(
        "t2pw.pipeline.pipeline", "filter_out_of_scope_reactions",
        "= _pipeline.filter_out_of_scope_reactions(stage_one)", "= _bound(stage_one)"), NEW),
    "bound-build_draft_graph": (SVC, *bound(
        "t2pw.pipeline.draft_graph", "build_draft_graph",
        "_rag_seed_graph = _draft_graph.build_draft_graph(", "_rag_seed_graph = _bound("), NEW),
    "bound-generate_qa_report": (SVC, *bound(
        "t2pw.pipeline.qa_graph", "generate_qa_report",
        '{"qa_graph": _qa_graph.generate_qa_report(', '{"qa_graph": _bound('), NEW),
    "bound-maybe_run_rag": (SVC, *bound(
        "t2pw.service.rag_orchestration", "maybe_run_rag",
        "rag_result = _rag_orchestration.maybe_run_rag(", "rag_result = _bound("), NEW),
    "bound-rag_config": (SVC, *bound(
        "t2pw.config", "rag_config", 'if _config.rag_config()["enabled"]:', 'if _bound()["enabled"]:'), NEW),
    "bound-diag-current-cleaning": (SVC, *bound(
        "t2pw.pipeline.extraction_diagnostics", "current",
        "artifacts = _diag.current().artifacts()\n    except Exception:  # noqa: BLE001\n        return {}",
        "artifacts = _bound().artifacts()\n    except Exception:  # noqa: BLE001\n        return {}"), NEW),
    # -- behaviour of the moved code
    "spinner-text": (SVC, 'reporter.spinner("Running Stage 1 extraction...")', 'reporter.spinner("Running Stage 1")', NEW),
    "artifact-dir-not-paths": (SVC, 'artifact_dir=paths.PROJECT_ROOT / "tmp",', 'artifact_dir=Path.cwd() / "tmp",', NEW),
    "chunk-size-not-int": (SVC, "chunk_word_limit=int(chunk_size),", "chunk_word_limit=chunk_size,", NEW),
    "mode-ignores-state": (SVC, 'mode=_export_mode.coerce_mode(state.get("export_mode")),',
                           "mode=_export_mode.coerce_mode(None),", NEW),
    "cleaning-report-dropped": (SVC, "cleaning_report=_last_cleaning_report(),", "cleaning_report={},", NEW),
    "cleaning-report-first-pass": (SVC, "return _safe_dict(passes[-1]) if passes else {}",
                                   "return _safe_dict(passes[0]) if passes else {}", NEW),
    "stage-one-not-settled": (SVC, "            stage_one = _boundary.payload\n", "            pass\n", NEW),
    "boundary-state-dropped": (SVC, '            state["stage_one_boundary"] = _boundary.to_summary()\n',
                               "            pass\n", NEW),
    "failure-sync-dropped": (SVC, "    except _pipeline.PipelineFailure as failure:\n        _sync_extraction_diagnostics(state=state)\n",
                             "    except _pipeline.PipelineFailure as failure:\n", NEW),
    "attempts-not-rendered": (SVC, '        render_attempts("Stage 1 attempts", failure.attempts, reporter=reporter)\n',
                              "", NEW),
    "failure-kind-wrong": (SVC, "failure=InputFailure(KIND_STAGE_ONE_FAILED, message)",
                           "failure=InputFailure(KIND_STAGE_ONE_INCOMPLETE, message)", NEW),
    "contract-json-dropped": (SVC, "        reporter.json(failure.report)\n", "", NEW),
    "contract-not-a-failure": (SVC, "        return StageOneResult(failure=InputFailure(KIND_STAGE_ONE_BOUNDARY_ERROR, message))\n",
                               "        return StageOneResult()\n", NEW),
    "post-try-sync-dropped": (SVC, "    _sync_extraction_diagnostics(state=state)\n    if not _boundary.ok:",
                              "    if not _boundary.ok:", NEW),
    "incomplete-failure-json-dropped": (SVC, "            reporter.json(_boundary.failure.report)\n", "            pass\n", NEW),
    "incomplete-falls-through": (SVC, "        return StageOneResult(failure=InputFailure(KIND_STAGE_ONE_INCOMPLETE, message))\n",
                                 "", NEW),
    "incomplete-info-no-dir": (SVC, "cleaning_report.json) are in {diagnostics_dir}.", "cleaning_report.json) are in tmp.", NEW),
    "reconstruction-info-dropped": (SVC, "    if _boundary.reconstruction:\n", "    if False:\n", NEW),
    "out-of-scope-info-dropped": (SVC, "    if out_of_scope_removed_reactions:\n", "    if False:\n", NEW),
    "seed-from-unfiltered": (SVC, "            seed_payload=stage_one_in_scope,\n", "            seed_payload=stage_one,\n", NEW),
    "user-flag-not-bool": (SVC, "user_flag=bool(rag_incomplete_flag),", "user_flag=rag_incomplete_flag,", NEW),
    "fold-order-swapped": (SVC, "for part in (user_task_context, rag_result.evidence_context)",
                           "for part in (rag_result.evidence_context, user_task_context)", NEW),
    "rag-result-state-dropped": (SVC, '            state["rag_result"] = rag_result\n', "            pass\n", NEW),
    "rag-evidence-state-dropped": (SVC, '                state["rag_evidence_context"] = rag_result.evidence_context  # S2\n',
                                   "                pass\n", NEW),
    "returns-unfolded-focus": (SVC, "        rag_result=rag_result,\n        user_task_context=user_task_context,\n",
                               "        rag_result=rag_result,\n        user_task_context=None,\n", NEW),
    "attempts-code-no-language": (SVC, 'reporter.code(log["raw"], language="json")', 'reporter.code(log["raw"])', NEW),
    "attempts-caption-dropped": (SVC, '                reporter.caption(log["error"])\n', "                pass\n", NEW),
    "state-and-reporter-defaulted": (SVC, "    rag_incomplete_flag: Any,\n    state: MutableMapping[str, Any],\n    reporter: Reporter,\n",
                                     "    rag_incomplete_flag: Any,\n    state: MutableMapping[str, Any] = None,\n    reporter: Reporter = None,\n", NEW),
    # -- the app's call site and shim
    "app-no-stop-on-failure": (APP, "    if _stage_one.failure is not None:\n        st.stop()\n",
                               "    if _stage_one.failure is not None:\n        pass\n", NEW),
    "app-focus-not-rebound": (APP, "    user_task_context = _stage_one.user_task_context\n", "", NEW),
    "app-in-scope-is-raw": (APP, "    stage_one_in_scope = _stage_one.stage_one_in_scope\n",
                            "    stage_one_in_scope = _stage_one.stage_one\n", NEW),
    "app-state-not-session": (APP, "        rag_incomplete_flag=rag_incomplete_flag,\n        state=st.session_state,\n",
                              "        rag_incomplete_flag=rag_incomplete_flag,\n        state={},\n", NEW),
    "app-shim-silent": (APP, "    _service_render_attempts(label, attempts, reporter=StreamlitReporter(st))\n",
                        "    pass\n", NEW),
    # -- reporter primitives
    "reporter-markdown-not-forwarded": (REP, "        return self._st.markdown(body)\n", "        return self._st.text(body)\n", NEW),
    "reporter-code-drops-kwargs": (REP, "        return self._st.code(body, **kwargs)\n", "        return self._st.code(body)\n", NEW),
}

if len(sys.argv) > 1:  # optional subset by index range, e.g. 0:11
    lo, hi = (int(x) for x in sys.argv[1].split(":"))
    MUTATIONS = dict(list(MUTATIONS.items())[lo:hi])
originals = {p: p.read_bytes() for p in (SVC, APP, REP)}
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
               "--timeout", "300", "--heavy-lock", "ws-R3c", "--json", str(EV / f"{name}.json"), "--",
               PY, "-u", f"{TREE}/docs/pwml_recovery_sprint/evidence/pinned_pytest.py", "--expect-tree", TREE,
               "-q", "-p", "no:cacheprovider", f"--basetemp=C:/t/wsR3ctmp/mut-tmp/{name}", *tests]
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
