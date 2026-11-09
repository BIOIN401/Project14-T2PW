"""R4b mutation check on a SCRATCH COPY of the tip (git archive, no .git): each single mutation
of service/post_run.py or of the app's call site must make the named tests fail. Every
pytest run goes through bounded_run + pinned_pytest with the heavy lock; the file is restored
after every mutation and in a finally.

usage: mutate.py SCRATCH_TREE OUTDIR [LO:HI]
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TREE = sys.argv[1].replace("\\", "/")
EV = Path(sys.argv[2])
PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
SVC = Path(TREE, "src/t2pw/service/post_run.py")
APP = Path(TREE, "src/t2pw/app/streamlit_app.py")
EV.mkdir(parents=True, exist_ok=True)
Path("C:/t/wsR4btmp/mut-tmp").mkdir(parents=True, exist_ok=True)
NEW = ["tests/test_service_post_run.py"]
EQ = ["tests/test_equivalence_app_vs_service.py"]
TOP = "from t2pw.service.reporter import Reporter\n"


def bound(module, name, call_old, call_new):
    """Revert one call-time lookup to an import-time binding at module level."""
    return [TOP, call_old], [TOP + f"from {module} import {name} as _bound\n", call_new]


MUTATIONS = {
    # -- every call-time lookup reverted to a module-level bound import
    "bound-run_post_pipeline_sbml_artifacts": (SVC, *bound(
        "t2pw.service.postpipeline", "run_post_pipeline_sbml_artifacts",
        "artifacts = _postpipeline.run_post_pipeline_sbml_artifacts(", "artifacts = _bound("), NEW),
    "bound-record_mapped_failure_snapshot": (SVC, *bound(
        "t2pw.service.postpipeline", "_record_mapped_failure_snapshot",
        "_postpipeline._record_mapped_failure_snapshot(", "_bound("), NEW),
    "bound-propagate_context_organism": (SVC, *bound(
        "t2pw.pipeline.pipeline", "propagate_context_organism",
        "final_payload = _pipeline.propagate_context_organism(", "final_payload = _bound("), NEW),
    "bound-reenter": (SVC, *bound(
        "t2pw.pipeline.extraction_diagnostics", "reenter", "_diag.reenter(", "_bound("), NEW),
    "bound-sync": (SVC, *bound(
        "t2pw.service.extraction", "_sync_extraction_diagnostics",
        "_extraction._sync_extraction_diagnostics(state=state)", "_bound(state=state)"), NEW),
    "bound-rag_config": (SVC, *bound("t2pw.config", "rag_config", "_config.rag_config()", "_bound()"), NEW),
    "bound-coerce_mode-call": (SVC, *bound(
        "t2pw.pipeline.export_mode", "coerce_mode",
        'export_mode=_export_mode.coerce_mode(state.get("export_mode"))',
        'export_mode=_bound(state.get("export_mode"))'), NEW),
    "bound-is_research": (SVC, *bound(
        "t2pw.pipeline.export_mode", "is_research",
        "_export_mode.is_research(", "_bound("), NEW),
    "bound-project_root": (SVC, [TOP, "outputs_dir=paths.PROJECT_ROOT"],
                           [TOP + "from t2pw.paths import PROJECT_ROOT as _bound\n", "outputs_dir=_bound"], NEW),
    # -- behaviour of the moved code
    "spinner-text": (SVC, 'reporter.spinner("Running audit and DB mapping...")', 'reporter.spinner("Running audit")', NEW),
    "reenter-into-try": (SVC, '    _diag.reenter(state.get("extraction_diagnostics_recorder"))\n    try:\n',
                         '    try:\n        _diag.reenter(state.get("extraction_diagnostics_recorder"))\n', NEW),
    "reenter-wrong-key": (SVC, 'state.get("extraction_diagnostics_recorder")', 'state.get("extraction_recorder")', NEW),
    "legacy-true": (SVC, "build_legacy_sbml=False", "build_legacy_sbml=True", NEW),
    "compartment-default": (SVC, '(request.default_compartment or "cell").strip() or "cell"',
                            '(request.default_compartment or "cell").strip() or "cytosol"', NEW),
    "compartment-no-strip": (SVC, '(request.default_compartment or "cell").strip() or "cell"',
                             'request.default_compartment or "cell"', NEW),
    "cache-default": (SVC, 'or "id_mapping_cache.json"', 'or "other.json"', NEW),
    "id-source-case": (SVC, '(request.id_source_mode or "hybrid").strip().lower()', '(request.id_source_mode or "hybrid").strip()', NEW),
    "db-port-not-int": (SVC, "db_port=int(request.db_port)", "db_port=request.db_port", NEW),
    "max-rounds-not-int": (SVC, "audit_max_rounds=int(request.audit_max_rounds)", "audit_max_rounds=request.audit_max_rounds", NEW),
    "timeout-not-int": (SVC, "audit_timeout_seconds=int(request.audit_timeout_seconds)", "audit_timeout_seconds=request.audit_timeout_seconds", NEW),
    "candidates-not-int": (SVC, "audit_candidate_count=int(request.audit_candidate_count)", "audit_candidate_count=request.audit_candidate_count", NEW),
    "top-k-not-int": (SVC, "example_top_k=int(request.example_top_k)", "example_top_k=request.example_top_k", NEW),
    "gap-items-not-int": (SVC, "gap_resolver_max_items=int(request.gap_resolver_max_items)", "gap_resolver_max_items=request.gap_resolver_max_items", NEW),
    "password-not-defaulted": (SVC, 'db_password=request.db_password or ""', "db_password=request.db_password", NEW),
    "schema-default": (SVC, '(request.db_schema or "pathbank").strip() or "pathbank"', '(request.db_schema or "pb").strip() or "pb"', NEW),
    "example-path-not-stripped": (SVC, '(request.example_index_path or "").strip()', 'request.example_index_path or ""', NEW),
    "llm-audit-not-bool": (SVC, "use_llm_audit=bool(request.use_llm_audit)", "use_llm_audit=request.use_llm_audit", NEW),
    "stoich-swapped": (SVC, "use_stoich_agent=bool(request.use_stoich_agent)", "use_stoich_agent=bool(request.use_gap_resolver)", NEW),
    "rag-always-on": (SVC, 'if _config.rag_config()["enabled"]', "if True", NEW),
    "rag-evidence-dropped": (SVC, 'state.get("rag_evidence_context") or ""\n                    if', '""\n                    if', NEW),
    "strict-db-default": (SVC, 'bool(state.get("pwml_strict_db", True))', 'bool(state.get("pwml_strict_db", False))', NEW),
    "mode-not-coerced": (SVC, 'export_mode=_export_mode.coerce_mode(state.get("export_mode"))', 'export_mode=state.get("export_mode")', NEW),
    "ctx-not-safe-dict": (SVC, 'pathway_context=_safe_dict(state.get("pathway_context"))', 'pathway_context=state.get("pathway_context")', NEW),
    "outputs-dir": (SVC, 'paths.PROJECT_ROOT / "outputs"', 'paths.PROJECT_ROOT / "output"', NEW),
    "state-not-passed": (SVC, "                state=state,\n            )\n        state[\"post_pipeline_artifacts\"]",
                         "                state={},\n            )\n        state[\"post_pipeline_artifacts\"]", NEW),
    "artifacts-not-stored": (SVC, '        state["post_pipeline_artifacts"] = artifacts\n', "", NEW),
    "pa-a-copy": (SVC, "        _pa = _safe_dict(artifacts)\n", "        _pa = dict(_safe_dict(artifacts))\n", NEW),
    "qa-overwritten-always": (SVC, '        if _pa.get("post_audit_qa_report"):\n            state["qa_report"] = _pa["post_audit_qa_report"]\n',
                              '        state["qa_report"] = _pa.get("post_audit_qa_report")\n', NEW),
    "summary-not-updated": (SVC, '            state["reaction_summary"] = _pa["post_audit_reaction_summary"]\n', "            pass\n", NEW),
    "snapshot-key": (SVC, '_pa["final_payload_snapshot"] = state.get("final_payload_snapshot", final_payload)',
                     '_pa["final_payload_snapshot"] = final_payload', NEW),
    "gate-failed-text": (SVC, "Post-pipeline stopped at hard-gate validation.", "Stopped at hard-gate validation.", NEW),
    "flags-dropped": (SVC, "                    reporter.json(_quarantine_flags)\n", "", NEW),
    "flags-text": (SVC, "research mode kept every process.", "research mode kept all processes.", NEW),
    "refusal-errors-not-set": (SVC, '                    state["refinement_gate_errors"] = [\n                        {"path": "/processes", "reason": f"quarantine refusal: {reason}"}\n                        for reason in _refusal_reasons\n                    ]\n', "", NEW),
    "refusal-reason-text": (SVC, 'f"quarantine refusal: {reason}"', 'f"refused: {reason}"', NEW),
    "refusal-message-join": (SVC, '+ "; ".join(_refusal_reasons)', '+ ", ".join(_refusal_reasons)', NEW),
    "refusal-coverage-dropped": (SVC, '                    reporter.json(_safe_dict(_pa.get("quarantine_coverage")))\n', "", NEW),
    "refusal-invariants-order": (SVC, '                    reporter.caption("Requested-core coverage")\n                    reporter.json(_safe_dict(_pa.get("quarantine_coverage")))\n                    reporter.caption("Strict invariants")\n                    reporter.json(_safe_dict(_pa.get("quarantine_report")).get("strict_invariants"))\n',
                                  '                    reporter.caption("Strict invariants")\n                    reporter.json(_safe_dict(_pa.get("quarantine_report")).get("strict_invariants"))\n                    reporter.caption("Requested-core coverage")\n                    reporter.json(_safe_dict(_pa.get("quarantine_coverage")))\n', NEW),
    "gate-key": (SVC, '_safe_dict(_pa.get(FINAL_GATE_REPORT_KEY))', '_safe_dict(_pa.get("final_gate"))', NEW),
    "research-open-branch-dropped": (SVC, "if _stage3_failed and _export_mode.is_research(", "if False and _export_mode.is_research(", NEW),
    "research-flags-not-stored": (SVC, '                        _pa["research_stage3_review_flags"] = _safe_list(final_gate.get("errors"))\n', "", NEW),
    "research-opened-not-marked": (SVC, '                        _pa["research_stage3_failed_open"] = True\n', "", NEW),
    "research-errors-not-cleared": (SVC, '                        _pa["research_stage3_failed_open"] = True\n                        state["refinement_gate_errors"] = []\n',
                                    '                        _pa["research_stage3_failed_open"] = True\n', NEW),
    "research-em-dash": (SVC, "did NOT pass — this candidate", "did NOT pass - this candidate", NEW),
    "research-markdown-em-dash": (SVC, '                                reporter.markdown(\n                                    f"- `{_gate_err.get(\'path\', \'\')}` — {_gate_err.get(\'reason\', _gate_err)}"\n                                )\n                            else:\n                                reporter.markdown(f"- {_gate_err}")\n                        reporter.json(final_gate)\n',
                                  '                                reporter.markdown(\n                                    f"- `{_gate_err.get(\'path\', \'\')}` - {_gate_err.get(\'reason\', _gate_err)}"\n                                )\n                            else:\n                                reporter.markdown(f"- {_gate_err}")\n                        reporter.json(final_gate)\n', NEW),
    "research-no-review": (SVC, "                        open_review(final_mapped_payload, mapping_report)\n                    elif _stage3_failed:",
                           "                    elif _stage3_failed:", NEW),
    "strict-errors-not-set": (SVC, '                        state["refinement_gate_errors"] = _safe_list(final_gate.get("errors"))\n', "", NEW),
    "strict-text": (SVC, '"Mapped pathway still fails the pre-export Stage 3 revalidation. "', '"Mapped pathway fails the pre-export Stage 3 revalidation. "', NEW),
    "strict-expander-collapsed": (SVC, "                            expanded=True,\n", "                            expanded=False,\n", NEW),
    "strict-detail-not-rendered": (SVC, '                                        render_issue_detail(_gate_err.get("detail"))\n', "", NEW),
    "strict-no-items-write": (SVC, '                                reporter.write("No itemized errors were returned; showing full gate report below.")\n', "                                pass\n", NEW),
    "strict-report-dropped": (SVC, "                                reporter.write(\"No itemized errors were returned; showing full gate report below.\")\n                            reporter.json(final_gate)\n",
                              "                                reporter.write(\"No itemized errors were returned; showing full gate report below.\")\n", NEW),
    "pass-errors-not-cleared": (SVC, '                        state["refinement_gate_errors"] = []\n                        open_review(final_mapped_payload, mapping_report)\n                        reporter.info(',
                                '                        open_review(final_mapped_payload, mapping_report)\n                        reporter.info(', NEW),
    "pass-info-before-review": (SVC, '                        open_review(final_mapped_payload, mapping_report)\n                        reporter.info("Mapped pathway is ready for review. PWML generation is paused until approval.")\n',
                                '                        reporter.info("Mapped pathway is ready for review. PWML generation is paused until approval.")\n                        open_review(final_mapped_payload, mapping_report)\n', NEW),
    "pass-info-text": (SVC, '"Mapped pathway is ready for review. PWML generation is paused until approval."', '"Mapped pathway is ready for review."', NEW),
    "fallback-final-mapped-dropped": (SVC, '_pa.get(CANONICAL_PAYLOAD_KEY) or _pa.get("final_mapped")', '_pa.get(CANONICAL_PAYLOAD_KEY)', NEW),
    "contract-not-rendered": (SVC, "        failure = exc\n        render_contract_failure(exc)\n", "        failure = exc\n", NEW),
    "contract-not-returned": (SVC, "        failure = exc\n        render_contract_failure(exc)\n", "        render_contract_failure(exc)\n", NEW),
    "contract-render-after-sync": (SVC, "        failure = exc\n        render_contract_failure(exc)\n    except Exception as exc:",
                                   "        failure = exc\n    except Exception as exc:", NEW),
    "contract-snapshot-added": (SVC, "        failure = exc\n        render_contract_failure(exc)\n",
                                '        failure = exc\n        _postpipeline._record_mapped_failure_snapshot(final_payload, stage="x", reason="y")\n        render_contract_failure(exc)\n', NEW),
    "generic-handler-dropped": (SVC, "    except Exception as exc:\n        failure = exc\n        _postpipeline._record_mapped_failure_snapshot(\n            final_payload,\n            stage=\"post_pipeline\",\n            reason=f\"{type(exc).__name__}: {exc}\",\n        )\n        reporter.error(f\"Post-pipeline conversion failed: {exc}\")\n        reporter.exception(exc)  # full traceback (file:line) so the cause isn't a mystery\n    finally:",
                                "    finally:", NEW),
    "snapshot-pre-rebind": (SVC, "        _postpipeline._record_mapped_failure_snapshot(\n            final_payload,", "        _postpipeline._record_mapped_failure_snapshot(\n            request.final_payload,", NEW),
    "snapshot-stage": (SVC, 'stage="post_pipeline",', 'stage="post",', NEW),
    "snapshot-reason": (SVC, 'reason=f"{type(exc).__name__}: {exc}",', 'reason=str(exc),', NEW),
    "generic-error-text": (SVC, 'f"Post-pipeline conversion failed: {exc}"', 'f"Conversion failed: {exc}"', NEW),
    "generic-exception-not-shown": (SVC, "        reporter.exception(exc)  # full traceback (file:line) so the cause isn't a mystery\n", "", NEW),
    "generic-error-after-exception": (SVC, '        reporter.error(f"Post-pipeline conversion failed: {exc}")\n        reporter.exception(exc)  # full traceback (file:line) so the cause isn\'t a mystery\n',
                                      '        reporter.exception(exc)  # full traceback (file:line) so the cause isn\'t a mystery\n        reporter.error(f"Post-pipeline conversion failed: {exc}")\n', NEW),
    "sync-before-handlers": (SVC, "    finally:\n        # Both the pass and the two failure paths above. The audit and gap\n        # rounds completed before an abort are already on disk; this is\n        # what carries them, and the mapping snapshot, to the batch driver.\n        _extraction._sync_extraction_diagnostics(state=state)\n\n    return",
                             "    finally:\n        pass\n    _extraction._sync_extraction_diagnostics(state=state)\n\n    return", NEW),
    "sync-state-wrong": (SVC, "_extraction._sync_extraction_diagnostics(state=state)", "_extraction._sync_extraction_diagnostics(state={})", NEW),
    "result-pre-rebind": (SVC, "return PostRunResult(final_payload=final_payload, failure=failure)", "return PostRunResult(final_payload=request.final_payload, failure=failure)", NEW),
    "result-no-failure": (SVC, "return PostRunResult(final_payload=final_payload, failure=failure)", "return PostRunResult(final_payload=final_payload)", NEW),
    "state-default": (SVC, "    state: MutableMapping[str, Any],\n    reporter: Reporter,\n    open_review:", "    state: MutableMapping[str, Any] = None,\n    reporter: Reporter = None,\n    open_review:", NEW),
    "request-field-default": (SVC, "    use_llm_audit: Any\n", "    use_llm_audit: Any = True\n", NEW),
    "password-in-repr": (SVC, "db_password: Any = field(repr=False)  # never in a log line or a traceback's repr", "db_password: Any", NEW),
    # -- the app's call site (real-app AppTest tests)
    "app-no-pop": (APP, '            st.session_state.pop("post_pipeline_artifacts", None)\n            # A second audit/mapping run', "            # A second audit/mapping run", NEW),
    "app-no-reset": (APP, '            reset_quarantine_state(PROJECT_ROOT / "outputs")\n            post_artifacts = {}\n            # R4b:', "            post_artifacts = {}\n            # R4b:", NEW),
    "app-guard-removed": (APP, '        if not isinstance(final_payload, dict) or not final_payload:\n            st.error("No pipeline output in session state. Run the pipeline first.")\n        else:\n            st.session_state.pop("post_pipeline_artifacts", None)\n            # A second',
                          '        if True:\n            st.session_state.pop("post_pipeline_artifacts", None)\n            # A second', NEW),
    "app-no-rebind": (APP, "            final_payload = _post_run.final_payload\n", "", NEW),
    "app-state-not-session": (APP, "                state=st.session_state,\n                reporter=StreamlitReporter(st),\n                open_review", "                state={},\n                reporter=StreamlitReporter(st),\n                open_review", NEW),
    "app-swapped-widget": (APP, "                    use_stoich_agent=use_stoich_agent,\n", "                    use_stoich_agent=use_gap_resolver,\n", NEW),
    "app-swapped-host": (APP, "                    db_host=db_host,\n", "                    db_host=db_user,\n", NEW),
    "app-swapped-rounds": (APP, "                    audit_max_rounds=audit_max_rounds,\n", "                    audit_max_rounds=audit_candidate_count,\n", NEW),
    "app-render-issue-noop": (APP, "                render_issue_detail=_render_issue_detail,\n", "                render_issue_detail=lambda detail: None,\n", NEW),
    "app-contract-label": (APP, 'operation_label="Audit and DB mapping"', 'operation_label="Audit"', NEW),
    "app-contract-key": (APP, 'download_key="dl_audit_mapping_stage_contract_error"', 'download_key="dl_x"', NEW),
    "app-review-noop": (APP, "                open_review=initialize_refinement_review_state,\n", "                open_review=lambda payload, report: None,\n", NEW),
    # -- the equivalence test is non-vacuous for the moved call
    "eq-no-audit-config": (SVC, "use_llm_audit=bool(request.use_llm_audit)", "use_llm_audit=not bool(request.use_llm_audit)", EQ),
    "eq-mapping-cache": (SVC, 'mapping_cache_path=request.mapping_cache_text.strip() or "id_mapping_cache.json"', 'mapping_cache_path="wrong.json"', EQ),
}

if len(sys.argv) > 3:
    lo, hi = (int(x) for x in sys.argv[3].split(":"))
    names = list(MUTATIONS)[lo:hi]
else:
    names = list(MUTATIONS)
if os.environ.get('ONLY'):
    names = [n for n in names if n in os.environ['ONLY'].split(',')]
originals = {p: p.read_bytes() for p in (SVC, APP)}


def run(name, tests):
    env = dict(os.environ, PYTHONPATH=f"{TREE}/src", T2PW_OFFLINE_CURATOR="1")
    cmd = [PY, f"{TREE}/docs/pwml_recovery_sprint/evidence/bounded_run.py", "--label", f"mut-{name}",
           "--timeout", "300", "--heavy-lock", "R4b", "--json", str(EV / f"{name}.json"), "--",
           PY, "-u", f"{TREE}/docs/pwml_recovery_sprint/evidence/pinned_pytest.py", "--expect-tree", TREE,
           "-q", "-p", "no:cacheprovider", f"--basetemp=C:/t/wsR4btmp/mut-tmp/{name}", *tests]
    while True:
        proc = subprocess.run(cmd, cwd=TREE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=400)
        if proc.returncode != 95:
            return proc.stdout
        print(f"  lock busy, waiting 8 s ({name})", flush=True)
        time.sleep(8)


results = {}
START = time.time()
try:
    for name in names:
        if time.time() - START > 420:
            print("STOP-AT", list(MUTATIONS).index(name), flush=True)
            break
        target, old, new, tests = MUTATIONS[name]
        raw = originals[target].decode("utf-8")
        crlf = "\r\n" in raw
        src = raw.replace("\r\n", "\n")
        for o, n in (zip(old, new) if isinstance(old, list) else [(old, new)]):
            assert src.count(o) == 1, (name, o[:60], src.count(o))
            src = src.replace(o, n)
        target.write_bytes((src.replace("\n", "\r\n") if crlf else src).encode("utf-8"))
        try:
            out = run(name, tests)
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
tail = f"killed {len(killed)}/{len(results)}; survived: {sorted(set(results) - set(killed))}"
print(tail)
with open(EV / "summary.txt", "a", encoding="utf-8") as fh:
    fh.write(tail + "\n")
sys.exit(0 if len(killed) == len(results) else 1)
