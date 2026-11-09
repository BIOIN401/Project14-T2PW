"""The ``run_pathway`` interface for the web server (W0) and its implementation (R5).

The two dataclasses and the vocabulary below are the contract every W-task builds
against and must not change shape without a new W0-style task.

Must stay cheap to import: no module-level import of ``streamlit``,
``t2pw.app.streamlit_app`` or ``t2pw.batch.driver`` (the app runs its UI on import, and
R6 makes the driver call this module); everything heavy is imported inside the functions.

The vocabulary strings MIRROR ``t2pw.batch.driver`` (``MODE_*``, ``STAGE_*``,
``_STATUS_*``, ``KIND_*``) rather than importing them, for the reason above;
``tests/test_service_pathway_interface.py`` asserts the two sets are equal.
"""

from __future__ import annotations

import dataclasses
import os
import re
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from t2pw.pipeline.release_status import ReleaseStatus
from t2pw.rag.eligibility import OUTCOME_SCOPE_CONFLICT, RequestedScope

# ── Mirrors of t2pw.batch.driver (equality is tested) ──────────────────────
MODE_STRICT = "strict"
MODE_RESEARCH = "research"

STAGE_INIT = "init"
STAGE_APP_LOAD = "app_load"
STAGE_INPUT = "input"
STAGE_STAGE1 = "stage1"
STAGE_POST_PIPELINE = "post_pipeline"
STAGE_PWML_EXPORT = "pwml_export"
STAGE_RESEARCH_REPORT = "research_report"

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"
STATUS_SCOPE_CONFLICT = OUTCOME_SCOPE_CONFLICT

KIND_CONTRACT = "contract"
KIND_LLM = "llm"
KIND_NETWORK = "network"
KIND_AMBIGUOUS = "ambiguous_review_scope"
KIND_NO_REACTIONS = "no_reactions"
KIND_CRASH = "crash"
KIND_TIMEOUT = "timeout"
KIND_UNKNOWN = "unknown"

#: The diagram the review step renders, saved beside the reports (Q-R5-1).
IMAGE_NAME = "pathway.png"


@dataclass(frozen=True)
class PathwayRequest:
    """One pathway to build. Frozen: the run may not rewrite its request.

    ``scope.requested_pathway`` is the pathway name (name mode) and
    ``scope.requested_organism`` the drop-down organism. Inputs are any mix of
    pasted ``texts``, ``pdf_paths`` and ``pmids``. ``out_dir=None`` lets the
    implementation choose (today the app writes to ``PROJECT_ROOT/outputs``).
    ``deadline`` is an absolute ``time.time()`` epoch second, ``None`` = no cap.
    ``extraction_focus`` is the user's own focus / task-context text (the app's
    "Optional extraction focus" box); empty means R5 derives it from ``scope``,
    as the batch driver does today.
    Consent, email and job ids are job-store fields, not pipeline inputs.
    """

    texts: Tuple[str, ...] = ()
    pdf_paths: Tuple[Path, ...] = ()
    pmids: Tuple[str, ...] = ()
    scope: RequestedScope = field(default_factory=RequestedScope)
    extraction_focus: str = ""
    export_mode: str = MODE_STRICT
    out_dir: Optional[Path] = None
    deadline: Optional[float] = None


@dataclass
class PathwayResult:
    """What one ``run_pathway`` call produced.

    Defaults mirror ``RunOutcome``: an unfilled result reads as an error at
    ``init``, never as a pass. ``release_status`` is the run's own release
    record (release_ready / review_required / diagnostic_only); a consumer
    must read it, never infer it from ``pwml_path``'s filename.

    ``warnings``, ``issue_codes`` and ``counts`` mirror ``RunOutcome``. A run
    can pass without producing one of its deliverables; that is a warning,
    and a result with warnings must never be shown as clean.

    ``report_paths`` maps a filename to the file written under ``out_dir``.
    Its keys are exactly the filenames ``t2pw.batch.driver`` puts in
    ``RunOutcome.artifacts`` (its diagnostic artifacts included), except the
    PWML itself (``pathway.pwml`` / ``pathway.review_required.pwml``), which
    is ``pwml_path``. Only files actually written are present; which ones
    depends on the mode and on how far the run got. The driver is the source
    of truth; today's names, by kind:

    * payloads and gates: ``stage1_payload.json``, ``merged_payload.json``,
      ``gate_fail_report.json``, ``initial_stage3_gate_report.json``,
      ``final_stage3_gate_report.json``, ``contract_reports.json``
    * identity: ``final_mapped.json``, ``rag_admission_report.json``
    * quarantine: ``quarantine_report.json``, ``removed_entity_report.json``,
      ``graph_closure_iterations.json``, ``coverage_summary.json``,
      ``coverage_diagnostics.json``
    * scope: ``scope_conflict.json``, ``scope_specialization.json``
    * diagnostics: ``stage0_attempts.json``,
      ``extraction_boundary_report.json``, ``cleaning_report.json``,
      ``mapped_failure_snapshot.json``, ``audit_iteration_summary.json``,
      ``gap_iteration_summary.json``
    * PWML export (strict mode only; research mode stops before the export):
      ``pwml_ir.json``, ``pwml_ir_report.json``,
      ``pwml_ir_validation.json``, ``pwml_validation_report.json``,
      ``pwml_qa.json``, ``pwml_required_field_gate_report.json``
    * research deliverables (research mode): ``review_flags.json``,
      ``research_pathway_report.txt``, ``research_pathway_citations.json``,
      ``research_pathway_elements.csv``

    The citation report already exists: ``research_pathway_citations.json``
    (with its ``.txt`` and ``.csv`` views), written in research mode only and
    only when RAG synthesis ran.

    ``attempt_log`` holds in-run attempts only: LLM / provider / stage retries
    inside ONE ``run_pathway`` call. Job-level retries (W4) are recorded in the
    W3 job store, not here. W4 decides a retry from ``failure_kind`` and
    ``pwml_path is None``.
    """

    status: str = STATUS_ERROR
    stage: str = STAGE_INIT
    failure_kind: str = ""
    message: str = ""
    warnings: List[str] = field(default_factory=list)
    issue_codes: List[str] = field(default_factory=list)
    counts: Dict[str, Any] = field(default_factory=dict)
    release_status: Optional[ReleaseStatus] = None
    pwml_path: Optional[Path] = None
    image_path: Optional[Path] = None
    report_paths: Dict[str, Path] = field(default_factory=dict)
    attempt_log: List[Dict[str, Any]] = field(default_factory=list)


def _server_paths(out_dir: Path, ignore_case: bool = os.name == "nt") -> "re.Pattern[str]":
    """Every spelling of a server folder a report can carry: native, ``/``, JSON-escaped."""

    import t2pw.paths as paths

    roots = {paths.PROJECT_ROOT, out_dir, Path(tempfile.gettempdir()), Path.home()}
    spellings = {form for root in roots for resolved in (Path(root), Path(root).resolve())
                 for form in (str(resolved), resolved.as_posix(), str(resolved).replace("\\", "\\\\"))}
    # A root counts only as a whole path: not glued to a word or path character before it
    # (``ACE2/TMPRSS2``, ``UMP/TMP``), and followed by a separator, quote, space or the end.
    alternatives = "|".join(r"(?<![\w/\\])" + re.escape(s) + r"(?=[\\/\"'\s]|$)[\\/]*"
                            for s in sorted((s for s in spellings if len(s) > 3), key=len, reverse=True))
    return re.compile(alternatives, re.IGNORECASE if ignore_case else 0)  # Windows paths only are case-blind


def _scrub_server_paths(artifacts: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """The artifacts as copied to ``out_dir``: no absolute server path in a text download.

    A root is cut, leaving the path relative (``extraction_boundary_report.json``'s
    ``report_path`` was the live case). Only the known roots, never by shape: a report's
    JSON pointers look absolute too (``/processes/reactions/0``). What the pipeline wrote
    under ``outputs/`` and ``tmp/`` is not touched, and bytes (the PWML) pass through.
    """

    pattern = _server_paths(out_dir)
    return {name: pattern.sub("", blob) if isinstance(blob, str) else blob for name, blob in artifacts.items()}


def _drive(
    request: PathwayRequest,
    paper: Any,
    outcome: Any,
    state: Dict[str, Any],
    reporter: Any,
    deadline: Optional[float],
) -> None:
    """The headless twin of ``batch.driver._drive``: its verdict logic, its helpers,
    the same order, with the three clicks replaced by the service calls the app makes.

    ``at`` is only ``session_state``, the one thing the driver's helpers read off AppTest.
    Keep this and ``_drive`` in step until R6 makes the driver call this module.
    """

    import functools

    from t2pw.batch import driver
    import t2pw.config as config
    import t2pw.paths as paths
    from t2pw.pipeline.deadline import BUDGET_SPENT_MARKER
    from t2pw.pipeline.export_mode import PATHWHIZ, RESEARCH
    from t2pw.pipeline.gate_reports import CANONICAL_PAYLOAD_KEY, FINAL_GATE_REPORT_KEY
    from t2pw.service import defaults as d
    from t2pw.service import extraction as ex
    from t2pw.service import post_run as pr
    from t2pw.service.driver_bridge import CODE_UNSUPPORTED
    from t2pw.service.export import run_export

    at = SimpleNamespace(session_state=state)
    outputs_dir = paths.PROJECT_ROOT / "outputs"
    classify = functools.partial(driver._classify, contract_signal=False, ambiguous=False, no_reactions=False)
    state[ex.DEADLINE_STATE_KEY] = deadline

    def out_of_time(message: str) -> bool:
        if deadline is None or time.time() < deadline:
            return False
        driver._finalize_timeout(outcome, message=message,
                                 detail=f"{BUDGET_SPENT_MARKER} this run was already spent")
        return True

    # Per-run isolation: the decision of a previous job must not reach this one.
    pr.reset_quarantine_state(outputs_dir, state=state)
    outcome.stage = STAGE_INPUT
    research = outcome.mode == MODE_RESEARCH
    try:
        extraction = ex.run_extraction(
            ex.ExtractionRequest(
                text_entries=[driver._paper_text(paper)], pdfs=[], uses_pdf_input=False,
                input_mode="Paste text", pdf_page_range="", pdf_skip_refs=True, pdf_ocr=False,
                user_task_context=request.extraction_focus or driver._extraction_focus(paper),
                export_mode=RESEARCH if research else PATHWHIZ,
                temperature=d.TEMPERATURE, run_inference=d.RUN_INFERENCE, enable_chunking=d.ENABLE_CHUNKING,
                chunk_size=d.CHUNK_SIZE, chunk_overlap=d.CHUNK_OVERLAP,
                extract_attempts=d.EXTRACT_ATTEMPTS, infer_attempts=d.INFER_ATTEMPTS, infer_rounds=d.INFER_ROUNDS,
                extract_tokens=d.extract_tokens(), infer_tokens=d.infer_tokens(),
                rag_incomplete_flag=d.RAG_INCOMPLETE_FLAG if config.rag_config()["enabled"] else False,
            ),
            reporter=reporter, state=state)
    except Exception as exc:  # noqa: BLE001 - the app's run shows it as at.exception; so does this
        reporter.exception(exc)
        extraction = None

    joined, errors, exceptions = reporter.text_seen(), reporter.errors, reporter.exceptions
    failure = getattr(extraction, "failure", None)
    if failure is not None and failure.kind == ex.KIND_AMBIGUOUS_REVIEW_SCOPE:
        pipeline_error = driver._safe_dict(state.get("pipeline_error"))
        driver._fail(outcome, status=STATUS_FAIL, kind=KIND_AMBIGUOUS,
                     message=driver._text(pipeline_error.get("message")) or failure.message,
                     detail="\n".join(p for p in (driver._json_text(pipeline_error), joined) if p),
                     codes=[KIND_AMBIGUOUS])
        outcome.stage = STAGE_STAGE1
        return
    if failure is not None and failure.kind == ex.KIND_DEADLINE_EXCEEDED:
        driver._finalize_timeout(outcome, message="extraction did not finish inside the time budget",
                                 detail=f"{BUDGET_SPENT_MARKER} this run was already spent")
        outcome.stage = STAGE_STAGE1
        driver._add_common_artifacts(at, {}, outcome.artifacts)
        return
    if extraction is None or failure is not None:
        kind = failure.kind if failure is not None else ""
        guard_codes = driver._structural_guard_codes(joined)
        message = errors[0].splitlines()[0] if errors else "extraction did not complete and reported no error"
        detail = joined or "(the run emitted no error, warning or exception)"
        outcome.stage = STAGE_STAGE1
        if kind in (ex.KIND_NO_PDF, ex.KIND_PDF_FAILED, ex.KIND_NO_TEXT):
            outcome.stage = STAGE_INPUT
            driver._fail(outcome, status=STATUS_ERROR, kind=KIND_CONTRACT, message=failure.message,
                         detail=detail, codes=[CODE_UNSUPPORTED])
        elif kind in (ex.KIND_STAGE_ONE_BOUNDARY_ERROR, ex.KIND_STAGE_ONE_INCOMPLETE):
            driver._fail(outcome, status=STATUS_FAIL, kind=KIND_CONTRACT, message=message, detail=detail,
                         codes=guard_codes)
        else:  # stage_one_failed / stage_two_failed wrap LLM, network and scope faults alike
            driver._fail(outcome, status=STATUS_ERROR if exceptions else STATUS_FAIL,
                         kind=classify(text=joined, issue_codes=guard_codes, crashed=bool(exceptions)),
                         message=message, detail=detail, codes=guard_codes)
        driver._add_common_artifacts(at, {}, outcome.artifacts)
        return

    outcome.stage = STAGE_STAGE1
    if driver._reconcile_stage0_scope(at, paper, outcome):
        driver._add_common_artifacts(at, {}, outcome.artifacts)
        return

    final_payload = extraction.final_payload
    if not isinstance(final_payload, dict) or not final_payload:
        driver._fail(outcome, status=STATUS_FAIL, kind=KIND_NO_REACTIONS,
                     message="No pipeline output in session state. Run the pipeline first.",
                     detail=joined or "(no error; the payload is simply absent)")
        driver._add_common_artifacts(at, {}, outcome.artifacts)
        return
    merged_counts = driver._payload_counts(final_payload)
    outcome.counts.update(merged_counts)
    if not merged_counts["reactions"] and not merged_counts["transports"]:
        driver._fail(outcome, status=STATUS_FAIL, kind=KIND_NO_REACTIONS,
                     message="the pipeline produced a pathway with no reactions and no transports",
                     detail=joined or "(no error; the payload is simply empty)")
        driver._add_common_artifacts(at, {}, outcome.artifacts)
        return

    if out_of_time("audit and DB mapping did not finish inside the time budget"):
        driver._add_common_artifacts(at, {}, outcome.artifacts)
        return
    # The app's caller duties: the not-a-dict guard is above; these two come next, in its order.
    state.pop("post_pipeline_artifacts", None)
    pr.reset_quarantine_state(outputs_dir, state=state)
    audit_timeout = d.AUDIT_TIMEOUT_SECONDS
    if deadline is not None:
        audit_timeout = max(1, int(min(audit_timeout, deadline - time.time())))
    reporter.clear()
    pr.run_post_pipeline(
        pr.PostRunRequest(
            final_payload=final_payload,
            use_llm_audit=d.USE_LLM_AUDIT, use_sbml_overwatch=d.USE_SBML_OVERWATCH,
            default_compartment=d.DEFAULT_COMPARTMENT, mapping_cache_text=d.MAPPING_CACHE_PATH,
            id_source_mode=d.id_source(), db_host=d.db_host(), db_port=d.db_port(), db_user=d.db_user(),
            db_password=d.db_password(), db_schema=d.db_schema(),
            audit_max_rounds=d.AUDIT_MAX_ROUNDS, audit_timeout_seconds=audit_timeout,
            audit_candidate_count=d.AUDIT_CANDIDATE_COUNT, use_example_retrieval=d.USE_EXAMPLE_RETRIEVAL,
            example_index_path=d.EXAMPLE_INDEX_PATH, example_top_k=d.EXAMPLE_TOP_K,
            use_gap_resolver=d.USE_GAP_RESOLVER, use_llm_gap_resolver=d.USE_LLM_GAP_RESOLVER,
            gap_resolver_max_items=d.GAP_RESOLVER_MAX_ITEMS, use_stoich_agent=d.USE_STOICH_AGENT,
        ),
        state=state, reporter=reporter,
        open_review=functools.partial(pr.open_refinement_review, state=state),
        render_issue_detail=lambda _detail: None,
        render_contract_failure=lambda exc: reporter.error(
            f"Audit and DB mapping stopped at the {exc.stage or 'unknown'} stage boundary: {exc}"),
    )

    outcome.stage = STAGE_POST_PIPELINE
    joined, errors, exceptions = reporter.text_seen(), reporter.errors, reporter.exceptions
    artifacts = driver._safe_dict(state.get("post_pipeline_artifacts"))
    superseded = driver._superseded_contract_reports(artifacts)
    if superseded:
        outcome.counts["superseded_contract_errors"] = sum(int(i["errors"]) for i in superseded)
    blocking_reports = driver._blocking_reports(artifacts)
    codes, code_lines, error_count = driver._collect_issue_codes(blocking_reports)
    blocking_contract_issues = driver._report_issues(blocking_reports)
    schema_warnings, schema_findings = driver._runtime_schema_warnings(driver._collect_reports(artifacts))
    outcome.warnings.extend(w for w in schema_warnings if w not in outcome.warnings)
    if error_count:
        outcome.counts["contract_errors"] = error_count
    if schema_findings:
        outcome.counts["runtime_schema_findings"] = schema_findings
    driver._add_common_artifacts(at, artifacts, outcome.artifacts)
    if out_of_time("audit and DB mapping ran past the time budget"):  # before any verdict is drawn from it
        return

    if not artifacts:
        codes.extend(c for c in driver._structural_guard_codes(joined) if c not in codes)
        driver._fail(outcome, status=STATUS_ERROR if exceptions else STATUS_FAIL,
                     kind=classify(text=joined, issue_codes=codes, crashed=bool(exceptions)),
                     message=errors[0].splitlines()[0] if errors
                     else "audit and DB mapping stored no artifacts and reported no error",
                     detail="\n".join(p for p in (joined, *code_lines) if p), codes=codes)
        return

    mapped = artifacts.get(CANONICAL_PAYLOAD_KEY) or artifacts.get("final_mapped_db") or artifacts.get("final_mapped")
    if isinstance(mapped, dict) and mapped:
        outcome.counts.update(driver._payload_counts(mapped))

    verdict = driver.gate_verdict(artifacts)
    gate_failed, gate_errors = verdict.failed, list(verdict.errors)
    codes.extend(c for c in map(driver._issue_code, gate_errors) if c and c not in codes)
    outcome.counts["gate_errors"] = len(gate_errors)

    research_gate_flags: List[Dict[str, str]] = []
    if research and (gate_failed or gate_errors):
        research_gate_flags, gate_warnings = driver._research_gate_flags(gate_errors)
        outcome.counts["gate_errors_not_blocking"] = len(gate_errors)
        outcome.warnings.extend(w for w in gate_warnings if w not in outcome.warnings)
        if gate_failed and not gate_errors:
            warning = (f"{driver.WARN_RESEARCH_GATE_PREFIX}the Stage-3 gate reported failure at "
                       f"{driver._text(verdict.stage) or 'a stage boundary'} with no itemised errors")
            if warning not in outcome.warnings:
                outcome.warnings.append(warning)

    blocking_gate = gate_failed and not research
    blocking_issues = (driver.blocking_findings(verdict, blocking_contract_issues) if blocking_gate
                       else driver.dedupe_findings(blocking_contract_issues))
    outcome.counts["blocking_issues"] = len(blocking_issues)
    if blocking_gate or error_count:
        driver._finalize_gate_failure(
            outcome, verdict=verdict, blocking_issues=blocking_issues, codes=codes, code_lines=code_lines,
            joined=joined, crashed=bool(exceptions), stage3_gate=driver._safe_dict(artifacts.get(FINAL_GATE_REPORT_KEY)),
            gate_fail=driver._safe_dict(artifacts.get("gate_fail_report")))
        return

    if research:
        outcome.stage = STAGE_RESEARCH_REPORT
        try:
            note = driver._add_research_artifacts(at, artifacts, outcome.artifacts, outcome.counts,
                                                  gate_flags=research_gate_flags)
        except Exception:  # noqa: BLE001 - a report bug must not be called a pass
            driver._fail(outcome, status=STATUS_ERROR, kind=KIND_CRASH,
                         message="research report generation failed after a clean pipeline run",
                         detail=traceback.format_exc(), codes=codes)
            return
        if "research_pathway_report.txt" not in outcome.artifacts and not note:
            driver._fail(outcome, status=STATUS_FAIL, kind=KIND_UNKNOWN,
                         message="no research report was produced and no reason was given",
                         detail=joined, codes=codes)
            return
        outcome.status, outcome.failure_kind = STATUS_PASS, ""
        outcome.message = "research run completed" + (f"; {note}" if note else " with a citation report")
        outcome.detail, outcome.issue_codes = joined, codes
        if "research_pathway_report.txt" not in outcome.artifacts and driver.WARN_NO_RESEARCH_REPORT not in outcome.warnings:
            outcome.warnings.append(driver.WARN_NO_RESEARCH_REPORT)
        return

    # Strict: the export the review step unlocks (the app's "Generate PWML" button).
    outcome.stage = STAGE_PWML_EXPORT
    if state.get("refinement_working_json") is None:
        driver._fail(outcome, status=STATUS_FAIL,
                     kind=classify(text=joined, issue_codes=codes, crashed=bool(exceptions)),
                     message="the review step that unlocks PWML export was not reached",
                     detail=joined, codes=codes)
        return
    if out_of_time("PWML export did not finish inside the time budget"):
        return
    reporter.clear()
    try:
        pwml_result = run_export(state=state, reporter=reporter, work_dir=paths.PROJECT_ROOT / "tmp",
                                 project_root=paths.PROJECT_ROOT)
    except Exception as exc:  # noqa: BLE001 - as the app: the failure becomes the result
        pwml_result = {"ok": False, "error": str(exc), "counts": {}, "issues": 0, "output_path": "",
                       "qa": {}, "grounding_report": {}}
    state["pwml_export_result"] = pwml_result
    if pwml_result.get("ok"):
        reporter.success(f"PWML generated from reviewed JSON: {pwml_result.get('output_path')}")
    else:
        reporter.error(f"PWML export failed: {pwml_result.get('error', 'unknown')}")

    joined, errors, exceptions = reporter.text_seen(), reporter.errors, reporter.exceptions
    artifacts = driver._safe_dict(state.get("post_pipeline_artifacts")) or artifacts
    driver._add_strict_artifacts(artifacts, pwml_result, outcome.artifacts, superseded)

    pwml_codes = list(codes)
    for report_key in ("required_gate_report", "stage3_contract_report", "pwml_contract_report"):
        pwml_codes.extend(c for c in map(driver._issue_code, driver._report_errors(pwml_result.get(report_key)))
                          if c and c not in pwml_codes)

    xml = driver._xml_bytes(pwml_result)
    if not pwml_result.get("ok") or not xml:
        driver._fail(outcome, status=STATUS_FAIL,
                     kind=driver._classify(text="\n".join([joined, driver._text(pwml_result.get("error"))]),
                                           issue_codes=pwml_codes, contract_signal=bool(pwml_codes), ambiguous=False,
                                           no_reactions=False, crashed=bool(exceptions)),
                     message="PWML export failed: "
                     + (driver._text(pwml_result.get("error")) or "no XML was produced and no error was reported"),
                     detail="\n".join(p for p in (joined, driver._json_text(pwml_result)) if p),
                     codes=pwml_codes)
        return
    driver._finalize_pwml_export(outcome, xml=xml, pwml_result=pwml_result, joined=joined,
                                 codes=pwml_codes, superseded=superseded)


def run_pathway(request: PathwayRequest) -> PathwayResult:
    """Run one pathway headless: Stage 0-2, audit and mapping, export, in one process.

    Input rules and the result are the driver bridge's (``driver_bridge._unsupported``,
    ``_paper``, ``_to_result``), so the job and result pages render the same; only a custom
    ``extraction_focus`` is honoured here, where the bridge refuses it. Everything the run
    writes to the fixed ``outputs/`` and ``tmp/`` stays there; ``out_dir`` gets a copy of
    the deliverables, so one worker per checkout (Q-W0-1). Never raises and never edits
    the request: a crash in the pipeline is an ``error``/``crash`` result.
    """

    from t2pw.batch import driver, runner
    from t2pw.service import driver_bridge as bridge
    from t2pw.service.reporter import CollectingReporter

    out_dir = Path(request.out_dir) if request.out_dir else Path(tempfile.mkdtemp(prefix="t2pw-job-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    problem = bridge._unsupported(dataclasses.replace(request, extraction_focus=""))
    paper: Any = None
    if problem is None:
        paper, problem = bridge._paper(request)
    if problem is not None:
        return bridge._refuse(*problem)

    deadline: Optional[float] = None
    if request.deadline is not None:
        if float(request.deadline) - time.time() <= 0:
            return bridge._refuse(STATUS_TIMEOUT, KIND_TIMEOUT,
                                  "the run's deadline passed before the pipeline started", bridge.CODE_DEADLINE)
        deadline = float(request.deadline) - bridge.GRACE_S

    runner.force_utf8_stdio()  # the worker child's stdout is a file (cp1252 on Windows)
    outcome = driver.RunOutcome(paper_id=driver._paper_id(paper), mode=driver._resolve_mode(request.export_mode)[0])
    state: Dict[str, Any] = {}
    started = time.monotonic()
    try:
        _drive(request, paper, outcome, state, CollectingReporter(), deadline)
    except Exception:  # noqa: BLE001 - a crash here is a result, never an unhandled error in the worker
        outcome.status, outcome.failure_kind = STATUS_ERROR, KIND_CRASH
        outcome.message = outcome.message or "the pipeline crashed before it could be judged"
        outcome.detail = ((outcome.detail + "\n\n") if outcome.detail else "") + traceback.format_exc()
    outcome.seconds = round(time.monotonic() - started, 2)
    outcome.artifacts = _scrub_server_paths(outcome.artifacts, out_dir)
    outcome.message, outcome.detail = (_scrub_server_paths({'m': t}, out_dir)['m'] for t in (outcome.message or '', outcome.detail or ''))
    outcome.warnings = [_scrub_server_paths({'m': t}, out_dir)['m'] for t in outcome.warnings]
    result = bridge._to_result(outcome, out_dir)
    record = out_dir / bridge.OUTCOME_NAME  # the bridge adds bridge_files[].error (an OSError text with its path) as it writes
    try:
        record.write_text(_scrub_server_paths({'m': record.read_text(encoding="utf-8")}, out_dir)['m'], encoding="utf-8")
    except OSError:
        pass  # the bridge already warned that the record could not be written
    result.attempt_log = [{**row, "runner": "t2pw.service.pathway.run_pathway"} for row in result.attempt_log]
    png = state.get("refinement_graph_bytes")
    if isinstance(png, (bytes, bytearray)) and png:
        try:
            (out_dir / IMAGE_NAME).write_bytes(bytes(png))
            result.image_path = out_dir / IMAGE_NAME
        except OSError:
            result.warnings.append("the pathway diagram could not be written")
    return result
