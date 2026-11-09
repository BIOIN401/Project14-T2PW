"""NEW TESTS (R5, new capability): ``run_pathway``'s failure mapping, deadline, per-run isolation,
shared defaults, delivered files and path scrubbing.

The service steps are faked at their source modules; the real ``run_pathway`` control flow and
the real batch-driver helpers it calls run. Equivalence with the app is
``tests/test_equivalence_run_pathway.py``.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import t2pw.paths as paths
from t2pw.pipeline.gate_reports import CANONICAL_PAYLOAD_KEY, FINAL_GATE_REPORT_KEY
from t2pw.pipeline.release_status import classify_release_status
from t2pw.rag.eligibility import RequestedScope
from t2pw.service import defaults
from t2pw.service import extraction as ex
from t2pw.service import pathway as pw
from t2pw.service import post_run as pr
from t2pw.service.extraction import ExtractionResult, InputFailure
from t2pw.service.pathway import PathwayRequest
from t2pw.service.reporter import CollectingReporter, LoggingReporter, StreamlitReporter
from t2pw.web import retry

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "t2pw" / "app" / "streamlit_app.py"
PAYLOAD = {"processes": {"reactions": [{"name": "r1"}], "transports": []}, "entities": {"compounds": [{"name": "c"}]}}
XML = b"<pwml/>"


class World:
    """Fakes the three service steps, recording what reached them."""

    def __init__(self, mp, root: Path) -> None:
        self.root, self.calls, self.post_requests, self.seen_stale = root, [], [], []
        self.extract = lambda request, reporter, state: ExtractionResult(final_payload=PAYLOAD)
        self.post = self.default_post
        self.export = lambda state, reporter: {"ok": True, "xml_bytes": XML, "output_path": "x", "counts": {},
                                               "quarantine_report": {"release": _released()}}
        mp.setattr(paths, "PROJECT_ROOT", root)
        mp.setattr(ex, "run_extraction", self._extract)
        mp.setattr(pr, "run_post_pipeline", self._post)
        import t2pw.service.export as export_module
        mp.setattr(export_module, "run_export", self._export)

    @property
    def stale(self) -> Path:
        return self.root / "outputs" / "quarantine_report.json"

    def _extract(self, request, *, reporter, state):
        self.calls.append("extract")
        self.request, self.state = request, state
        return self.extract(request, reporter, state)

    def _post(self, request, *, state, reporter, open_review, render_issue_detail, render_contract_failure):
        self.calls.append("post")
        self.post_requests.append(request)
        self.seen_stale.append(self.stale.exists())
        self.open_review, self.render_contract_failure = open_review, render_contract_failure
        return self.post(request, state, reporter)

    def default_post(self, request, state, reporter):
        state["refinement_working_json"] = {"reviewed": True}
        state["refinement_graph_bytes"] = b"\x89PNG"
        state["post_pipeline_artifacts"] = {CANONICAL_PAYLOAD_KEY: PAYLOAD, FINAL_GATE_REPORT_KEY: {"ok": True}}
        return pr.PostRunResult(final_payload=request.final_payload)

    def _export(self, *, state, reporter, work_dir, project_root):
        self.calls.append("export")
        return self.export(state, reporter)


def _released() -> dict:
    return {**classify_release_status(pipeline_executed=True, strict_gates_passed=True).to_dict(), "status": "release_ready"}


@pytest.fixture
def world(monkeypatch, tmp_path):
    return World(monkeypatch, tmp_path)


def _run(tmp_path: Path, **kw):
    fields = dict(texts=("The paper.",), out_dir=tmp_path / "out", scope=RequestedScope())
    fields.update(kw)
    return pw.run_pathway(PathwayRequest(**fields))


def _fail(kind: str, *errors: str, exception: Exception = None):
    def extract(request, reporter, state):
        for text in errors:
            reporter.error(text)
        if exception is not None:
            raise exception
        return ExtractionResult(failure=InputFailure(kind, errors[0] if errors else kind))
    return extract


# -- happy paths ------------------------------------------------------------------------
def test_strict_success_delivers_the_pwml_named_by_its_release_record(world, tmp_path):
    res = _run(tmp_path)
    assert (res.status, res.failure_kind, res.stage) == ("pass", "", "pwml_export")
    assert world.calls == ["extract", "post", "export"]
    assert res.pwml_path.name == "pathway.pwml" and res.pwml_path.read_bytes() == XML
    assert res.pwml_path.parent == tmp_path / "out"
    assert res.release_status.status == "release_ready"
    assert res.counts["reactions"] == 1 and res.counts["pwml_bytes"] == len(XML)
    assert res.attempt_log[0]["runner"] == "t2pw.service.pathway.run_pathway"
    assert "final_mapped.json" in res.report_paths and "pathway.pwml" not in res.report_paths
    assert retry.decide({"status": res.status, "failure_kind": "", "pwml_path": "x"}, 1, 0.0).retry is False


def test_research_mode_stops_before_the_export(world, tmp_path):
    res = _run(tmp_path, export_mode="research")
    assert (res.status, res.stage) == ("pass", "research_report")
    assert "export" not in world.calls and res.pwml_path is None
    assert "review_flags.json" in res.report_paths
    assert not {"pwml_ir.json", "pwml_qa.json"} & set(res.report_paths)


def test_the_form_mode_reaches_extraction(world, tmp_path):
    _run(tmp_path)
    assert world.request.export_mode == "pathwhiz"
    _run(tmp_path, export_mode="research")
    assert world.request.export_mode == "research"


def test_focus_comes_from_the_request_else_from_the_scope(world, tmp_path):
    scope = RequestedScope(requested_pathway="heme biosynthesis", requested_organism="Homo sapiens")
    _run(tmp_path, scope=scope)
    assert world.request.user_task_context == "heme biosynthesis in Homo sapiens"
    _run(tmp_path, scope=scope, extraction_focus="only the first three steps")
    assert world.request.user_task_context == "only the first three steps"


def test_the_bridges_refusals_still_apply(world, tmp_path):
    res = _run(tmp_path, texts=(), scope=RequestedScope())
    assert (res.status, res.failure_kind) == ("error", "contract") and world.calls == []


# -- failure mapping --------------------------------------------------------------------
@pytest.mark.parametrize("kind", [ex.KIND_NO_PDF, ex.KIND_PDF_FAILED, ex.KIND_NO_TEXT])
def test_input_failures_are_final_contract_errors(world, tmp_path, kind):
    world.extract = _fail(kind, "bad input")
    res = _run(tmp_path)
    assert (res.status, res.failure_kind, res.stage) == ("error", "contract", "input")
    assert res.issue_codes == ["bridge_unsupported_request"] and res.message == "bad input"
    assert retry.decide({"status": res.status, "failure_kind": res.failure_kind}, 1, 0.0).retry is False


@pytest.mark.parametrize("kind", [ex.KIND_STAGE_ONE_BOUNDARY_ERROR, ex.KIND_STAGE_ONE_INCOMPLETE])
def test_stage_one_verdicts_are_final_contract_failures(world, tmp_path, kind):
    world.extract = _fail(kind, "no usable result")  # no classifier marker: the kind alone decides
    res = _run(tmp_path)
    assert (res.status, res.failure_kind, res.stage) == ("fail", "contract", "stage1")
    assert retry.decide({"status": res.status, "failure_kind": res.failure_kind}, 1, 0.0).retry is False


@pytest.mark.parametrize("kind", [ex.KIND_STAGE_ONE_FAILED, ex.KIND_STAGE_TWO_FAILED])
@pytest.mark.parametrize("text,want", [("Extraction failed: invalid json from the model", "llm"),
                                       ("Inference stage failed: connection refused", "network")])
def test_wrapped_stage_failures_are_classified_and_retried(world, tmp_path, kind, text, want):
    world.extract = _fail(kind, text)
    res = _run(tmp_path)
    assert (res.status, res.failure_kind, res.stage) == ("fail", want, "stage1")
    assert res.message == text
    assert retry.decide({"status": res.status, "failure_kind": res.failure_kind}, 1, 0.0).retry is True


def test_the_ambiguous_gate_is_a_final_refusal_with_its_own_kind(world, tmp_path):
    def extract(request, reporter, state):
        state["pipeline_error"] = {"status": "ambiguous_review_scope", "message": "pick an example"}
        return ExtractionResult(failure=InputFailure(ex.KIND_AMBIGUOUS_REVIEW_SCOPE, "x"))
    world.extract = extract
    res = _run(tmp_path)
    assert (res.status, res.failure_kind, res.stage) == ("fail", "ambiguous_review_scope", "stage1")
    assert res.message == "pick an example" and res.issue_codes == ["ambiguous_review_scope"]
    assert retry.decide({"status": res.status, "failure_kind": res.failure_kind}, 1, 0.0).retry is False


def test_an_exception_in_the_run_is_a_classified_error_result(world, tmp_path):
    world.extract = _fail("", exception=RuntimeError("boom"))
    res = _run(tmp_path)
    assert (res.status, res.failure_kind) == ("error", "crash") and res.stage == "stage1"
    assert retry.decide({"status": res.status, "failure_kind": res.failure_kind}, 1, 0.0).retry is True


def test_a_bug_in_the_drive_is_a_crash_result_not_an_exception(world, tmp_path, monkeypatch):
    monkeypatch.setattr(pw, "_drive", lambda *a, **k: 1 / 0)
    res = _run(tmp_path)
    assert (res.status, res.failure_kind) == ("error", "crash")
    outcome = json.loads((tmp_path / "out" / "run_outcome.json").read_text(encoding="utf-8"))
    assert "ZeroDivisionError" in outcome["detail"]


def test_no_reactions_and_absent_payload_are_final(world, tmp_path):
    world.extract = lambda r, rep, s: ExtractionResult(final_payload={"processes": {"reactions": []}})
    res = _run(tmp_path)
    assert (res.status, res.failure_kind) == ("fail", "no_reactions") and world.calls == ["extract"]
    world.extract = lambda r, rep, s: ExtractionResult(final_payload=None)
    res = _run(tmp_path)
    assert res.failure_kind == "no_reactions" and "No pipeline output" in res.message


def test_a_scope_conflict_stops_before_the_audit(world, tmp_path, monkeypatch):
    def extract(request, reporter, state):
        state["pathway_context"] = {"pathway_name": "cholesterol biosynthesis", "likely_organism": "Homo sapiens"}
        return ExtractionResult(final_payload=PAYLOAD)
    world.extract = extract
    res = _run(tmp_path, scope=RequestedScope(requested_pathway="lipid A biosynthesis",
                                              requested_organism="Escherichia coli"))
    assert res.status == "scope_conflict" and world.calls == ["extract"]
    assert "scope_conflict.json" in res.report_paths


def test_post_pipeline_without_artifacts_is_classified(world, tmp_path):
    def post(request, state, reporter):
        reporter.error("Post-pipeline conversion failed: mysql went away")
        return pr.PostRunResult(final_payload=request.final_payload, failure=RuntimeError("x"))
    world.post = post
    res = _run(tmp_path)
    assert (res.status, res.failure_kind, res.stage) == ("fail", "network", "post_pipeline")
    assert "export" not in world.calls


def test_a_stage_contract_failure_is_a_final_contract_failure(world, tmp_path):
    def post(request, state, reporter):
        world.render_contract_failure(SimpleNamespace(stage="post_mapping", __str__=lambda s: "x"))
        return pr.PostRunResult(final_payload=request.final_payload, failure=RuntimeError("x"))
    world.post = post
    res = _run(tmp_path)
    assert (res.status, res.failure_kind) == ("fail", "contract")


def test_a_strict_gate_failure_blocks_the_export_with_a_release_record(world, tmp_path):
    def post(request, state, reporter):
        state["post_pipeline_artifacts"] = {
            CANONICAL_PAYLOAD_KEY: PAYLOAD,
            FINAL_GATE_REPORT_KEY: {"ok": False, "errors": [{"path": "/p", "reason": "bad", "code": "E1"}]}}
        return pr.PostRunResult(final_payload=request.final_payload)
    world.post = post
    res = _run(tmp_path)
    assert res.status == "fail" and res.stage == "post_pipeline" and "export" not in world.calls
    assert res.release_status.status == "diagnostic_only" and res.pwml_path is None


def test_the_review_step_not_opening_stops_the_export(world, tmp_path):
    def post(request, state, reporter):
        state["post_pipeline_artifacts"] = {CANONICAL_PAYLOAD_KEY: PAYLOAD, FINAL_GATE_REPORT_KEY: {"ok": True}}
        return pr.PostRunResult(final_payload=request.final_payload)
    world.post = post
    res = _run(tmp_path)
    assert res.status == "fail" and "review step" in res.message and "export" not in world.calls


def test_export_refusal_and_export_exception_are_failures(world, tmp_path):
    world.export = lambda state, reporter: {"ok": False, "error": "no reviewed payload"}
    res = _run(tmp_path)
    assert res.status == "fail" and res.message == "PWML export failed: no reviewed payload" and res.pwml_path is None

    def boom(state, reporter):
        raise RuntimeError("exporter crashed")
    world.export = boom
    res = _run(tmp_path)
    assert res.status == "fail" and "exporter crashed" in res.message


# -- deadline ---------------------------------------------------------------------------
def test_no_time_at_the_start_runs_nothing(world, tmp_path):
    res = _run(tmp_path, deadline=time.time() - 1)
    assert (res.status, res.failure_kind) == ("timeout", "timeout") and world.calls == []
    assert retry.decide({"status": res.status, "failure_kind": res.failure_kind}, 1, 0.0).retry is False


def test_extraction_gets_the_deadline_less_the_grace(world, tmp_path):
    from t2pw.service import driver_bridge as bridge

    end = time.time() + 5000
    _run(tmp_path, deadline=end)
    assert world.state[ex.DEADLINE_STATE_KEY] == pytest.approx(end - bridge.GRACE_S)


def test_no_deadline_changes_nothing_downstream(world, tmp_path):
    _run(tmp_path)
    assert world.state[ex.DEADLINE_STATE_KEY] is None
    assert world.post_requests[0].audit_timeout_seconds == defaults.AUDIT_TIMEOUT_SECONDS


def test_the_audit_timeout_is_capped_by_the_time_left(world, tmp_path):
    from t2pw.service import driver_bridge as bridge

    _run(tmp_path, deadline=time.time() + bridge.GRACE_S + 100)
    assert 90 <= world.post_requests[0].audit_timeout_seconds <= 100


def test_the_default_audit_timeout_survives_a_long_deadline(world, tmp_path):
    _run(tmp_path, deadline=time.time() + 5000)
    assert world.post_requests[0].audit_timeout_seconds == defaults.AUDIT_TIMEOUT_SECONDS


def test_extraction_running_out_stops_cleanly_as_a_timeout(world, tmp_path):
    world.extract = _fail(ex.KIND_DEADLINE_EXCEEDED)
    res = _run(tmp_path, deadline=time.time() + 5000)
    assert (res.status, res.failure_kind) == ("timeout", "timeout") and world.calls == ["extract"]


def test_time_running_out_before_the_audit_stops_cleanly(world, tmp_path, monkeypatch):
    from t2pw.service import driver_bridge as bridge

    def extract(request, reporter, state):
        state[ex.DEADLINE_STATE_KEY] = None  # the run's own copy; the check reads the deadline it was given
        monkeypatch.setattr(pw.time, "time", lambda: end + 1)
        return ExtractionResult(final_payload=PAYLOAD)
    end = time.time() + 5000
    world.extract = extract
    res = _run(tmp_path, deadline=end + bridge.GRACE_S)
    assert (res.status, res.failure_kind) == ("timeout", "timeout") and world.calls == ["extract"]
    assert res.attempt_log[0]["termination_reason"] == "budget_exhausted"


def test_time_running_out_before_the_export_stops_cleanly(world, tmp_path, monkeypatch):
    from t2pw.service import driver_bridge as bridge

    end = time.time() + 5000

    def post(request, state, reporter):
        monkeypatch.setattr(pw.time, "time", lambda: end + 1)
        return world.default_post(request, state, reporter)
    world.post = post
    res = _run(tmp_path, deadline=end + bridge.GRACE_S)
    assert (res.status, res.failure_kind, res.stage) == ("timeout", "timeout", "pwml_export")
    assert "export" not in world.calls


@pytest.mark.parametrize("step", ["before stage one", "before stage two"])
def test_run_extraction_stops_between_stages_when_the_deadline_has_passed(monkeypatch, step):
    state = {ex.DEADLINE_STATE_KEY: time.time() + 1000}
    ran = []
    assembled = SimpleNamespace(failure=None, text="t")
    monkeypatch.setattr(ex, "assemble_inputs", lambda *a, **k: assembled)
    monkeypatch.setattr(ex, "run_stage_zero", lambda *a, **k: SimpleNamespace(pathway_context={}, diagnostics_dir=None))
    monkeypatch.setattr(ex, "ambiguous_review_gate", lambda *a, **k: None)

    def stage_one(*a, **k):
        ran.append("one")
        state[ex.DEADLINE_STATE_KEY] = time.time() - 1
        return SimpleNamespace(failure=None)
    if step == "before stage one":
        state[ex.DEADLINE_STATE_KEY] = time.time() - 1
    monkeypatch.setattr(ex, "run_stage_one", stage_one)
    monkeypatch.setattr(ex, "run_stage_two", lambda *a, **k: ran.append("two"))
    request = SimpleNamespace(export_mode="pathwhiz", user_task_context="", text_entries=[], pdfs=[],
                              uses_pdf_input=False, input_mode="Paste text", pdf_page_range="", pdf_skip_refs=True,
                              pdf_ocr=False, temperature=0.0, enable_chunking=True, chunk_size=1, chunk_overlap=1,
                              extract_attempts=1, extract_tokens=1, rag_incomplete_flag=False)
    result = ex.run_extraction(request, reporter=LoggingReporter(), state=state)
    assert result.failure.kind == ex.KIND_DEADLINE_EXCEEDED
    assert ran == ([] if step == "before stage one" else ["one"])


def test_the_rag_rounds_get_the_time_left_only_with_a_deadline():
    assert ex._rag_deadline_kwargs({}) == {}
    assert ex._rag_deadline_kwargs({ex.DEADLINE_STATE_KEY: None}) == {}
    left = ex._rag_deadline_kwargs({ex.DEADLINE_STATE_KEY: time.time() + 500})["deadline_seconds"]
    assert 495 <= left <= 500
    assert ex._rag_deadline_kwargs({ex.DEADLINE_STATE_KEY: time.time() - 50}) == {"deadline_seconds": 1.0}


# -- per-run isolation ------------------------------------------------------------------
def test_a_stale_decision_on_disk_is_gone_before_the_run_starts(world, tmp_path):
    world.stale.parent.mkdir(parents=True)
    world.stale.write_text("{}", encoding="utf-8")
    seen = []

    def extract(request, reporter, state):  # recorded, not asserted: run_pathway catches exceptions
        seen.append(world.stale.exists())
        return ExtractionResult(failure=InputFailure(ex.KIND_NO_TEXT, "nothing"))
    world.extract = extract
    _run(tmp_path)
    assert seen == [False], "the extraction started under the previous job's decision"


def test_two_runs_in_sequence_do_not_share_a_decision(world, tmp_path):
    """The second run dies before the boundary: it must carry nothing of the first."""

    decisions = []

    def first_post(request, state, reporter):
        world.stale.parent.mkdir(parents=True, exist_ok=True)
        world.stale.write_text(json.dumps({"decision": "job-one"}), encoding="utf-8")
        state["quarantine_report"] = {"decision": "job-one"}
        state["quarantine_artifacts"] = {"quarantine_report.json": str(world.stale)}
        decisions.append(dict(state))
        return world.default_post(request, state, reporter)
    world.post = first_post
    first = _run(tmp_path, out_dir=tmp_path / "one")
    assert "quarantine_report.json" in first.report_paths

    def second_post(request, state, reporter):  # recorded, not asserted: run_pathway catches exceptions
        decisions.append(dict(state))
        decisions.append(world.stale.exists())
        return pr.PostRunResult(final_payload=request.final_payload, failure=RuntimeError("mapping died"))
    world.post = second_post
    second = _run(tmp_path, out_dir=tmp_path / "two")
    assert "quarantine_report.json" not in second.report_paths
    assert "quarantine_report" not in decisions[1] and decisions[2] is False


def test_the_pop_and_reset_happen_again_before_the_audit(world, tmp_path):
    def extract(request, reporter, state):
        world.stale.parent.mkdir(parents=True, exist_ok=True)
        world.stale.write_text("{}", encoding="utf-8")
        state["post_pipeline_artifacts"] = {"from": "stage two"}
        state["quarantine_report"] = {"stale": True}
        return ExtractionResult(final_payload=PAYLOAD)

    seen = []

    def post(request, state, reporter):  # recorded, not asserted: run_pathway catches exceptions
        seen.append(("post_pipeline_artifacts" in state, "quarantine_report" in state))
        return world.default_post(request, state, reporter)
    world.extract, world.post = extract, post
    _run(tmp_path)
    assert seen == [(False, False)]
    assert world.seen_stale == [False]


def test_a_fresh_state_for_each_run(world, tmp_path):
    states = []

    def extract(request, reporter, state):
        states.append(state)
        return ExtractionResult(failure=InputFailure(ex.KIND_NO_TEXT, "x"))
    world.extract = extract
    _run(tmp_path)
    _run(tmp_path)
    assert states[0] is not states[1]


def test_reset_quarantine_state_clears_the_keys_and_survives_a_locked_file(tmp_path, monkeypatch):
    state = {key: 1 for key in pr.QUARANTINE_STATE_KEYS} | {"other": 2}
    pr.reset_quarantine_state(None, state=state)
    assert state == {"other": 2}

    def locked(_):
        raise OSError("locked")
    monkeypatch.setattr(pr, "clear_quarantine_artifacts", locked)
    pr.reset_quarantine_state(tmp_path, state=state)


# -- the review state, the gate and the reporter -----------------------------------------
def test_open_refinement_review_writes_what_the_export_reads(monkeypatch):
    from t2pw.curation import interactive_curator as curator
    from t2pw.pipeline import draft_graph, draft_graph_render, qa_graph

    graph = SimpleNamespace(to_dict=lambda: {"g": 1})
    monkeypatch.setattr(draft_graph, "build_draft_graph", lambda payload: graph)
    monkeypatch.setattr(draft_graph_render, "render_draft_graph_to_png_bytes", lambda d, dpi: b"png")
    monkeypatch.setattr(qa_graph, "generate_qa_report", lambda g, p: {"qa": 1})
    monkeypatch.setattr(curator, "compact_mapping_misses", lambda report: ["miss"])
    payload, report = {"p": [1]}, {"r": [1]}
    state = {}
    pr.open_refinement_review(payload, report, state=state)
    assert state == {
        "refinement_working_json": payload, "refinement_graph_dict": {"g": 1}, "refinement_graph_bytes": b"png",
        "refinement_qa_report": {"qa": 1}, "refinement_mapping_report": report,
        "refinement_mapping_misses": ["miss"], "refinement_history": [], "refinement_round": 0,
        "refinement_pwml_ready": False, "refinement_last_error": None, "refinement_last_warnings": [],
        "refinement_gate_errors": [], "refinement_checkpoints": [],
    }
    assert state["refinement_working_json"] is not payload and state["refinement_mapping_report"] is not report


AMBIGUOUS = {"document_type": "multi_example_review", "selected_example": "",
             "candidate_examples": [{"name": "A"}], "warning": "  watch out "}


def test_the_gate_refuses_and_writes_the_apps_state():
    state, reporter = {}, CollectingReporter()
    failure = ex.ambiguous_review_gate(AMBIGUOUS, state=state, reporter=reporter)
    assert failure.kind == "ambiguous_review_scope"
    assert state["pipeline_ready"] is False and state["pathway_context"] is AMBIGUOUS
    assert state["pipeline_error"]["status"] == "ambiguous_review_scope"
    assert state["pipeline_error"]["candidate_examples"] == [{"name": "A"}]
    assert reporter.errors[0].startswith("Ambiguous review scope:") and reporter.warnings == ["watch out"]


def test_the_gate_passes_a_clear_context():
    state = {}
    assert ex.ambiguous_review_gate({"document_type": "primary"}, state=state, reporter=CollectingReporter()) is None
    assert state == {}


def test_run_extraction_applies_the_gate_before_stage_one(monkeypatch):
    monkeypatch.setattr(ex, "assemble_inputs", lambda *a, **k: SimpleNamespace(failure=None, text="t"))
    monkeypatch.setattr(ex, "run_stage_zero",
                        lambda *a, **k: SimpleNamespace(pathway_context=AMBIGUOUS, diagnostics_dir=None))
    monkeypatch.setattr(ex, "run_stage_one", lambda *a, **k: pytest.fail("Stage 1 ran past the gate"))
    request = SimpleNamespace(export_mode="pathwhiz", user_task_context="", text_entries=[], pdfs=[],
                              uses_pdf_input=False, input_mode="Paste text", pdf_page_range="", pdf_skip_refs=True,
                              pdf_ocr=False, temperature=0.0)
    result = ex.run_extraction(request, reporter=LoggingReporter(), state={})
    assert result.failure.kind == ex.KIND_AMBIGUOUS_REVIEW_SCOPE


def test_the_collecting_reporter_reads_like_the_app_text():
    reporter = CollectingReporter()
    reporter.warning("w")
    reporter.error(" e ")
    reporter.exception(ValueError("x"))
    assert reporter.text_seen() == "e\nx\nw"
    reporter.clear()
    assert reporter.text_seen() == ""


def test_subheader_forwards_and_logs(caplog):
    calls = []
    StreamlitReporter(SimpleNamespace(subheader=calls.append)).subheader("Candidate examples")
    assert calls == ["Candidate examples"]
    with caplog.at_level("INFO", logger="t2pw.service.reporter"):
        LoggingReporter().subheader("Candidate examples")
    assert "Candidate examples" in caplog.text


# -- defaults ---------------------------------------------------------------------------
def _widget_values() -> dict:
    """The ``value=`` expression of every form widget that names a run setting, by label."""

    source = APP.read_text(encoding="utf-8")
    found = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            for keyword in node.keywords:
                if keyword.arg == "value":
                    found[node.args[0].value] = ast.get_source_segment(source, keyword.value)
    return found


FORM = {
    "Run inference/enrichment stage": "RUN_INFERENCE", "Enable automatic chunking for long inputs": "ENABLE_CHUNKING",
    "Multi-paper RAG (novel pathway from multiple papers)": "RAG_INCOMPLETE_FLAG",
    "Stage 1 auto-repair attempts": "EXTRACT_ATTEMPTS", "Stage 2 auto-repair attempts": "INFER_ATTEMPTS",
    "LLM temperature": "TEMPERATURE", "Stage 2 QA rounds": "INFER_ROUNDS", "Stage 1 max tokens": "extract_tokens()",
    "Stage 2 max tokens": "infer_tokens()", "Chunk size (approx. words)": "CHUNK_SIZE",
    "Chunk overlap (words)": "CHUNK_OVERLAP", "Use LLM in audit stage": "USE_LLM_AUDIT",
    "Use SBML semantic overwatch for legacy export": "USE_SBML_OVERWATCH",
    "Stoichiometry agent (fill missing ATP/ADP/NAD etc.)": "USE_STOICH_AGENT",
    "Default compartment": "DEFAULT_COMPARTMENT", "ID mapping cache path": "MAPPING_CACHE_PATH",
    "Audit repair max rounds": "AUDIT_MAX_ROUNDS", "Audit repair timeout (seconds)": "AUDIT_TIMEOUT_SECONDS",
    "Audit candidates / round": "AUDIT_CANDIDATE_COUNT", "Use SBML motif retrieval during audit": "USE_EXAMPLE_RETRIEVAL",
    "SBML motif index path": "EXAMPLE_INDEX_PATH", "SBML motifs top-k": "EXAMPLE_TOP_K",
    "Use Stage 3 Targeted Resolution (LLM-planned, code-executed)": "USE_GAP_RESOLVER",
    "Use LLM in Stage 3 planner/selection": "USE_LLM_GAP_RESOLVER", "Gap resolver max entities": "GAP_RESOLVER_MAX_ITEMS",
    "DB host": "db_host()", "DB port": "db_port()", "DB user": "db_user()", "DB schema": "db_schema()",
    "DB password": "db_password()",
}


@pytest.mark.parametrize("label,name", sorted(FORM.items()))
def test_each_form_widget_reads_its_shared_default(label, name):
    assert _widget_values()[label] == f"_defaults.{name}"


def test_the_id_source_selectbox_reads_the_shared_default():
    source = APP.read_text(encoding="utf-8")
    assert "index=list(_defaults.ID_SOURCES).index(_defaults.id_source())" in source


#: The form's values before R5, verbatim: a changed constant changes every benchmark.
SNAPSHOT = {
    "RUN_INFERENCE": True, "ENABLE_CHUNKING": True, "RAG_INCOMPLETE_FLAG": True, "EXTRACT_ATTEMPTS": 2,
    "INFER_ATTEMPTS": 2, "INFER_ROUNDS": 2, "TEMPERATURE": 0.0, "CHUNK_SIZE": 10000, "CHUNK_OVERLAP": 1600,
    "USE_LLM_AUDIT": True, "USE_SBML_OVERWATCH": False, "USE_STOICH_AGENT": False, "DEFAULT_COMPARTMENT": "cell",
    "MAPPING_CACHE_PATH": "data/id_mapping_cache.json", "AUDIT_MAX_ROUNDS": 4, "AUDIT_TIMEOUT_SECONDS": 240,
    "AUDIT_CANDIDATE_COUNT": 3, "USE_EXAMPLE_RETRIEVAL": False, "EXAMPLE_INDEX_PATH": "src/tmp/sbml_motif_index.json",
    "EXAMPLE_TOP_K": 3, "USE_GAP_RESOLVER": True, "USE_LLM_GAP_RESOLVER": True, "GAP_RESOLVER_MAX_ITEMS": 80,
}


def test_the_defaults_are_the_forms_historic_values():
    assert {name: getattr(defaults, name) for name in SNAPSHOT} == SNAPSHOT


@pytest.mark.parametrize("raw,want", [(None, "hybrid"), ("", "hybrid"), (" DB ", "db"), ("nonsense", "hybrid")])
def test_id_source_follows_the_environment_as_the_form_did(monkeypatch, raw, want):
    monkeypatch.delenv("PATHBANK_ID_SOURCE", raising=False)
    if raw is not None:
        monkeypatch.setenv("PATHBANK_ID_SOURCE", raw)
    assert defaults.id_source() == want


@pytest.mark.parametrize("raw,want", [(None, 64000), ("40000", 40000), ("bad", 64000), ("9", 500), ("999999", 128000)])
def test_token_defaults_are_bounded(monkeypatch, raw, want):
    monkeypatch.delenv("OPENROUTER_EXTRACTION_MAX_TOKENS", raising=False)
    if raw is not None:
        monkeypatch.setenv("OPENROUTER_EXTRACTION_MAX_TOKENS", raw)
    assert defaults.extract_tokens() == want


def test_run_pathway_states_the_shared_values_to_each_step(world, tmp_path, monkeypatch):
    for name, value in (("CHUNK_SIZE", 123), ("AUDIT_MAX_ROUNDS", 9), ("USE_LLM_AUDIT", False), ("MAPPING_CACHE_PATH", "m")):
        monkeypatch.setattr(defaults, name, value)
    monkeypatch.setenv("PATHBANK_DB_PORT", "4000")
    monkeypatch.setenv("PATHBANK_ID_SOURCE", "api")
    _run(tmp_path)
    assert world.request.chunk_size == 123
    post = world.post_requests[0]
    assert (post.audit_max_rounds, post.use_llm_audit, post.mapping_cache_text) == (9, False, "m")
    assert (post.db_port, post.id_source_mode) == (4000, "api")


# -- delivered files --------------------------------------------------------------------
def test_the_rendered_diagram_is_saved_and_filled(world, tmp_path):
    res = _run(tmp_path)
    assert res.image_path == tmp_path / "out" / pw.IMAGE_NAME
    assert res.image_path.read_bytes() == b"\x89PNG"
    assert pw.IMAGE_NAME not in res.report_paths


def test_no_diagram_when_review_never_rendered_one(world, tmp_path):
    world.extract = _fail(ex.KIND_NO_TEXT, "x")
    assert _run(tmp_path).image_path is None


def test_an_unwritable_diagram_is_a_warning_not_a_lost_run(world, tmp_path, monkeypatch):
    real = Path.write_bytes

    def refuse(self, data):
        if self.name == pw.IMAGE_NAME:
            raise OSError("disk full")
        return real(self, data)
    monkeypatch.setattr(Path, "write_bytes", refuse)
    res = _run(tmp_path)
    assert res.status == "pass" and res.image_path is None
    assert "the pathway diagram could not be written" in res.warnings


def _boundary_report(world, tmp_path, value: str):
    from t2pw.pipeline import extraction_diagnostics as diag

    def extract(request, reporter, state):
        state["extraction_diagnostics"] = {diag.BOUNDARY_REPORT_NAME: {"report_path": value, "pointer": "/processes/reactions/0"}}
        return ExtractionResult(failure=InputFailure(ex.KIND_STAGE_ONE_INCOMPLETE, "x"))
    world.extract = extract
    return _run(tmp_path)


@pytest.mark.parametrize("sep", ["native", "posix"])
def test_absolute_server_paths_are_cut_from_a_delivered_report(world, tmp_path, sep):
    from t2pw.pipeline import extraction_diagnostics as diag

    target = tmp_path / "tmp" / "diag" / "report.json"
    res = _boundary_report(world, tmp_path, str(target) if sep == "native" else target.as_posix())
    text = res.report_paths[diag.BOUNDARY_REPORT_NAME].read_text(encoding="utf-8")
    report = json.loads(text)
    assert Path(report["report_path"].replace("\\", "/")) == Path("tmp/diag/report.json")
    assert tmp_path.name not in text
    assert report["pointer"] == "/processes/reactions/0", "a JSON pointer is not a path"


def test_the_pwml_bytes_are_never_scrubbed(tmp_path):
    blob = str(tmp_path).encode()
    assert pw._scrub_server_paths({"pathway.pwml": blob}, tmp_path) == {"pathway.pwml": blob}


def test_scrubbing_covers_the_temp_dir_and_home(tmp_path):
    import tempfile

    text = f"{tempfile.gettempdir()}/a {Path.home() / 'b'}"
    out = pw._scrub_server_paths({"r.txt": text}, tmp_path)["r.txt"]
    assert tempfile.gettempdir() not in out and str(Path.home()) not in out and out.split() == ["a", "b"]


def test_a_write_failure_on_a_required_artifact_is_a_crash(world, tmp_path, monkeypatch):
    from t2pw.batch import runner

    real = runner.write_artifacts

    def lose(out_dir, artifacts):
        return [{**row, "error": "disk full"} if row["name"] == "pathway.pwml" else row for row in real(out_dir, artifacts)]
    monkeypatch.setattr(runner, "write_artifacts", lose)
    res = _run(tmp_path)
    assert (res.status, res.failure_kind) == ("fail", "crash")
