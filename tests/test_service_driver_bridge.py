"""NEW ACCEPTANCE TEST (BR, new capability): the driver bridge ``t2pw.service.driver_bridge``.

``batch.driver.run_one`` and the W1 adapters are stubbed: no AppTest, no LLM, no network,
no database. Every result is also checked through ``jobs.result_to_dict`` against the
real W4 retry rule (``retry.decide``) and W7 qualification (``curator_queue.qualify``),
because those read ``status``, ``failure_kind``, ``pwml_path`` and ``release_status``.
"""

from __future__ import annotations

import functools
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import fields
from pathlib import Path

import pytest

from t2pw.batch import driver
from t2pw.batch.driver import RunOutcome
from t2pw.batch.fetch import BatchPaper
from t2pw.pipeline.release_status import (
    DIAGNOSTIC_ONLY,
    DISPOSITION_EXTRACTED_NOT_SERIALIZED,
    RELEASE_READY,
    ReleaseStatus,
    classify_release_status,
)
from t2pw.rag.eligibility import RequestedScope
from t2pw.service import driver_bridge as br
from t2pw.service import inputs
from t2pw.service.pathway import (
    KIND_CONTRACT,
    KIND_CRASH,
    KIND_NETWORK,
    KIND_NO_REACTIONS,
    KIND_TIMEOUT,
    STAGE_INIT,
    STAGE_PWML_EXPORT,
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_TIMEOUT,
    PathwayRequest,
    PathwayResult,
)
from t2pw.web import curator_queue, worker
from t2pw.web.jobs import request_to_dict, result_to_dict
from t2pw.web.retry import decide

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
SCOPE = RequestedScope(requested_pathway="lipid A biosynthesis",
                       requested_organism="Escherichia coli")


class FakeDriver:
    """Stands in for ``driver.run_one``; records every call."""

    def __init__(self, outcome: RunOutcome | None = None):
        self.outcome = outcome or RunOutcome(status=STATUS_FAIL, failure_kind=KIND_NO_REACTIONS)
        self.calls = []

    def __call__(self, paper, mode, **kwargs):
        self.calls.append((paper, mode, kwargs))
        return self.outcome


@pytest.fixture
def fake(monkeypatch):
    f = FakeDriver()
    monkeypatch.setattr(driver, "run_one", f)
    return f


def _req(tmp_path, name="out", **kw) -> PathwayRequest:
    kw.setdefault("scope", SCOPE)
    return PathwayRequest(out_dir=tmp_path / name, **kw)


def _retry(res: PathwayResult) -> bool:
    return decide(result_to_dict(res), 1, 0.0).retry


def _pass_outcome(release, name="pathway.pwml") -> RunOutcome:
    return RunOutcome(
        paper_id="web:text1", mode="strict", status=STATUS_PASS, stage=STAGE_PWML_EXPORT,
        seconds=12.5, message="strict run completed", detail="LONG DETAIL",
        issue_codes=["c1"], counts={"reactions": 4}, warnings=["w1"],
        artifacts={name: b"<pwml/>", "final_mapped.json": "{}",
                   "quarantine_report.json": '{"q": 1}'},
        release_status=release, pwml_artifact=name,
        observed_context={"observed_pathways": ["lipid A"]})


# ── import, worker default ────────────────────────────────────────────────


def test_import_loads_no_streamlit_app_or_driver():
    """Importing the bridge (fresh process) loads no streamlit, no app, no batch driver."""
    code = ("import sys, t2pw.service.driver_bridge; "
            "bad = [m for m in sys.modules if m.split('.')[0] == 'streamlit' "
            "or m.startswith(('t2pw.app', 't2pw.batch'))]; print(bad); sys.exit(bool(bad))")
    env = dict(os.environ, PYTHONPATH=str(SRC_DIR))
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr


def test_worker_default_is_the_bridge_and_the_stub_stays_selectable(tmp_path):
    assert worker.DEFAULT_RUN_FN == "t2pw.service.pathway:run_pathway"
    for spec, fn in (("t2pw.service.driver_bridge:run_pathway", br.run_pathway),
                     ("t2pw.service.pathway:run_pathway", importlib.import_module(
                         "t2pw.service.pathway").run_pathway)):
        module, _, attr = spec.partition(":")
        assert getattr(importlib.import_module(module), attr) is fn
    assert worker.Worker(object(), tmp_path, postrun_cfg=object()).run_fn == worker.DEFAULT_RUN_FN


def test_worker_child_runs_the_bridge_end_to_end(tmp_path, fake):
    """The worker's child entry point, in-process, with the default run function."""
    fake.outcome = _pass_outcome(classify_release_status(strict_gates_passed=True).to_dict())
    req = _req(tmp_path, texts=("some text",), deadline=time.time() + 3600)
    rq, rs = tmp_path / "request.json", tmp_path / "result.json"
    rq.write_text(json.dumps(request_to_dict(req)), encoding="utf-8")
    assert worker._child_main("t2pw.service.driver_bridge:run_pathway", str(rq), str(rs)) == 0
    got = json.loads(rs.read_text(encoding="utf-8"))
    assert got["status"] == STATUS_PASS and got["pwml_path"] == str(tmp_path / "out" / "pathway.pwml")
    assert len(fake.calls) == 1


# ── request -> driver ─────────────────────────────────────────────────────


def test_several_articles_are_joined_exactly_as_the_app_joins_them(tmp_path, fake, monkeypatch):
    """texts, then PMIDs, then PDFs; stripped; empty boxes skipped; '\\n\\n' joiner."""
    seen = {}

    def fake_pmids(pmids, **kw):
        seen["pmid_fetch"] = kw["fetch_text_fn"]
        return [inputs.PmidResult(p, inputs.OK, text=f"  pmid {p} text \n") for p in pmids]

    def fake_pdfs(paths, *, parse_fn):
        seen["parse_fn"] = parse_fn
        return [inputs.PdfResult(str(p), inputs.OK, text=f"pdf {Path(p).name}") for p in paths]

    monkeypatch.setattr(inputs, "resolve_pmids", fake_pmids)
    monkeypatch.setattr(inputs, "read_pdfs", fake_pdfs)
    req = _req(tmp_path, texts=("  first \n", "   ", "second"), pmids=("111", "222"),
               pdf_paths=(Path("u/a.pdf"),), export_mode="research")
    br.run_pathway(req)

    (paper, mode, kwargs), = fake.calls
    entries = ["  first \n", "   ", "second", " pmid 111 text \n", "pmid 222 text", "pdf a.pdf"]
    app_text = "\n\n".join(e.strip() for e in entries if e.strip())  # the app's rule
    assert paper["full_text"] == app_text
    assert paper["full_text"].strip() == paper["full_text"]  # one box gives the app the same text
    assert paper["requested_pathway"] == SCOPE.requested_pathway
    assert paper["requested_organism"] == SCOPE.requested_organism
    assert paper["paper_id"] == "web:text1+text2+PMID111+PMID222+a.pdf"
    assert mode == "research" and kwargs == {}  # no deadline -> the driver's own default
    # PDFs are parsed with the app's defaults: OCR fallback off, references skipped.
    assert isinstance(seen["parse_fn"], functools.partial)
    assert seen["parse_fn"].func is inputs.parse_pdf
    assert seen["parse_fn"].keywords == {"enable_ocr_fallback": False}
    # Focus derived by the driver from the request, as for a batch topic leg.
    assert driver._extraction_focus(paper) == "lipid A biosynthesis in Escherichia coli"


def test_app_still_joins_text_parts_that_way():
    """Guard: if the app's combining rule changes, the bridge must change with it.

    Searches the app package and the service package, so a pure R3 move of the handler
    out of ``streamlit_app.py`` still passes and a rule change still fails.
    """
    pkg = SRC_DIR / "t2pw"
    src = "\n".join(p.read_text(encoding="utf-8")
                    for d in ("app", "service") for p in sorted((pkg / d).rglob("*.py"))
                    if p.name != "driver_bridge.py")  # never let the bridge vouch for itself
    assert 'text_parts = [entry.strip() for entry in text_entries if entry.strip()]' in src
    assert 'text_parts.append(_pdf["text"].strip())' in src
    assert 'text = "\\n\\n".join(text_parts)' in src
    assert br.APP_JOIN == "\n\n"


def test_name_mode_runs_one_fetched_paper_unchanged(tmp_path, fake, monkeypatch):
    paper = BatchPaper(paper_id="PMC1", requested_pathway=SCOPE.requested_pathway,
                       requested_organism=SCOPE.requested_organism,
                       topic=SCOPE.requested_pathway, full_text="body")
    seen = {}

    def fake_name(pathway, organism, count, **kw):
        seen.update(pathway=pathway, organism=organism, count=count)
        return inputs.NameInputs(spec=None, scope=None, papers=[paper])

    monkeypatch.setattr(inputs, "resolve_name", fake_name)
    br.run_pathway(_req(tmp_path))
    assert seen == {"pathway": SCOPE.requested_pathway, "organism": SCOPE.requested_organism,
                    "count": 1}
    assert fake.calls[0][0] == paper and fake.calls[0][1] == "strict"  # same record, same text


@pytest.mark.parametrize("skipped,status,kind,retry", [
    ([{"reason": "no_full_text"}], STATUS_FAIL, KIND_NO_REACTIONS, False),
    ([{"reason": "search_failed"}], STATUS_ERROR, KIND_NETWORK, True),
])
def test_name_mode_with_no_paper(tmp_path, fake, monkeypatch, skipped, status, kind, retry):
    monkeypatch.setattr(inputs, "resolve_name", lambda *a, **k: inputs.NameInputs(
        spec=None, scope=None, papers=[], skipped=skipped, message="W1 MESSAGE"))
    res = br.run_pathway(_req(tmp_path))
    assert (res.status, res.failure_kind, res.message) == (status, kind, "W1 MESSAGE")
    assert res.issue_codes == [br.CODE_NO_INPUT] and _retry(res) is retry and not fake.calls


@pytest.mark.parametrize("kw", [
    dict(export_mode="Strict PWML"),
    dict(scope=RequestedScope(requested_organism="Homo sapiens"), texts=("t",)),
    dict(scope=RequestedScope(requested_pathway="x", pinned=True), texts=("t",)),
    dict(extraction_focus="only the first half", texts=("t",)),
    dict(scope=RequestedScope()),
])
def test_unhonourable_fields_fail_loudly_and_are_final(tmp_path, fake, kw):
    res = br.run_pathway(_req(tmp_path, **kw))
    assert (res.status, res.stage, res.failure_kind) == (STATUS_ERROR, STAGE_INIT, KIND_CONTRACT)
    assert res.message and res.issue_codes == [br.CODE_UNSUPPORTED]
    assert not _retry(res) and not fake.calls


def test_extraction_focus_equal_to_the_derived_one_is_accepted(tmp_path, fake):
    br.run_pathway(_req(tmp_path, texts=("t",),
                        extraction_focus="lipid A biosynthesis in Escherichia coli"))
    assert len(fake.calls) == 1


@pytest.mark.parametrize("status,kind,retry", [
    (inputs.LOOKUP_FAILED, KIND_NETWORK, True),
    (inputs.NO_FULL_TEXT, KIND_NO_REACTIONS, False),
    (inputs.NO_PMCID, KIND_NO_REACTIONS, False),
    (inputs.INVALID, KIND_CONTRACT, False),
])
def test_a_pmid_without_text_fails_the_job(tmp_path, fake, monkeypatch, status, kind, retry):
    monkeypatch.setattr(inputs, "resolve_pmids", lambda pmids, **k: [
        inputs.PmidResult("1", inputs.OK, text="good"),
        inputs.PmidResult("2", status, message="PMID 2 MESSAGE")])
    res = br.run_pathway(_req(tmp_path, pmids=("1", "2")))
    assert res.failure_kind == kind and res.message == "PMID 2 MESSAGE"
    assert _retry(res) is retry and not fake.calls


def test_a_permanent_input_problem_outranks_a_retryable_one(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(inputs, "resolve_pmids", lambda pmids, **k: [
        inputs.PmidResult("1", inputs.LOOKUP_FAILED, message="down"),
        inputs.PmidResult("2", inputs.NO_FULL_TEXT, message="no oa")])
    res = br.run_pathway(_req(tmp_path, pmids=("1", "2")))
    assert res.failure_kind == KIND_NO_REACTIONS and not _retry(res)
    assert res.message == "no oa; down"  # both inputs' messages, highest-ranked first


def _batch_leg_text(tmp_path, text: str) -> str:
    """What a batch leg hands the driver: the runner's own write, then its own read."""
    from t2pw.batch import runner

    path = tmp_path / "01_source_text.txt"
    runner._write_text(path, text)
    got, err = runner._read_text_or_error(path)
    assert err == ""
    return got


RAW = "line one\r\nline two\rline three\nend\r\n\r\nlast"


def test_name_mode_text_is_byte_identical_to_a_batch_leg(tmp_path, fake, monkeypatch):
    """Review F3: the batch reads its source text back with universal newlines."""
    paper = BatchPaper(paper_id="PMC1", requested_pathway="x", topic="x", full_text=RAW)
    monkeypatch.setattr(inputs, "resolve_name", lambda *a, **k: inputs.NameInputs(
        spec=None, scope=None, papers=[paper]))
    br.run_pathway(_req(tmp_path, scope=RequestedScope(requested_pathway="x")))
    sent = fake.calls[0][0]
    assert sent.full_text == _batch_leg_text(tmp_path, RAW) and "\r" not in sent.full_text
    assert paper.full_text == RAW  # the fetched record itself is not mutated


def test_source_mode_text_is_byte_identical_to_a_batch_leg(tmp_path, fake):
    br.run_pathway(_req(tmp_path, texts=(RAW, "second\r\npart")))
    joined = br.APP_JOIN.join([RAW.strip(), "second\r\npart"])
    assert fake.calls[0][0]["full_text"] == _batch_leg_text(tmp_path, joined)
    assert "\r" not in fake.calls[0][0]["full_text"]


def test_a_failed_run_record_write_keeps_the_mapped_result(tmp_path, fake):
    """Review F6: a disk fault on run_outcome.json must not turn a final gate refusal
    into a retried crash."""
    rs = classify_release_status(pipeline_executed=True, strict_gates_passed=False)
    fake.outcome = RunOutcome(status=STATUS_FAIL, failure_kind=KIND_CONTRACT, message="gate",
                              release_status=rs)
    (tmp_path / "out" / br.OUTCOME_NAME).mkdir(parents=True)  # the write cannot happen
    res = br.run_pathway(_req(tmp_path, texts=("t",)))
    assert (res.status, res.failure_kind, res.release_status) == (STATUS_FAIL, KIND_CONTRACT, rs)
    assert f"the run record {br.OUTCOME_NAME} could not be written" in res.warnings
    assert res.attempt_log and not _retry(res)


@pytest.mark.parametrize("status,kind", [(inputs.EMPTY, KIND_NO_REACTIONS),
                                          (inputs.ERROR, KIND_CONTRACT)])
def test_an_unreadable_pdf_fails_the_job(tmp_path, fake, monkeypatch, status, kind):
    monkeypatch.setattr(inputs, "read_pdfs", lambda paths, parse_fn: [
        inputs.PdfResult("a.pdf", status, message="PDF MESSAGE")])
    res = br.run_pathway(_req(tmp_path, pdf_paths=(Path("a.pdf"),)))
    assert (res.failure_kind, res.message) == (kind, "PDF MESSAGE")
    assert not _retry(res) and not fake.calls


def test_deadline_becomes_the_driver_timeout_less_the_batch_grace(tmp_path, fake):
    br.run_pathway(_req(tmp_path, texts=("t",), deadline=time.time() + 1000))
    timeout = fake.calls[0][2]["timeout"]
    assert 1000 - br.GRACE_S - 5 < timeout <= 1000 - br.GRACE_S


def test_a_passed_deadline_is_a_final_timeout_without_a_run(tmp_path, fake):
    res = br.run_pathway(_req(tmp_path, texts=("t",), deadline=time.time() - 1))
    assert (res.status, res.failure_kind) == (STATUS_TIMEOUT, KIND_TIMEOUT)
    assert not _retry(res) and not fake.calls  # Q-E2E-1: a timeout is final


# ── RunOutcome -> result ──────────────────────────────────────────────────


def test_strict_pass_maps_every_field(tmp_path, fake):
    # Every ReleaseStatus field set away from its default, so a dropped field shows.
    record = ReleaseStatus(
        status=RELEASE_READY, pipeline_executed=True, strict_gates_passed=True,
        semantic_evaluation="failed", semantic_not_evaluated_reason="why",
        semantic_failed_checks=("k",), semantic_check_evaluability=(("c", True, ""),),
        strict_acceptance_eligible=True, completeness=0.9, missing_anchors=("m",),
        retrieval_attempts=2, expansion_blocked_reason="e", coverage_evaluated=True,
        reasons=("r",), disposition=DISPOSITION_EXTRACTED_NOT_SERIALIZED).to_dict()
    assert len(record) == len(fields(ReleaseStatus))
    frozen = {**record, "applied": True, "review_reasons": [], "retrieval_attempts_source": "x"}
    fake.outcome = _pass_outcome(frozen)
    res = br.run_pathway(_req(tmp_path, texts=("t",)))
    out = tmp_path / "out"

    assert (res.status, res.stage, res.failure_kind, res.message) == (
        STATUS_PASS, STAGE_PWML_EXPORT, "", "strict run completed")
    assert res.warnings == ["w1"] and res.issue_codes == ["c1"] and res.counts == {"reactions": 4}
    assert isinstance(res.release_status, ReleaseStatus)
    assert res.release_status.to_dict() == record  # lossless for every ReleaseStatus field
    assert res.pwml_path == out / "pathway.pwml" and res.pwml_path.read_bytes() == b"<pwml/>"
    assert res.report_paths == {"final_mapped.json": out / "final_mapped.json",
                                "quarantine_report.json": out / "quarantine_report.json"}
    assert res.image_path is None  # the driver renders no image
    # Fields with no PathwayResult slot: the driver's own row, on disk and in attempt_log.
    row = json.loads((out / br.OUTCOME_NAME).read_text(encoding="utf-8"))
    assert row["detail"] == "LONG DETAIL" and row["seconds"] == 12.5
    assert row["release_status"]["applied"] is True
    (entry,) = res.attempt_log
    assert entry["attempt"] == 1 and entry["paper_id"] == "web:text1" and "detail" not in entry
    assert entry["observed_context"] == {"observed_pathways": ["lipid A"]}
    assert entry["release_status"]["retrieval_attempts_source"] == "x"
    assert br.OUTCOME_NAME not in res.report_paths

    d = result_to_dict(res)
    json.dumps(d)  # the worker's result.json
    assert not decide(d, 1, 0.0).retry
    item = {"state": "finished", "request": PathwayRequest(), "result": d}
    assert curator_queue.qualify({"consent": 1}, item) == (True, "ok")


def test_review_required_pwml_is_mapped_and_never_queued(tmp_path, fake):
    rec = classify_release_status(strict_gates_passed=True).to_dict()
    rec["status"] = "review_required"
    fake.outcome = _pass_outcome(rec, name="pathway.review_required.pwml")
    res = br.run_pathway(_req(tmp_path, texts=("t",)))
    assert res.pwml_path == tmp_path / "out" / "pathway.review_required.pwml"
    assert "pathway.review_required.pwml" not in res.report_paths
    item = {"state": "finished", "request": PathwayRequest(), "result": result_to_dict(res)}
    assert curator_queue.qualify({"consent": 1}, item) == (False, "not release_ready")


def test_gate_failure_keeps_its_release_object_and_is_final(tmp_path, fake):
    rs = classify_release_status(pipeline_executed=True, strict_gates_passed=False)
    assert rs.status == DIAGNOSTIC_ONLY
    fake.outcome = RunOutcome(status=STATUS_FAIL, stage="post_pipeline",
                              failure_kind=KIND_CONTRACT, message="gate", release_status=rs,
                              artifacts={"gate_fail_report.json": "{}"})
    res = br.run_pathway(_req(tmp_path, texts=("t",)))
    assert res.release_status is rs and res.pwml_path is None
    assert set(res.report_paths) == {"gate_fail_report.json"}
    assert not _retry(res)


@pytest.mark.parametrize("status,kind,retry", [
    (STATUS_TIMEOUT, KIND_TIMEOUT, False), (STATUS_ERROR, KIND_CRASH, True),
    (STATUS_ERROR, "", True), (STATUS_FAIL, "unknown", False)])
def test_failed_outcome_kinds_pass_through_for_the_retry_rule(tmp_path, fake, status, kind, retry):
    fake.outcome = RunOutcome(status=status, failure_kind=kind, message="m")
    res = br.run_pathway(_req(tmp_path, texts=("t",)))
    assert (res.status, res.failure_kind, res.release_status, res.pwml_path) == (
        status, kind, None, None)
    assert _retry(res) is retry


def test_unreadable_release_record_is_dropped_loudly(tmp_path, fake):
    fake.outcome = _pass_outcome({"status": RELEASE_READY})  # required fields missing
    res = br.run_pathway(_req(tmp_path, texts=("t",)))
    assert res.release_status is None and br.WARN_RELEASE_UNREADABLE in res.warnings


def test_a_lost_pwml_write_fails_the_run_like_the_batch(tmp_path, fake):
    fake.outcome = _pass_outcome(classify_release_status(strict_gates_passed=True).to_dict())
    (tmp_path / "out" / "pathway.pwml").mkdir(parents=True)  # the write cannot happen
    res = br.run_pathway(_req(tmp_path, texts=("t",)))
    assert (res.status, res.failure_kind, res.pwml_path) == (STATUS_FAIL, KIND_CRASH, None)
    assert "pathway.pwml" in res.message


# ── isolation ─────────────────────────────────────────────────────────────


def test_one_jobs_files_cannot_reach_another_jobs_result(tmp_path, fake):
    fake.outcome = _pass_outcome(classify_release_status(strict_gates_passed=True).to_dict())
    first = br.run_pathway(_req(tmp_path, "job1", texts=("t",)))
    assert first.pwml_path.parent == tmp_path / "job1"

    stale = tmp_path / "job2"
    stale.mkdir()
    (stale / "pathway.pwml").write_bytes(b"<stale/>")  # left over, not this run's
    (stale / "final_mapped.json").write_text("stale", encoding="utf-8")
    fake.outcome = RunOutcome(status=STATUS_FAIL, failure_kind=KIND_NO_REACTIONS,
                              artifacts={"../escape.json": "{}"})
    second = br.run_pathway(_req(tmp_path, "job2", texts=("t",)))
    assert second.pwml_path is None
    assert second.report_paths == {"escape.json": stale / "escape.json"}
    assert not (tmp_path / "escape.json").exists()
    for p in second.report_paths.values():
        assert p.parent == stale
    assert (tmp_path / "job1" / "pathway.pwml").read_bytes() == b"<pwml/>"  # untouched


def test_no_out_dir_gets_a_fresh_private_directory(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    fake.outcome = RunOutcome(status=STATUS_FAIL, artifacts={"final_mapped.json": "{}"})
    a, b = (br.run_pathway(PathwayRequest(texts=("t",))) for _ in range(2))
    da, db = (r.report_paths["final_mapped.json"].parent for r in (a, b))
    assert da != db and da.parent == db.parent == tmp_path
