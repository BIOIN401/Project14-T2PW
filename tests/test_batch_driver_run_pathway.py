"""NEW TESTS (R6, new capability): a batch leg runs through the service engine by default.

``driver.run_one_pathway`` is ``run_one``'s twin on ``service.pathway.run_paper`` (the core
``run_pathway`` is built on); ``runner.run_single`` picks it, or the AppTest ``run_one``, from the
engine stored in the run's plan.json. The LLM-facing steps are faked at their source modules (the
same seam ``tests/test_service_run_pathway.py`` uses); no LLM, no network, no Streamlit app. The
real driver verdict logic and the real runner leg writer run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import t2pw.paths as paths
from t2pw.batch import driver, runner
from t2pw.batch.fetch import BatchPaper
from t2pw.pipeline.gate_reports import CANONICAL_PAYLOAD_KEY, FINAL_GATE_REPORT_KEY
from t2pw.pipeline.release_status import classify_release_status
from t2pw.service import extraction as ex
from t2pw.service import pathway as pw
from t2pw.service import post_run as pr
from t2pw.service.extraction import ExtractionResult, InputFailure

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = {"processes": {"reactions": [{"name": "r1"}], "transports": []}, "entities": {"compounds": [{"name": "c"}]}}
XML = b"<pwml/>"
PAPER = {"paper_id": "PMC1", "slug": "PMC1__alpha", "title": "T", "topic": "lipid A biosynthesis",
         "organism": "Escherichia coli", "full_text": "The paper text.\nSecond line."}


def _released() -> dict:
    return {**classify_release_status(pipeline_executed=True, strict_gates_passed=True).to_dict(), "status": "release_ready"}


class World:
    """The three service steps, faked at their source modules; what reached them is recorded."""

    def __init__(self, mp, root: Path) -> None:
        self.calls: List[str] = []
        self.extract = lambda request, reporter, state: ExtractionResult(final_payload=PAYLOAD)
        mp.setattr(paths, "PROJECT_ROOT", root)
        mp.setattr(ex, "run_extraction", self._extract)
        mp.setattr(pr, "run_post_pipeline", self._post)
        import t2pw.service.export as export_module
        mp.setattr(export_module, "run_export", self._export)

    def _extract(self, request, *, reporter, state):
        self.calls.append("extract")
        self.request, self.state = request, state
        return self.extract(request, reporter, state)

    def _post(self, request, *, state, reporter, open_review, render_issue_detail, render_contract_failure):
        self.calls.append("post")
        state["refinement_working_json"] = {"reviewed": True}
        state["post_pipeline_artifacts"] = {CANONICAL_PAYLOAD_KEY: PAYLOAD, FINAL_GATE_REPORT_KEY: {"ok": True}}
        return pr.PostRunResult(final_payload=request.final_payload)

    def _export(self, *, state, reporter, work_dir, project_root):
        self.calls.append("export")
        return {"ok": True, "xml_bytes": XML, "output_path": "x", "counts": {},
                "quarantine_report": {"release": _released()}}


@pytest.fixture
def world(monkeypatch, tmp_path):
    return World(monkeypatch, tmp_path)


def _failing(kind: str, message: str):
    def extract(request, reporter, state):
        reporter.error(message)
        return ExtractionResult(failure=InputFailure(kind, message))
    return extract


def _run_dir(tmp_path: Path, engine: Optional[str] = None, text: str = "The paper text.") -> Path:
    run_dir = tmp_path / "runs" / "2020-01-01_0000"
    folder = run_dir / runner.PAPERS_DIRNAME / "PMC1__alpha"
    folder.mkdir(parents=True)
    (folder / runner.SOURCE_TEXT_NAME).write_text(text, encoding="utf-8")
    plan: Dict[str, Any] = {"modes": ["strict", "research"], "papers": [
        {"paper_id": "PMC1", "slug": "PMC1__alpha", "title": "T", "topic": "lipid A biosynthesis"}]}
    if engine:
        plan["engine"] = engine
    runner.save_plan(run_dir, plan)
    return run_dir


# -- the engine names ---------------------------------------------------------------------
def test_engine_names_default_to_run_pathway_and_reject_a_typo():
    assert driver.resolve_engine(None) == driver.resolve_engine("") == driver.ENGINE_PATHWAY == "run_pathway"
    assert driver.resolve_engine(" AppTest ") == driver.ENGINE_APPTEST == "apptest"
    with pytest.raises(ValueError, match="run_pathway, apptest"):
        driver.resolve_engine("apptset")


def test_only_a_stored_run_pathway_selects_the_service_engine():
    assert driver.engine_fn("run_pathway") is driver.run_one_pathway
    for stored in ("apptest", None, "", "something else"):
        assert driver.engine_fn(stored) is driver.run_one


def test_the_driver_and_the_service_do_not_import_each_other_at_module_level():
    code = ("import sys\n"
            "import t2pw.batch.driver\n"
            "assert 't2pw.service.pathway' not in sys.modules, 'driver imports the service'\n"
            "for name in [m for m in sys.modules if m.startswith('t2pw')]: del sys.modules[name]\n"
            "import t2pw.service.pathway\n"
            "assert 't2pw.batch.driver' not in sys.modules, 'the service imports the driver'\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert done.returncode == 0, done.stderr


# -- run_one_pathway ----------------------------------------------------------------------
def test_a_strict_leg_through_the_service_returns_the_drivers_own_outcome(world):
    outcome = driver.run_one_pathway(PAPER, "strict", timeout=600)
    assert (outcome.status, outcome.stage, outcome.failure_kind) == ("pass", "pwml_export", "")
    assert world.calls == ["extract", "post", "export"]
    assert outcome.paper_id == "PMC1", "the plan's id, not the web's 'web:text1'"
    assert outcome.mode == "strict" and outcome.pwml_artifact == "pathway.pwml"
    assert outcome.artifacts["pathway.pwml"] == XML and outcome.release_status["status"] == "release_ready"
    assert outcome.seconds >= 0 and outcome.counts["reactions"] == 1
    # The input is the paper's text exactly as the plan holds it (no strip, no re-join).
    assert world.request.text_entries == [PAPER["full_text"]]
    assert world.request.user_task_context == "lipid A biosynthesis in Escherichia coli"
    assert world.request.export_mode == "pathwhiz"


def test_a_research_leg_stops_before_the_export(world):
    outcome = driver.run_one_pathway(PAPER, "Research (relaxed)", timeout=600)
    assert (outcome.mode, outcome.status, outcome.stage) == ("research", "pass", "research_report")
    assert "export" not in world.calls and world.request.export_mode == "research"
    assert "review_flags.json" in outcome.artifacts


def test_the_leg_budget_becomes_an_absolute_deadline(world):
    before = time.time()
    driver.run_one_pathway(PAPER, "strict", timeout=300)
    deadline = world.state[ex.DEADLINE_STATE_KEY]
    assert before + 300 <= deadline <= time.time() + 300


def test_a_leg_with_no_text_is_judged_before_any_work_as_the_apptest_path_judges_it(world):
    outcome = driver.run_one_pathway({**PAPER, "full_text": ""}, "strict")
    assert (outcome.status, outcome.failure_kind) == ("fail", "no_reactions")
    assert outcome.message == "paper has no full text, so there was nothing to extract"
    assert world.calls == []


def test_a_stage_one_failure_is_reported_as_the_batch_reports_it(world):
    world.extract = _failing(ex.KIND_STAGE_ONE_FAILED, "provider returned 429 rate limit")
    row = driver.run_one_pathway(PAPER, "strict").to_dict()
    assert (row["status"], row["stage"]) == ("fail", "stage1")
    assert row["failure_kind"] == "llm" and "429" in row["message"]
    assert "post" not in world.calls


def test_a_deadline_the_stage_ran_past_is_a_timeout_row_with_its_clock(world):
    world.extract = lambda request, reporter, state: ExtractionResult(failure=InputFailure(ex.KIND_DEADLINE_EXCEEDED, "late"))
    row = driver.run_one_pathway(PAPER, "strict", timeout=600).to_dict()
    assert (row["status"], row["failure_kind"], row["stage"]) == ("timeout", "timeout", "stage1")
    assert row["termination_reason"] and "operational_failure" in row


def test_a_budget_spent_before_the_audit_is_a_timeout_not_a_verdict(world):
    row = driver.run_one_pathway(PAPER, "strict", timeout=-5).to_dict()
    assert (row["status"], row["failure_kind"]) == ("timeout", "timeout")
    assert world.calls == ["extract"]


def test_a_bug_in_the_run_is_a_crash_outcome_not_an_exception(world, monkeypatch):
    monkeypatch.setattr(pw, "_drive", lambda *a, **k: 1 / 0)
    outcome = driver.run_one_pathway(PAPER, "strict")
    assert (outcome.status, outcome.failure_kind) == ("error", "crash") and "ZeroDivisionError" in outcome.detail
    monkeypatch.setattr(pw, "run_paper", lambda *a, **k: 1 / 0)
    outcome = driver.run_one_pathway(PAPER, "strict")
    assert (outcome.status, outcome.failure_kind) == ("error", "crash") and "ZeroDivisionError" in outcome.detail


# -- run_single: which engine ---------------------------------------------------------------
@pytest.fixture
def engines(monkeypatch):
    seen: List[Any] = []

    def make(name):
        def fn(paper, mode, **kwargs):
            seen.append((name, kwargs.get("timeout")))
            return driver.RunOutcome(paper_id="PMC1", mode="strict", status="pass", stage="pwml_export",
                                     artifacts={"pathway.pwml": b"x"})
        return fn
    monkeypatch.setattr(driver, "run_one", make("apptest"))
    monkeypatch.setattr(driver, "run_one_pathway", make("run_pathway"))
    return seen


@pytest.mark.parametrize("stored,asked,want", [
    ("run_pathway", None, "run_pathway"),
    ("apptest", None, "apptest"),
    (None, None, "apptest"),            # called bare on a plan that names none: as it always did
    (None, "run_pathway", "run_pathway"),
    ("run_pathway", "apptest", "apptest"),
])
def test_run_single_takes_the_engine_from_the_plan_or_the_argument(tmp_path, engines, stored, asked, want):
    runner.run_single(_run_dir(tmp_path, stored), "PMC1__alpha", "strict", timeout=900, engine=asked)
    assert engines == [(want, 900.0)]


def test_run_fn_still_wins_over_every_engine(tmp_path, engines):
    runner.run_single(_run_dir(tmp_path, "run_pathway"), "PMC1__alpha", "strict",
                      run_fn=lambda *a, **k: driver.RunOutcome(status="pass"), engine="apptest")
    assert engines == []


def test_a_leg_through_the_service_writes_the_files_and_row_batch_reads(world, tmp_path):
    run_dir = _run_dir(tmp_path, "run_pathway")
    row = runner.run_single(run_dir, "PMC1__alpha", "strict", timeout=900)
    leg = run_dir / runner.PAPERS_DIRNAME / "PMC1__alpha" / "strict"
    assert (row["status"], row["stage"], row["pwml_artifact"]) == ("pass", "pwml_export", "pathway.pwml")
    assert row["release_status"]["status"] == "release_ready" and row["slug"] == "PMC1__alpha"
    assert (leg / "pathway.pwml").read_bytes() == XML
    assert {"pathway.pwml", "final_mapped.json"} <= {Path(f["name"]).name for f in row["files"]}
    assert (leg / runner.RESULT_NAME).read_text(encoding="utf-8").startswith("RESULT: PASS")
    assert runner.failed_writes(row["files"]) == []


def test_a_failed_leg_through_the_service_is_a_failed_row_with_its_result_file(world, tmp_path):
    world.extract = _failing(ex.KIND_STAGE_ONE_FAILED, "boom")
    run_dir = _run_dir(tmp_path, "run_pathway")
    row = runner.run_single(run_dir, "PMC1__alpha", "strict", timeout=900)
    assert row["status"] == "fail" and row["stage"] == "stage1"
    assert (run_dir / runner.PAPERS_DIRNAME / "PMC1__alpha" / "strict" / runner.RESULT_NAME).read_text(
        encoding="utf-8").startswith("RESULT: FAIL")


def test_a_timed_out_leg_through_the_service_is_a_timeout_row(world, tmp_path):
    world.extract = lambda request, reporter, state: ExtractionResult(failure=InputFailure(ex.KIND_DEADLINE_EXCEEDED, "late"))
    row = runner.run_single(_run_dir(tmp_path, "run_pathway"), "PMC1__alpha", "strict", timeout=900)
    assert (row["status"], row["failure_kind"]) == ("timeout", "timeout") and row["termination_reason"]


# -- run_batch: where the engine comes from and is stored --------------------------------------
def _batch(tmp_path: Path, monkeypatch, *, fresh: bool = True, child=None, env: Optional[str] = None, **kw):
    monkeypatch.delenv(driver.ENGINE_ENV, raising=False)
    if env is not None:
        monkeypatch.setenv(driver.ENGINE_ENV, env)
    paper = BatchPaper(paper_id="PMC9", title="t", source_uri="u", topic="x", full_text="body", slug="PMC9__gamma")
    row = {"paper_id": "PMC9", "slug": "PMC9__gamma", "mode": "strict", "status": "pass", "stage": "pwml_export",
           "seconds": 1.0, "failure_kind": "", "message": "ok", "detail": "", "issue_codes": [], "counts": {}, "files": []}
    ok = lambda cmd, timeout: runner.ChildResult(0, f"{runner.OUTCOME_SENTINEL} {json.dumps(row)}\n", "", False)  # noqa: E731
    code = runner.run_batch(out=tmp_path, modes="strict", fresh=fresh, child_fn=child or ok,
                            fetch_fn=lambda text, limit=None: ([paper], []), script_path=ROOT / "scripts" / "batch_run.py",
                            echo=False, **kw)
    return code, runner.load_plan(runner.latest_run_dir(tmp_path))


def test_a_fresh_batch_defaults_to_run_pathway_and_stores_it(tmp_path, monkeypatch):
    code, plan = _batch(tmp_path, monkeypatch)
    assert code == 0 and plan["engine"] == "run_pathway"
    log = (runner.latest_run_dir(tmp_path) / runner.LOG_NAME).read_text(encoding="utf-8")
    assert "leg engine       : run_pathway" in log


def test_the_environment_or_the_argument_selects_apptest(tmp_path, monkeypatch):
    assert _batch(tmp_path / "a", monkeypatch, env="apptest")[1]["engine"] == "apptest"
    assert _batch(tmp_path / "b", monkeypatch, env="apptest", engine="run_pathway")[1]["engine"] == "run_pathway"
    assert _batch(tmp_path / "c", monkeypatch, engine="AppTest")[1]["engine"] == "apptest"


def test_an_unknown_engine_is_a_usage_error_and_creates_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(driver.ENGINE_ENV, "apptset")
    code = runner.run_batch(out=tmp_path, modes="strict", fresh=True, child_fn=lambda *a: pytest.fail("ran"),
                            fetch_fn=lambda text, limit=None: pytest.fail("fetched"), echo=True)
    assert code == 2 and "apptset" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def _interrupted(cmd, timeout):
    raise KeyboardInterrupt


def test_a_resumed_batch_keeps_the_plans_engine_unless_one_is_given(tmp_path, monkeypatch):
    code, plan = _batch(tmp_path, monkeypatch, child=_interrupted, env="apptest")
    assert code == 1 and plan["engine"] == "apptest"
    code, plan = _batch(tmp_path, monkeypatch, fresh=False)  # no env, no argument: the plan decides
    assert plan["engine"] == "apptest"
    code, plan = _batch(tmp_path, monkeypatch, fresh=False, child=_interrupted, engine="run_pathway")
    assert plan["engine"] == "run_pathway"


def test_a_plan_from_before_r6_resumes_on_the_apptest_path(tmp_path, monkeypatch):
    code, plan = _batch(tmp_path, monkeypatch, child=_interrupted)
    plan.pop("engine")
    runner.save_plan(runner.latest_run_dir(tmp_path), plan)
    code, plan = _batch(tmp_path, monkeypatch, fresh=False)
    assert code == 0 and "engine" not in plan
    log = (runner.latest_run_dir(tmp_path) / runner.LOG_NAME).read_text(encoding="utf-8")
    assert "this plan names none, so the AppTest path" in log
