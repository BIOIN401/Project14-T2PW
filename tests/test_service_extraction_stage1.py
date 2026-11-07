"""R3c: ``t2pw.service.extraction.run_stage_one``, the app's Stage 1 and RAG seed moved out.

NEW TESTS (R3c, new module). ``run_stage_one`` is a pure move of the app's submit block
from the "Running Stage 1 extraction..." spinner through the RAG seed's
``user_task_context`` fold, plus ``render_attempts`` and ``_last_cleaning_report``.
These tests pin what that code did: the Stage-1 and boundary arguments, every message
and its order, the three failure exits (now returned with a kind), the session keys
written into ``state``, the out-of-scope filter, the RAG seed and the fold, and that
every callee is looked up on its SOURCE module at call time (the R3b review F1 rule).

The unit tests are offline: the callees are patched on their source modules and a
recording reporter stands in for Streamlit. The last two tests drive the REAL
``streamlit_app.py`` through ``AppTest``, so this file runs in its own bounded process.
"""

from __future__ import annotations

import inspect
import logging
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
APP_PATH = SRC / "t2pw" / "app" / "streamlit_app.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import t2pw.config as config  # noqa: E402
import t2pw.paths as paths  # noqa: E402
import t2pw.pipeline.draft_graph as draft_graph  # noqa: E402
import t2pw.pipeline.extraction_diagnostics as diag  # noqa: E402
import t2pw.pipeline.pipeline as pipeline  # noqa: E402
import t2pw.pipeline.preprocessor as preprocessor  # noqa: E402
import t2pw.pipeline.qa_graph as qa_graph  # noqa: E402
import t2pw.pipeline.stage_one_boundary as boundary  # noqa: E402
from t2pw.pipeline.stage_contracts import StageContractError  # noqa: E402
from t2pw.service import extraction as ex  # noqa: E402
from t2pw.service.reporter import LoggingReporter, StreamlitReporter  # noqa: E402

DIAG_DIR = Path("C:/diag/run-1")


class _Ctx:
    def __init__(self, calls: List[tuple], name: str) -> None:
        self._calls, self._name = calls, name

    def __enter__(self) -> None:
        self._calls.append((f"{self._name}:enter",))

    def __exit__(self, *exc: Any) -> None:
        self._calls.append((f"{self._name}:exit",))


class RecordingReporter:
    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def __getattr__(self, name: str) -> Any:
        def _call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs) if kwargs else (name, args))
            return _Ctx(self.calls, name) if name in ("spinner", "expander") else None
        return _call


class _Boundary:
    def __init__(self, payload, ok=True, failure=None, reconstruction=None, reason=""):
        self.payload, self.ok, self.failure = payload, ok, failure
        self.reconstruction, self.incomplete_reason = reconstruction or {}, reason

    def to_summary(self):
        return {"ok": self.ok, "reason": self.incomplete_reason}


def _payload(*names, out=()):
    rx = [{"name": n} for n in names] + [{"name": n, "scope_membership": "out_of_scope"} for n in out]
    return {"entities": {}, "processes": {"reactions": rx}}


ATTEMPTS = [{"attempt": 1, "raw": "{bad", "error": "Expecting value", "phase": "extract", "note": "cut"},
            {"attempt": 2, "raw": "{}"}]
CLEANING = {diag.CLEANING_REPORT_NAME: {"passes": [{"pass": 1}, {"pass": 2, "dropped": 3}]}}


@pytest.fixture()
def world(monkeypatch, tmp_path):
    """Every callee patched on its SOURCE module, recording into ``w.calls``."""

    w = types.SimpleNamespace(calls=[], stage1=_payload("a"), chunks=[], boundary={}, artifacts=CLEANING,
                              rag=False, rag_result=None, stage1_raises=None)
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)

    def stage1(text, **kw):
        w.calls.append(("stage1", text, kw))
        if w.stage1_raises is not None:
            raise w.stage1_raises
        return deepcopy(w.stage1), w.chunks

    def settle(payload, **kw):
        w.calls.append(("settle", payload, kw))
        return _Boundary(dict(payload, settled=True), **w.boundary)

    def current():
        w.calls.append(("diag.current",))
        return types.SimpleNamespace(artifacts=lambda: deepcopy(w.artifacts))

    def build(payload):
        w.calls.append(("build_draft_graph", payload))
        return "GRAPH"

    def qa(graph, payload):
        w.calls.append(("generate_qa_report", graph, payload))
        return {"qa": 1}

    def maybe_run_rag(**kw):
        w.calls.append(("maybe_run_rag", kw))
        return w.rag_result

    real_filter = pipeline.filter_out_of_scope_reactions

    def filt(payload):
        w.calls.append(("filter", payload))
        return real_filter(payload)

    import t2pw.service.rag_orchestration as rag

    monkeypatch.setattr(pipeline, "run_stage_one_with_chunking", stage1)
    monkeypatch.setattr(pipeline, "filter_out_of_scope_reactions", filt)
    monkeypatch.setattr(boundary, "settle_stage_one", settle)
    monkeypatch.setattr(diag, "current", current)
    monkeypatch.setattr(draft_graph, "build_draft_graph", build)
    monkeypatch.setattr(qa_graph, "generate_qa_report", qa)
    monkeypatch.setattr(rag, "maybe_run_rag", maybe_run_rag)
    monkeypatch.setattr(config, "rag_config", lambda *a, **k: {"enabled": w.rag})
    w.root = tmp_path
    return w


def _run(w, *, utc="focus", state=None, flag=0, **kw):
    rep, state = RecordingReporter(), ({} if state is None else state)
    args = dict(pathway_context={"pathway_name": "Menaquinone"}, user_task_context=utc,
                diagnostics_dir=DIAG_DIR, enable_chunking=True, chunk_size=1200.0, chunk_overlap=200.0,
                extract_attempts=3.0, temperature=0.25, extract_tokens=8000.0,
                rag_incomplete_flag=flag, state=state, reporter=rep)
    args.update(kw)
    res = ex.run_stage_one("the paper", **args)
    return res, rep, state


SPIN = [("spinner", ("Running Stage 1 extraction...",)), ("spinner:enter",), ("spinner:exit",)]


# ── success path ─────────────────────────────────────────────────────────────
def test_success_passes_stage_one_its_arguments_and_returns_what_the_block_reads(world):
    world.chunks = [{"chunk": 1}]
    res, rep, state = _run(world, state={"export_mode": "research"})

    name, text, kw = world.calls[0]
    assert (name, text) == ("stage1", "the paper")
    assert kw == dict(pathway_context={"pathway_name": "Menaquinone"}, user_task_context="focus",
                      artifact_dir=world.root / "tmp", enable_chunking=True, chunk_word_limit=1200,
                      chunk_overlap=200, max_attempts=3, temperature=0.25, max_tokens=8000)
    assert all(type(kw[k]) is int for k in ("chunk_word_limit", "chunk_overlap", "max_attempts", "max_tokens"))
    settle = world.calls[2]
    assert world.calls[1] == ("diag.current",)  # the cleaning report is read after Stage 1 ran
    assert settle[0] == "settle" and settle[1] == _payload("a")
    assert settle[2] == {"mode": "research", "cleaning_report": {"pass": 2, "dropped": 3}}
    assert rep.calls == SPIN
    assert list(state) == ["export_mode", "stage_one_boundary", ex.EXTRACTION_DIAGNOSTICS_KEY]
    assert state["stage_one_boundary"] == {"ok": True, "reason": ""}
    assert state[ex.EXTRACTION_DIAGNOSTICS_KEY] == CLEANING
    settled = dict(_payload("a"), settled=True)
    assert res.failure is None
    assert res.stage_one == settled and res.stage_one_in_scope == settled
    assert res.chunk_details is world.chunks
    assert res.out_of_scope_removed_reactions == []
    assert res.rag_result is None and res.user_task_context == "focus"


def test_export_mode_is_coerced_and_defaults_when_unset(world):
    _run(world)
    assert [c for c in world.calls if c[0] == "settle"][0][2]["mode"] == "pathwhiz"


def test_cleaning_report_is_empty_when_the_recorder_fails_or_has_no_passes(world, monkeypatch):
    world.artifacts = {}
    _run(world)
    assert [c for c in world.calls if c[0] == "settle"][0][2]["cleaning_report"] == {}

    def boom():
        raise RuntimeError("no recorder")

    monkeypatch.setattr(diag, "current", boom)
    world.calls.clear()
    res, _rep, state = _run(world)
    assert [c for c in world.calls if c[0] == "settle"][0][2]["cleaning_report"] == {}
    assert ex.EXTRACTION_DIAGNOSTICS_KEY not in state and res.failure is None


def test_reconstruction_is_reported(world):
    world.boundary = {"reconstruction": {"shell_count": 2}}
    _res, rep, _state = _run(world)
    assert rep.calls == SPIN + [("info", (
        "Reconstructed 2 unresolved entity shell(s) for participants named by reactions but "
        "missing from the registry. They carry no identifier and no function; every identity "
        "gate downstream still applies to them.",))]


def test_out_of_scope_reactions_are_removed_and_reported(world):
    world.stage1 = _payload("a", out=("x", "y"))
    world.boundary = {"reconstruction": {}}
    res, rep, _state = _run(world)
    assert res.out_of_scope_removed_reactions == ["x", "y"]
    assert res.stage_one_in_scope == dict(_payload("a"), settled=True)
    assert len(res.stage_one["processes"]["reactions"]) == 3  # stage_one keeps the raw output
    assert [c[0] for c in world.calls] == ["stage1", "diag.current", "settle", "diag.current", "filter"]
    assert rep.calls == SPIN + [("info", ("Removed 2 out-of-scope reaction(s) before Stage 2: x, y",))]


# ── the three failure exits ──────────────────────────────────────────────────
def test_pipeline_failure_reports_attempts_and_returns_a_failure(world):
    world.stage1_raises = pipeline.PipelineFailure("stage1", "no JSON after 3 attempts", ATTEMPTS)
    res, rep, state = _run(world)
    assert res.failure == ex.InputFailure(ex.KIND_STAGE_ONE_FAILED, "Extraction failed: no JSON after 3 attempts")
    assert res.stage_one is None and res.user_task_context is None
    assert rep.calls == SPIN + [
        ("error", ("Extraction failed: no JSON after 3 attempts",)),
        ("expander", ("Stage 1 attempts",), {"expanded": False}), ("expander:enter",),
        ("markdown", ("**Attempt 1 (extract)** - retry [cut]",)),
        ("code", ("{bad",), {"language": "json"}),
        ("caption", ("Expecting value",)),
        ("markdown", ("**Attempt 2** - success",)),
        ("code", ("{}",), {"language": "json"}),
        ("expander:exit",)]
    assert state == {ex.EXTRACTION_DIAGNOSTICS_KEY: CLEANING}  # synced before the exit
    assert [c[0] for c in world.calls] == ["stage1", "diag.current"]


def test_contract_error_shows_its_report_and_returns_a_failure(world):
    world.stage1_raises = StageContractError("post_extraction", "Payload must be a dict.", {"errors": ["e"]})
    res, rep, state = _run(world)
    assert res.failure == ex.InputFailure(ex.KIND_STAGE_ONE_BOUNDARY_ERROR,
                                          "Extraction boundary failed: Payload must be a dict.")
    assert rep.calls == SPIN + [("error", ("Extraction boundary failed: Payload must be a dict.",)),
                                ("json", ({"errors": ["e"]},))]
    assert state == {ex.EXTRACTION_DIAGNOSTICS_KEY: CLEANING}


@pytest.mark.parametrize("failure", [None, types.SimpleNamespace(report={"errors": ["r"]})])
def test_incomplete_boundary_reports_and_returns_a_failure(world, failure):
    world.boundary = {"ok": False, "reason": "rows rejected", "failure": failure}
    world.rag = True
    res, rep, state = _run(world)
    message = "Extraction boundary failed and could not be recovered: rows rejected"
    assert res.failure == ex.InputFailure(ex.KIND_STAGE_ONE_INCOMPLETE, message)
    tail = [("json", ({"errors": ["r"]},))] if failure is not None else []
    assert rep.calls == SPIN + [("error", (message,)), ("info", (
        "Localized repair and deterministic registry reconstruction were both applied; the "
        "payload below is as far as the run got. No reaction was invented to satisfy the "
        "contract. The diagnostics for this run (stage0_attempts.json, "
        "extraction_boundary_report.json, cleaning_report.json) are in "
        f"{DIAG_DIR}.",)), ("json", ({"ok": False, "reason": "rows rejected"},))] + tail
    assert list(state) == ["stage_one_boundary", ex.EXTRACTION_DIAGNOSTICS_KEY]
    assert "filter" not in [c[0] for c in world.calls] and "maybe_run_rag" not in [c[0] for c in world.calls]


# ── the RAG seed ─────────────────────────────────────────────────────────────
def test_rag_off_runs_no_seed(world):
    res, _rep, state = _run(world, flag=1)
    assert [c[0] for c in world.calls] == ["stage1", "diag.current", "settle", "diag.current", "filter"]
    assert res.rag_result is None and res.user_task_context == "focus"
    assert "rag_result" not in state


def test_rag_on_seeds_from_the_in_scope_payload_and_folds_the_evidence(world):
    world.stage1 = _payload("a", out=("x",))
    world.rag = True
    world.rag_result = types.SimpleNamespace(evidence_context="PMC1: MenG.")
    res, rep, state = _run(world, flag=1)
    in_scope = dict(_payload("a"), settled=True)
    assert world.calls[-3:-1] == [("build_draft_graph", in_scope), ("generate_qa_report", "GRAPH", in_scope)]
    name, kw = world.calls[-1]
    assert name == "maybe_run_rag"
    assert kw == dict(pathway_context={"pathway_name": "Menaquinone"}, user_flag=True,
                      seed_payload=in_scope, reports={"qa_graph": {"qa": 1}}, seed_text="the paper",
                      reporter=rep)
    assert kw["seed_payload"] is res.stage_one_in_scope
    assert res.rag_result is world.rag_result
    assert res.user_task_context == "focus\n\nPMC1: MenG."
    assert list(state)[-2:] == ["rag_result", "rag_evidence_context"]
    assert state["rag_result"] is world.rag_result and state["rag_evidence_context"] == "PMC1: MenG."
    assert rep.calls == SPIN + [("info", ("Removed 1 out-of-scope reaction(s) before Stage 2: x",))]


@pytest.mark.parametrize("utc", [None, ""])
def test_rag_fold_without_a_focus_is_the_evidence_alone(world, utc):
    world.rag = True
    world.rag_result = types.SimpleNamespace(evidence_context="E")
    res, _rep, _state = _run(world, utc=utc)
    assert res.user_task_context == "E"
    assert [c for c in world.calls if c[0] == "maybe_run_rag"][0][1]["user_flag"] is False


def test_rag_without_evidence_records_the_result_and_keeps_the_focus(world):
    world.rag = True
    world.rag_result = types.SimpleNamespace(evidence_context="")
    res, _rep, state = _run(world)
    assert res.user_task_context == "focus" and state["rag_result"] is world.rag_result
    assert "rag_evidence_context" not in state


def test_rag_declined_writes_nothing(world):
    world.rag = True
    res, _rep, state = _run(world)
    assert res.rag_result is None and res.user_task_context == "focus"
    assert "rag_result" not in state and "maybe_run_rag" in [c[0] for c in world.calls]


def test_rag_error_propagates(world, monkeypatch):
    import t2pw.service.rag_orchestration as rag

    def boom(**_):
        raise RuntimeError("chain blew up")

    world.rag = True
    monkeypatch.setattr(rag, "maybe_run_rag", boom)
    with pytest.raises(RuntimeError, match="chain blew up"):
        _run(world)


# ── wiring ───────────────────────────────────────────────────────────────────
def test_patches_on_the_source_modules_reach_run_stage_one(world):
    """R3b review F1 rule. Before R3c the app script re-ran ``from ... import`` on every
    AppTest run, so a patch on the source module (c055 patches
    ``pipeline.run_stage_one_with_chunking`` and ``stage_one_boundary.settle_stage_one``)
    reached the call. Every callee of the moved code must still be looked up there."""

    world.rag = True
    world.rag_result = types.SimpleNamespace(evidence_context="")
    _run(world)
    assert [c[0] for c in world.calls] == ["stage1", "diag.current", "settle", "diag.current", "filter",
                                          "build_draft_graph", "generate_qa_report", "maybe_run_rag"]


def test_importing_the_service_loads_no_stage_one_module():
    """The Stage-1 imports are function-local, so importing the module loads what it did before."""

    import subprocess

    code = ("import sys, t2pw.service.extraction\n"
            "print(sorted(n for n in ('t2pw.pipeline.pipeline', 't2pw.pipeline.stage_one_boundary',"
            " 't2pw.service.rag_orchestration', 't2pw.service.postpipeline', 't2pw.config')"
            " if n in sys.modules))\n")
    import os

    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=str(SRC)), cwd=str(ROOT), timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "[]"


def test_state_and_reporter_are_required_keywords():
    params = inspect.signature(ex.run_stage_one).parameters
    assert list(params)[0] == "text"
    for name, p in list(params.items())[1:]:
        assert p.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert p.default is inspect.Parameter.empty, name
    p = inspect.signature(ex.render_attempts).parameters["reporter"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty


def test_reporter_markdown_and_code_forward_unchanged_and_log(caplog):
    calls = []

    class FakeSt:
        def __getattr__(self, name):
            return lambda *a, **k: calls.append((name, a, k)) or f"<{name}>"

    rep = StreamlitReporter(FakeSt())
    assert rep.markdown("**m**") == "<markdown>"
    assert rep.code("{}", language="json") == "<code>"
    assert calls == [("markdown", ("**m**",), {}), ("code", ("{}",), {"language": "json"})]
    with caplog.at_level(logging.INFO, logger="r3c"):
        log = LoggingReporter(logging.getLogger("r3c"))
        log.markdown("**m**")
        log.code("{}", language="json")
    assert [r.getMessage() for r in caplog.records] == ["**m**", "{}"]


# ── the real app ─────────────────────────────────────────────────────────────
def _real_app(monkeypatch, tmp_path, *, focus=""):
    from streamlit.testing.v1 import AppTest

    import t2pw.pwml.writer  # noqa: F401 - reads reference data under the real root

    monkeypatch.setattr(preprocessor, "preprocess",
                        lambda *_a, **_k: {"pathway_name": "Menaquinone biosynthesis",
                                           "likely_organism": "Escherichia coli"})
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("RAG_ELIGIBILITY_ENABLED", "0")
    try:
        at = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
        assert not at.exception
        at.text_area(key="pathway_text_0").set_value("MenG demethylates demethylmenaquinone.")
        if focus:
            next(t for t in at.text_area if t.label.startswith("Optional extraction focus")).set_value(focus)
        next(b for b in at.button if b.label == "Run pipeline").click().run()
    finally:
        diag.deactivate()
    return at


def test_real_app_stops_at_a_stage_one_failure(monkeypatch, tmp_path):
    """The app calls ``st.stop()`` on the returned failure: Stage 2 never runs."""

    stage2: List[Any] = []

    def stage1(*_a, **_k):
        raise pipeline.PipelineFailure("stage1", "empty reply after 3 attempts", ATTEMPTS)

    monkeypatch.setattr(pipeline, "run_stage_one_with_chunking", stage1)
    monkeypatch.setattr(pipeline, "run_stage_two_with_feedback_loop",
                        lambda *a, **k: stage2.append(a) or ({}, [], []))
    monkeypatch.setenv("RAG_ENABLED", "0")
    at = _real_app(monkeypatch, tmp_path)

    assert [str(e.value) for e in at.exception] == []
    assert [e.value for e in at.error] == ["Extraction failed: empty reply after 3 attempts"]
    assert [x.label for x in at.expander] == ["Stage 1 attempts"]
    assert "**Attempt 1 (extract)** - retry [cut]" in [m.value for m in at.markdown]
    assert stage2 == []
    assert "stage_one" not in at.session_state and "stage_one_boundary" not in at.session_state


def test_real_app_hands_stage_two_the_filtered_payload_and_the_folded_focus(monkeypatch, tmp_path):
    """The app rebinds what the rest of the block reads from the returned result."""

    import t2pw.service.rag_orchestration as rag

    stage2: List[Any] = []
    evidence = types.SimpleNamespace(evidence_context="PMC1: MenG demethylates.", synthesized=False,
                                     payload=None)

    def stage_two(text, payload, **kw):
        stage2.append((payload, kw))
        raise pipeline.PipelineFailure("stage2", "stop here", ATTEMPTS[1:])

    monkeypatch.setattr(pipeline, "run_stage_one_with_chunking",
                        lambda *_a, **_k: (_payload("a", out=("x",)), [{"chunk": 1}]))
    monkeypatch.setattr(boundary, "settle_stage_one",
                        lambda payload, **_k: _Boundary(payload))
    monkeypatch.setattr(pipeline, "run_stage_two_with_feedback_loop", stage_two)
    monkeypatch.setattr(draft_graph, "build_draft_graph", lambda payload: "GRAPH")
    monkeypatch.setattr(qa_graph, "generate_qa_report", lambda graph, payload: {"qa": graph})
    monkeypatch.setattr(rag, "maybe_run_rag", lambda **_k: evidence)
    monkeypatch.setenv("RAG_ENABLED", "1")
    at = _real_app(monkeypatch, tmp_path, focus="  menaquinone  ")

    assert [str(e.value) for e in at.exception] == []
    assert [e.value for e in at.error] == ["Inference stage failed: stop here"]
    # the app's render_attempts shim (Stage 2 still calls it) renders through the service
    assert [x.label for x in at.expander] == ["stage2 attempts"]
    assert "**Attempt 2** - success" in [m.value for m in at.markdown]
    assert "Removed 1 out-of-scope reaction(s) before Stage 2: x" in [i.value for i in at.info]
    assert len(stage2) == 1
    payload, kw = stage2[0]
    assert payload == _payload("a")
    assert kw["chunk_details"] == [{"chunk": 1}]
    assert kw["user_task_context"] == "menaquinone\n\nPMC1: MenG demethylates."
    assert at.session_state["rag_evidence_context"] == "PMC1: MenG demethylates."
    assert at.session_state["stage_one_boundary"] == {"ok": True, "reason": ""}
