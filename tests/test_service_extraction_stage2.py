"""R3d: ``t2pw.service.extraction.run_stage_two`` and ``run_extraction``.

NEW TESTS (R3d, new module). ``run_stage_two`` is a pure move of the app's submit block
from ``final_payload = stage_one_in_scope`` through the session-state publish: Stage 2
and its failure exit (now returned with a kind), the Stage-2 ``merge_additions``, the
RAG rounds with their merge site, the reaction-preservation report, the draft graph and
every session key, plus ``emit_rag_loop_messages`` and ``rag_loop_session_record``.
``run_extraction`` is the new headless composition of the four steps (new capability).

These tests pin what that code did: the Stage-2 arguments, the merge arguments, every
message and its order, every state write and its order, and that every callee is looked
up on its SOURCE module at call time (the R3b review F1 rule). The unit tests are
offline: callees patched on their source modules, a recording reporter for Streamlit.
The last two tests drive the REAL ``streamlit_app.py`` through ``AppTest``, so this
file runs in its own bounded process.
"""

from __future__ import annotations

import inspect
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
APP_PATH = SRC / "t2pw" / "app" / "streamlit_app.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import t2pw.paths as paths  # noqa: E402
import t2pw.pipeline.extraction_diagnostics as diag  # noqa: E402
import t2pw.pipeline.pipeline as pipeline  # noqa: E402
import t2pw.pipeline.preprocessor as preprocessor  # noqa: E402
import t2pw.pipeline.stage_one_boundary as boundary  # noqa: E402
import t2pw.service.postpipeline as postpipeline  # noqa: E402
import t2pw.service.rag_orchestration as rag  # noqa: E402
from t2pw.pipeline.cofactor_policy import PathwayContext  # noqa: E402
from t2pw.service import extraction as ex  # noqa: E402


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


class RecordingState(dict):
    """``state`` that records every write and pop, in order."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.log: List[tuple] = []

    def __setitem__(self, key: str, value: Any) -> None:
        self.log.append(("set", key))
        super().__setitem__(key, value)

    def pop(self, key: str, *default: Any) -> Any:
        self.log.append(("pop", key))
        return super().pop(key, *default)


class _Graph:
    def to_dict(self) -> dict:
        return {"graph": 1}


def _payload(*names):
    return {"entities": {}, "processes": {"reactions": [{"name": n} for n in names]}}


ATTEMPTS = [{"attempt": 1, "raw": "{bad", "error": "Expecting value", "phase": "infer"}, {"attempt": 2, "raw": "{}"}]
CTX = {"pathway_name": "Menaquinone biosynthesis", "likely_organism": "Escherichia coli"}
RECORDER = object()


@pytest.fixture()
def world(monkeypatch):
    """Every callee patched on its SOURCE module, recording into ``w.calls``."""

    w = types.SimpleNamespace(calls=[], additions={"additions": {}, "qa_hints": {"h": 1}}, raises=None,
                              loop=None)

    def stage_two(text, payload, **kw):
        w.calls.append(("stage2", text, payload, kw))
        if w.raises is not None:
            raise w.raises
        return deepcopy(w.additions), [{"chunk": 1}], [{"round": 1}]

    def merge(base, additions, **kw):
        w.calls.append(("merge", base, additions, kw))
        return {"merged": [base, additions]}

    def preservation(stage, payload):
        w.calls.append(("preservation", stage, payload))
        return {"stage": stage}

    def build(payload):
        w.calls.append(("draft_graph", payload))
        return _Graph(), {"qa": 1}, "summary"

    def rounds(graph, **kw):
        w.calls.append(("rag_rounds", graph, kw))
        return w.loop

    monkeypatch.setattr(pipeline, "run_stage_two_with_feedback_loop", stage_two)
    monkeypatch.setattr(pipeline, "merge_additions", merge)
    monkeypatch.setattr(pipeline, "build_and_save_draft_graph", build)
    monkeypatch.setattr(postpipeline, "_write_reaction_preservation_report", preservation)
    monkeypatch.setattr(rag, "run_rag_rounds", rounds)
    monkeypatch.setattr(diag, "current", lambda: w.calls.append(("diag.current",)) or RECORDER)
    return w


def _stage_one(rag_result=None, **kw):
    fields = dict(stage_one=_payload("a", "x"), chunk_details=[{"c": 1}], stage_one_in_scope=_payload("a"),
                  out_of_scope_removed_reactions=["x"], rag_result=rag_result, user_task_context="focus")
    fields.update(kw)
    return ex.StageOneResult(**fields)


def _run(w, *, stage_one=None, state=None, **kw):
    rep = RecordingReporter()
    state = RecordingState() if state is None else state
    args = dict(pathway_context=dict(CTX), run_inference=True, enable_chunking=True, chunk_size=1200.0,
                chunk_overlap=200.0, infer_rounds=2.0, infer_attempts=3.0, temperature=0.25,
                infer_tokens=9000.0, rag_incomplete_flag=0, export_mode="pathwhiz", state=state, reporter=rep)
    args.update(kw)
    res = ex.run_stage_two("the paper", stage_one or _stage_one(), **args)
    return res, rep, state


PUBLISH = [("set", "draft_graph"), ("pop", "draft_graph_png_bytes"), ("pop", "draft_graph_render_error"),
           ("set", "qa_report"), ("set", "reaction_summary"), ("set", "pipeline_ready"),
           ("set", "run_inference_enabled"), ("set", "pathway_context"), ("set", "user_task_context"),
           ("set", "stage_one"), ("set", "extraction_diagnostics_recorder"),
           ("set", "out_of_scope_removed_reactions"), ("set", "chunk_details"), ("set", "stage_two"),
           ("set", "stage_two_chunks"), ("set", "stage_two_rounds"), ("set", "qa_hints"),
           ("set", "final_payload"), ("set", "final_payload_snapshot"), ("set", "pipeline_source_text"),
           ("set", "reaction_preservation_after_stage2"), ("pop", "post_pipeline_artifacts")]


# ── success path ─────────────────────────────────────────────────────────────
def test_stage_two_gets_its_arguments_and_the_merge_builds_the_final_payload(world):
    res, rep, state = _run(world)

    name, text, payload, kw = world.calls[0]
    assert (name, text, payload) == ("stage2", "the paper", _payload("a"))
    assert kw == dict(chunk_details=[{"c": 1}], pathway_context=CTX, user_task_context="focus", qa_rounds=2,
                      enable_chunking=True, chunk_word_limit=1200, chunk_overlap=200, max_attempts=3,
                      temperature=0.25, max_tokens=9000)
    assert all(type(kw[k]) is int for k in ("qa_rounds", "chunk_word_limit", "chunk_overlap", "max_attempts",
                                            "max_tokens"))
    merge = world.calls[1]
    assert merge[:3] == ("merge", _payload("a"), world.additions)
    assert set(merge[3]) == {"pathway_context", "source_text"} and merge[3]["source_text"] == "the paper"
    assert merge[3]["pathway_context"] == PathwayContext("Menaquinone biosynthesis", "Escherichia coli")
    assert rep.calls == [("spinner", ("Running Stage 2 inference...",)), ("spinner:enter",), ("spinner:exit",)]
    assert res == ex.ExtractionResult(pathway_context=CTX, final_payload={"merged": [_payload("a"),
                                                                                       world.additions]})


def test_the_publish_writes_every_key_in_the_apps_order(world):
    state = RecordingState(draft_graph_png_bytes=b"png", post_pipeline_artifacts={"old": 1}, keep=1)
    res, _rep, state = _run(world, state=state)

    assert [c[0] for c in world.calls] == ["stage2", "merge", "preservation", "draft_graph", "diag.current"]
    assert world.calls[2] == ("preservation", "after_stage2", res.final_payload)
    assert world.calls[3] == ("draft_graph", res.final_payload)
    assert state.log == PUBLISH
    assert state["draft_graph"] == {"graph": 1} and state["qa_report"] == {"qa": 1}
    assert state["reaction_summary"] == "summary" and state["pipeline_ready"] is True
    assert state["run_inference_enabled"] is True and state["pathway_context"] == CTX
    assert state["user_task_context"] == "focus" and state["stage_one"] == _payload("a", "x")
    assert state["extraction_diagnostics_recorder"] is RECORDER
    assert state["out_of_scope_removed_reactions"] == ["x"] and state["chunk_details"] == [{"c": 1}]
    assert state["stage_two"] == world.additions
    assert state["stage_two_chunks"] == [{"chunk": 1}] and state["stage_two_rounds"] == [{"round": 1}]
    assert state["qa_hints"] == {"h": 1}
    assert state["final_payload"] is res.final_payload and state["final_payload_snapshot"] is res.final_payload
    assert state["pipeline_source_text"] == "the paper"
    assert state["reaction_preservation_after_stage2"] == {"stage": "after_stage2"}
    assert "draft_graph_png_bytes" not in state and "post_pipeline_artifacts" not in state
    assert state["keep"] == 1


def test_inference_off_publishes_the_in_scope_payload_untouched(world):
    res, rep, state = _run(world, run_inference=0)

    assert [c[0] for c in world.calls] == ["preservation", "draft_graph", "diag.current"]
    assert rep.calls == []
    assert res.final_payload == _payload("a") and state["final_payload"] is res.final_payload
    assert state["stage_two"] is None and state["stage_two_chunks"] == [] and state["stage_two_rounds"] == []
    assert state["qa_hints"] is None and state["run_inference_enabled"] is False


def test_a_non_dict_stage_two_merges_nothing_and_has_empty_hints(world):
    world.additions = ["not", "a", "dict"]
    _res, _rep, state = _run(world)

    assert world.calls[1][2] == {}
    assert state["qa_hints"] == {}


# ── the failure exit ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("stage, label", [("stage2-chunk-1", "stage2-chunk-1 attempts"),
                                          (None, "Stage 2 attempts")])
def test_a_stage_two_failure_reports_and_is_returned_before_anything_is_published(world, stage, label):
    failure = pipeline.PipelineFailure("stage2", "no JSON after 3", ATTEMPTS)
    failure.stage = stage
    world.raises = failure
    res, rep, state = _run(world)

    message = "Inference stage failed: no JSON after 3"
    assert res == ex.ExtractionResult(failure=ex.InputFailure(ex.KIND_STAGE_TWO_FAILED, message))
    assert ex.KIND_STAGE_TWO_FAILED == "stage_two_failed"
    assert rep.calls == [
        ("spinner", ("Running Stage 2 inference...",)), ("spinner:enter",), ("spinner:exit",),
        ("error", (message,)), ("expander", (label,), {"expanded": False}), ("expander:enter",),
        ("markdown", ("**Attempt 1 (infer)** - retry",)), ("code", ("{bad",), {"language": "json"}),
        ("caption", ("Expecting value",)), ("markdown", ("**Attempt 2** - success",)),
        ("code", ("{}",), {"language": "json"}), ("expander:exit",)]
    assert [c[0] for c in world.calls] == ["stage2"]
    assert state.log == []


def test_any_other_stage_two_exception_propagates(world):
    world.raises = RuntimeError("LLM down")
    with pytest.raises(RuntimeError, match="LLM down"):
        _run(world)


# ── the RAG rounds ───────────────────────────────────────────────────────────
def _rag_result(**kw):
    fields = dict(synthesized=True, payload={"p": 1}, evidence_context="E")
    fields.update(kw)
    return types.SimpleNamespace(**fields)


LOOP = {"payload": {"looped": 1}, "messages": [("info", "i"), ("warning", "w"), ("error", "e"), ("json", {"j": 1}),
                                               ("other", "dropped")], "rounds": [1], "max_rounds": 1}


def test_a_synthesized_rag_result_runs_the_rounds_on_the_merged_payload(world):
    world.loop = deepcopy(LOOP)
    result = _rag_result()
    state = RecordingState(pwml_strict_db=0)
    res, rep, state = _run(world, stage_one=_stage_one(rag_result=result), state=state, rag_incomplete_flag=1,
                           export_mode="research")

    assert [c[0] for c in world.calls] == ["stage2", "merge", "rag_rounds", "preservation", "draft_graph",
                                          "diag.current"]
    _name, graph, kw = world.calls[2]
    assert graph == {"merged": [_payload("a"), world.additions]}
    assert set(kw) == {"first_result", "merge_site", "pathway_context", "user_flag", "seed_text", "export_mode",
                       "strict_db", "reporter", "state"}
    assert kw["first_result"] is result and kw["pathway_context"] == CTX and kw["user_flag"] is True
    assert kw["seed_text"] == "the paper" and kw["export_mode"] == "research" and kw["strict_db"] is False
    assert kw["reporter"] is rep and kw["state"] is state
    assert res.final_payload == {"looped": 1}
    assert rep.calls[3:] == [("info", ("i",)), ("warning", ("w",)), ("error", ("e",)), ("json", ({"j": 1},))]
    assert state["rag_loop"] == {"rounds": [1], "max_rounds": 1} and result.loop is state["rag_loop"]
    assert state.log.index(("set", "rag_loop")) < state.log.index(("set", "draft_graph"))


def test_strict_db_defaults_to_true_when_the_checkbox_never_rendered(world):
    world.loop = deepcopy(LOOP)
    _run(world, stage_one=_stage_one(rag_result=_rag_result()))
    assert world.calls[2][2]["strict_db"] is True


def test_the_rag_merge_site_merges_with_a_real_context_and_no_source_text(world):
    world.loop = deepcopy(LOOP)
    _run(world, stage_one=_stage_one(rag_result=_rag_result()))
    merge_site = world.calls[2][2]["merge_site"]

    assert merge_site({"base": 1}, {"env": 1}, {"pathway_name": "Heme biosynthesis"}) == {
        "merged": [{"base": 1}, {"env": 1}]}
    assert world.calls[-1] == ("merge", {"base": 1}, {"env": 1},
                               {"pathway_context": PathwayContext("Heme biosynthesis", "")})


def test_a_loop_record_that_cannot_be_attached_is_ignored(world):
    world.loop = deepcopy(LOOP)

    class Frozen(types.SimpleNamespace):
        def __setattr__(self, name, value):
            if name == "loop":
                raise AttributeError("read-only")
            super().__setattr__(name, value)

    res, _rep, state = _run(world, stage_one=_stage_one(rag_result=Frozen(synthesized=True, payload={"p": 1})))
    assert res.final_payload == {"looped": 1} and "rag_loop" in state


@pytest.mark.parametrize("result", [None, _rag_result(synthesized=False), _rag_result(payload=None),
                                    _rag_result(payload={})])
def test_no_rounds_without_a_synthesized_payload(world, result):
    _res, _rep, state = _run(world, stage_one=_stage_one(rag_result=result))
    assert "rag_rounds" not in [c[0] for c in world.calls] and "rag_loop" not in state


def test_emit_and_session_record_helpers():
    rep = RecordingReporter()
    ex.emit_rag_loop_messages([("info", "i"), ("json", [1]), ("nope", "x")], reporter=rep)
    ex.emit_rag_loop_messages(None, reporter=rep)
    assert rep.calls == [("info", ("i",)), ("json", ([1],))]
    assert ex.rag_loop_session_record({"payload": 1, "messages": 2, "rounds": 3}) == {"rounds": 3}
    assert ex.rag_loop_session_record(None) == {}


# ── wiring ───────────────────────────────────────────────────────────────────
def test_patches_on_the_source_modules_reach_run_stage_two(world):
    """R3b review F1 rule: the app script re-ran ``from ... import`` on every AppTest run, so a
    patch on the source module (c055 patches ``pipeline.run_stage_two_with_feedback_loop`` and
    ``pipeline.build_and_save_draft_graph``) reached the call. It must still."""

    world.loop = deepcopy(LOOP)
    _run(world, stage_one=_stage_one(rag_result=_rag_result()))
    merge_site = world.calls[2][2]["merge_site"]
    merge_site({}, {}, {})
    assert [c[0] for c in world.calls] == ["stage2", "merge", "rag_rounds", "preservation", "draft_graph",
                                          "diag.current", "merge"]


def test_state_and_reporter_are_required_keywords():
    for fn, positional in ((ex.run_stage_two, ["text", "stage_one_result"]), (ex.run_extraction, ["request"])):
        params = inspect.signature(fn).parameters
        assert list(params)[:len(positional)] == positional
        for name, p in list(params.items())[len(positional):]:
            assert p.kind is inspect.Parameter.KEYWORD_ONLY, name
            assert p.default is inspect.Parameter.empty, name
    p = inspect.signature(ex.emit_rag_loop_messages).parameters["reporter"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty
    assert all(f.default is inspect.Parameter.empty
               for f in inspect.signature(ex.ExtractionRequest).parameters.values())


# ── run_extraction (new capability: the headless composition) ────────────────
def _request(**kw):
    fields = dict(text_entries=["  The paper.  "], pdfs=[], uses_pdf_input=False, input_mode="Paste text",
                  pdf_page_range="", pdf_skip_refs=True, pdf_ocr=False, user_task_context="  focus  ",
                  export_mode="research", temperature=0.25, run_inference=True, enable_chunking=False,
                  chunk_size=1200, chunk_overlap=200, extract_attempts=2, infer_attempts=3, infer_rounds=2,
                  extract_tokens=8000, infer_tokens=9000, rag_incomplete_flag=False)
    fields.update(kw)
    return ex.ExtractionRequest(**fields)


@pytest.fixture()
def steps(monkeypatch):
    """The four steps patched on the service module, recording their arguments."""

    s = types.SimpleNamespace(calls=[], assembled=ex.AssembledInputs(text="The paper."),
                              stage_one=_stage_one(), stage_two=ex.ExtractionResult(final_payload={"f": 1}))
    zero = ex.StageZeroResult(pathway_context=dict(CTX), diagnostics_dir=Path("C:/diag/run-1"))

    def record(name, result):
        def call(*a, **k):
            s.calls.append((name, a, k))
            return result() if callable(result) else result
        return call

    monkeypatch.setattr(ex, "assemble_inputs", record("assemble", lambda: s.assembled))
    monkeypatch.setattr(ex, "run_stage_zero", record("zero", zero))
    monkeypatch.setattr(ex, "run_stage_one", record("one", lambda: s.stage_one))
    monkeypatch.setattr(ex, "run_stage_two", record("two", lambda: s.stage_two))
    s.zero = zero
    return s


def test_run_extraction_composes_the_four_steps_in_the_apps_order(steps):
    rep, state = RecordingReporter(), {}
    res = ex.run_extraction(_request(), reporter=rep, state=state)

    assert [c[0] for c in steps.calls] == ["assemble", "zero", "one", "two"]
    assert res is steps.stage_two and state == {"export_mode": "research"}
    assemble, zero, one, two = steps.calls
    assert assemble[1] == (["  The paper.  "], [])
    assert assemble[2] == dict(uses_pdf_input=False, input_mode="Paste text", pdf_page_range="",
                               pdf_skip_refs=True, pdf_ocr=False, reporter=rep)
    assert zero[1] == ("The paper.",)
    assert zero[2] == dict(temperature=0.25, user_task_context="focus", state=state, reporter=rep)
    assert one[1] == ("The paper.",)
    assert one[2] == dict(pathway_context=CTX, user_task_context="focus", diagnostics_dir=steps.zero.diagnostics_dir,
                          enable_chunking=False, chunk_size=1200, chunk_overlap=200, extract_attempts=2,
                          temperature=0.25, extract_tokens=8000, rag_incomplete_flag=False, state=state,
                          reporter=rep)
    assert two[1] == ("The paper.", steps.stage_one)
    assert two[2] == dict(pathway_context=CTX, run_inference=True, enable_chunking=False, chunk_size=1200,
                          chunk_overlap=200, infer_rounds=2, infer_attempts=3, temperature=0.25,
                          infer_tokens=9000, rag_incomplete_flag=False, export_mode="research", state=state,
                          reporter=rep)


@pytest.mark.parametrize("focus, passed", [("", None), ("   ", None), (None, None), (" x ", "x")])
def test_run_extraction_strips_the_focus_as_the_app_does(steps, focus, passed):
    ex.run_extraction(_request(user_task_context=focus), reporter=RecordingReporter(), state={})
    assert steps.calls[1][2]["user_task_context"] == passed


def test_run_extraction_returns_each_steps_failure_and_runs_nothing_after_it(steps):
    steps.assembled = ex.AssembledInputs(failure=ex.InputFailure(ex.KIND_NO_TEXT, "No text"))
    res = ex.run_extraction(_request(), reporter=RecordingReporter(), state={})
    assert res == ex.ExtractionResult(failure=steps.assembled.failure)
    assert [c[0] for c in steps.calls] == ["assemble"]

    steps.calls.clear()
    steps.assembled = ex.AssembledInputs(text="The paper.")
    steps.stage_one = ex.StageOneResult(failure=ex.InputFailure(ex.KIND_STAGE_ONE_FAILED, "Extraction failed: x"))
    res = ex.run_extraction(_request(), reporter=RecordingReporter(), state={})
    assert res == ex.ExtractionResult(failure=steps.stage_one.failure)
    assert [c[0] for c in steps.calls] == ["assemble", "zero", "one"]


def test_run_extraction_gates_an_ambiguous_review_before_any_llm_call(monkeypatch, tmp_path):
    """``run_extraction`` runs the ambiguous-review gate after Stage 0 (R3d.md, open question 3,
    reversed by Q-R3d-2): a Case-C context is refused before the first Stage-1 LLM call."""

    case_c = {"document_type": "multi_example_review", "selected_example": "", "candidate_examples": ["A", "B"]}
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(preprocessor, "preprocess", lambda *_a, **_k: deepcopy(case_c))
    monkeypatch.setattr(pipeline, "run_extraction_pipeline",
                        lambda *_a, **_k: pytest.fail("Stage 1 reached the LLM on an ambiguous review"))
    rep, state = RecordingReporter(), {}
    try:
        res = ex.run_extraction(_request(text_entries=["short review"]), reporter=rep, state=state)
    finally:
        diag.deactivate()

    assert res.failure == ex.InputFailure(
        ex.KIND_AMBIGUOUS_REVIEW_SCOPE, "multi_example_review detected with no selected_example. "
        "Extraction skipped to prevent mixed-pathway output.")
    assert state["pipeline_ready"] is False and "stage_one_boundary" not in state


def test_run_extraction_end_to_end_offline(monkeypatch, tmp_path):
    """The real four steps with only the LLM stages stubbed on their source modules."""

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "TMP_DIR", tmp_path / "tmp")
    monkeypatch.setattr(postpipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(preprocessor, "preprocess", lambda *_a, **_k: dict(CTX))
    seen = {}

    def stage1(text, **kw):
        seen["stage1"] = (text, kw["user_task_context"])
        return _payload("menaquinone demethylation"), []

    def stage2(text, payload, **kw):
        seen["stage2"] = (text, payload, kw["user_task_context"])
        return {}, [], []

    monkeypatch.setattr(pipeline, "run_stage_one_with_chunking", stage1)
    monkeypatch.setattr(boundary, "settle_stage_one", lambda payload, **_k: types.SimpleNamespace(
        ok=True, payload=payload, failure=None, reconstruction=None, incomplete_reason="",
        to_summary=lambda: {"ok": True}))
    monkeypatch.setattr(pipeline, "run_stage_two_with_feedback_loop", stage2)
    monkeypatch.setenv("RAG_ENABLED", "0")
    state: dict = {}
    try:
        res = ex.run_extraction(_request(text_entries=["MenG demethylates."]), reporter=RecordingReporter(),
                                state=state)
    finally:
        diag.deactivate()

    assert res.failure is None and res.pathway_context == CTX
    assert seen["stage1"] == ("MenG demethylates.", "focus")
    assert seen["stage2"][0] == "MenG demethylates." and seen["stage2"][2] == "focus"
    assert seen["stage2"][1] == _payload("menaquinone demethylation")  # the settled, filtered Stage-1 payload
    assert isinstance(res.final_payload, dict) and "processes" in res.final_payload  # the real merge's output
    assert state["pipeline_ready"] is True and state["final_payload"] is res.final_payload
    assert state["stage_one_boundary"] == {"ok": True} and state["export_mode"] == "research"
    assert (tmp_path / "tmp" / "draft_graph.json").is_file()


# ── the real app ─────────────────────────────────────────────────────────────
def _real_app(monkeypatch, tmp_path, *, seed=None):
    from streamlit.testing.v1 import AppTest

    import t2pw.pwml.writer  # noqa: F401 - reads reference data under the real root

    monkeypatch.setattr(preprocessor, "preprocess", lambda *_a, **_k: dict(CTX))
    monkeypatch.setattr(pipeline, "run_stage_one_with_chunking",
                        lambda *_a, **_k: (_payload("menaquinone demethylation"), []))
    monkeypatch.setattr(boundary, "settle_stage_one", lambda payload, **_k: types.SimpleNamespace(
        ok=True, payload=payload, failure=None, reconstruction=None, incomplete_reason="",
        to_summary=lambda: {"ok": True}))
    real_build = pipeline.build_and_save_draft_graph
    monkeypatch.setattr(pipeline, "build_and_save_draft_graph",
                        lambda payload: real_build(payload, output_path=tmp_path / "draft_graph.json"))
    monkeypatch.setattr(postpipeline, "_write_reaction_preservation_report", lambda stage, payload: {"stage": stage})
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("RAG_ENABLED", "0")
    monkeypatch.setenv("RAG_ELIGIBILITY_ENABLED", "0")
    try:
        at = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
        assert not at.exception
        for key, value in (seed or {}).items():
            at.session_state[key] = value
        at.text_area(key="pathway_text_0").set_value("MenG demethylates demethylmenaquinone.")
        next(b for b in at.button if b.label == "Run pipeline").click().run()
    finally:
        diag.deactivate()
    return at


def test_real_app_publishes_the_run_then_resets_the_quarantine_and_counts_tokens(monkeypatch, tmp_path):
    additions = {"additions": {"processes": {"reactions": [{"name": "menaquinone prenylation"}]}}}
    monkeypatch.setattr(pipeline, "run_stage_two_with_feedback_loop", lambda *_a, **_k: (additions, [], [{"r": 1}]))
    at = _real_app(monkeypatch, tmp_path, seed={"quarantine_report": {"stale": True},
                                                "post_pipeline_artifacts": {"stale": True}})

    assert [str(e.value) for e in at.exception] == []
    assert [e.value for e in at.error] == []
    ss = at.session_state
    assert ss["pipeline_ready"] is True and ss["run_inference_enabled"] is True
    assert ss["pathway_context"] == CTX and ss["stage_two"] == additions and ss["stage_two_rounds"] == [{"r": 1}]
    assert ss["reaction_preservation_after_stage2"] == {"stage": "after_stage2"}
    assert ss["pipeline_source_text"] == "MenG demethylates demethylmenaquinone."
    assert ss["final_payload"] == ss["final_payload_snapshot"]
    assert (tmp_path / "draft_graph.json").is_file() and "draft_graph" in ss
    # after the service returned: the app's quarantine reset, then the token stats
    assert "quarantine_report" not in ss and "post_pipeline_artifacts" not in ss
    assert "token_stats" in ss


def test_real_app_stops_at_a_stage_two_failure_before_the_publish(monkeypatch, tmp_path):
    def stage_two(*_a, **_k):
        raise pipeline.PipelineFailure("stage2", "no JSON after 2", ATTEMPTS)

    monkeypatch.setattr(pipeline, "run_stage_two_with_feedback_loop", stage_two)
    at = _real_app(monkeypatch, tmp_path, seed={"quarantine_report": {"stale": True}})

    assert [str(e.value) for e in at.exception] == []
    assert [e.value for e in at.error] == ["Inference stage failed: no JSON after 2"]
    assert [x.label for x in at.expander] == ["stage2 attempts"]
    ss = at.session_state
    assert "pipeline_ready" not in ss and "final_payload" not in ss and "token_stats" not in ss
    assert ss["quarantine_report"] == {"stale": True}  # the reset is after the stop, as before
