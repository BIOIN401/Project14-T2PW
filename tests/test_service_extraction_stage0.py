"""R3b: ``t2pw.service.extraction.run_stage_zero``, the app's Stage 0 moved out.

NEW TESTS (R3b, new module). ``run_stage_zero`` and the Stage-0 helpers are a pure move
of the app's submit block (diagnostics start through the post-Stage-0 sync) and of
``_run_stage_zero_with_retry`` / ``_has_usable_context`` / ``_seed_context_from_user_scope``
/ ``_stage0_context_rank`` / the diagnostics start and sync. These tests pin what that
code did: how many Stage-0 calls run and on what text, which context is kept, every
message and its order, the session keys written into ``state``, and where the
diagnostics directory lands.

The unit tests are offline: a fake ``preprocess`` is passed in and a recording
reporter stands in for Streamlit. The last test drives the REAL ``streamlit_app.py``
through ``AppTest`` (a Case-C draw that the app's own ambiguous-review gate then stops),
so this file runs in its own bounded process, like every AppTest-driving file.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
APP_PATH = SRC / "t2pw" / "app" / "streamlit_app.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import t2pw.paths as paths  # noqa: E402
import t2pw.pipeline.extraction_diagnostics as diag  # noqa: E402
import t2pw.pipeline.preprocessor as preprocessor  # noqa: E402
from t2pw.pipeline.preprocessor import PREPROCESS_STATUS_KEY  # noqa: E402
from t2pw.service import extraction as ex  # noqa: E402

LONG = "x" * (ex._PREPROCESS_RETRY_CHARS + 500)
SHORT = "a short paper about menaquinone"


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


def _failed() -> Dict[str, Any]:
    return {"pathway_name": "", "likely_organism": "", "key_compounds": [],
            PREPROCESS_STATUS_KEY: {"status": "llm_error", "detail": "timed out"}}


def _usable() -> Dict[str, Any]:
    return {"pathway_name": "Menaquinone biosynthesis", "likely_organism": "E. coli"}


def _case_c() -> Dict[str, Any]:
    return {"document_type": "multi_example_review", "selected_example": "",
            "candidate_examples": [{"name": "A"}]}


def _recovered() -> Dict[str, Any]:
    return {"pathway_name": "Heme biosynthesis",
            PREPROCESS_STATUS_KEY: {"status": "ok", "recovered": True, "detail": "cut at 800"}}


def _fake(draws: List[Any]):
    seen: List[tuple] = []
    queue = list(draws)

    def preprocess(text: str, **kwargs: Any) -> Any:
        seen.append((text, kwargs))
        assert queue, "preprocess called more often than staged"
        return queue.pop(0)

    return preprocess, seen


@pytest.fixture()
def root(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    yield tmp_path
    diag.deactivate()


def _run(text, draws, utc=None, **kw):
    fn, seen = _fake(draws)
    rep, state = RecordingReporter(), {}
    res = ex.run_stage_zero(text, temperature=0.25, user_task_context=utc, state=state,
                            reporter=rep, preprocess_fn=fn, **kw)
    return res, rep, state, seen


SPIN = [("spinner", ("Running preprocessor...",)), ("spinner:enter",)]


# ── Stage 0 branches ─────────────────────────────────────────────────────────
def test_usable_context_one_call_no_message(root):
    first = _usable()
    res, rep, state, seen = _run(SHORT, [first], utc="focus")
    assert res.pathway_context is first
    assert seen == [(SHORT, {"temperature": 0.25, "user_task_context": "focus"})]
    assert rep.calls == SPIN + [("spinner:exit",)]


def test_seeded_from_user_scope_with_organism(root):
    res, rep, _, seen = _run(SHORT, [_failed()], utc="menaquinone biosynthesis in Lactococcus lactis")
    assert len(seen) == 1  # short text: no retry
    assert res.pathway_context["pathway_name"] == "menaquinone biosynthesis"
    assert res.pathway_context["likely_organism"] == "Lactococcus lactis"
    assert res.pathway_context[PREPROCESS_STATUS_KEY] == _failed()[PREPROCESS_STATUS_KEY]
    assert rep.calls == SPIN + [
        ("info", ("Stage 0 returned no usable context, so the pipeline is falling back to the scope "
                  "you provided — pathway \"menaquinone biosynthesis\", organism \"Lactococcus lactis\". "
                  "Extraction and multi-paper RAG will use it. Re-run if you'd rather the preprocessor "
                  "try to read the scope from the paper again.",)),
        ("spinner:exit",)]


def test_seeded_without_organism_omits_the_organism_clause(root):
    res, rep, _, _ = _run(SHORT, [_failed()], utc="lipid A biosynthesis")
    assert res.pathway_context["pathway_name"] == "lipid A biosynthesis"
    assert "organism \"" not in rep.calls[2][1][0]
    assert "pathway \"lipid A biosynthesis\". Extraction" in rep.calls[2][1][0]


def test_failed_without_scope_warns_with_the_status(root):
    first = _failed()
    res, rep, _, _ = _run(SHORT, [first], utc=None)
    assert res.pathway_context is first
    assert rep.calls == SPIN + [
        ("warning", ("Stage 0 failed: llm_error - timed out\n\nIt returned no usable context — no "
                     "pathway name, organism, compounds or proteins, and no scope was provided in the "
                     "extraction-focus box to fall back to. Extraction will be unguided, and multi-paper "
                     "RAG cannot build a literature query, so no papers will be fetched. Re-run (usually "
                     "a transient LLM failure), or type the pathway/organism into the focus box so the "
                     "pipeline can proceed without the preprocessor.",)),
        ("spinner:exit",)]


def test_recovered_context_warns(root):
    first = _recovered()
    res, rep, _, _ = _run(SHORT, [first])
    assert res.pathway_context is first
    assert rep.calls == SPIN + [
        ("warning", ("Stage 0 partially recovered: ok (recovered from a truncated reply) - cut at 800\n\n"
                     "The reply was cut off and repaired, so some context fields (e.g. candidate "
                     "examples, key compounds) may be missing. Re-run if the context below looks "
                     "incomplete.",)),
        ("spinner:exit",)]


def test_case_c_is_quiet_and_returned_untouched(root):
    first = _case_c()
    res, rep, _, seen = _run(LONG, [first], utc="lipid A biosynthesis")
    assert res.pathway_context is first
    assert len(seen) == 1  # no retry on a deliberate Case C, even on a long text
    assert rep.calls == SPIN + [("spinner:exit",)]


# ── the bounded-head retry ───────────────────────────────────────────────────
def test_retry_on_long_failed_text_reads_the_head_and_adopts_a_better_draw(root):
    second = _usable()
    res, rep, _, seen = _run(LONG, [_failed(), second], utc="u")
    assert [t for t, _ in seen] == [LONG, LONG[: ex._PREPROCESS_RETRY_CHARS]]
    assert seen[1][1] == {"temperature": 0.25, "user_task_context": "u"}
    assert res.pathway_context is second
    assert rep.calls == SPIN + [("spinner:exit",)]


def test_retry_keeps_the_first_draw_when_the_second_is_no_better(root):
    first = _failed()
    res, _, _, seen = _run(LONG, [first, _failed()], utc=None)
    assert len(seen) == 2
    assert res.pathway_context is first


def test_retry_adopts_a_case_c_head(root):
    second = _case_c()
    res, _, _, _ = _run(LONG, [_failed(), second])
    assert res.pathway_context is second


def test_no_retry_at_exactly_the_head_length(root):
    text = "y" * ex._PREPROCESS_RETRY_CHARS
    _, _, _, seen = _run(text, [_failed()])
    assert len(seen) == 1


def test_seed_runs_after_a_failed_retry(root):
    res, rep, _, seen = _run(LONG, [_failed(), _failed()], utc="heme biosynthesis in yeast")
    assert len(seen) == 2
    assert res.pathway_context["likely_organism"] == "yeast"
    assert rep.calls[2][0] == "info"


def test_default_preprocess_is_the_source_module_attribute_at_call_time(root, monkeypatch):
    fn, seen = _fake([_usable()])
    monkeypatch.setattr(preprocessor, "preprocess", fn)
    ex.run_stage_zero(SHORT, temperature=0.0, user_task_context=None, state={},
                      reporter=RecordingReporter())
    assert len(seen) == 1


def test_helpers_rank_and_usable():
    assert ex._stage0_context_rank(_case_c()) == 2
    assert ex._stage0_context_rank(_usable()) == 1
    assert ex._stage0_context_rank(_failed()) == 0
    assert ex._has_usable_context({"key_proteins": ["", "MenA"]})
    assert not ex._has_usable_context({"key_proteins": ["", " "]})
    # a 4-word tail is not an organism: the whole scope stays the pathway name
    assert ex._seed_context_from_user_scope({}, "x in vitro reconstitution of y") == {
        "pathway_name": "x in vitro reconstitution of y"}
    # an organism Stage 0 already found is kept
    assert ex._seed_context_from_user_scope({"likely_organism": "E. coli"}, "heme in yeast") == {
        "pathway_name": "heme", "likely_organism": "E. coli"}
    assert ex._seed_context_from_user_scope({}, "  ") is None


# ── diagnostics and state ────────────────────────────────────────────────────
def test_diagnostics_open_before_stage_zero_and_land_under_project_root(root):
    order: List[str] = []
    real_activate = ex.activate_extraction_diagnostics

    def activate(**kw):
        order.append("activate")
        return real_activate(**kw)

    def preprocess(text, **_):
        order.append("preprocess")
        return _usable()

    ex.activate_extraction_diagnostics = activate
    try:
        state: Dict[str, Any] = {}
        res = ex.run_stage_zero(SHORT, temperature=0.0, user_task_context=None, state=state,
                                reporter=RecordingReporter(), preprocess_fn=preprocess)
    finally:
        ex.activate_extraction_diagnostics = real_activate
    assert order == ["activate", "preprocess"]
    base = root / "tmp" / ex.DIAGNOSTICS_DIRNAME
    assert res.diagnostics_dir.parent == base  # unique=True: its own subdirectory
    assert state[ex.EXTRACTION_DIAGNOSTICS_DIR_KEY] == str(res.diagnostics_dir)
    assert diag.current().run_id == diag.payload_hash(SHORT)
    # the post-Stage-0 sync published the recorder's artifacts
    assert state[ex.EXTRACTION_DIAGNOSTICS_KEY] == diag.current().artifacts()


def test_start_falls_back_to_tmp_when_the_recorder_has_no_dir(root, monkeypatch):
    class _Rec:
        artifact_dir = None

    monkeypatch.setattr(ex, "activate_extraction_diagnostics", lambda **_: _Rec())
    state: Dict[str, Any] = {}
    assert ex._start_extraction_diagnostics("t", state=state) == root / "tmp"
    assert state == {ex.EXTRACTION_DIAGNOSTICS_KEY: {}, ex.EXTRACTION_DIAGNOSTICS_DIR_KEY: ""}


def test_sync_never_raises_and_leaves_state_alone(monkeypatch):
    def boom():
        raise RuntimeError("no recorder")

    monkeypatch.setattr(ex, "current_extraction_diagnostics", boom)
    state = {"keep": 1}
    assert ex._sync_extraction_diagnostics(state=state) == {}
    assert state == {"keep": 1}


def test_state_and_reporter_are_required_keywords():
    params = inspect.signature(ex.run_stage_zero).parameters
    for name in ("state", "reporter", "temperature", "user_task_context"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is inspect.Parameter.empty, name
    for fn in (ex._start_extraction_diagnostics, ex._sync_extraction_diagnostics):
        p = inspect.signature(fn).parameters["state"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty


def test_the_service_has_no_project_root_copy():
    """The equivalence harness isolates through ``t2pw.paths``; a by-value copy here would escape it."""
    assert not hasattr(ex, "PROJECT_ROOT") and not hasattr(ex, "TMP_DIR")


# ── the real app ─────────────────────────────────────────────────────────────
def test_real_app_case_c_opens_diagnostics_then_stops_at_the_ambiguous_gate(monkeypatch, tmp_path):
    """The app passes ``st.session_state`` as ``state`` and still stops at its own gate.

    ``preprocess`` is patched on its source module, which the app's ``from ... import
    preprocess`` binds on every script run. A Case-C draw must end in the app's
    ambiguous-review error, with the diagnostics directory the moved code opened
    (under the patched project root) published in session state.
    """

    from streamlit.testing.v1 import AppTest

    import t2pw.config  # noqa: F401 - bind the real PROJECT_ROOT before it is redirected
    import t2pw.pwml.writer  # noqa: F401 - reads reference data under the real root

    fn, seen = _fake([_case_c()])
    monkeypatch.setattr(preprocessor, "preprocess", fn)
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    try:
        at = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
        assert not at.exception
        at.text_area(key="pathway_text_0").set_value(SHORT)
        next(b for b in at.button if b.label == "Run pipeline").click().run()
    finally:
        diag.deactivate()

    assert [str(e.value) for e in at.exception] == []
    assert len(seen) == 1
    assert any("Ambiguous review scope" in str(e.value) for e in at.error)
    assert at.session_state["pipeline_error"]["status"] == "ambiguous_review_scope"
    assert at.session_state["pipeline_ready"] is False
    published = Path(at.session_state[ex.EXTRACTION_DIAGNOSTICS_DIR_KEY])
    assert published.parent == tmp_path / "tmp" / ex.DIAGNOSTICS_DIRNAME
    assert isinstance(at.session_state[ex.EXTRACTION_DIAGNOSTICS_KEY], dict)
