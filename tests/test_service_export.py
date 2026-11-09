"""R4a: the "Generate PWML" step in ``t2pw.service.export``.

NEW TESTS (R4a, new module). ``run_export`` is a new public interface over code moved
verbatim out of ``streamlit_app.py``; the behaviour pinned here is the app's, the same
values the app read and wrote before the move.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

from t2pw.pipeline import export_mode as export_mode_module
from t2pw.service import export, postpipeline
from t2pw.service.reporter import StreamlitReporter

APP = Path(__file__).resolve().parents[1] / "src" / "t2pw" / "app" / "streamlit_app.py"
SPINNER = "Building PWML from reviewed pathway JSON..."
NO_PAYLOAD = {
    "ok": False,
    "error": "No reviewed mapped pathway is available for PWML export.",
    "counts": {},
    "issues": 0,
    "output_path": "",
    "qa": {},
    "grounding_report": {},
}


class Recorder:
    def __init__(self) -> None:
        self.events: List[Any] = []

    @contextlib.contextmanager
    def spinner(self, text: str, **kwargs: Any) -> Iterator[None]:
        self.events.append(("enter", text, kwargs))
        try:
            yield
        finally:
            self.events.append(("exit", text))


@pytest.fixture
def spy(monkeypatch):
    calls: List[Dict[str, Any]] = []
    result = {"ok": True, "output_path": "x.pwml"}

    def fake(payload, **kwargs):
        calls.append({"payload": payload, "kwargs": kwargs,
                      "ready_at_call": fake.state.get("refinement_pwml_ready")})
        return result

    fake.calls, fake.result, fake.state = calls, result, {}
    monkeypatch.setattr(postpipeline, "run_pwml_export", fake)
    return fake


def _run(state: Dict[str, Any], tmp_path: Path, reporter: Any = None, spy: Any = None) -> Dict[str, Any]:
    if spy is not None:
        spy.state = state
    return export.run_export(state=state, reporter=reporter or Recorder(), work_dir=tmp_path / "work",
                             project_root=tmp_path)


def test_new_run_export_arguments_are_keyword_only_and_required() -> None:
    """NEW (R4a): no fallback to Streamlit, and no ``out_dir`` (Q-W0-1)."""

    params = inspect.signature(export.run_export).parameters
    assert list(params) == ["state", "reporter", "work_dir", "project_root"]
    assert {p.kind for p in params.values()} == {inspect.Parameter.KEYWORD_ONLY}
    assert {p.default for p in params.values()} == {inspect.Parameter.empty}


@pytest.mark.parametrize("working", ["<absent>", {}, [], "text", None])
def test_no_reviewed_payload_is_refused_before_any_write(tmp_path, spy, working) -> None:
    state = {} if working == "<absent>" else {"refinement_working_json": working}
    assert _run(state, tmp_path, spy=spy) == NO_PAYLOAD
    assert not spy.calls and "refinement_pwml_ready" not in state and not (tmp_path / "work").exists()


def test_gate_errors_refuse_and_carry_the_quarantine_decision(tmp_path, spy) -> None:
    state = {"refinement_working_json": {"a": 1}, "refinement_gate_errors": ["e1", "e2"],
             "quarantine_report": {"r": 1}, "quarantine_artifacts": "not a dict"}
    assert _run(state, tmp_path, spy=spy) == {
        "ok": False,
        "error": "PWML export stopped by pre-export Stage 3 revalidation. "
                 "Resolve the refinement gate errors first.",
        "counts": {},
        "issues": 2,
        "output_path": "",
        "qa": {},
        "grounding_report": {},
        "stage3_gate_errors": ["e1", "e2"],
        "quarantine_report": {"r": 1},
        "quarantine_artifacts": {},
    }
    assert not spy.calls and "refinement_pwml_ready" not in state and not (tmp_path / "work").exists()


def test_defaults_reach_the_export(tmp_path, spy) -> None:
    working = {"a": 1}
    state: Dict[str, Any] = {"refinement_working_json": working}
    assert _run(state, tmp_path, spy=spy) is spy.result
    call = spy.calls[0]
    assert call["payload"] is working  # the session's object, not a copy
    assert call["kwargs"] == {
        "pathway_name": "Generated Pathway",
        "pathway_description": "",
        "pathway_subject": "Metabolic",
        "project_root": tmp_path,
        "ref_path": tmp_path / "reference" / "PW000001.pwml",
        "vis_width": 3200,
        "vis_height": 1400,
        "background_color": "#FFFFFF",
        "grounding_dict": None,
        "strict_db": True,
        "pathway_context": {},
        "export_mode": export_mode_module.coerce_mode(None),
        "quarantine_report": {},
        "quarantine_artifacts": {},
    }


def test_session_settings_reach_the_export(tmp_path, spy) -> None:
    state = {"refinement_working_json": {"a": 1}, "pwml_name": "N", "pwml_description": "D", "pwml_subject": "S",
             "pwml_width": "10", "pwml_height": 20, "pwml_bg": "#000000", "pwml_strict_db": False,
             "pathway_context": {"o": 1}, "export_mode": "research", "quarantine_report": {"r": 1},
             "quarantine_artifacts": {"q": 2}}
    _run(state, tmp_path, spy=spy)
    kwargs = spy.calls[0]["kwargs"]
    assert (kwargs["pathway_name"], kwargs["pathway_description"], kwargs["pathway_subject"]) == ("N", "D", "S")
    assert (kwargs["vis_width"], kwargs["vis_height"], kwargs["background_color"]) == (10, 20, "#000000")
    assert kwargs["strict_db"] is False
    assert kwargs["pathway_context"] == {"o": 1}
    assert kwargs["export_mode"] == export_mode_module.coerce_mode("research") != export_mode_module.coerce_mode(None)
    assert (kwargs["quarantine_report"], kwargs["quarantine_artifacts"]) == ({"r": 1}, {"q": 2})


def test_falsy_settings_take_the_fallbacks(tmp_path, spy) -> None:
    state = {"refinement_working_json": {"a": 1}, "pwml_name": "", "pwml_subject": None, "pwml_width": 0,
             "pwml_height": 0, "pwml_bg": "", "pathway_context": "not a dict"}
    _run(state, tmp_path, spy=spy)
    kwargs = spy.calls[0]["kwargs"]
    assert (kwargs["pathway_name"], kwargs["pathway_subject"]) == ("Generated Pathway", "Metabolic")
    assert (kwargs["vis_width"], kwargs["vis_height"], kwargs["background_color"]) == (3200, 1400, "#FFFFFF")
    assert kwargs["pathway_context"] == {}


@pytest.mark.parametrize("on, expected", [(True, {"g": 1}), (False, None), ("<absent>", None)])
def test_grounding_dict_only_when_grounding_is_on(tmp_path, spy, on, expected) -> None:
    state: Dict[str, Any] = {"refinement_working_json": {"a": 1}, "pwml_grounding_dict": {"g": 1}}
    if on != "<absent>":
        state["pwml_grounding"] = on
    _run(state, tmp_path, spy=spy)
    assert spy.calls[0]["kwargs"]["grounding_dict"] == expected


def test_ready_flag_is_set_before_the_export_and_survives_its_failure(tmp_path, monkeypatch) -> None:
    state: Dict[str, Any] = {"refinement_working_json": {"a": 1}}
    seen = []

    def boom(payload, **kwargs):
        seen.append(state.get("refinement_pwml_ready"))
        raise RuntimeError("export blew up")

    monkeypatch.setattr(postpipeline, "run_pwml_export", boom)
    reporter = Recorder()
    with pytest.raises(RuntimeError, match="export blew up"):
        _run(state, tmp_path, reporter)
    assert seen == [True] and state["refinement_pwml_ready"] is True
    assert reporter.events == [("enter", SPINNER, {}), ("exit", SPINNER)]


def test_spinner_wraps_the_export(tmp_path, monkeypatch) -> None:
    reporter = Recorder()

    def fake(payload, **kwargs):
        reporter.events.append("export")
        return {"ok": True}

    monkeypatch.setattr(postpipeline, "run_pwml_export", fake)
    _run({"refinement_working_json": {"a": 1}}, tmp_path, reporter)
    assert reporter.events == [("enter", SPINNER, {}), "export", ("exit", SPINNER)]


def test_a_failed_snapshot_sets_nothing_and_never_exports(tmp_path, spy) -> None:
    (tmp_path / "work").write_text("a file, not a directory", encoding="utf-8")
    state: Dict[str, Any] = {"refinement_working_json": {"a": 1}}
    with pytest.raises(OSError):
        _run(state, tmp_path, spy=spy)
    assert not spy.calls and "refinement_pwml_ready" not in state


def test_snapshot_is_written_and_mutates_the_stored_artifacts(tmp_path, spy) -> None:
    working = {"name": "β-carotene"}
    artifacts = {"keep": 1}
    state = {"refinement_working_json": working, "post_pipeline_artifacts": artifacts}
    _run(state, tmp_path, spy=spy)
    path = tmp_path / "work" / "final.mapped.json"
    assert path.read_text(encoding="utf-8") == json.dumps(working, indent=2, ensure_ascii=False)
    assert state["post_pipeline_artifacts"] is artifacts and artifacts["keep"] == 1
    assert artifacts["final_mapped_db"] == working and artifacts["final_export_input"] == working
    assert artifacts["final_mapped_db"] is not working
    assert artifacts["final_export_input"] is not artifacts["final_mapped_db"]
    assert artifacts["reviewed_final_mapped_path"] == str(path)


@pytest.mark.parametrize("stored", ["<absent>", ["a", "list"], None])
def test_non_dict_artifacts_are_left_alone(tmp_path, spy, stored) -> None:
    state: Dict[str, Any] = {"refinement_working_json": {"a": 1}}
    if stored != "<absent>":
        state["post_pipeline_artifacts"] = stored
    _run(state, tmp_path, spy=spy)
    assert (tmp_path / "work" / "final.mapped.json").is_file()
    assert state.get("post_pipeline_artifacts", "<absent>") == stored


@pytest.mark.parametrize("present, expected", [
    ([], "PW000001.pwml"),
    (["PW012926.pwml"], "PW012926.pwml"),
    (["PW000001.pwml", "PW012926.pwml"], "PW000001.pwml"),
])
def test_reference_path_prefers_the_first_existing_candidate(tmp_path, present, expected) -> None:
    (tmp_path / "reference").mkdir()
    for name in present:
        (tmp_path / "reference" / name).write_text("x", encoding="utf-8")
    assert export._pwml_reference_path(tmp_path) == tmp_path / "reference" / expected


def test_helpers_are_looked_up_on_their_source_modules_at_call_time(tmp_path, spy, monkeypatch) -> None:
    monkeypatch.setattr(export_mode_module, "coerce_mode", lambda value: "COERCED")
    monkeypatch.setattr(postpipeline, "_safe_dict", lambda value: {"safe": 1})
    monkeypatch.setattr(postpipeline, "_safe_list", lambda value: [])
    state = {"refinement_working_json": {"a": 1}, "refinement_gate_errors": ["ignored by the patched _safe_list"]}
    _run(state, tmp_path, spy=spy)
    kwargs = spy.calls[0]["kwargs"]
    assert kwargs["export_mode"] == "COERCED"
    assert kwargs["pathway_context"] == kwargs["quarantine_report"] == kwargs["quarantine_artifacts"] == {"safe": 1}


# The app's call site, lifted out of the real file and run against a recording fake.
class FakeColumn:
    def __init__(self, clicked: bool) -> None:
        self.clicked = clicked
        self.labels: List[tuple] = []

    def button(self, label: str, key: str) -> bool:
        self.labels.append((label, key))
        return self.clicked


class FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: Dict[str, Any] = {}
        self.messages: List[tuple] = []

    def success(self, body: Any) -> None:
        self.messages.append(("success", body))

    def error(self, body: Any) -> None:
        self.messages.append(("error", body))


def _run_app_button(run_export: Any, clicked: bool = True) -> tuple:
    function = next(node for node in ast.parse(APP.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_render_review_refine_section")
    block = next(node for node in ast.walk(function)
                 if isinstance(node, ast.If) and isinstance(node.test, ast.Call)
                 and getattr(node.test.func, "attr", "") == "button"
                 and getattr(node.test.args[0], "value", "") == "Generate PWML")
    st, column = FakeStreamlit(), FakeColumn(clicked)
    namespace = {"st": st, "pwml_col": column, "work_dir": "the-work-dir", "PROJECT_ROOT": Path("the-root"),
                 "run_export": run_export, "StreamlitReporter": StreamlitReporter}
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(APP), "exec"), namespace)
    return st, column


def test_app_button_passes_its_session_reporter_and_folders() -> None:
    seen: Dict[str, Any] = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"ok": True, "output_path": "out.pwml"}

    st, column = _run_app_button(fake)
    assert column.labels == [("Generate PWML", "refinement_generate_pwml")]
    assert set(seen) == {"state", "reporter", "work_dir", "project_root"}
    assert seen["state"] is st.session_state
    assert isinstance(seen["reporter"], StreamlitReporter) and seen["reporter"]._st is st
    assert (seen["work_dir"], seen["project_root"]) == ("the-work-dir", Path("the-root"))
    assert st.session_state["pwml_export_result"] == {"ok": True, "output_path": "out.pwml"}
    assert st.messages == [("success", "PWML generated from reviewed JSON: out.pwml")]


@pytest.mark.parametrize("result, text", [
    ({"ok": False, "error": "the gate rejected the payload"}, "PWML export failed: the gate rejected the payload"),
    ({"ok": False}, "PWML export failed: unknown"),
])
def test_app_button_reports_a_refusal(result, text) -> None:
    st, _ = _run_app_button(lambda **kwargs: result)
    assert st.session_state["pwml_export_result"] == result
    assert st.messages == [("error", text)]


def test_app_button_turns_an_exception_into_the_failure_dict() -> None:
    def boom(**kwargs):
        raise RuntimeError("no disk")

    st, _ = _run_app_button(boom)
    assert st.session_state["pwml_export_result"] == {
        "ok": False, "error": "no disk", "counts": {}, "issues": 0, "output_path": "", "qa": {}, "grounding_report": {},
    }
    assert st.messages == [("error", "PWML export failed: no disk")]


def test_app_button_does_nothing_until_clicked() -> None:
    def never(**kwargs):
        raise AssertionError("run_export must not be called")

    st, _ = _run_app_button(never, clicked=False)
    assert st.session_state == {} and st.messages == []


def test_the_moved_helpers_are_gone_from_the_app() -> None:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert not defined & {"_pwml_reference_path", "_write_reviewed_payload_snapshot",
                          "_generate_pwml_from_refinement_working_json", "quarantine_session_state"}
    # Archived evidence scripts still read app._pwml_reference_path (orch729, orch768, orch787).
    imported = {alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
                and node.module == "t2pw.service.export" for alias in node.names}
    assert imported == {"_pwml_reference_path", "run_export"}
