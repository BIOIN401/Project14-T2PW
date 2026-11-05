"""R2: the Reporter protocol and the explicit ``state`` mapping.

NEW ACCEPTANCE TEST (R2, new capability): nothing under ``src/t2pw/service/`` imports
``streamlit``, so the service runs headless. The other tests here are unit tests of the
two reporters and of the state-dict path that replaced ``st.session_state``.
"""

from __future__ import annotations

import ast
import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SERVICE_DIR = SRC / "t2pw" / "service"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

QUARANTINE_KEYS = ["quarantine_report", "quarantine_artifacts", "quarantine_coverage",
                   "quarantine_ok", "quarantine_decision_controls"]


def _mod(name: str) -> Any:
    return importlib.import_module(f"t2pw.service.{name}")


# ── NEW ACCEPTANCE TEST (R2, new capability) ─────────────────────────────────
def test_new_acceptance_r2_no_service_module_imports_streamlit() -> None:
    """NEW ACCEPTANCE TEST (R2, new capability). Importing every service module in a
    fresh interpreter loads no ``streamlit``, and no service source names it in an
    import statement (lazy, function-local imports included)."""

    modules = sorted(p.stem for p in SERVICE_DIR.glob("*.py") if p.stem != "__init__")
    assert {"postpipeline", "rag_orchestration", "reporter"} <= set(modules)
    for path in SERVICE_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not [n for n in names if n.split(".")[0] == "streamlit"], path.name
            assert not [n for n in names if n.startswith("t2pw.app")], path.name

    code = ("import importlib, sys\n"
            f"for m in {modules!r}:\n"
            "    importlib.import_module('t2pw.service.' + m)\n"
            "print(sorted(n for n in sys.modules if n.split('.')[0] == 'streamlit'))\n")
    env = dict(os.environ, PYTHONPATH=str(SRC))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, cwd=str(ROOT), timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "[]"


# ── the two reporters ────────────────────────────────────────────────────────
class _RecordingStreamlit:
    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def __getattr__(self, name: str) -> Any:
        def _call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            return f"<{name}>"
        return _call


def test_streamlit_reporter_forwards_every_call_unchanged() -> None:
    fake = _RecordingStreamlit()
    reporter = _mod("reporter").StreamlitReporter(fake)
    assert reporter.progress("searching…", expanded=True) == "<status>"
    reporter.info("i")
    reporter.warning("w")
    reporter.error("e")
    reporter.json({"k": 1})
    assert fake.calls == [
        ("status", ("searching…",), {"expanded": True}),
        ("info", ("i",), {}), ("warning", ("w",), {}), ("error", ("e",), {}),
        ("json", ({"k": 1},), {}),
    ]


def test_logging_reporter_logs_messages_and_progress(caplog: pytest.LogCaptureFixture) -> None:
    reporter = _mod("reporter").LoggingReporter(logging.getLogger("r2-test"))
    with caplog.at_level(logging.INFO, logger="r2-test"):
        status = reporter.progress("Multi-paper RAG: searching", expanded=True)
        status.write("Fetched 0 candidate paper(s).")
        status.update(label="Multi-paper RAG: selecting")
        reporter.info("an info")
        reporter.warning("a warning")
        reporter.json({"failed": ["x"]})
        status.update(label="could not complete", state="error")
        reporter.error("an error")
    seen = [(r.levelname, r.getMessage()) for r in caplog.records]
    assert seen[0] == ("INFO", "Multi-paper RAG: searching")
    assert ("INFO", "  Fetched 0 candidate paper(s).") in seen
    assert ("INFO", "Multi-paper RAG: selecting [running]") in seen
    assert ("WARNING", "a warning") in seen
    assert any(level == "INFO" and '"failed"' in msg for level, msg in seen)
    assert ("ERROR", "could not complete [error]") in seen
    assert seen[-1] == ("ERROR", "an error")
    assert (status.label, status.state) == ("could not complete", "error")


def test_defaults_are_headless_until_the_app_installs_its_own(monkeypatch) -> None:
    reporter_mod = _mod("reporter")
    monkeypatch.setattr(reporter_mod, "_APP_DEFAULTS", {})
    assert isinstance(reporter_mod.default_reporter(), reporter_mod.LoggingReporter)
    first, second = reporter_mod.default_state(), reporter_mod.default_state()
    assert first == {} and first is not second
    session: Dict[str, Any] = {}
    fake = _RecordingStreamlit()
    reporter_mod.install_app_defaults(reporter=lambda: reporter_mod.StreamlitReporter(fake),
                                      state=lambda: session)
    assert reporter_mod.default_state() is session
    reporter_mod.default_reporter().warning("w")
    assert fake.calls == [("warning", ("w",), {})]


# ── the state-dict path ──────────────────────────────────────────────────────
class _Result:
    ok = True
    payload = {"processes": {"reactions": []}}
    quarantine_report = {"decision_input_hash": "h", "decision_id": "d"}
    coverage = {"c": 1}
    refusal_reasons: List[str] = []


def _stub_boundary_deps(monkeypatch) -> Any:
    pp = _mod("postpipeline")
    monkeypatch.setattr(pp, "quarantine_and_close", lambda payload, **_k: _Result())
    monkeypatch.setattr(pp, "write_quarantine_artifacts", lambda result, outputs_dir: {"a": 1})
    monkeypatch.setattr(pp, "quarantine_review_flags", lambda report: [])
    return pp


def test_quarantine_boundary_records_its_decision_in_the_given_state(monkeypatch, tmp_path) -> None:
    pp = _stub_boundary_deps(monkeypatch)
    state: Dict[str, Any] = {"unrelated": 1}
    result = pp.run_quarantine_boundary({}, strict_db=True, outputs_dir=tmp_path,
                                        pathway_context={"pathway_name": "p"}, state=state)
    assert list(state) == ["unrelated", *QUARANTINE_KEYS]
    assert state["quarantine_report"] is _Result.quarantine_report
    assert state["quarantine_artifacts"] == {"a": 1}
    assert state["quarantine_ok"] is True
    assert state["quarantine_decision_controls"]["decision_id"] == "d"
    assert state["quarantine_decision_controls"]["pathway_context"] == {"pathway_name": "p"}
    # No state given: same result, written to a throwaway mapping, nothing else touched.
    again = pp.run_quarantine_boundary({}, strict_db=True, outputs_dir=tmp_path,
                                       pathway_context={"pathway_name": "p"})
    assert again == result


def test_freeze_passes_its_state_through_to_the_boundary(monkeypatch, tmp_path) -> None:
    pp = _stub_boundary_deps(monkeypatch)
    state: Dict[str, Any] = {}
    pp.freeze_canonical_payload({"processes": {}}, strict_db=True, outputs_dir=tmp_path,
                                pathway_context=None, export_mode="pathwhiz",
                                research_mode=False, tmp=tmp_path, state=state)
    assert list(state) == QUARANTINE_KEYS


def test_a_rag_round_restores_the_state_it_was_given(monkeypatch, tmp_path) -> None:
    """The classification adapter hands its OWN state to the boundary and restores it
    afterwards: sentinels come back, an absent key stays absent."""

    rag = _mod("rag_orchestration")
    seen: List[Any] = []

    def _boundary(graph, *, state, **_kwargs):
        seen.append(state)
        for key in QUARANTINE_KEYS:
            state[key] = "written by the round"
        return {"quarantine_report": {"r": 1}, "coverage": {"c": 1}}

    monkeypatch.setattr(rag, "run_quarantine_boundary", _boundary)
    session: Dict[str, Any] = {"quarantine_report": "sentinel-report", "quarantine_ok": "sentinel-ok"}
    channel: Dict[str, Any] = {"round_index": 0}
    stages = rag.build_rag_round_stages(channel, scratch_root=tmp_path, state=session)
    graph = {"processes": {"reactions": []}}
    assert stages.classification(graph) is graph
    assert seen == [session]
    assert session == {"quarantine_report": "sentinel-report", "quarantine_ok": "sentinel-ok"}
    assert channel["classification"] == {"r": 1} and channel["coverage"] == {"c": 1}
    assert (tmp_path / "rag_rounds" / "round_00" / "quarantine").is_dir()


def test_maybe_run_rag_narrates_through_the_reporter_it_is_given(monkeypatch) -> None:
    """Same labels and arguments ``st.status`` received before R2, on a chain failure."""

    rag = _mod("rag_orchestration")
    from t2pw.rag import acquire, triage

    monkeypatch.setattr(rag, "rag_config", lambda *a, **k: {"enabled": True})
    monkeypatch.setattr(triage, "should_run_rag",
                        lambda *_a, **_k: type("D", (), {"run": True})())

    def _boom(*_a, **_k):
        raise RuntimeError("network down")

    monkeypatch.setattr(acquire, "search_candidates", _boom)
    calls: List[tuple] = []

    class _Status:
        def write(self, body):
            calls.append(("write", (body,), {}))

        def update(self, **kwargs):
            calls.append(("update", (), kwargs))

    class _Reporter:
        def progress(self, label, **kwargs):
            calls.append(("status", (label,), kwargs))
            return _Status()

        def warning(self, body):
            calls.append(("warning", (body,), {}))

    result = rag.maybe_run_rag(pathway_context={"pathway_name": "p"}, user_flag=True,
                               seed_payload={}, reporter=_Reporter())
    assert result.error == "network down"
    assert calls == [
        ("status", ("Multi-paper RAG: searching literature for candidate papers…",),
         {"expanded": True}),
        ("update", (), {"label": "Multi-paper RAG could not complete (continuing without it): "
                                  "network down", "state": "error"}),
    ]
