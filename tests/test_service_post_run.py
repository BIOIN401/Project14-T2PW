"""R4b: ``t2pw.service.post_run.run_post_pipeline``.

NEW TESTS (R4b, new module). ``run_post_pipeline`` is a pure move of the app's "Run audit
and DB mapping" handler after the quarantine reset: the recorder re-entry, the audit and
mapping call with every coercion, the presentation of each verdict, the two failure
handlers and the diagnostics sync.

These tests pin what that code did: every argument, every message and its order, every
state write and its order against the messages and callbacks, the rebound payload, and
that every callee is looked up on its SOURCE module at call time (the R3b review F1 rule;
the harness spies rebind it there). The unit tests are offline. The last tests drive the
REAL ``streamlit_app.py`` through ``AppTest``, including the audit F5 check that the app's
call passes every widget value through. This file runs in its own bounded process.
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
import types
from pathlib import Path
from typing import Any, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
APP_PATH = SRC / "t2pw" / "app" / "streamlit_app.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import t2pw.config as config  # noqa: E402
import t2pw.paths as paths  # noqa: E402
import t2pw.pipeline.export_mode as export_mode  # noqa: E402
import t2pw.pipeline.extraction_diagnostics as diag  # noqa: E402
import t2pw.pipeline.pipeline as pipeline  # noqa: E402
import t2pw.service.extraction as extraction  # noqa: E402
import t2pw.service.postpipeline as postpipeline  # noqa: E402
from t2pw.pipeline.gate_reports import CANONICAL_PAYLOAD_KEY, FINAL_GATE_REPORT_KEY  # noqa: E402
from t2pw.pipeline.stage_contracts import StageContractError  # noqa: E402
from t2pw.service import post_run  # noqa: E402

CTX = {"pathway_name": "Menaquinone", "likely_organism": "Escherichia coli"}
PAYLOAD = {"entities": {}, "processes": {"reactions": [{"name": "a"}]}}
REBOUND = {"entities": {}, "processes": {"reactions": [{"name": "a"}]}, "organism": "E. coli"}
RECORDER = object()


def _contract_error() -> StageContractError:
    return StageContractError("post_mapping", "contract broke", {"errors": [], "summary": {}, "stage": "post_mapping"})


class _Ctx:
    def __init__(self, events: List[tuple], name: str) -> None:
        self._events, self._name = events, name

    def __enter__(self) -> None:
        self._events.append((f"{self._name}:enter",))

    def __exit__(self, *exc: Any) -> None:
        self._events.append((f"{self._name}:exit",))


class _Reporter:
    def __init__(self, events: List[tuple]) -> None:
        self._events = events

    def __getattr__(self, name: str) -> Any:
        def _call(*args: Any, **kwargs: Any) -> Any:
            self._events.append((name, args, kwargs) if kwargs else (name, args))
            return _Ctx(self._events, name) if name in ("spinner", "expander") else None
        return _call


class _State(dict):
    """``state`` that logs writes and pops into the shared event list."""

    def __init__(self, events: List[tuple], *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.events = events

    def __setitem__(self, key: str, value: Any) -> None:
        self.events.append(("set", key))
        super().__setitem__(key, value)


def _request(**kw: Any) -> post_run.PostRunRequest:
    values = {
        "final_payload": PAYLOAD,
        "use_llm_audit": 0,
        "use_sbml_overwatch": 1,
        "default_compartment": "  nucleus  ",
        "mapping_cache_text": "  cache.json ",
        "id_source_mode": " DB ",
        "db_host": " host ",
        "db_port": 3307.0,
        "db_user": " user ",
        "db_password": "pw",
        "db_schema": " schema ",
        "audit_max_rounds": 7.0,
        "audit_timeout_seconds": 300.0,
        "audit_candidate_count": 5.0,
        "use_example_retrieval": 1,
        "example_index_path": " idx.json ",
        "example_top_k": 6.0,
        "use_gap_resolver": 0,
        "use_llm_gap_resolver": 1,
        "gap_resolver_max_items": 120.0,
        "use_stoich_agent": 1,
    }
    values.update(kw)
    return post_run.PostRunRequest(**values)


@pytest.fixture()
def world(monkeypatch, tmp_path):
    """Every callee patched on its SOURCE module, one shared ordered event log."""

    w = types.SimpleNamespace(events=[], artifacts={}, raises=None, propagate_raises=None, review_raises=None,
                              rag=False, call=None, tmp=tmp_path)

    def propagate(payload, ctx):
        w.events.append(("propagate", payload, ctx))
        if w.propagate_raises is not None:
            raise w.propagate_raises
        return REBOUND

    def artifacts(payload, **kw):
        w.call = (payload, kw)
        w.events.append(("artifacts", payload))
        if w.raises is not None:
            raise w.raises
        return w.artifacts

    def snapshot(payload, *, stage, reason):
        w.events.append(("snapshot", payload, stage, reason))

    def reenter(recorder):
        w.events.append(("reenter", recorder))

    def sync(*, state):
        w.events.append(("sync", state is w.state))

    monkeypatch.setattr(pipeline, "propagate_context_organism", propagate)
    monkeypatch.setattr(postpipeline, "run_post_pipeline_sbml_artifacts", artifacts)
    monkeypatch.setattr(postpipeline, "_record_mapped_failure_snapshot", snapshot)
    monkeypatch.setattr(diag, "reenter", reenter)
    monkeypatch.setattr(extraction, "_sync_extraction_diagnostics", sync)
    monkeypatch.setattr(config, "rag_config", lambda *a, **k: {"enabled": w.rag})
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    w.state = _State(w.events, {"pathway_context": dict(CTX), "extraction_diagnostics_recorder": RECORDER})

    def review(payload, report):
        w.events.append(("open_review", payload, report))
        if w.review_raises is not None:
            raise w.review_raises

    w.review = review
    w.run = lambda **kw: post_run.run_post_pipeline(
        kw.pop("request", _request()),
        state=w.state,
        reporter=_Reporter(w.events),
        open_review=w.review,
        render_issue_detail=lambda detail: w.events.append(("render_issue_detail", detail)),
        render_contract_failure=lambda failure: w.events.append(("render_contract_failure", failure)),
    )
    return w


def _names(events: List[tuple], *skip: str) -> List[str]:
    """Event kinds, in order, without the two the spinner adds around every run."""

    drop = {"propagate", "artifacts", "reenter", "spinner", "spinner:enter", "spinner:exit", *skip}
    return [e[0] for e in events if e[0] not in drop]


def test_the_handler_spends_the_run_in_order_and_coerces_every_argument(world):
    world.state.update(qa_report={"qa": 1}, reaction_summary="sum", export_mode="research", pwml_strict_db=0)
    result = world.run()

    assert world.events[:5] == [
        ("reenter", RECORDER),
        ("spinner", ("Running audit and DB mapping...",)),
        ("spinner:enter",),
        ("propagate", PAYLOAD, CTX),
        ("artifacts", REBOUND),
    ]
    assert world.events[5] == ("spinner:exit",)
    payload, kwargs = world.call
    assert payload is REBOUND
    assert kwargs == {
        "build_legacy_sbml": False,
        "use_llm_audit": False,
        "use_sbml_overwatch": True,
        "default_compartment": "nucleus",
        "mapping_cache_path": "cache.json",
        "id_source": "db",
        "db_host": "host",
        "db_port": 3307,
        "db_user": "user",
        "db_password": "pw",
        "db_schema": "schema",
        "audit_max_rounds": 7,
        "audit_timeout_seconds": 300,
        "audit_candidate_count": 5,
        "use_example_retrieval": True,
        "example_index_path": "idx.json",
        "example_top_k": 6,
        "use_gap_resolver": False,
        "use_llm_gap_resolver": True,
        "gap_resolver_max_items": 120,
        "qa_report": {"qa": 1},
        "reaction_summary": "sum",
        "use_stoich_agent": True,
        "rag_evidence_context": "",
        "export_mode": export_mode.coerce_mode("research"),
        "strict_db": False,
        "pathway_context": CTX,
        "outputs_dir": world.tmp / "outputs",
        "state": world.state,
    }
    assert result.final_payload is REBOUND and result.failure is None


@pytest.mark.parametrize("field, value, kw, expected", [
    ("default_compartment", None, "default_compartment", "cell"),
    ("default_compartment", "   ", "default_compartment", "cell"),
    ("mapping_cache_text", "   ", "mapping_cache_path", "id_mapping_cache.json"),
    ("id_source_mode", None, "id_source", "hybrid"),
    ("db_host", None, "db_host", ""),
    ("db_user", None, "db_user", ""),
    ("db_password", None, "db_password", ""),
    ("db_schema", None, "db_schema", "pathbank"),
    ("db_schema", "  ", "db_schema", "pathbank"),
    ("example_index_path", None, "example_index_path", ""),
])
def test_blank_widget_values_get_the_apps_defaults(world, field, value, kw, expected):
    world.run(request=_request(**{field: value}))
    assert world.call[1][kw] == expected


def test_the_rag_context_is_read_at_call_time_and_only_with_rag_on(world):
    world.state["rag_evidence_context"] = "EVIDENCE"
    world.run()
    assert world.call[1]["rag_evidence_context"] == ""
    world.rag = True
    world.run()
    assert world.call[1]["rag_evidence_context"] == "EVIDENCE"
    del world.state["rag_evidence_context"]
    world.run()
    assert world.call[1]["rag_evidence_context"] == ""


def test_strict_db_defaults_to_true_when_the_checkbox_never_rendered(world):
    world.run()
    assert world.call[1]["strict_db"] is True


def test_the_stored_artifacts_are_the_object_later_writes_mutate(world):
    world.state["final_payload_snapshot"] = {"snap": 1}
    world.artifacts.update(gate_failed=True)
    world.run()
    assert world.state["post_pipeline_artifacts"] is world.artifacts
    assert world.artifacts["final_payload_snapshot"] == {"snap": 1}


def test_the_snapshot_defaults_to_the_rebound_payload(world):
    world.artifacts.update(gate_failed=True)
    world.run()
    assert world.artifacts["final_payload_snapshot"] is REBOUND


def test_post_audit_reports_replace_the_session_copies_only_when_present(world):
    world.state.update(qa_report={"old": 1}, reaction_summary="old")
    world.artifacts.update(gate_failed=True, post_audit_qa_report={"new": 1})
    world.run()
    assert world.state["qa_report"] == {"new": 1} and world.state["reaction_summary"] == "old"
    world.artifacts.update(post_audit_qa_report=None, post_audit_reaction_summary="new")
    world.run()
    assert world.state["qa_report"] == {"new": 1} and world.state["reaction_summary"] == "new"


def test_a_gate_failure_only_warns(world):
    world.artifacts.update(gate_failed=True)
    world.run()
    assert [e for e in world.events if e[0] in ("warning", "info", "error", "open_review")] == [
        ("warning", ("Post-pipeline stopped at hard-gate validation. Review gate_fail_report.json.",))]
    assert world.events[-1] == ("sync", True)


def test_no_mapped_payload_says_nothing_and_syncs(world):
    world.run()
    assert _names(world.events) == ["set", "sync"]
    assert world.state.get("refinement_gate_errors") is None


def test_quarantine_flags_are_reported_before_the_verdict(world):
    world.artifacts.update({CANONICAL_PAYLOAD_KEY: {"p": 1}, "quarantine_ok": True,
                            "quarantine_review_flags": ["f1", "f2"]})
    world.run()
    assert world.events[6:9] == [
        ("set", "post_pipeline_artifacts"),
        ("warning", ("Quarantine review flags (2): research mode kept every process. "
                     "A strict export would have removed the ones listed below.",)),
        ("json", (["f1", "f2"],)),
    ]


def test_a_quarantine_refusal_stops_before_review(world):
    world.artifacts.update({
        CANONICAL_PAYLOAD_KEY: {"p": 1}, "quarantine_ok": False, "quarantine_refusal_reasons": ["no core", 7],
        "quarantine_coverage": {"cov": 1}, "quarantine_report": {"strict_invariants": {"inv": 1}},
    })
    world.run()
    assert world.state["refinement_gate_errors"] == [
        {"path": "/processes", "reason": "quarantine refusal: no core"},
        {"path": "/processes", "reason": "quarantine refusal: 7"},
    ]
    assert [e for e in world.events if e[0] in ("error", "caption", "json", "open_review", "sync")] == [
        ("error", ("Quarantine could not produce a viable requested-pathway core: no core; 7",)),
        ("caption", ("Requested-core coverage",)),
        ("json", ({"cov": 1},)),
        ("caption", ("Strict invariants",)),
        ("json", ({"inv": 1},)),
        ("sync", True),
    ]


GATE = {"ok": False, "errors": [{"path": "/a", "reason": "bad", "detail": {"d": 1}}, "plain"]}


def test_research_mode_fails_open_on_stage_three_and_opens_review(world):
    world.state["export_mode"] = "research"
    world.artifacts.update({CANONICAL_PAYLOAD_KEY: {"p": 1}, "quarantine_ok": True, FINAL_GATE_REPORT_KEY: GATE,
                            "mapping_report": {"m": 1}})
    world.run()
    assert world.artifacts["research_stage3_review_flags"] == GATE["errors"]
    assert world.artifacts["research_stage3_failed_open"] is True
    assert world.state["refinement_gate_errors"] == []
    tail = [e for e in world.events if e[0] in ("warning", "caption", "markdown", "json", "open_review", "sync")]
    assert tail == [
        ("warning", ("Research mode: the pre-export Stage 3 revalidation FAILED (2 error(s)) and was not enforced. "
                     "The strict gates did NOT pass — this candidate is not PathWhiz-exportable as it stands. "
                     "Refinement review is open so the findings below can be reviewed against the pathway itself.",)),
        ("caption", ("Stage 3 findings retained as review flags",)),
        ("markdown", ("- `/a` — bad",)),
        ("markdown", ("- plain",)),
        ("json", (GATE,)),
        ("open_review", {"p": 1}, {"m": 1}),
        ("sync", True),
    ]


def test_strict_stage_three_failure_shows_the_errors_and_no_review(world):
    world.artifacts.update({CANONICAL_PAYLOAD_KEY: {"p": 1}, "quarantine_ok": True, FINAL_GATE_REPORT_KEY: GATE})
    world.run()
    assert world.state["refinement_gate_errors"] == GATE["errors"]
    names = [e for e in world.events if e[0] not in ("reenter", "propagate", "artifacts", "spinner", "spinner:enter",
                                                     "spinner:exit", "set")]
    assert names == [
        ("error", ("Mapped pathway still fails the pre-export Stage 3 revalidation. Refinement review was not opened.",)),
        ("expander", ("Why Stage 3 failed (2 error(s))",), {"expanded": True}),
        ("expander:enter",),
        ("markdown", ("- `/a` — bad",)),
        ("render_issue_detail", {"d": 1}),
        ("markdown", ("- plain",)),
        ("json", (GATE,)),
        ("expander:exit",),
        ("sync", True),
    ]


def test_strict_stage_three_failure_with_no_itemised_errors(world):
    world.artifacts.update({CANONICAL_PAYLOAD_KEY: {"p": 1}, "quarantine_ok": True,
                            FINAL_GATE_REPORT_KEY: {"ok": False, "errors": []}})
    world.run()
    assert ("write", ("No itemized errors were returned; showing full gate report below.",)) in world.events


def test_a_passing_run_clears_the_errors_then_opens_review_then_says_so(world):
    world.artifacts.update({CANONICAL_PAYLOAD_KEY: {"p": 1}, "quarantine_ok": True,
                            FINAL_GATE_REPORT_KEY: {"ok": True}, "mapping_report": {"m": 2}})
    world.state["refinement_gate_errors"] = ["stale"]
    world.run()
    assert world.state["refinement_gate_errors"] == []
    # the seed above, post_pipeline_artifacts, refinement_gate_errors
    assert _names(world.events) == ["set", "set", "set", "open_review", "info", "sync"]
    assert world.events[-2:] == [
        ("info", ("Mapped pathway is ready for review. PWML generation is paused until approval.",)),
        ("sync", True),
    ]


def test_final_mapped_is_the_fallback_review_payload(world):
    world.artifacts.update({"final_mapped": {"fallback": 1}, "quarantine_ok": True, FINAL_GATE_REPORT_KEY: {"ok": True}})
    world.run()
    assert ("open_review", {"fallback": 1}, {}) in world.events


def test_a_stage_contract_failure_is_rendered_before_the_sync_and_returned(world):
    world.raises = _contract_error()
    result = world.run()
    assert result.failure is world.raises and result.final_payload is REBOUND
    assert not any(e[0] == "snapshot" for e in world.events)
    assert world.events[-2:] == [("render_contract_failure", world.raises), ("sync", True)]
    assert "post_pipeline_artifacts" not in world.state


def test_an_exception_is_recorded_against_the_rebound_payload_and_shown(world):
    world.raises = RuntimeError("boom")
    result = world.run()
    assert result.failure is world.raises and result.final_payload is REBOUND
    assert world.events[-4:] == [
        ("snapshot", REBOUND, "post_pipeline", "RuntimeError: boom"),
        ("error", ("Post-pipeline conversion failed: boom",)),
        ("exception", (world.raises,)),
        ("sync", True),
    ]


def test_a_propagation_failure_keeps_the_original_payload(world):
    world.propagate_raises = RuntimeError("no organism")
    result = world.run()
    assert result.final_payload is PAYLOAD
    assert world.events[-4:][0] == ("snapshot", PAYLOAD, "post_pipeline", "RuntimeError: no organism")
    assert not any(e[0] == "artifacts" for e in world.events)


def test_a_failing_review_lands_in_the_generic_handler(world):
    world.artifacts.update({CANONICAL_PAYLOAD_KEY: {"p": 1}, "quarantine_ok": True, FINAL_GATE_REPORT_KEY: {"ok": True}})
    world.review_raises = RuntimeError("render failed")
    result = world.run()
    assert result.failure is world.review_raises
    assert ("error", ("Post-pipeline conversion failed: render failed",)) in world.events
    assert not any(e[0] == "info" for e in world.events)
    assert world.events[-1] == ("sync", True)


def test_a_failing_contract_render_still_syncs_and_propagates(world):
    world.raises = _contract_error()
    boom = ValueError("render")

    def bad(failure):
        raise boom

    with pytest.raises(ValueError):
        post_run.run_post_pipeline(_request(), state=world.state, reporter=_Reporter(world.events), open_review=world.review,
                                   render_issue_detail=lambda d: None, render_contract_failure=bad)
    assert world.events[-1] == ("sync", True)


def test_reenter_sits_outside_the_try(world, monkeypatch):
    boom = RuntimeError("recorder")

    def reenter(recorder):
        raise boom

    monkeypatch.setattr(diag, "reenter", reenter)
    with pytest.raises(RuntimeError):
        world.run()
    assert world.events == []


def test_the_handler_does_not_write_the_final_payload_key(world):
    world.run()
    assert "final_payload" not in world.state


def test_mode_lookups_are_made_on_the_source_module_at_call_time(world, monkeypatch):
    """The R3b review F1 rule for the two names the world fixture does not already patch."""

    monkeypatch.setattr(export_mode, "coerce_mode", lambda value: "sentinel-mode")
    research = []
    monkeypatch.setattr(export_mode, "is_research", lambda mode: research.append(mode) or True)
    world.artifacts.update({CANONICAL_PAYLOAD_KEY: {"p": 1}, "quarantine_ok": True,
                            FINAL_GATE_REPORT_KEY: {"ok": False, "errors": []}})
    world.run()
    assert world.call[1]["export_mode"] == "sentinel-mode"
    assert research == ["sentinel-mode"]
    assert world.artifacts["research_stage3_failed_open"] is True

def test_the_signature_is_keyword_only_with_no_defaults():
    params = inspect.signature(post_run.run_post_pipeline).parameters
    for name in ("state", "reporter", "open_review", "render_issue_detail", "render_contract_failure"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY and params[name].default is inspect.Parameter.empty, name
    assert not any(f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
                   for f in dataclasses.fields(post_run.PostRunRequest))


def test_the_request_repr_never_shows_the_password():
    assert "s3cret" not in repr(_request(db_password="s3cret"))


# ---------------------------------------------------------------------------
# The real app. No LLM, no DB: the post-pipeline call itself is the fake.
# ---------------------------------------------------------------------------
WIDGETS = {
    "post_use_llm_audit": False,
    "post_use_sbml_overwatch": True,
    "post_use_stoich_agent": True,
    "post_default_compartment": "  nucleus ",
    "post_mapping_cache": "  my_cache.json ",
    "post_audit_max_rounds": 7,
    "post_audit_timeout_seconds": 300,
    "post_audit_candidate_count": 5,
    "post_use_example_retrieval": True,
    "post_example_index_path": " idx.json ",
    "post_example_top_k": 6,
    "post_use_gap_resolver": False,
    "post_use_llm_gap_resolver": False,
    "post_gap_resolver_max_items": 120,
    "post_mapping_source": "db",
    "post_db_host": " host ",
    "post_db_port": 3307,
    "post_db_user": " user ",
    "post_db_schema": " schema ",
    "post_db_password": "pw",
}
EXPECTED = {
    "use_llm_audit": False, "use_sbml_overwatch": True, "use_stoich_agent": True, "default_compartment": "nucleus",
    "mapping_cache_path": "my_cache.json", "audit_max_rounds": 7, "audit_timeout_seconds": 300,
    "audit_candidate_count": 5, "use_example_retrieval": True, "example_index_path": "idx.json", "example_top_k": 6,
    "use_gap_resolver": False, "use_llm_gap_resolver": False, "gap_resolver_max_items": 120, "id_source": "db",
    "db_host": "host", "db_port": 3307, "db_user": "user", "db_schema": "schema", "db_password": "pw",
}


def _real_app(monkeypatch, tmp_path, *, payload, widgets=None, patched=None):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("RAG_ENABLED", "0")
    calls = []

    def fake(final_payload, **kw):
        calls.append((final_payload, kw))
        if patched is not None:
            patched()
        return {"gate_failed": True}

    monkeypatch.setattr(postpipeline, "run_post_pipeline_sbml_artifacts", fake)
    at = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
    at.session_state["pipeline_ready"] = True
    at.session_state["final_payload"] = payload
    at.session_state["final_payload_snapshot"] = payload
    at.session_state["pathway_context"] = dict(CTX)
    at.session_state["stage_one"] = payload
    at.session_state["stage_two"] = None
    at.session_state["stage_two_chunks"] = []
    at.session_state["stage_two_rounds"] = []
    at.session_state["chunk_details"] = []
    at.session_state["qa_hints"] = None
    at.session_state["pipeline_source_text"] = "seeded"
    for key, value in (widgets or {}).items():
        at.session_state[key] = value
    at.run()
    at.button(key="pwml_generate_btn").click().run()
    return at, calls


def _graph_payload():
    return {"entities": {"compounds": [{"name": "A"}]},
            "processes": {"reactions": [{"name": "r", "inputs": ["A"], "outputs": ["A"]}]}}


def test_real_app_passes_every_widget_value_through_to_the_pipeline(monkeypatch, tmp_path):
    """The audit's F5 gap, for this call site: each widget, set to a non-default, reaches the
    pipeline through the app's PostRunRequest, by name, with no widget wired to another's value."""

    payload = _graph_payload()
    at, calls = _real_app(monkeypatch, tmp_path, payload=payload, widgets=WIDGETS)
    assert not at.exception
    assert len(calls) == 1
    passed = calls[0][1]
    assert {key: passed[key] for key in EXPECTED} == EXPECTED
    assert any("hard-gate" in str(w.value) for w in at.warning)


def test_real_app_warns_when_there_is_no_payload(monkeypatch, tmp_path):
    at, calls = _real_app(monkeypatch, tmp_path, payload={})
    assert calls == []
    assert any("No pipeline output in session state" in str(e.value) for e in at.error)


def test_real_app_renders_a_contract_failure_and_keeps_the_rebound_payload(monkeypatch, tmp_path):
    payload = _graph_payload()

    def raise_it():
        raise StageContractError("post_mapping", "contract broke", {"errors": [], "summary": {}, "stage": "post_mapping"})

    at, _calls = _real_app(monkeypatch, tmp_path, payload=payload, patched=raise_it)
    assert not at.exception
    assert any("Audit and DB mapping stopped at the" in str(e.value) for e in at.error)
    assert "post_pipeline_artifacts" not in at.session_state


def test_real_app_reads_the_rebound_payload_after_the_handler(monkeypatch, tmp_path):
    """Hazard: the failure path's rebound ``final_payload`` is what the rest of the page shows."""

    payload = _graph_payload()
    seen = []
    real = postpipeline.graph_summary

    def spy(p):
        seen.append(p)
        return real(p)

    monkeypatch.setattr(postpipeline, "graph_summary", spy)
    monkeypatch.setattr(pipeline, "propagate_context_organism",
                        lambda p, ctx: {**p, "rebound_marker": True})

    def raise_it():
        raise StageContractError("post_mapping", "contract broke", {"errors": [], "summary": {}, "stage": "post_mapping"})

    at, _calls = _real_app(monkeypatch, tmp_path, payload=payload, patched=raise_it)
    assert not at.exception
    assert seen[-1].get("rebound_marker") is True
