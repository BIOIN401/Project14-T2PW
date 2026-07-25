"""WP7 — R0 triage + the S5 orchestration guard (offline, deterministic).

Two things are proven here:

1. ``t2pw.rag.triage.should_run_rag`` — explicit flag on -> run; auto-triggers
   on a low ``scope_clarity_score`` or a strong gap from a *reliable* (post-
   resolution ``gate`` / ``mapping``) source -> run with a reason. Gaps derived
   from a pre-mapping ``qa_graph`` (ubiquitous: every enzyme unmapped, every
   pathway has open ends) do NOT auto-start RAG; a clean, in-scope pathway -> no
   run.
2. The orchestrator's guard: with ``RAG_ENABLED`` off, the thin app helper
   ``maybe_run_rag`` returns ``None`` (today's flow, untouched) and never enters
   the RAG chain (``acquire`` is not called); with RAG on + the flag, the chain
   *is* entered.

The app module runs a Streamlit script at import, and neither ``streamlit`` nor
``openai`` is present in the test env, so both are stubbed at the top (the
``openai`` stub mirrors tests/test_audit_json_llm_payload.py). Importing the app
pulls in no chromadb (the RAG chain imports are lazy, inside the guard).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *_: object, **__: object) -> None:
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **__: None)
            )

    openai_stub.OpenAI = _OpenAI
    openai_stub.RateLimitError = RuntimeError
    openai_stub.APIError = RuntimeError
    openai_stub.APITimeoutError = RuntimeError
    openai_stub.AuthenticationError = RuntimeError
    openai_stub.BadRequestError = RuntimeError
    sys.modules["openai"] = openai_stub

if "streamlit" not in sys.modules:
    _st = MagicMock(name="streamlit")
    _st.columns.side_effect = lambda spec=1, **__: [
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    _st.tabs.side_effect = lambda labels, **__: [MagicMock() for _ in labels]
    _st.form_submit_button.return_value = False
    _st.checkbox.return_value = False
    _st.session_state = MagicMock()
    _st.session_state.get.return_value = None
    _st.session_state.__contains__ = lambda _s, _k: False
    sys.modules["streamlit"] = _st

import t2pw.app.streamlit_app as app  # noqa: E402
from t2pw.rag import acquire  # noqa: E402
from t2pw.rag.triage import LOW_SCOPE_CLARITY, TriageDecision, should_run_rag  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Triage decision logic (lives in t2pw.rag, not the app).
# ---------------------------------------------------------------------------
def test_explicit_flag_runs_rag() -> None:
    decision = should_run_rag({"scope_clarity_score": 0.99}, True)
    assert isinstance(decision, TriageDecision)
    assert decision.run is True
    assert bool(decision) is True
    assert "user_flag" in decision.signals


def test_low_scope_clarity_auto_triggers() -> None:
    decision = should_run_rag({"scope_clarity_score": 0.2}, False)
    assert decision.run is True
    assert any("scope_clarity_score" in s for s in decision.signals)


def test_dangling_reaction_qa_report_does_not_auto_trigger() -> None:
    # A dangling_reaction derived from a pre-mapping qa_graph (source "qa_graph")
    # must NOT auto-start RAG: every pathway's entry substrate / terminal product
    # is an "open end", so connectivity gaps are ubiquitous on the seed payload.
    # They drive gap RETRIEVAL once RAG runs; they are not a reliable trigger.
    reports = {"qa_graph": {"dangling_nodes": [{"node": "reaction:#1"}]}}
    decision = should_run_rag({"scope_clarity_score": 0.9}, False, reports)
    assert decision.run is False
    assert not any("dangling_reaction" in s for s in decision.signals)


def test_orphan_metabolite_report_does_not_auto_trigger() -> None:
    # AMENDED SEMANTICS (audit §6.B): orphan_metabolite is a high-frequency signal
    # (terminal/boundary metabolites emit it in most healthy pathways), so it is
    # deliberately NOT in the auto-trigger set. With the toggle OFF (user_flag
    # False) and a clear scope, an orphan-metabolite-only report must NOT start RAG.
    reports = {"qa_graph": {"dangling_nodes": [{"node": "compound:pyruvate"}]}}
    decision = should_run_rag({"scope_clarity_score": 0.9}, False, reports)
    assert decision.run is False
    assert not any("orphan_metabolite" in s for s in decision.signals)


def test_missing_compartment_only_report_does_not_auto_trigger() -> None:
    # missing_compartment is emitted (via detect_gaps) for any entity with no
    # recorded subcellular location — common in healthy pathways — so it too is
    # excluded from the auto-trigger set. A report carrying ONLY that signal must
    # NOT start RAG.
    reports = {"qa_graph": {"flags": {"missing_compartments": [{"entity": "pyruvate"}]}}}
    decision = should_run_rag({"scope_clarity_score": 0.9}, False, reports)
    assert decision.run is False
    assert not any("missing_compartment" in s for s in decision.signals)


def test_missing_precursor_qa_report_does_not_auto_trigger() -> None:
    # missing_precursor derived from a pre-mapping qa_graph (flags.empty_reactions,
    # source "qa_graph") is shape-fragile and must NOT auto-start RAG. Only a
    # post-resolution gate/mapping source is trusted to auto-start.
    reports = {"qa_graph": {"flags": {"empty_reactions": [{"reaction": "queuine synthase reaction"}]}}}
    decision = should_run_rag({"scope_clarity_score": 0.9}, False, reports)
    assert decision.run is False
    assert not any("missing_precursor" in s for s in decision.signals)


def test_unmapped_enzyme_mapping_report_auto_triggers() -> None:
    # A genuinely-unmapped enzyme from a POST-resolution `mapping` report (source
    # "mapping", status="unmapped" == failed to resolve) IS a reliable auto-trigger.
    # This is the reliable-source counterpart to the pre-mapping qa_graph case below.
    reports = {"mapping": {"unmapped": [{"name": "NdmA", "type": "protein", "status": "unmapped"}]}}
    decision = should_run_rag({"scope_clarity_score": 0.9}, False, reports)
    assert decision.run is True
    assert any("unmapped_enzyme" in s for s in decision.signals)


def test_pre_mapping_qa_report_does_not_auto_trigger() -> None:
    # REGRESSION (adversarial review 2026-07-25): the orchestrator feeds triage a
    # qa_graph computed on the PRE-mapping Stage-1 payload, where NO entity has
    # mapped_ids yet -> flags.missing_ids lists every protein -> detect_gaps emits
    # unmapped_enzyme for all of them (and missing_precursor for empty reactions).
    # Before the source filter this auto-fired the whole R1-R5 chain on essentially
    # every protein-containing pathway with the toggle OFF. A clear-scope pathway
    # whose only "gaps" come from this pre-mapping qa_graph must NOT auto-start RAG.
    reports = {
        "qa_graph": {
            "flags": {
                "missing_ids": [
                    {"entity": f"Enzyme{i}", "type": "protein"} for i in range(17)
                ],
                "empty_reactions": [{"reaction": "rxn 1"}, {"reaction": "rxn 2"}],
            }
        }
    }
    decision = should_run_rag({"scope_clarity_score": 1.0}, False, reports)
    assert decision.run is False, f"pre-mapping qa_graph must not auto-fire: {decision.reason}"
    assert decision.signals == []


def test_clean_complete_pathway_does_not_run() -> None:
    decision = should_run_rag({"scope_clarity_score": 0.9}, False, reports=None)
    assert decision.run is False
    assert bool(decision) is False
    assert decision.signals == []


def test_scope_clarity_boundary_is_inclusive_of_clear() -> None:
    # Exactly at the floor is treated as clear (strict less-than triggers).
    decision = should_run_rag({"scope_clarity_score": LOW_SCOPE_CLARITY}, False)
    assert decision.run is False


def test_missing_or_garbage_context_is_safe() -> None:
    assert should_run_rag(None, False).run is False
    assert should_run_rag({"scope_clarity_score": "n/a"}, False).run is False


# ---------------------------------------------------------------------------
# 2. The S5 orchestration guard — RAG-off byte-identity of the code path.
# ---------------------------------------------------------------------------
def test_maybe_run_rag_returns_none_and_skips_chain_when_disabled(monkeypatch) -> None:
    """RAG off -> maybe_run_rag returns None and never touches the RAG chain."""
    called: list[bool] = []

    def _boom(*_a, **_k):
        called.append(True)
        raise AssertionError("acquire.search_candidates must NOT run with RAG off")

    monkeypatch.setattr(acquire, "search_candidates", _boom)
    # Force the master switch off regardless of the ambient environment.
    monkeypatch.setattr(app, "rag_config", lambda *a, **k: {"enabled": False})

    result = app.maybe_run_rag(
        pathway_context={"scope_clarity_score": 0.1},  # would auto-trigger IF enabled
        user_flag=True,                                 # and the flag is on, too
        seed_payload={},
        reports=None,
    )
    assert result is None
    assert called == []  # the chain was never entered


def test_maybe_run_rag_returns_none_when_triage_declines(monkeypatch) -> None:
    """RAG on but a clean pathway -> triage declines -> None, chain not entered."""
    called: list[bool] = []

    def _boom(*_a, **_k):
        called.append(True)
        raise AssertionError("acquire must not run when triage declines")

    monkeypatch.setattr(acquire, "search_candidates", _boom)
    monkeypatch.setattr(app, "rag_config", lambda *a, **k: {"enabled": True})

    result = app.maybe_run_rag(
        pathway_context={"scope_clarity_score": 0.95},  # clear scope, no flag
        user_flag=False,
        seed_payload={},
        reports=None,
    )
    assert result is None
    assert called == []


def test_maybe_run_rag_enters_chain_when_enabled_and_flagged(monkeypatch) -> None:
    """RAG on + flag -> the chain IS entered (acquire is called)."""
    called: list[bool] = []

    def _boom(*_a, **_k):
        called.append(True)
        raise RuntimeError("network down")  # exercised, then degraded to a result

    monkeypatch.setattr(acquire, "search_candidates", _boom)
    monkeypatch.setattr(app, "rag_config", lambda *a, **k: {"enabled": True})

    result = app.maybe_run_rag(
        pathway_context={"pathway_name": "novel pathway"},
        user_flag=True,
        seed_payload={},
        reports=None,
    )
    # acquire WAS called (chain entered); the network failure degraded to a
    # result object (never None, never a raised exception) so the core run is safe.
    assert called == [True]
    assert result is not None
    assert result.payload is None
    assert result.synthesized is False
    assert result.error
