"""Stage R0 — triage: decide when the RAG subsystem runs (WP7).

This module holds the **one piece of RAG decision logic** the separation
invariant permits outside the orchestrator (docs/rag/03_separation_invariant.md).
The orchestrator (``streamlit_app.py``, seam **S5**) is wiring only; it *calls*
:func:`should_run_rag` and branches on the returned decision. It never re-derives
that decision itself.

R0 reads two read-only inputs and decides whether the seed pathway is
novel/incomplete enough to warrant multi-paper RAG:

* **the user's explicit flag** — "this pathway is unknown / incomplete" — which
  always wins; and
* **auto-triage signals** — a low Stage-0 ``scope_clarity_score`` (the target is
  ambiguously defined) and, when post-extraction reports are supplied, the WP4
  gap signals (dangling reactions, orphan metabolites, unmapped enzymes) that
  mark an incomplete graph.

Everything here is a *read* of Stage-0 context and the core's read-only reports
(seam **S4**): it never mutates Stage 0, a report, or a payload. Importing this
module is dependency-light — it pulls in no chromadb, no LLM client, and defers
the WP4 gap detector to a lazy import used only when reports are passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["TriageDecision", "should_run_rag", "LOW_SCOPE_CLARITY"]

# ``scope_clarity_score`` is a 0.0–1.0 Stage-0 measure of how unambiguously the
# extraction TARGET is defined (see llm/prompts/preprocess_system.txt: a clear
# primary paper scores >= 0.7, an ambiguous review as low as 0.2). Below this
# floor the seed pathway's scope is unclear enough that pulling in related
# papers is warranted.
LOW_SCOPE_CLARITY = 0.5

# Gap kinds (WP4 classification) that AUTO-TRIGGER RAG when the UI toggle is OFF
# and should_run_rag is left to decide from graph signals alone. These are the
# *strong* structural-incompleteness signals only: a reaction with no partner
# (dangling_reaction), an enzyme that never resolved (unmapped_enzyme), or a
# missing upstream feeder (missing_precursor). ``missing_compartment`` /
# ``orphan_metabolite`` are excluded — they occur in most healthy pathways.
_INCOMPLETE_GAP_KINDS = {
    "dangling_reaction",
    "unmapped_enzyme",
    "missing_precursor",
}

# Gap SOURCES trustworthy enough to AUTO-START RAG unattended. This is the
# load-bearing filter, not the kind set above.
#
# The orchestrator runs triage at the R0 seam — BEFORE Stage-6 id mapping — and
# feeds it a ``qa_graph`` computed on the raw Stage-1 payload (streamlit_app.py:
# maybe_run_rag call site). On that pre-mapping payload EVERY structural gap kind
# is ubiquitous, so the kind set alone cannot tell an incomplete pathway from a
# healthy one:
#   * ``unmapped_enzyme`` — derived from ``flags.missing_ids``: no entity carries
#     mapped_ids yet (mapping is Stage 6), so it fires for *every* protein.
#   * ``missing_precursor`` — from ``flags.empty_reactions``: shape-fragile.
#   * ``dangling_reaction`` / ``orphan_metabolite`` — from connectivity: *every*
#     pathway has an entry substrate and a terminal product (open ends), which are
#     exactly what gap RETRIEVAL later searches to extend the pathway. Universal
#     by design, not a sign of incompleteness.
# So a gap is only allowed to START RAG when it comes from a POST-resolution
# report — the Stage-3 ``gate`` errors or the ``mapping`` report — where
# "unmapped" means an entity that genuinely failed to resolve, not one that simply
# has not been mapped yet. The pre-mapping ``qa_graph`` / payload-connectivity
# gaps still drive gap RETRIEVAL once RAG is running; they just must not be the
# thing that starts it. Because the current orchestrator supplies only a
# pre-mapping ``qa_graph``, the effective auto-trigger today is
# ``scope_clarity_score`` (an ambiguous/novel seed) — the one reliable signal at
# this seam — plus the explicit user flag; a reliable gap-based auto-start becomes
# live the moment a gate/mapping report is wired into the reports mapping.
_AUTO_TRIGGER_GAP_SOURCES = {"gate", "mapping"}


@dataclass
class TriageDecision:
    """The R0 outcome: whether RAG runs, and why.

    ``run`` is the boolean the orchestrator branches on; ``reason`` is a
    human-readable explanation for the UI / logs; ``signals`` is the list of the
    individual triggers that fired (empty when ``run`` is ``False``). The
    instance is truthy iff ``run`` so it drops straight into the orchestrator's
    ``rag_config()["enabled"] and should_run_rag(...)`` guard.
    """

    run: bool
    reason: str
    signals: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.run


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any) -> Optional[float]:
    """Parse ``value`` to a float, or ``None`` if it is absent/uninterpretable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gap_signals(reports: Dict[str, Any]) -> List[str]:
    """Summarize the incomplete-graph gap kinds present in ``reports`` (read-only).

    Delegates classification to the WP4 gap detector (imported lazily so this
    module stays import-cheap and chromadb-free) and returns one signal string
    per distinct incompleteness kind found. Any failure degrades to an empty
    list — triage must never raise on a malformed report.
    """
    try:
        from t2pw.rag.retrieve import detect_gaps
    except Exception:  # noqa: BLE001 - a missing/broken retriever must not sink triage
        return []

    seed_payload = _safe_dict(reports.get("payload"))
    try:
        gaps = detect_gaps(seed_payload, reports)
    except Exception:  # noqa: BLE001
        return []

    counts: Dict[str, int] = {}
    for gap in gaps:
        kind = getattr(gap, "kind", "")
        source = getattr(gap, "source", "")
        # Both filters are required: a *strong* kind AND a *reliable* (post-
        # resolution) source. Pre-mapping qa_graph / payload-connectivity gaps are
        # ubiquitous and must not auto-start RAG — see _AUTO_TRIGGER_GAP_SOURCES.
        if kind in _INCOMPLETE_GAP_KINDS and source in _AUTO_TRIGGER_GAP_SOURCES:
            counts[kind] = counts.get(kind, 0) + 1

    return [f"{kind} x{counts[kind]}" for kind in sorted(counts)]


def should_run_rag(
    context: Any,
    user_flag: Any,
    reports: Optional[Dict[str, Any]] = None,
) -> TriageDecision:
    """Decide whether to run the RAG subsystem for this seed pathway (R0).

    ``context`` is the Stage-0 preprocessor output dict (read for
    ``scope_clarity_score``). ``user_flag`` is the user's explicit
    "unknown / incomplete" checkbox. ``reports`` (optional) is a mapping of the
    core's read-only gap reports — ``qa_graph`` / ``gate`` / ``mapping`` and,
    optionally, the seed ``payload`` — inspected via the WP4 gap detector.

    An explicit flag always triggers RAG. Otherwise RAG auto-triggers when the
    Stage-0 scope clarity is below :data:`LOW_SCOPE_CLARITY` (an ambiguous/novel
    seed), or when a *strong* structural-incompleteness gap
    (:data:`_INCOMPLETE_GAP_KINDS`) is reported from a *reliable, post-resolution*
    source (:data:`_AUTO_TRIGGER_GAP_SOURCES` — the Stage-3 ``gate`` or the
    ``mapping`` report). Gaps derived from a pre-mapping ``qa_graph`` / payload
    connectivity are intentionally NOT auto-triggers: they are ubiquitous on the
    unmapped Stage-1 payload the orchestrator supplies (every enzyme is unmapped,
    every pathway has open ends) and would fire on essentially every pathway.
    Scope clarity is therefore the effective auto-trigger at the current R0 seam.
    A clear, complete pathway returns ``run=False`` and RAG never runs — the
    orchestrator falls through to today's single-paper flow, untouched.
    """
    if user_flag:
        return TriageDecision(
            run=True,
            reason="explicit user flag: pathway marked unknown / incomplete",
            signals=["user_flag"],
        )

    signals: List[str] = []

    ctx = _safe_dict(context)
    score = _as_float(ctx.get("scope_clarity_score"))
    if score is not None and score < LOW_SCOPE_CLARITY:
        signals.append(f"low scope_clarity_score={score:g} (< {LOW_SCOPE_CLARITY:g})")

    if reports:
        signals.extend(_gap_signals(_safe_dict(reports)))

    if signals:
        return TriageDecision(
            run=True,
            reason="auto-triage: " + "; ".join(signals),
            signals=signals,
        )

    return TriageDecision(
        run=False,
        reason="pathway looks complete and in-scope; RAG not triggered",
        signals=[],
    )
