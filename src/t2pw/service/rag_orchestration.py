"""RAG orchestration, moved out of ``streamlit_app.py`` (R2).

``maybe_run_rag``, ``run_rag_rounds`` and their RAG-round-only helpers, verbatim. The
only edits are the UI and session seams: ``st.status`` / ``st.warning`` became calls on
a :class:`~t2pw.service.reporter.Reporter`, and the quarantine-key save/restore in the
classification adapter works on an explicit ``state`` mapping instead of
``st.session_state``. ``reporter=None`` / ``state=None`` resolve through
:mod:`t2pw.service.reporter`. The app re-imports every name here under the same name.

Must not import ``streamlit`` or ``t2pw.app.streamlit_app``.
"""

import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Tuple

from t2pw.paths import PROJECT_ROOT
from t2pw.mapping.map_ids import map_payload
from t2pw.pipeline.process_normalizer import (
    GateValidationError,
    normalize_process_payload,
    run_strict_post_normalization_gates,
)
from t2pw.pipeline.stage_contracts import (
    StageContractError,
    validate_runtime_payload_contract,
)
from t2pw.pipeline.export_mode import (
    DEFAULT_EXPORT_MODE,
    ExportMode,
    is_research,
)
from t2pw.pipeline.pipeline import (
    apply_post_merge_cleanup,
    build_and_save_draft_graph,
)
from t2pw.pipeline.reaction_preservation_validator import load_locked_reaction_manifest
from t2pw.pipeline.preprocessor import format_context_header
from t2pw.config import rag_config
from t2pw.service.postpipeline import _safe_dict, _safe_list, run_quarantine_boundary
from t2pw.service.reporter import Reporter, default_reporter, default_state


# ── RAG orchestration (seam S5) — WIRING ONLY, NO LOGIC ─────────────────────
# Per the separation invariant (docs/rag/03_separation_invariant.md), the
# orchestrator *calls* t2pw.rag functions and passes their results between the
# existing seams; it contains no acquisition / selection / retrieval / synthesis
# logic itself. Every entry point below is guarded so that with RAG_ENABLED off
# (the default) the app takes exactly today's branch: ``maybe_run_rag`` returns
# ``None`` before importing or calling any t2pw.rag function, and every call site
# is behind ``if rag_config()["enabled"]``.
class RagOrchestrationResult(SimpleNamespace):
    """Bundle the RAG chain's outputs the app wires into seams S1/S2/S3 + UI.

    ``evidence_context`` is the formatted retrieval string injected via S1
    (``user_task_context``) and S2 (``retrieval_context=``). ``payload`` is the
    synthesized standard Payload for S3 (``None`` when nothing was synthesized).
    ``synthesized`` marks whether ``payload`` should replace the seed. The
    remaining fields (``decision``, ``selection``, ``candidates``,
    ``provenance``, ``synthesis``) drive the fetched-papers panel and provenance
    viewer. Nothing here is computed in the app — each is a t2pw.rag return value.
    """


def _count_reactions(payload: Any) -> int:
    """Number of ``processes.reactions`` rows in a payload (0 for any bad shape)."""
    processes = payload.get("processes") if isinstance(payload, dict) else None
    reactions = processes.get("reactions") if isinstance(processes, dict) else None
    return len(reactions) if isinstance(reactions, list) else 0


def _post_extraction_error_keys(payload: Any) -> set[tuple[str, str]]:
    """Enforce-mode post-extraction schema error keys for ``payload``.

    Runs ``validate_runtime_payload_contract(payload, boundary="post_extraction",
    mode="enforce")`` and, on failure, returns the set of ``(code,
    normalized_pointer)`` identities of the raised errors. Array indices in the
    JSON pointer are collapsed to ``/*`` (``re.sub(r"/\\d+", "/*", pointer)``) so
    that a row shifting position between the seed and the merged payload does not
    read as a brand-new error. An empty set means the payload passes the gate.
    """
    try:
        validate_runtime_payload_contract(
            payload, boundary="post_extraction", mode="enforce"
        )
    except StageContractError as failure:
        keys: set[tuple[str, str]] = set()
        for issue in failure.errors:
            if not isinstance(issue, dict):
                continue
            code = str(issue.get("code") or "")
            pointer = str(issue.get("pointer") or issue.get("path") or "")
            keys.add((code, re.sub(r"/\d+", "/*", pointer)))
        return keys
    return set()


def _post_extraction_new_error_keys(
    seed_payload: Any, merged_payload: Any
) -> set[tuple[str, str]]:
    """Schema-error keys the merge INTRODUCES relative to the seed.

    Empty means the merged payload adds no post-extraction runtime-schema
    violation that the seed did not already carry, so the merge is safe to adopt
    even though the wider pipeline runs this schema in report (non-enforcing)
    mode. Non-empty means the merge introduced a genuinely new violation.
    """
    return _post_extraction_error_keys(merged_payload) - _post_extraction_error_keys(
        seed_payload
    )


def maybe_run_rag(
    *,
    pathway_context: Dict[str, Any],
    user_flag: bool,
    seed_payload: Dict[str, Any],
    reports: Optional[Dict[str, Any]] = None,
    seed_text: str = "",
    reporter: Optional[Reporter] = None,
) -> Optional[RagOrchestrationResult]:
    """Run R0–R5 and return their outputs, or ``None`` for today's flow.

    Returns ``None`` — the untouched single-paper path — whenever RAG is
    disabled (``rag_config()["enabled"]`` false) or triage declines
    (``should_run_rag`` false); in that case no t2pw.rag chain function
    (``acquire`` etc.) is imported or called. Otherwise it wires the chain:
    ``acquire.search_candidates`` -> ``acquire.fetch_full_text`` (per kept paper)
    -> ``select.select`` -> ``ingest.ingest`` -> ``retrieve.detect_gaps`` /
    ``retrieve.retrieve_evidence`` -> ``retrieve.format_retrieval_context`` (S1/S2
    evidence) -> ``synthesize.synthesize_with_report`` (S3 Payload) ->
    ``provenance.validate_provenance``. Any failure in the optional network chain
    degrades to ``None`` so RAG never breaks the core run.
    """
    if not rag_config()["enabled"]:
        return None

    # Triage (the one piece of RAG decision logic — lives in t2pw.rag).
    from t2pw.rag.triage import should_run_rag

    decision = should_run_rag(pathway_context, user_flag, reports=reports)
    if not decision.run:
        return None

    # Live progress: the RAG chain (acquire -> select -> ingest -> retrieve ->
    # synthesize) can run for minutes with no UI output, so the app looks frozen
    # even though it is working. A single st.status container narrates each phase
    # and is marked complete/error on the way out. Best-effort UI only — it never
    # changes control flow, honoring the "RAG must never break the core run" rule.
    if reporter is None:
        reporter = default_reporter()
    status = reporter.progress(
        "Multi-paper RAG: searching literature for candidate papers…",
        expanded=True,
    )
    try:
        from t2pw.rag import (
            acquire,
            extract,
            identity,
            ingest,
            provenance,
            retrieve,
            select,
            synonyms,
            synthesize,
        )

        # R1 acquire -> R2 select -> (full text for kept) -> R3 ingest.
        # The REQUEST -- what this run asked for. Derived once, before anything
        # reads it: the eligibility screen, gap detection and the admission gate
        # all compare against it, and none of them may re-derive it differently.
        _requested_pathway = str(_safe_dict(pathway_context).get("pathway_name") or "")
        _requested_organism = str(
            _safe_dict(pathway_context).get("likely_organism")
            or _safe_dict(pathway_context).get("organism")
            or ""
        )
        acquire_status: Dict[str, Any] = {}
        candidates = acquire.search_candidates(pathway_context, status=acquire_status)
        # Display-only: a zero-candidate acquisition used to be a silent no-op, and
        # a rate-limited API looked identical to an empty search. ``acquire_status``
        # records what actually happened, so name the REAL cause. No control flow
        # changes — the chain continues exactly as before.
        if not candidates:
            if acquire_status.get("empty_query"):
                _why = (
                    "Stage 0 produced no usable search terms, so the literature "
                    "query was empty and no search was performed. Re-run — this is "
                    "usually a transient preprocessing LLM failure."
                )
            elif acquire_status.get("rate_limited"):
                _why = (
                    "the literature APIs rate-limited us (HTTP "
                    f"{acquire_status.get('http_statuses')}). NCBI allows only 3 "
                    "requests/second without an API key — set NCBI_API_KEY and "
                    "NCBI_EMAIL in .env to raise it to 10/s, then re-run."
                )
            elif acquire_status.get("transport_failed"):
                _why = (
                    "the literature APIs were unreachable "
                    f"({acquire_status.get('errors')}). Check network access."
                )
            elif acquire_status.get("from_cache"):
                _why = "a cached empty result was served for this query."
            else:
                _why = (
                    "the search ran but matched nothing (HTTP "
                    f"{acquire_status.get('http_statuses')}). The query may be too "
                    "narrow — check the organism and pathway name."
                )
            reporter.warning(f"RAG acquisition fetched 0 candidate papers: {_why}")
        status.write(f"Fetched {len(candidates)} candidate paper(s).")

        # R1b eligibility — the deterministic title/abstract screen, run on the
        # RAW acquired candidates BEFORE selection or any full-text fetch. It is
        # what fills each paper's observed_pathways / observed_organisms /
        # organism_match (via ``apply_decision``), which the admission gate later
        # compares against the request; without this call every paper reaches the
        # gate with empty observations, i.e. "unknown", i.e. the permissive
        # verdict. Screening also drops the papers the batch fetcher would have
        # dropped, so a case report or a wrong-species paper never gets its full
        # text downloaded, chunked or indexed.
        eligibility_report: Dict[str, Any] = {
            "enabled": bool(rag_config()["eligibility_enabled"]),
            "screened": 0,
            "kept": 0,
            "dropped": 0,
            "by_outcome": {},
        }
        eligible = list(candidates)
        if eligibility_report["enabled"] and candidates:
            status.update(label="Multi-paper RAG: screening candidate papers…")
            from t2pw.rag import eligibility as _eligibility

            scope = _eligibility.RequestedScope(
                requested_pathway=_requested_pathway,
                requested_organism=_requested_organism,
            )
            limits = _eligibility.thresholds_from_config(rag_config())
            eligible = []
            for _paper in candidates:
                try:
                    # apply=True writes the REQUESTED fields and the OBSERVED
                    # fields onto the candidate, in their separate slots. The
                    # requested organism is never written into an observed one --
                    # that stamping bug is why the split exists.
                    decision = _eligibility.screen_candidate(
                        _paper, scope, thresholds=limits
                    )
                except Exception:  # noqa: BLE001 - one bad row must not sink the run
                    eligible.append(_paper)
                    continue
                outcome = str(getattr(decision, "outcome", "") or "")
                eligibility_report["screened"] += 1
                eligibility_report["by_outcome"][outcome] = (
                    eligibility_report["by_outcome"].get(outcome, 0) + 1
                )
                if _eligibility.outcome_is_ineligible(outcome):
                    eligibility_report["dropped"] += 1
                    continue
                eligibility_report["kept"] += 1
                eligible.append(_paper)
            status.write(
                f"Screened {eligibility_report['screened']} paper(s): "
                f"{eligibility_report['kept']} eligible, "
                f"{eligibility_report['dropped']} dropped "
                f"({eligibility_report['by_outcome']})."
            )

        status.update(label="Multi-paper RAG: selecting relevant papers…")
        selection = select.select(eligible, pathway_context)
        full_text_papers = 0
        for paper in getattr(selection, "selected", []) or []:
            paper.full_text = acquire.fetch_full_text(paper)
            if str(getattr(paper, "full_text", "") or "").strip():
                full_text_papers += 1
        status.write(
            f"Selected {len(getattr(selection, 'selected', []) or [])} paper(s); "
            f"{full_text_papers} with full text."
        )
        status.update(label="Multi-paper RAG: embedding & indexing passages…")
        embedder = ingest.Embedder(config=rag_config())
        store = ingest.get_vector_store(embed_fn=embedder.embed)
        # Index the uploaded seed alongside the retrieved papers, so seed-derived
        # claims get a real section and a quotable passage instead of citing
        # "the seed paper" with nothing to show, and can corroborate like any
        # other identified source.
        ingest_report = ingest.ingest(
            selection,
            store=store,
            embedder=embedder,
            seed=ingest.seed_candidate(
                seed_text,
                title=str(_safe_dict(pathway_context).get("pathway_name") or ""),
                organism=str(_safe_dict(pathway_context).get("likely_organism") or ""),
            ),
        )
        status.write(
            f"Indexed {int(getattr(ingest_report, 'chunks', 0) or 0)} passage(s)."
        )

        # R4 gap-retrieve: detect gaps from read-only reports, retrieve per gap.
        # The request stamped onto every gap so a retrieved reaction can later be
        # compared against it — never against whatever the paper is about.
        seed_context_text = format_context_header(pathway_context)
        status.update(label="Multi-paper RAG: detecting connectivity gaps…")
        gaps = retrieve.detect_gaps(
            seed_payload,
            reports,
            requested_pathway=_requested_pathway,
            requested_organism=_requested_organism,
        )
        status.write(f"Detected {len(gaps)} connectivity gap(s) to fill.")
        bundles = []
        for _gi, gap in enumerate(gaps, start=1):
            status.update(
                label=f"Multi-paper RAG: retrieving evidence for gap {_gi}/{len(gaps)}…"
            )
            bundles.append(
                retrieve.retrieve_evidence(gap, store, seed_context=seed_context_text)
            )
        evidence_context = retrieve.format_retrieval_context(bundles)

        # R5 synthesize -> standard Payload (S3); validate provenance (WP6).
        # The seed context must be a *dict* carrying a source descriptor: the
        # uploaded seed paper is itself evidence for its own reactions, so its
        # identity has to reach synthesis. Passing the plain text header instead
        # makes _seed_source_descriptor return None, which omits every seed
        # reaction as "unsupported" and yields an empty pathway.
        _seed_name = ""
        if isinstance(pathway_context, dict):
            _seed_name = str(
                pathway_context.get("pathway_name")
                or pathway_context.get("title")
                or ""
            ).strip()
        seed_source_context = {
            "text": seed_context_text,
            "source": {
                "source_id": "seed_paper",
                "source_title": _seed_name or "uploaded seed paper",
                "source_type": "paper",
            },
        }
        # Prose→reaction extraction: recover reactions stated in paper prose (not
        # just arrow equations). Opt-in via RAG_EXTRACT_REACTIONS; fails closed so
        # it can only add reactions, never break synthesis.
        _base_prose_extractor = (
            extract.make_prose_extractor()
            if rag_config()["extract_reactions"]
            else None
        )
        # Prose extraction is the slow phase: one LLM chat call per evidence
        # passage, run sequentially inside synthesis. Wrap it so each call ticks
        # the status label — this is the step that previously left the UI blank.
        if _base_prose_extractor is not None:
            _prose_progress = {"n": 0}

            def prose_extractor(
                text,
                _base=_base_prose_extractor,
                _status=status,
                _counter=_prose_progress,
            ):
                _counter["n"] += 1
                _status.update(
                    label=(
                        "Multi-paper RAG: extracting reactions from evidence "
                        f"(passage {_counter['n']})…"
                    )
                )
                return _base(text)
        else:
            prose_extractor = None
        # Synonym-canonical merge: collapse cross-paper reaction duplicates that
        # differ only by a compound/enzyme SYNONYM, reusing the project's EXISTING
        # offline id-mapping cache. GROUPING-ONLY (drives merge keys, never rewrites
        # emitted names) so locked-reaction matching is unaffected. Offline +
        # deterministic — no network, no live LLM — and fails safe to a plain
        # normalizer if the cache file is absent, so it can only add merges, never
        # break synthesis. Enabled unconditionally for real runs (mirrors the
        # opt-in ``prose_extractor`` seam; no config.py flag is added).
        synonym_resolver = synonyms.build_offline_synonym_resolver()
        _identity_attempts: list = []
        status.update(label="Multi-paper RAG: synthesizing one connected pathway…")
        synthesis = synthesize.synthesize_with_report(
            seed_payload,
            bundles,
            seed_source_context,
            prose_extractor=prose_extractor,
            synonym_resolver=synonym_resolver,
            # Gap admission: a retrieved reaction enters the pathway only by
            # filling one of THESE gaps and passing pathway / organism / evidence
            # admission. The gap list and the request are passed explicitly so the
            # gate compares against what the run asked for.
            gaps=gaps,
            requested_pathway=_requested_pathway,
            requested_organism=_requested_organism,
            # Protein identity is FAIL-CLOSED in production. The ladder
            # (``verify_real_protein_identity``) judges a shipped accession
            # against local resolver candidates, and the RAG chain produces
            # passages, not resolver candidates -- so nothing here can verify an
            # identity, and nothing here pretends to. Every unmapped-enzyme gap
            # therefore stays open and keeps the PathBank Unknown-protein
            # fallback; ``identity_attempts`` records that the ladder was asked
            # and why it declined, so "no identities were resolved" is a reported
            # outcome rather than a silence.
            identity_verifier=identity.fail_closed_verifier(_identity_attempts),
        )
        prov_report = provenance.validate_provenance(synthesis.payload)

        # Funnel diagnostics — read-only counts of the t2pw.rag return values, so
        # "RAG added nothing" can be traced to the stage it stopped at (no papers
        # / no full text / no gaps / no evidence / nothing extracted) instead of
        # being an unexplained empty result. Computes nothing about the pathway.
        _gap_counts: Dict[str, int] = {}
        for _gap in gaps:
            _kind = str(getattr(_gap, "kind", "") or "unknown")
            _gap_counts[_kind] = _gap_counts.get(_kind, 0) + 1
        diagnostics = {
            "stage0_pathway_name": str(
                (pathway_context or {}).get("pathway_name") or ""
            ),
            "stage0_organism": str(
                (pathway_context or {}).get("likely_organism") or ""
            ),
            "acquire_query": str(acquire_status.get("query") or "")[:300],
            "acquire_http": acquire_status.get("http_statuses"),
            "acquire_rate_limited": acquire_status.get("rate_limited"),
            "acquire_from_cache": acquire_status.get("from_cache"),
            "candidates_fetched": len(candidates),
            "eligibility": eligibility_report,
            "papers_eligible": len(eligible),
            "papers_selected": len(getattr(selection, "selected", []) or []),
            "papers_with_full_text": full_text_papers,
            "chunks_indexed": int(getattr(ingest_report, "chunks", 0) or 0),
            "gaps_detected": len(gaps),
            "gaps_by_kind": _gap_counts,
            "evidence_hits": sum(len(getattr(b, "hits", []) or []) for b in bundles),
            "prose_extraction_enabled": prose_extractor is not None,
            "seed_reactions": _count_reactions(seed_payload),
            "synthesized_reactions": _count_reactions(synthesis.payload),
            "cross_paper_stitches": len(getattr(synthesis, "stitched", []) or []),
            "unresolved_gaps": len(getattr(synthesis, "unresolved_gaps", []) or []),
            # Gap admission — "RAG added nothing" now separates "nothing was
            # retrieved" from "everything retrieved was refused, for these reasons".
            "candidates_considered": int(
                _safe_dict(_safe_dict(synthesis.admission).get("counts")).get(
                    "considered", 0
                )
            ),
            "candidates_accepted": int(
                _safe_dict(_safe_dict(synthesis.admission).get("counts")).get(
                    "accepted", 0
                )
            ),
            "candidates_rejected": int(
                _safe_dict(_safe_dict(synthesis.admission).get("counts")).get(
                    "rejected", 0
                )
            ),
            "admission_reason_counts": _safe_dict(
                _safe_dict(synthesis.admission).get("reason_counts")
            ),
            # Typed-gap resolutions, and what the identity ladder actually did.
            "typed_resolutions_proposed": len(
                _safe_list(_safe_dict(synthesis.admission).get("resolutions"))
            ),
            "typed_resolutions_applied": sum(
                1
                for r in _safe_list(_safe_dict(synthesis.admission).get("resolutions"))
                if isinstance(r, dict) and r.get("applied")
            ),
            "typed_resolutions_rejected": sum(
                1
                for r in _safe_list(_safe_dict(synthesis.admission).get("resolutions"))
                if isinstance(r, dict) and r.get("status") == "rejected"
            ),
            # Verified identities, EC-only annotations, refusals and gaps awaiting
            # the Unknown fallback, counted APART and in their own units
            # (``proposals`` / ``targets`` / gap counts). Only ``verified`` means a
            # protein gained an exportable identity.
            #
            # Note ``unverified_no_resolver`` vs ``rejected_by_identity_resolver``:
            # this path wires ``fail_closed_verifier``, which has no candidate
            # provider, so every attempt lands in the FORMER. A count in the latter
            # would mean a real resolver had been wired and had judged a claim.
            "identity_outcomes": _safe_dict(
                _safe_dict(synthesis.admission).get("identity_outcomes")
            ),
            "identity_attempts": list(_identity_attempts),
        }
        status.update(
            label=(
                "Multi-paper RAG complete: "
                f"{diagnostics['synthesized_reactions']} reaction(s) synthesized "
                f"from {diagnostics['papers_selected']} paper(s)."
            ),
            state="complete",
        )
    except Exception as exc:  # noqa: BLE001 - RAG must never break the core run
        try:
            status.update(
                label=f"Multi-paper RAG could not complete (continuing without it): {exc}",
                state="error",
            )
        except Exception:  # noqa: BLE001 - status is best-effort UI only
            pass
        return RagOrchestrationResult(
            decision=decision,
            evidence_context="",
            payload=None,
            synthesized=False,
            selection=None,
            candidates=[],
            ingest_report=None,
            synthesis=None,
            provenance=None,
            gaps=[],
            bundles=[],
            diagnostics={},
            error=str(exc),
        )

    return RagOrchestrationResult(
        decision=decision,
        evidence_context=evidence_context,
        payload=synthesis.payload,
        synthesized=bool(synthesis.payload) and bool(bundles),
        selection=selection,
        candidates=candidates,
        ingest_report=ingest_report,
        synthesis=synthesis,
        provenance=prov_report,
        gaps=gaps,
        bundles=bundles,
        diagnostics=diagnostics,
        error="",
    )

# ── RAG round loop (D-008, PRODUCT_CONTRACT §10) — WIRING ONLY, NO LOGIC ────
# The loop is ``t2pw.rag.controller.run_rag_loop``. Nothing below decides whether
# to go round again, what a round's delta is worth, or why the loop stopped —
# those are ``loop_policy.decide``, ``graph_delta.validate_graph_delta`` and
# ``LoopOutcome.reason``. The app supplies exactly the four inputs the separation
# invariant (``docs/rag/03_separation_invariant.md``, and the note at :161) allows:
# ``stages=``, ``round_runner=``, ``deadline=`` and ``checkpoint=``. There is no
# ``while`` here and there must never be one.
#
# THE FIVE STAGES ARE CALLED, NEVER EDITED, AND NEVER FEED THE CANONICAL GRAPH.
# D-008 requires every round to RE-ENTER normalization, mapping, gates,
# persistence and classification. Three of the five cannot return a graph at all:
# ``run_strict_post_normalization_gates`` returns gate details,
# ``build_and_save_draft_graph`` returns ``(graph, qa_report, summary)`` and
# ``quarantine_and_close`` returns a ``StrictQuarantineResult`` whose ``payload``
# is a REDUCED graph; ``map_payload`` returns ``{"payload", "report"}``. So each
# adapter runs the real stage, publishes its REPORT on the round channel, and
# returns the graph it was handed.
#
# That is exactly what round N+1 needs. ``retrieve.detect_gaps`` reads THREE
# report channels (``rag/retrieve.py:699-701`` — ``qa``, ``gate``, ``mapping``)
# and at base only ``qa_graph`` was ever populated, from the Stage-1 payload, once
# (``:5520``). A second round reusing that frozen report detects the identical gap
# set and converges for the wrong reason. Re-entering the five stages is what
# recomputes all three channels from the PREVIOUS round's merged payload.
#
# Feeding a stage's OUTPUT forward instead would (a) hand
# ``run_post_pipeline_sbml_artifacts`` an already normalized, mapped and
# quarantined payload that it then re-normalizes and re-maps, and (b) make every
# round's delta a rewrite rather than the additive merge ``validate_graph_delta``
# exists to judge (``rag/graph_delta.py:29-32``: "the seam is additive by
# construction … so a removal or a biology change on a pre-existing row is
# reported and never repaired").
#
# Every per-round side effect is confined to ``tmp/rag_rounds/``. The authoritative
# draft graph / QA report / reaction summary stay the ones the existing flow writes
# at ``:5659``, and the authoritative classification stays the one the quarantine
# boundary writes. The round path therefore calls ``run_quarantine_boundary``
# (``:1790``) -- NOT ``quarantine_and_close``, which would add callers that
# ``tests/test_streamlit_quarantine_boundary.py:761-779`` forbids, and that file is
# hotspot 9. Its two extra effects are neutralized rather than inherited: artifacts
# go to the ROUND's directory, never ``outputs/``, and the FIVE session keys it
# writes (``:1839-1843``, and :data:`QUARANTINE_SESSION_KEYS` below) are saved and
# restored in a ``finally``, so a round leaves no trace. See ``:1120``.
#
# The LLM audit is NOT re-entered per round: it lives inside
# ``run_post_pipeline_sbml_artifacts``, and calling the five concrete stage
# functions excludes it by construction. Keep it that way — a per-round LLM audit
# would make the loop nondeterministic and unaffordable.

#: Where every per-round artifact goes. ``tmp/`` is gitignored, and this is a fresh
#: subdirectory: it cannot collide with the protected scratch files
#: (``tmp/draft_graph.json``, ``tmp/qa_report.json``, ``tmp/reaction_summary.txt``).
RAG_ROUND_SCRATCH_DIRNAME = "rag_rounds"

#: Rounds this leg may run. Default 1 — today's exact single-pass behaviour, so
#: wiring the loop changes nothing until an operator asks for more.
RAG_LOOP_MAX_ROUNDS_ENV = "RAG_LOOP_MAX_ROUNDS"

#: The loop's own monotonic budget, in seconds. Default is D-005's child deadline.
RAG_LOOP_DEADLINE_ENV = "RAG_LOOP_DEADLINE_SECONDS"

#: Sentinel for "this session key was absent before the round".
_MISSING = object()

#: Every ``st.session_state`` key ``run_quarantine_boundary`` writes. The round
#: restores each one, so a per-round classification cannot leave a boundary decision
#: in the session for a payload version that is not the one the boundary judged.
#: (Measured: FIVE keys at ``:1839-1849``, not the six a record claimed.)
QUARANTINE_SESSION_KEYS = (
    "quarantine_report", "quarantine_artifacts", "quarantine_coverage",
    "quarantine_ok", "quarantine_decision_controls",
)


def rag_loop_max_rounds() -> int:
    """How many rounds the loop may run. ``1`` unless an operator raises it."""

    raw = str(os.getenv(RAG_LOOP_MAX_ROUNDS_ENV, "") or "").strip()
    try:
        value = int(raw) if raw else 1
    except ValueError:
        value = 1
    return max(1, value)


def rag_loop_deadline_seconds() -> float:
    """The loop's wall-clock budget, from D-005's numbers (``pipeline/deadline.py``).

    Not invented here: the default is ``CHILD_DEADLINE_SECONDS`` (the per-leg
    ceiling minus the parent/child grace) and the floor is
    ``MIN_CHILD_DEADLINE_SECONDS``, both of which that module owns.
    """

    from t2pw.pipeline.deadline import (  # noqa: PLC0415 - lazy: RAG-off imports nothing
        CHILD_DEADLINE_SECONDS,
        MIN_CHILD_DEADLINE_SECONDS,
    )

    raw = str(os.getenv(RAG_LOOP_DEADLINE_ENV, "") or "").strip()
    try:
        value = float(raw) if raw else float(CHILD_DEADLINE_SECONDS)
    except ValueError:
        value = float(CHILD_DEADLINE_SECONDS)
    return max(float(MIN_CHILD_DEADLINE_SECONDS), value)


def rag_round_scratch_dir(
    round_index: Optional[int] = None, *, root: Optional[Path] = None
) -> Path:
    """The round-scoped scratch directory, created on demand."""

    base = (Path(root) if root is not None else PROJECT_ROOT / "tmp") / RAG_ROUND_SCRATCH_DIRNAME
    path = base if round_index is None else base / f"round_{int(round_index):02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def rag_round_mapping_cache_path(*, root: Optional[Path] = None) -> Path:
    """A round-scoped mapping cache, seeded ONCE from the authoritative one.

    TRAP 11. ``map_payload`` ends with ``cache.save()`` (``map_ids.py:8856``), and
    ``MappingCache.save`` (``map_ids.py:780-783``) is a non-atomic, unlocked,
    whole-file overwrite of a ~4.2 MB file. On 2026-08-18 a T-100 leg died on
    exactly that write — ``[Errno 22] Invalid argument`` on
    ``data/id_mapping_cache.json`` — after 456.8 s and after it had already
    produced 5 reactions and 20 entities.

    Re-entering mapping once per round would turn one such overwrite of the shared
    file into N. Pointing the rounds at a copy under ``tmp/`` instead means the loop
    performs **zero** writes to ``data/id_mapping_cache.json`` — the authoritative
    cache is still written exactly once per leg, by the unchanged post-pipeline
    path. Seeding the copy (rather than passing ``use_cache=False``) keeps the reads
    warm, so a round does not re-resolve every entity over the network; and a failed
    scratch write aborts one round through ``RoundAborted``, which carries the
    checkpoint, instead of killing a leg that had already produced real biology.

    ``map_ids.py`` is not touched: it is outside this card's boundary.
    """

    scratch = rag_round_scratch_dir(root=root) / "id_mapping_cache.json"
    authoritative = PROJECT_ROOT / "data" / "id_mapping_cache.json"
    if not authoritative.exists():
        return scratch
    # Re-seed whenever the authoritative cache has moved on. The copy outlives the leg,
    # and a scratch cache that is never refreshed would drift away from what the
    # post-pipeline mapping sees -- which would make the round's ``mapping`` report,
    # the very channel this card exists to create, describe a different resolution
    # than production's.
    if (not scratch.exists()
            or authoritative.stat().st_mtime_ns > scratch.stat().st_mtime_ns):
        shutil.copyfile(authoritative, scratch)
    return scratch


def rag_round_reports(channel: Dict[str, Any]) -> Dict[str, Any]:
    """The three report channels ``retrieve.detect_gaps`` reads (``retrieve.py:699-701``).

    Only non-empty channels are passed on, so a stage that produced nothing is
    absent rather than present-and-empty — ``detect_gaps`` treats the two the same,
    but a diagnostic reading this dict should not see a report that never existed.
    """

    reports: Dict[str, Any] = {}
    for key in ("qa_graph", "gate", "mapping"):
        value = channel.get(key)
        if isinstance(value, dict) and value:
            reports[key] = value
    return reports


def build_rag_round_stages(
    channel: Dict[str, Any],
    *,
    pathway_context: Optional[Dict[str, Any]] = None,
    export_mode: "ExportMode" = DEFAULT_EXPORT_MODE,
    strict_db: bool = True,
    mapping_cache_path: Optional[Path] = None,
    id_source: str = "",
    scratch_root: Optional[Path] = None,
    state: Optional[MutableMapping[str, Any]] = None,
) -> Any:
    """The five D-008 stage adapters, as ``RoundStages``.

    Each adapter CALLS the real stage function, publishes its report on ``channel``
    and returns the graph unchanged (see the block comment above). ``channel`` is
    also how the round index reaches the persistence adapter: ``RoundStages`` is
    built once for the whole loop, so the runner stamps ``channel["round_index"]``
    before the controller drives the stages.
    """

    from t2pw.rag.controller import RoundStages  # noqa: PLC0415 - lazy, see :452 and :436

    def _cache_path() -> Path:
        # Resolved on the mapping stage's first call, never at build time: seeding
        # the round-scoped copy reads and writes ~4.2 MB, and a round aborted before
        # mapping must not pay for it.
        return (
            Path(mapping_cache_path)
            if mapping_cache_path is not None
            else rag_round_mapping_cache_path(root=scratch_root)
        )

    resolved_id_source = (
        id_source or os.getenv("PATHBANK_ID_SOURCE", "hybrid") or "hybrid"
    ).strip().lower()

    def _round_dir() -> Path:
        return rag_round_scratch_dir(int(channel.get("round_index") or 0), root=scratch_root)

    def _normalization(graph: Any) -> Any:
        normalized, report = normalize_process_payload(graph, mode=export_mode)
        channel["normalized_payload"] = normalized
        channel["normalization"] = _safe_dict(report)
        return graph

    def _mapping(graph: Any) -> Any:
        # COST, stated rather than suppressed. This is a full ``map_payload`` pass per
        # round: PathBank/UniProt/ChEBI resolution plus ``hydrate_species_references``,
        # which issues an LLM call unless ``T2PW_SPECIES_LLM=0``. D-008 asks for
        # re-entry, not a cheaper imitation, so no resolver flag is flipped behind the
        # operator's back — the knobs stay ``T2PW_SPECIES_LLM`` / ``PATHBANK_ID_SOURCE``
        # / ``RAG_LOOP_MAX_ROUNDS``, and at the default of one round this is exactly one
        # extra mapping pass per RAG leg.
        result = map_payload(
            graph,
            cache_path=_cache_path(),
            id_source=resolved_id_source,
            use_cache=True,
            mode=export_mode,
        )
        if not isinstance(result, dict):
            raise ValueError("map_payload returned a malformed result: expected an object.")
        channel["mapped_payload"] = result.get("payload")
        channel["mapping"] = _safe_dict(result.get("report"))
        return graph

    def _gates(graph: Any) -> Any:
        # Run on THIS round's normalized payload when normalization produced one:
        # gating the un-normalized graph would report errors normalization had
        # already fixed. A GateValidationError is a scientific verdict, not an
        # infrastructure failure, so it is RECORDED and the round continues — the
        # same {"ok": False, **details} shape ``normalize_process_payload`` itself
        # records at ``process_normalizer.py:5532``. Anything else propagates and
        # the controller turns it into RoundAborted, which is what it is for.
        target = channel.get("normalized_payload")
        target = target if isinstance(target, dict) else graph
        try:
            details = run_strict_post_normalization_gates(
                target,
                enforce_all_proteins_connected=not is_research(export_mode),
            )
            channel["gate"] = {"ok": True, **_safe_dict(details)}
        except GateValidationError as exc:
            channel["gate"] = {"ok": False, **_safe_dict(exc.details)}
        return graph

    def _persistence(graph: Any) -> Any:
        # ``output_path`` is round-scoped on purpose: build_and_save_draft_graph
        # writes draft_graph.json, qa_report.json and reaction_summary.txt next to
        # it, and those three are protected scratch files whose authoritative write
        # is the one at :5659.
        _graph, qa_report, _summary = build_and_save_draft_graph(
            graph, output_path=_round_dir() / "draft_graph.json"
        )
        channel["qa_graph"] = _safe_dict(qa_report)
        return graph

    def _classification(graph: Any) -> Any:
        # DEVIATION FROM THIS CARD'S RULING, disclosed and measured. The ruling said
        # to call ``quarantine_and_close`` directly rather than
        # ``run_quarantine_boundary``, because the latter additionally writes
        # artifacts and session-state keys. But
        # ``tests/test_streamlit_quarantine_boundary.py:761-779`` (hotspot 9, which
        # this card may not edit) asserts that the ONLY functions in THIS FILE calling
        # ``quarantine_and_close`` are ``run_quarantine_boundary`` and
        # ``run_pwml_export`` -- "exactly one function in the app decides; everything
        # else carries it" -- so a stage adapter that called it directly became a third
        # decision point and turned that gate red (measured: Chunk D qb node19).
        #
        # Both harms the ruling named are removed instead of the guard being dodged:
        # the artifacts go to the ROUND's own directory, never ``outputs/``, and every
        # session key the boundary writes is restored afterwards, so a round re-enters
        # the app's real classification decision and leaves no trace of having done so.
        # The authoritative boundary decision is still the one the post-pipeline pass
        # makes, on the payload it judges, under its own controls.
        target = _round_dir() / "quarantine"
        target.mkdir(parents=True, exist_ok=True)
        session = state if state is not None else default_state()
        saved = {key: session.get(key, _MISSING) for key in QUARANTINE_SESSION_KEYS}
        try:
            result = run_quarantine_boundary(
                graph,
                strict_db=bool(strict_db),
                outputs_dir=target,
                pathway_context=pathway_context,
                mode=export_mode,
                state=session,
            )
        finally:
            for key, value in saved.items():
                if value is _MISSING:
                    session.pop(key, None)
                else:
                    session[key] = value
        channel["classification"] = _safe_dict(_safe_dict(result).get("quarantine_report"))
        channel["coverage"] = _safe_dict(_safe_dict(result).get("coverage"))
        return graph

    return RoundStages(
        normalization=_normalization,
        mapping=_mapping,
        gates=_gates,
        persistence=_persistence,
        classification=_classification,
    )


def rag_round_merge(
    graph: Dict[str, Any],
    rag_payload: Any,
    *,
    pathway_context: Optional[Dict[str, Any]],
    merge_site: Callable[[Any, Any, Any], Dict[str, Any]],
    locked_manifest: Any = None,
    quarantine_output_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[str], int]:
    """conform -> merge -> post-merge cleanup, against the LIVE merge base.

    Lifted verbatim from the single-pass site this card replaced, with one thing
    parameterized: the merge base. ``conform_rag_additions_for_merge``'s second
    argument must be the very object ``merge_additions`` is about to be handed, and
    per round that object moves — round 2 merges into round 1's result, not into
    the Stage-1 seed. Everything the original comment there established still holds:
    the base is Stage 1 PLUS Stage 2 whenever inference ran, and ``build_citation_report``
    reads the PRE-MERGE ``rag_result.payload``, which conform never mutates.

    ``merge_site`` is the ``merge_additions`` statement itself, supplied by the
    caller rather than written here. C-060a's wiring test lifts the app's two
    ``merge_additions`` statements out of ``streamlit_app.py`` BY AST and executes
    the real source bytes to prove each passes a real ``PathwayContext``
    (``tests/test_c060a_requested_pathway_wiring.py:68-104``, floor
    ``OWNED_FLOOR = 4640``). Writing the RAG statement here would move it out of
    that instrument's reach and silently disarm the protection, so it stays in the
    script body where C-060a can still lift it, and this function calls it.
    """

    from t2pw.rag.conform import conform_rag_additions_for_merge  # noqa: PLC0415

    seed_reaction_count = _count_reactions(graph)
    envelope = conform_rag_additions_for_merge(rag_payload, graph)
    # The RAG leg runs merge_additions a SECOND time, so the same request has to
    # reach the gate on this pass too — and on every later round's pass.
    merged = merge_site(graph, envelope, pathway_context)
    # Preserve the locked-reaction quarantine the old replace-path performed:
    # merge_additions runs apply_post_merge_cleanup WITHOUT the manifest.
    merged, removed = apply_post_merge_cleanup(
        merged,
        locked_manifest=locked_manifest,
        quarantine_output_path=quarantine_output_path,
    )
    return merged, list(removed), _count_reactions(merged) - seed_reaction_count


def _rag_round_claims(rag_result: Any) -> Tuple[List[Any], List[Any]]:
    """``(admitted, rejected)`` claim rows for the round, from the admission report.

    Read-only: these are ``RagReactionCandidate.to_dict()`` rows, which carry the
    ``inputs`` / ``outputs`` / ``enzymes`` / ``reversible`` fields
    ``loop_policy.claim_identity_key`` reads. Only the gate's verdict makes a claim
    rejected — nothing here promotes a retrieval failure into one.
    """

    admission = _safe_dict(getattr(getattr(rag_result, "synthesis", None), "admission", None))
    return _safe_list(admission.get("accepted")), _safe_list(admission.get("rejected"))


def _rag_round_gap_ids(rag_result: Any) -> frozenset:
    """The round's detected-gap ids. An empty set refuses every addition (§10)."""

    ids = {
        str(getattr(gap, "gap_id", "") or "").strip()
        for gap in _safe_list(getattr(rag_result, "gaps", None))
    }
    return frozenset(gap_id for gap_id in ids if gap_id)


def _rag_reentry_map(reentered: Any) -> Dict[str, str]:
    """``stage -> true/false/not_evaluated``. §11: ``not_evaluated`` is NEVER ``false``."""

    from t2pw.rag.graph_delta import (  # noqa: PLC0415
        FALSE,
        NOT_EVALUATED,
        REENTRY_STAGES,
        TRUE,
    )

    known = _safe_dict(reentered)
    out: Dict[str, str] = {}
    for stage in REENTRY_STAGES:
        value = known.get(stage)
        out[stage] = TRUE if value is True else FALSE if value is False else NOT_EVALUATED
    return out


def run_rag_rounds(
    graph: Dict[str, Any],
    *,
    first_result: Any,
    merge_site: Callable[[Any, Any, Any], Dict[str, Any]],
    pathway_context: Optional[Dict[str, Any]],
    user_flag: bool,
    seed_text: str = "",
    export_mode: "ExportMode" = DEFAULT_EXPORT_MODE,
    strict_db: bool = True,
    max_rounds: Optional[int] = None,
    deadline_seconds: Optional[float] = None,
    scratch_root: Optional[Path] = None,
    locked_manifest_path: Optional[Path] = None,
    quarantine_output_path: Optional[Path] = None,
    reporter: Optional[Reporter] = None,
    state: Optional[MutableMapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Drive ``run_rag_loop`` over the RAG chain and return what it settled on.

    ``first_result`` is round 0's already-computed chain result. It is reused rather
    than re-run because the chain has to happen BEFORE Stage 2 — its
    ``evidence_context`` is folded into Stage 2's prompt through S1/S2 — while the
    merge has to happen AFTER it, so the merge base is Stage 1 PLUS Stage 2. Later
    rounds run the chain fresh against the round's own graph and the round's own
    recomputed reports.

    Returns a record, never raises: ``payload`` is the graph to carry forward (the
    loop's last GOOD graph, or the checkpoint's graph if a stage aborted the round),
    and ``messages`` are the UI lines for the caller to emit, so this function does
    not have to be a Streamlit widget to be testable.
    """

    from t2pw.rag.controller import RoundAborted, RoundResult, run_rag_loop  # noqa: PLC0415
    from t2pw.pipeline.deadline import LegDeadline  # noqa: PLC0415

    rounds_allowed = int(max_rounds) if max_rounds is not None else rag_loop_max_rounds()
    leg = LegDeadline(
        total_seconds=(
            float(deadline_seconds) if deadline_seconds is not None else rag_loop_deadline_seconds()
        )
    )
    channel: Dict[str, Any] = {}
    record: Dict[str, Any] = {
        "payload": graph,
        "rounds": [],
        "checkpoints": [],
        "messages": [],
        "reason": None,
        "aborted": None,
        "max_rounds": rounds_allowed,
        "budget": {},
    }
    manifest = load_locked_reaction_manifest(
        locked_manifest_path
        if locked_manifest_path is not None
        else PROJECT_ROOT / "tmp" / "locked_reaction_manifest.json"
    )
    quarantine_path = (
        quarantine_output_path
        if quarantine_output_path is not None
        else PROJECT_ROOT / "tmp" / "quarantined_rag_reactions.json"
    )
    pending: Dict[str, Any] = {"first": first_result}

    def _checkpoint(mark: Any) -> None:
        """§9's pre-retrieval checkpoint: the last good graph, the claim history and
        the budget, on disk BEFORE anything long runs."""

        state = {
            "round_index": int(mark.round_index),
            "reactions": _count_reactions(mark.graph),
            "seen_claims": sorted(mark.seen_claims.keys),
            "rejected_claim_keys": sorted(mark.rejected_claim_keys),
            "decision": {
                "should_continue": bool(mark.decision.should_continue),
                "reason": mark.decision.reason,
                "also_true": list(mark.decision.also_true),
                "counts": dict(mark.decision.counts),
            },
        }

        def _persist(budget_record: Any) -> None:
            target = rag_round_scratch_dir(mark.round_index, root=scratch_root)
            (target / "checkpoint.json").write_text(
                json.dumps(
                    {"budget": budget_record.to_dict(), "loop": state},
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )
            (target / "checkpoint_graph.json").write_text(
                json.dumps(mark.graph, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        leg.checkpoint(
            f"rag_round_{int(mark.round_index)}",
            stage="rag_loop",
            state=state,
            persist=_persist,
        )
        record["checkpoints"].append(state)

    def _round_runner(context: Any) -> Any:
        channel["round_index"] = int(context.round_index)
        rag = pending.pop("first", None)
        if rag is None:
            # Round N>0 re-runs the chain against THIS round's graph and THIS
            # round's recomputed reports — the whole point of re-entering the five
            # stages. Reusing round 0's frozen seed payload and its single qa_graph
            # report would re-detect the identical gap set.
            rag = maybe_run_rag(
                pathway_context=pathway_context or {},
                user_flag=bool(user_flag),
                seed_payload=context.graph,
                reports=rag_round_reports(channel),
                seed_text=seed_text,
                reporter=reporter,
            )
        admitted, rejected = _rag_round_claims(rag)
        observations = dict(
            claims=tuple(admitted) + tuple(rejected),
            admitted_claims=tuple(admitted),
            rejected_claims=tuple(rejected),
            detected_gap_ids=_rag_round_gap_ids(rag),
            retrieval_completed=bool(rag is not None and not str(getattr(rag, "error", "") or "")),
        )
        candidate = context.graph
        synthesized = bool(
            rag is not None and getattr(rag, "synthesized", False) and getattr(rag, "payload", None)
        )
        if synthesized:
            merged, removed, added = rag_round_merge(
                context.graph,
                rag.payload,
                pathway_context=pathway_context,
                merge_site=merge_site,
                locked_manifest=manifest,
                quarantine_output_path=quarantine_path,
            )
            if added > 0:
                # Differential schema gate at the RAG boundary: reject only when the
                # merge INTRODUCES a violation the base did not already carry.
                if _post_extraction_new_error_keys(context.graph, merged):
                    try:
                        validate_runtime_payload_contract(
                            merged, boundary="post_extraction", mode="enforce"
                        )
                    except StageContractError as failure:
                        record["messages"].append(
                            (
                                "error",
                                "Multi-paper RAG merge produced a payload that failed the "
                                "post-extraction runtime schema; keeping the single-paper "
                                "extraction rather than passing a malformed payload downstream.",
                            )
                        )
                        record["messages"].append(("json", failure.report))
                else:
                    candidate = merged
                    message = (
                        f"Merged {added} RAG reaction(s) into the single-paper extraction."
                    )
                    if removed:
                        message += (
                            f" {len(removed)} synthesized reaction(s) were dropped/quarantined "
                            f"as unusable: {', '.join(removed)}."
                        )
                    record["messages"].append(("info", message))
            else:
                record["messages"].append(
                    (
                        "warning",
                        "Multi-paper RAG produced no additional usable reactions (no evidence "
                        "retrieved — check that the embeddings endpoint is running and papers "
                        "were fetched). Keeping the single-paper extraction."
                        + (
                            f" {len(removed)} synthesized reaction(s) were dropped as unusable: "
                            f"{', '.join(removed)}."
                            if removed
                            else ""
                        ),
                    )
                )
        return RoundResult(graph=candidate, **observations)

    try:
        outcome = run_rag_loop(
            graph,
            stages=build_rag_round_stages(
                channel,
                pathway_context=pathway_context,
                export_mode=export_mode,
                strict_db=strict_db,
                scratch_root=scratch_root,
                state=state,
            ),
            round_runner=_round_runner,
            deadline=leg.started + leg.total,
            max_rounds=rounds_allowed,
            next_round_reserve_seconds=leg.reserve,
            checkpoint=_checkpoint,
        )
    except RoundAborted as exc:
        # TRAP 3. RoundAborted carries the failing stage, the PARTIAL re-entry map
        # and the checkpoint written before the round. Funnelling it through a
        # blanket ``except Exception`` would collapse all three into a UI string and
        # destroy the checkpoint. It carries NO termination reason on purpose: an
        # infrastructure failure is not one of D-005's six, and the un-run stages
        # report ``not_evaluated``, never ``false``.
        record["aborted"] = {
            "stage": exc.stage,
            "round_index": int(exc.checkpoint.round_index),
            "reentry": _rag_reentry_map(exc.reentered),
            "error": str(exc),
        }
        record["payload"] = exc.checkpoint.graph
        record["budget"] = leg.snapshot()
        record["messages"].append(
            (
                "error",
                f"Multi-paper RAG round {int(exc.checkpoint.round_index)} was aborted in the "
                f"{exc.stage!r} stage; the last checkpointed graph was kept and the round was "
                "not merged.",
            )
        )
        return record

    record["payload"] = outcome.graph
    record["reason"] = outcome.reason
    record["budget"] = leg.snapshot()
    record["seen_claims"] = len(outcome.seen_claims.keys)
    record["rejected_claim_keys"] = sorted(outcome.rejected_claim_keys)
    for round_outcome in outcome.rounds:
        row = {
            "index": int(round_outcome.index),
            "accepted": bool(round_outcome.accepted),
            "graph_delta": int(round_outcome.graph_delta),
            "new_admissible_claims": int(round_outcome.new_admissible_claims),
            "reentry": dict(round_outcome.verdict.reentry),
            "rules": list(round_outcome.verdict.rules()),
            "violations": [
                {
                    "rule": v.rule,
                    "element": v.element,
                    "observed": v.observed,
                    "required": v.required,
                    "detail": v.detail,
                }
                for v in round_outcome.verdict.violations
            ],
        }
        record["rounds"].append(row)
        if not round_outcome.accepted:
            # A refused delta never advances the canonical graph — the controller's
            # rule, not a choice made here. Say WHICH rules refused it, so a refusal
            # is diagnosable rather than a silent disappearance of RAG output.
            record["messages"].append(
                (
                    "warning",
                    f"Multi-paper RAG round {row['index']} produced a delta the graph-delta "
                    f"validator refused ({', '.join(row['rules']) or 'no rule reported'}); the "
                    "previous graph was kept. See the RAG loop record for the affected elements.",
                )
            )
    return record
