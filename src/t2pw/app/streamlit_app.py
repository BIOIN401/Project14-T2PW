import json
import logging
import os
import hashlib
import re
import sys
import time
import shutil
import copy
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# Streamlit executes this file directly, so Python puts src/t2pw/app on
# sys.path instead of the project src directory that contains the t2pw package.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import streamlit as st
from lxml import etree

import t2pw.llm.client as llm_client_module
from t2pw.paths import PROJECT_ROOT
from t2pw.curation.apply_audit_patch import committed_change_count, run_apply
from t2pw.curation.audit_json_llm import run_audit
from t2pw.curation.interactive_curator import (
    apply_patch_and_rerender,
    compact_mapping_misses,
    run_interactive_curator_round,
)
from t2pw.mapping.enrich_entities import run_enrichment
from t2pw.mapping.grounding import apply_grounding
from t2pw.curation.gap_resolver import run_gap_resolution
from t2pw.curation.pathway_curator import run_pathway_curator
from t2pw.sbml.json_to_sbml import build_sbml
from t2pw.mapping.map_ids import map_payload, PathBankDbResolver, resolve_mapping_gaps
from t2pw.sbml.render_pathwhiz_like import build_render_artifacts
from t2pw.pipeline.draft_graph_render import render_draft_graph_to_png_bytes
from t2pw.sbml.strip_unmapped import strip_unmapped
from t2pw.sbml.overwatch import run_sbml_overwatch
from t2pw.sbml.examples import build_retrieval_context, load_motif_index, payload_to_query_text
from t2pw.tools.pathwhiz_converter.ui import render_pathwhiz_converter_section
from t2pw.pipeline.process_normalizer import (
    GateValidationError,
    compute_normalization_stats,
    normalize_process_payload,
    run_strict_post_normalization_gates,
)
from t2pw.pipeline.strict_quarantine import (
    clear_quarantine_artifacts,
    decision_inputs_match,
    decision_matches,
    quarantine_and_close,
    quarantine_review_flags,
    write_quarantine_artifacts,
)
from t2pw.pipeline.stage_contracts import (
    StageContractError,
    run_stage_contract,
    validate_post_audit,
    validate_post_extraction,
    validate_post_mapping,
    validate_post_normalization,
    validate_post_remap,
    validate_pre_export,
    validate_runtime_payload_contract,
)
from t2pw.pipeline.export_mode import (  # noqa: E402  (follows the sys.path shim above)
    DEFAULT_EXPORT_MODE,
    PATHWHIZ,
    RESEARCH,
    ExportMode,
    coerce_mode,
    format_gaps,
    is_research,
    review_flags,
)
from t2pw.pipeline.pipeline import (
    PipelineFailure,
    apply_post_merge_cleanup,
    build_qa_feedback,
    build_and_save_draft_graph,
    filter_out_of_scope_reactions,
    merge_additions,
    propagate_context_organism,
    run_stage_two_with_feedback_loop,
    run_stage_one_with_chunking,
)
from t2pw.pipeline.extraction_diagnostics import (
    CLEANING_REPORT_NAME,
    activate as activate_extraction_diagnostics,
    count_entities as diagnostic_entity_counts,
    count_processes as diagnostic_process_counts,
    current as current_extraction_diagnostics,
    payload_hash as diagnostic_payload_hash,
)
from t2pw.pipeline.stage_one_boundary import settle_stage_one
from t2pw.pipeline.gate_reports import (
    CANONICAL_PAYLOAD_KEY,
    FINAL_GATE_REPORT_KEY,
    INITIAL_GATE_REPORT_KEY,
    PHASE_AUDIT_ROUND,
    PHASE_FINAL_PRE_EXPORT,
    PHASE_INITIAL_POST_NORMALIZATION,
    payload_sha256,
    stamp_artifact_set,
    stamp_report,
)
from t2pw.pipeline.failure_detail import headline as detail_headline
from t2pw.pipeline.draft_graph import build_draft_graph
from t2pw.pipeline.qa_graph import generate_qa_report
from t2pw.pipeline.reaction_summary import generate_reaction_summary
from t2pw.pipeline.reaction_preservation_validator import (
    load_locked_reaction_manifest,
    write_reaction_preservation_report_if_manifest,
)
from t2pw.curation.completeness_audit import run_final_completeness_audit
from t2pw.pipeline.preprocessor import (
    describe_preprocess_status,
    format_context_header,
    is_ambiguous_multi_example_review_context,
    preprocess,
    preprocess_was_recovered,
    strip_preprocess_status,
)
from t2pw.extraction.pdf_parser import parse_pdf, SKIP_SECTIONS
from t2pw.pwml.validate import discover_structure_signature, repair_tree, validate_generated_tree
from t2pw.pwml.writer import (
    DeterministicPwmlBuilder,
    blocking_pwml_ir_errors,
    is_non_blocking_pwml_ir_error,
)
from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir, validate_required_pwml_contract
from t2pw.pwml.qa import run_pwml_qa
from t2pw.pipeline.qa_graph import build_graph, connected_components, degrees, get_entities, node
from t2pw.config import rag_config  # noqa: E402  (all imports here follow the sys.path shim above)

st.set_page_config(page_title="PWML Multi-Stage Pipeline", layout="wide")
st.title("PWML Extraction -> Inference Pipeline (LM Studio)")

REFINEMENT_STATE_DEFAULTS = {
    "refinement_working_json": None,
    "refinement_graph_dict": None,
    "refinement_graph_bytes": None,
    "refinement_qa_report": None,
    "refinement_mapping_report": None,
    "refinement_mapping_misses": [],
    "refinement_history": [],
    "refinement_round": 0,
    "refinement_pwml_ready": False,
    "refinement_last_error": None,
    "refinement_last_warnings": [],
    "refinement_gate_errors": [],
    "refinement_checkpoints": [],
}


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


# Bounded head of the document used when retrying a failed Stage-0 pass. The
# pathway name / organism / key entities live in the title, abstract and intro,
# and ``t2pw.rag.select`` already caps this same ``preprocess`` call at 6000
# chars for exactly this reason.
_PREPROCESS_RETRY_CHARS = 20000


def _has_usable_context(ctx: Any) -> bool:
    """True when Stage 0 produced at least one term usable as a search anchor.

    Mirrors precisely what ``t2pw.rag.acquire.build_query`` needs: without a
    pathway name, an organism, or any compound/protein/gap term, the literature
    query is the empty string and no paper can ever be fetched.
    """
    if not isinstance(ctx, dict):
        return False
    if str(ctx.get("pathway_name") or "").strip():
        return True
    if str(ctx.get("likely_organism") or ctx.get("organism") or "").strip():
        return True
    for key in ("key_compounds", "key_proteins", "gap_terms"):
        value = ctx.get(key)
        if isinstance(value, list) and any(str(v or "").strip() for v in value):
            return True
    return False


def _seed_context_from_user_scope(
    ctx: Any, user_scope: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Derive a usable Stage-0 context from an explicit user scope string.

    Stage 0 (the preprocessor LLM) fails CLOSED to an empty context, which
    silently disables RAG (``build_query`` returns "", so zero papers are ever
    fetched) and leaves extraction unguided. But the pipeline is not actually
    blind in that case: the user routinely types the scope into the extraction
    focus box (e.g. ``"menaquinone biosynthesis in Lactococcus lactis"``). This
    fallback parses that into a ``pathway_name`` and — from a trailing
    ``" in <organism>"`` — a ``likely_organism``, so a flaky preprocessor call
    can no longer nuke a scope the user already provided.

    Returns an augmented COPY of ``ctx`` (all original keys preserved) that
    ``_has_usable_context`` accepts, or ``None`` when there is no usable scope to
    seed from. Heuristic and transparent: the caller surfaces exactly what
    pathway/organism was inferred so a mis-parse is visible, not silent.
    """
    scope = str(user_scope or "").strip()
    if not scope:
        return None
    head, sep, tail = scope.rpartition(" in ")
    # Treat the trailing "in <X>" as the organism only when it is short (organism
    # names are 1–3 words); otherwise keep the whole scope as the pathway name so
    # a phrase like "... in vitro reconstitution" is not mistaken for an organism.
    if sep and head.strip() and 1 <= len(tail.split()) <= 3:
        pathway_name = head.strip()
        organism_name = tail.strip()
    else:
        pathway_name = scope
        organism_name = ""
    seeded = dict(ctx) if isinstance(ctx, dict) else {}
    seeded["pathway_name"] = pathway_name
    if organism_name and not str(seeded.get("likely_organism") or "").strip():
        seeded["likely_organism"] = organism_name
    if not _has_usable_context(seeded):
        return None
    return seeded


def _stage0_context_rank(ctx: Any) -> int:
    """Order two Stage-0 draws so a retry can only ever move the context UPWARD.

    Higher is both more informative and more fail-safe:

    * ``2`` — a well-formed **Case C**: ``document_type == "multi_example_review"``
      with a blank ``selected_example``. This is the Stage-0 prompt's own refusal
      verdict, and it is the strongest state precisely because it is the only one
      that stops the run. It outranks a usable context on purpose: if one draw
      says "this document is a multi-example review and I could not pick a
      target" and another says "the pathway is X", the safe reading is the
      refusal — extracting X from a review that describes six unrelated examples
      is how a mixed-pathway payload gets built. A second draw must never be able
      to disarm the refusal gate.
    * ``1`` — a usable context: ``_has_usable_context`` accepts it, so extraction
      is guided and ``t2pw.rag.acquire.build_query`` can form a non-empty query.
    * ``0`` — neither: Stage 0 failed closed (swallowed LLM error, unparseable
      reply) and left every field blank.

    Note that Case C and rank 0 look IDENTICAL to ``_has_usable_context``: the
    Stage-0 prompt mandates that Case C leave pathway_name / likely_organism /
    key_compounds / key_proteins / gap_terms blank, so the only thing separating a
    deliberate refusal from a crashed call is ``document_type``. That is exactly
    what the 2026-07-28 clobber destroyed (see ``_run_stage_zero_with_retry``).
    """
    if is_ambiguous_multi_example_review_context(ctx):
        return 2
    if _has_usable_context(ctx):
        return 1
    return 0


def _run_stage_zero_with_retry(
    text: str,
    *,
    temperature: float,
    user_task_context: Optional[str],
) -> Dict[str, Any]:
    """Run Stage 0, retrying once on a bounded head only when that can help.

    Two separate defects lived in the inline version of this block, and both were
    load-bearing in run ``runs/2026-07-28_0919``.

    1. *The retry fired on a correct refusal.* Stage 0 fails CLOSED to an empty
       context (a swallowed LLM error or an unparseable reply are both logged, not
       raised), and an empty context silently disables RAG — ``build_query()``
       returns "" so zero papers can ever be fetched. Hence the retry on a bounded
       head of the document: the pathway name / organism / key entities live in
       the title, abstract and intro, and the smaller payload also dodges
       transient failures. But the retry's trigger was ``not
       _has_usable_context(...)``, and the prompt's Case C — a multi-example
       review with no target selected — MANDATES those same fields be blank. A
       CORRECT Case C was therefore indistinguishable from a FAILED Stage 0 and
       fired the retry every time. With ``_PREPROCESS_RETRY_CHARS`` at 20,000 the
       length condition is met by essentially every topic-fetched review (the
       PMC13278307 leg below is 50,377 chars per that run's plan.json), so this
       was not a rare path.

    2. *The retry's result was adopted unconditionally.* ``pathway_context =
       preprocess(head)`` overwrote the first draw whatever came back. When the
       first draw was a Case C and the second was blank, the overwrite dropped
       ``document_type``, which is the one field
       ``is_ambiguous_multi_example_review_context`` requires. That disarms the
       refusal gate further down and lets the run proceed to an unguided
       extraction over a multi-example review. Because the extraction-focus box is
       now filled in by the batch driver, such a run no longer aborts loudly — it
       reports PASS. Leg PMC13278307/strict of ``runs/2026-07-28_0919`` is exactly
       that shape: the paper is the review "An Overview of Mobile Colistin
       Resistance (mcr) Genes in Gram-Negative Bacilli", the focus box said
       ``lipid A biosynthesis``, and the leg came back **pass** in 2098.21s with
       14 reactions and 0 gate errors — every one of them an mcr / PEtN / L-Ara4N
       lipid A MODIFICATION step, and not one reaction of the lipid A
       biosynthesis pathway it was asked for.

    The fix is the two rules below: skip the retry entirely for a deliberate Case
    C (it is deterministic — re-running never helps, only the user naming a target
    does), and, whenever a retry does run, keep its result in a LOCAL and adopt it
    only when ``_stage0_context_rank`` says it is strictly better. A second draw
    can now still rescue a genuinely failed Stage 0, but it can no longer degrade
    the context it was meant to improve.
    """
    first_context = preprocess(
        text, temperature=temperature, user_task_context=user_task_context
    )
    if _has_usable_context(first_context):
        # Nothing to rescue: extraction is guided and RAG can build a query.
        return first_context
    if is_ambiguous_multi_example_review_context(first_context):
        # Defect 1: a deliberate Case C has blank pathway fields BY DESIGN. The
        # caller's refusal gate needs the document_type this draw carries, so
        # return it untouched rather than spending a second LLM call that can only
        # return the same verdict or something weaker.
        return first_context
    if len(text) <= _PREPROCESS_RETRY_CHARS:
        # The retry re-reads a bounded head; on a document already shorter than
        # that head it would re-send identical input for an identical answer.
        return first_context

    retry_context = preprocess(
        text[:_PREPROCESS_RETRY_CHARS],
        temperature=temperature,
        user_task_context=user_task_context,
    )
    # Defect 2: adopt only on a strict improvement. ``first_context`` is known to
    # be rank 0 here, so this admits both a usable retry and a retry that is
    # itself a well-formed Case C (the head of a review is often enough for Stage
    # 0 to recognise the review it could not classify from the full text), while
    # rejecting a second blank draw.
    if _stage0_context_rank(retry_context) > _stage0_context_rank(first_context):
        return retry_context
    return first_context


def maybe_run_rag(
    *,
    pathway_context: Dict[str, Any],
    user_flag: bool,
    seed_payload: Dict[str, Any],
    reports: Optional[Dict[str, Any]] = None,
    seed_text: str = "",
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
    status = st.status(
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
            st.warning(f"RAG acquisition fetched 0 candidate papers: {_why}")
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


def _rag_funnel_verdict(diag: Dict[str, Any]) -> str:
    """Plain-language read of the RAG funnel: the first zero is the cause.

    Display-only. Each check mirrors one stage of the chain in order, so the
    returned sentence names the stage that stopped it — turning "RAG added
    nothing" into an actionable diagnosis instead of an unexplained empty result.
    """
    if diag.get("candidates_fetched", 0) == 0:
        return (
            "No candidate papers were fetched — the literature query returned nothing. "
            "The query is organism-scoped, so check the pathway name / organism in the "
            "Stage-0 context (a vague or missing organism narrows it to nothing), or "
            "network access to EuropePMC/NCBI."
        )
    if diag.get("papers_selected", 0) == 0:
        return (
            "Papers were fetched but every one was dropped in selection — see the drop "
            "reasons below. Common cause: candidates classified as a multi_example_review "
            "with no seed-matching example are penalized below the score floor and dropped."
        )
    if diag.get("papers_with_full_text", 0) == 0:
        return (
            "Papers were selected but none had retrievable full text (all abstract-only). "
            "Abstracts rarely state reaction mechanisms, so there is little to extract — "
            "this is the PMC open-access limit, not a bug."
        )
    if diag.get("gaps_detected", 0) == 0:
        return (
            "No gaps were detected in the seed pathway, so RAG had nothing to search for. "
            "This is a legitimate no-op — the seed extraction was already connected. Use a "
            "sparser / more incomplete seed to create the gaps RAG is meant to fill."
        )
    if diag.get("evidence_hits", 0) == 0:
        return (
            "Gaps were detected but retrieval returned no evidence — the indexed passages "
            "did not match the gap queries."
        )
    if diag.get("candidates_considered", 0) > 0 and diag.get(
        "candidates_accepted", 0
    ) == 0:
        return (
            f"Evidence was retrieved and {diag.get('candidates_considered', 0)} candidate "
            "reaction(s) were transcribed, but none passed gap admission: "
            f"{diag.get('admission_reason_counts')}. A retrieved reaction enters the "
            "pathway only when it fills a specific detected gap and passes pathway, "
            "organism and evidence admission — see the admission report for the "
            "per-candidate reasons."
        )
    if diag.get("synthesized_reactions", 0) <= diag.get("seed_reactions", 0):
        return (
            "Evidence was retrieved but produced no NEW reactions: the passages did not "
            "state a transcribable reaction, or prose extraction is off "
            f"(enabled={diag.get('prose_extraction_enabled')}). Synthesized "
            f"({diag.get('synthesized_reactions', 0)}) did not exceed the seed "
            f"({diag.get('seed_reactions', 0)}), so the single-paper extraction was kept."
        )
    return (
        f"RAG added reactions: {diag.get('seed_reactions', 0)} seed -> "
        f"{diag.get('synthesized_reactions', 0)} synthesized, with "
        f"{diag.get('cross_paper_stitches', 0)} cross-paper stitch(es)."
    )


def render_rag_panels(rag_result: "RagOrchestrationResult") -> None:
    """Render the RAG UI: triage note, fetched/selected papers, provenance viewer.

    Pure presentation of :func:`maybe_run_rag`'s outputs — it only *reads* the
    t2pw.rag return values (selection report, candidate list, the synthesized
    payload's additive provenance keys, the WP6 provenance report). It computes
    nothing about the pathway. Callers guard it behind ``rag_config()["enabled"]``.
    """
    if rag_result is None:
        return

    decision = getattr(rag_result, "decision", None)
    st.subheader("Multi-paper RAG")
    if decision is not None:
        st.caption(f"Triage: {'ON' if getattr(decision, 'run', False) else 'off'} - {getattr(decision, 'reason', '')}")
    if getattr(rag_result, "error", ""):
        st.warning(f"RAG chain degraded (core run unaffected): {rag_result.error}")

    # Panel 0 — the funnel. Read this first when RAG "added nothing": each number
    # feeds the next stage, so the first zero is the cause.
    diag = getattr(rag_result, "diagnostics", None) or {}
    if diag:
        with st.expander("RAG funnel — why RAG did / didn't add reactions", expanded=True):
            st.caption(_rag_funnel_verdict(diag))
            row1 = st.columns(4)
            row1[0].metric("Papers fetched", diag.get("candidates_fetched", 0))
            row1[1].metric("Selected", diag.get("papers_selected", 0))
            row1[2].metric("With full text", diag.get("papers_with_full_text", 0))
            row1[3].metric("Chunks indexed", diag.get("chunks_indexed", 0))
            row2 = st.columns(4)
            row2[0].metric("Gaps detected", diag.get("gaps_detected", 0))
            row2[1].metric("Evidence hits", diag.get("evidence_hits", 0))
            row2[2].metric("Seed reactions", diag.get("seed_reactions", 0))
            row2[3].metric("Synthesized", diag.get("synthesized_reactions", 0))
            row3 = st.columns(3)
            row3[0].metric("Candidates", diag.get("candidates_considered", 0))
            row3[1].metric("Admitted", diag.get("candidates_accepted", 0))
            row3[2].metric("Refused", diag.get("candidates_rejected", 0))
            st.json(diag)

    # Panel 0b — gap admission. Both halves are shown: an accepted-only view
    # cannot answer "why is the reaction I expected missing?", which is the
    # question a gap-filling run actually raises.
    synthesis = getattr(rag_result, "synthesis", None)
    admission = getattr(synthesis, "admission", None) if synthesis is not None else None
    if admission:
        with st.expander("Gap admission — accepted / refused candidates", expanded=False):
            st.caption(
                "A retrieved reaction enters the pathway only when it fills a "
                "specific detected gap and passes pathway, organism and evidence "
                "admission."
            )
            st.json(admission)

    ingest_report = getattr(rag_result, "ingest_report", None)
    if ingest_report is not None and hasattr(ingest_report, "to_dict"):
        with st.expander("Ingest report", expanded=False):
            st.json(ingest_report.to_dict())

    # Panel 1 — fetched + selected papers (WP1/WP2 reports).
    selection = getattr(rag_result, "selection", None)
    candidates = getattr(rag_result, "candidates", None) or []
    with st.expander(f"Fetched papers ({len(candidates)}) and selection", expanded=False):
        if candidates:
            st.markdown("**Fetched candidates (WP1 acquire)**")
            st.json([c.to_dict() for c in candidates if hasattr(c, "to_dict")])
        if selection is not None:
            kept = selection.kept_entries()
            st.markdown(f"**Selected papers (WP2): {len(kept)} kept**")
            st.json([e.to_dict() for e in kept])
            dropped = selection.dropped_entries()
            if dropped:
                st.markdown(f"**Dropped ({len(dropped)}) - with reason**")
                st.json([e.to_dict() for e in dropped])

    # Panel 2 — provenance viewer: source papers per reaction / entity (WP5/WP6).
    payload = getattr(rag_result, "payload", None)
    prov = getattr(rag_result, "provenance", None)
    with st.expander("Provenance viewer (source papers per element)", expanded=False):
        if prov is not None:
            st.caption(
                f"Evidence-bound check (WP6): {'OK - every element sourced' if getattr(prov, 'ok', False) else 'ISSUES'} "
                f"(reactions checked={getattr(prov, 'checked_reactions', 0)}, "
                f"entities checked={getattr(prov, 'checked_entities', 0)})"
            )
            issues = getattr(prov, "issues", None) or []
            if issues:
                st.json([
                    {"kind": i.kind, "label": i.label, "pointer": i.pointer, "reason": i.reason}
                    for i in issues
                ])
        rows = _rag_provenance_rows(payload)
        if rows:
            st.table(rows)
        else:
            st.caption("No synthesized elements with provenance to display.")


def _rag_provenance_rows(payload: Any) -> List[Dict[str, Any]]:
    """Flatten a synthesized payload into (element, type, source papers) rows.

    Read-only presentation helper: it reads the additive provenance carriers
    (``source_papers`` / ``rag_provenance`` / ``evidence`` / ``source_refs``) WP5
    attaches to each reaction and entity. No pipeline logic — display only.
    """
    if not isinstance(payload, dict):
        return []

    def _sources(row: Dict[str, Any]) -> str:
        ids: List[str] = []
        for paper in _safe_list(row.get("source_papers")):
            if isinstance(paper, dict):
                ident = str(paper.get("source_id") or paper.get("uri") or "").strip()
                if ident:
                    ids.append(ident)
        prov = row.get("rag_provenance")
        if isinstance(prov, dict):
            ident = str(prov.get("source_id") or prov.get("source_uri") or "").strip()
            if ident:
                ids.append(ident)
        for ev in _safe_list(row.get("evidence")):
            if isinstance(ev, dict):
                ident = str(ev.get("source_id") or ev.get("source_uri") or "").strip()
                if ident:
                    ids.append(ident)
        for ref in _safe_list(row.get("source_refs")):
            if isinstance(ref, str) and ref.strip():
                ids.append(ref.strip())
        # Stable de-dupe for display.
        seen: List[str] = []
        for ident in ids:
            if ident not in seen:
                seen.append(ident)
        return ", ".join(seen) if seen else "(none)"

    rows: List[Dict[str, Any]] = []
    processes = _safe_dict(payload.get("processes"))
    for rxn in _safe_list(processes.get("reactions")):
        if isinstance(rxn, dict):
            rows.append({
                "element": str(rxn.get("name") or "(unnamed reaction)"),
                "type": "reaction",
                "source_papers": _sources(rxn),
            })
    entities = _safe_dict(payload.get("entities"))
    for bucket in ("compounds", "proteins", "protein_complexes"):
        for ent in _safe_list(entities.get(bucket)):
            if isinstance(ent, dict):
                rows.append({
                    "element": str(ent.get("name") or f"(unnamed {bucket[:-1]})"),
                    "type": bucket[:-1],
                    "source_papers": _sources(ent),
                })
    return rows


def reset_refinement_state() -> None:
    for key, default in REFINEMENT_STATE_DEFAULTS.items():
        st.session_state[key] = deepcopy(default)


#: Session keys holding the single quarantine decision for this payload.
QUARANTINE_STATE_KEYS = (
    "quarantine_report",
    "quarantine_artifacts",
    "quarantine_coverage",
    "quarantine_ok",
    "quarantine_decision_controls",
)


def reset_quarantine_state(outputs_dir: Optional[Path] = None) -> None:
    """Drop every stored quarantine decision. Called when a new run begins.

    Session keys AND the artifacts they point at. A decision outlives the payload
    it was made about in two directions: ``st.session_state`` survives a rerun,
    and ``outputs/quarantine_report.json`` survives the whole session. Without
    this, a second pipeline run that dies before reaching the boundary would show
    the first run's coverage summary, and a caller reading the session would find
    a report authorizing a pathway that is no longer loaded.

    History under ``quarantine_history/`` is deliberately left in place.
    """

    for key in QUARANTINE_STATE_KEYS:
        st.session_state.pop(key, None)
    if outputs_dir is not None:
        try:
            clear_quarantine_artifacts(outputs_dir)
        except OSError as exc:  # pragma: no cover - a locked file must not kill the run
            logger.warning("reset_quarantine_state: could not clear artifacts: %s", exc)


def run_quarantine_boundary(
    payload: Dict[str, Any],
    *,
    strict_db: bool,
    outputs_dir: Path,
    pathway_context: Optional[Dict[str, Any]] = None,
    mode: ExportMode = DEFAULT_EXPORT_MODE,
) -> Dict[str, Any]:
    """The one place a payload is quarantined. Runs once per PAYLOAD VERSION.

    It sits immediately before the post-mapping Stage 3 revalidation because that
    is the FIRST blocking pre-export decision, not the last: a payload that fails
    there never initializes refinement review, so
    ``_generate_pwml_from_refinement_working_json`` returns on
    ``refinement_gate_errors`` and ``run_pwml_export`` -- where quarantine first
    lived -- is never reached at all. Quarantine placed inside the exporter was
    unreachable in exactly the case it exists for.

    "Once per payload version, under one set of rules" and not "once per session"
    is the invariant, and the difference is load-bearing. Refinement can delete the
    requested core between here and the export, and grounding rewrites entity
    identifiers on the way in; a decision made before either is a statement about a
    graph that no longer exists. Every decision therefore carries the hash of the
    payload it judged *and* the hash of the controls it judged under, and the
    exporter re-runs the boundary when the payload has moved. See
    :func:`strict_quarantine.decision_matches`.

    The controls are also **locked** into the session here. They cannot be
    re-evaluated later from the payload in hand: after a strict pass that payload
    is an already reduced graph, and re-judging it under research rules would
    produce a "research decision" that quarantined things, which is the one thing
    research mode promises never to do. So a control change voids the run rather
    than being absorbed -- see :func:`run_pwml_export`.

    ``pathway_context`` is the Stage-0 context and is passed explicitly: the final
    mapped payload carries no copy of it, so leaving quarantine to find one would
    put every production run in the undeclared-core regime.

    ``mode="research"`` runs the same decisions and applies none of them; the
    candidate comes back unchanged and the findings are review flags.
    """

    result = quarantine_and_close(
        payload,
        strict_db=bool(strict_db),
        pathway_context=pathway_context,
        mode=mode,
    )
    artifacts = write_quarantine_artifacts(result, outputs_dir)
    st.session_state["quarantine_report"] = result.quarantine_report
    st.session_state["quarantine_artifacts"] = artifacts
    st.session_state["quarantine_coverage"] = result.coverage
    st.session_state["quarantine_ok"] = bool(result.ok)
    st.session_state["quarantine_decision_controls"] = {
        "strict_db": bool(strict_db),
        "export_mode": coerce_mode(mode),
        "pathway_context": deepcopy(_safe_dict(pathway_context)),
        "decision_input_hash": result.quarantine_report.get("decision_input_hash"),
        "decision_id": result.quarantine_report.get("decision_id"),
    }
    return {
        "ok": bool(result.ok),
        "payload": result.payload,
        "quarantine_report": result.quarantine_report,
        "quarantine_artifacts": artifacts,
        "coverage": result.coverage,
        "refusal_reasons": list(result.refusal_reasons),
        "review_flags": quarantine_review_flags(result.quarantine_report),
    }


def quarantine_session_state() -> Dict[str, Any]:
    """The stored boundary decision, for callers downstream of it."""

    return {
        "quarantine_report": _safe_dict(st.session_state.get("quarantine_report")),
        "quarantine_artifacts": _safe_dict(st.session_state.get("quarantine_artifacts")),
    }


def initialize_refinement_review_state(
    final_mapped_payload: Dict[str, Any],
    mapping_report: Dict[str, Any],
) -> None:
    graph = build_draft_graph(final_mapped_payload)
    graph_dict = graph.to_dict()
    graph_bytes = render_draft_graph_to_png_bytes(graph_dict, dpi=100)
    qa_report = generate_qa_report(graph, final_mapped_payload)

    st.session_state.refinement_working_json = deepcopy(final_mapped_payload)
    st.session_state.refinement_graph_dict = graph_dict
    st.session_state.refinement_graph_bytes = graph_bytes
    st.session_state.refinement_qa_report = qa_report
    st.session_state.refinement_mapping_report = deepcopy(mapping_report)
    st.session_state.refinement_mapping_misses = compact_mapping_misses(mapping_report)
    st.session_state.refinement_history = []
    st.session_state.refinement_round = 0
    st.session_state.refinement_pwml_ready = False
    st.session_state.refinement_last_error = None
    st.session_state.refinement_last_warnings = []
    st.session_state.refinement_gate_errors = []
    st.session_state.refinement_checkpoints = []


for _refinement_key, _refinement_default in REFINEMENT_STATE_DEFAULTS.items():
    if _refinement_key not in st.session_state:
        st.session_state[_refinement_key] = deepcopy(_refinement_default)


def render_attempts(label: str, attempts: List[Dict[str, Any]]) -> None:
    with st.expander(label, expanded=False):
        for log in attempts:
            status = "success" if not log.get("error") else "retry"
            phase = log.get("phase")
            phase_label = f" ({phase})" if phase else ""
            note = log.get("note")
            note_label = f" [{note}]" if note else ""
            st.markdown(f"**Attempt {log['attempt']}{phase_label}** - {status}{note_label}")
            st.code(log["raw"], language="json")
            if log.get("error"):
                st.caption(log["error"])


def graph_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    adj, meta = build_graph(payload)
    entities = get_entities(payload)

    # Add every declared entity as an isolated node if it has no process connections,
    # so connected_components and degree counts include genuinely disconnected entities.
    for _name in entities.get("compounds", set()):
        adj.setdefault(node("compound", _name), set())
    for _name in entities.get("proteins", set()):
        adj.setdefault(node("protein", _name), set())
    for _name in entities.get("nucleic_acids", set()):
        adj.setdefault(node("nucleic_acid", _name), set())
    for _name in entities.get("element_collections", set()):
        adj.setdefault(node("element_collection", _name), set())
    for _name in entities.get("protein_complexes", set()):
        adj.setdefault(node("protein_complex", _name), set())

    comps = connected_components(adj)
    deg = degrees(adj)
    n_edges = sum(len(v) for v in adj.values()) // 2
    main_size = max((len(c) for c in comps), default=0)
    n_nodes = len(adj)
    protein_nodes = [node("protein", name) for name in sorted(entities.get("proteins", set()))]
    proteins_degree0 = sum(1 for n in protein_nodes if deg.get(n, 0) == 0)
    proteins_total = len(protein_nodes)
    proteins_attached = max(0, proteins_total - proteins_degree0)
    isolated_nodes = sum(1 for _, d in deg.items() if d == 0)
    return {
        **meta,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_components": len(comps),
        "main_component_size": main_size,
        "largest_component_pct": round((100.0 * main_size / n_nodes), 2) if n_nodes else 0.0,
        "n_isolated_nodes": isolated_nodes,
        "proteins_degree0": proteins_degree0,
        "proteins_attached_pct": round((100.0 * proteins_attached / proteins_total), 2) if proteins_total else 100.0,
    }


def qa_summary_line(payload: Dict[str, Any]) -> str:
    stats = graph_summary(payload)
    orphans = max(stats["n_components"] - 1, 0)
    return (
        f"Components: {stats['n_components']} | "
        f"Main size: {stats['main_component_size']} | "
        f"Orphans: {orphans}"
    )


def resolve_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.exists():
        return candidate
    project_root = PROJECT_ROOT
    rooted = project_root / path_text
    if rooted.exists():
        return rooted
    return candidate


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


#: Session-state key holding this run's diagnostic artifacts as a
#: ``{filename: object}`` map. The batch driver reads it to write those files into
#: the per-leg directory, and it is populated on the FAILURE paths too -- which is
#: the whole point. The 2026-07-28 legs died with ``files: []`` because every
#: artifact hand-off in the app sat downstream of an ``st.stop()``.
logger = logging.getLogger(__name__)

EXTRACTION_DIAGNOSTICS_KEY = "extraction_diagnostics"

#: Session-state key holding the directory this invocation's artifacts were
#: written to. Each run gets its own, so the path has to be reported rather than
#: assumed -- "look in tmp/extraction_diagnostics" is no longer an instruction
#: anyone can follow.
EXTRACTION_DIAGNOSTICS_DIR_KEY = "extraction_diagnostics_dir"

#: Where a run's diagnostics land on disk, under the project ``tmp`` tree the app
#: already uses for Stage-1 lock artifacts. Writing to disk is independent of the
#: session-state hand-off on purpose: an interactive user who never runs the batch
#: driver still gets the files, and a leg that dies before the driver reads
#: session state still leaves them behind.
DIAGNOSTICS_DIRNAME = "extraction_diagnostics"


def _start_extraction_diagnostics(source_text: str) -> Path:
    """Begin a fresh diagnostics record for one pipeline run. Returns its directory.

    Called once, before Stage 0. Replacing the recorder rather than clearing it
    means diagnostics from a previous paper can never leak into this run's
    artifacts -- a real hazard in the batch, where one process drives the app
    repeatedly.

    ``unique=True`` is not optional here. Streamlit serves every browser session
    from the same process, so two people (or two batch legs sharing an
    interpreter) extracting different papers at the same moment would write
    ``extraction_boundary_report.json`` over each other. The loser would then be
    diagnosed with the winner's evidence, which is worse than having none: it
    looks authoritative. Each invocation gets its own subdirectory, and the path
    is surfaced in the failure message so an operator knows where to look.
    """

    run_id = diagnostic_payload_hash(source_text)
    recorder = activate_extraction_diagnostics(
        run_id=run_id,
        artifact_dir=PROJECT_ROOT / "tmp" / DIAGNOSTICS_DIRNAME,
        unique=True,
    )
    st.session_state[EXTRACTION_DIAGNOSTICS_KEY] = {}
    st.session_state[EXTRACTION_DIAGNOSTICS_DIR_KEY] = str(recorder.artifact_dir or "")
    return Path(recorder.artifact_dir) if recorder.artifact_dir else PROJECT_ROOT / "tmp"


def _sync_extraction_diagnostics() -> Dict[str, Any]:
    """Publish the diagnostics collected so far into session state.

    Called at every point the run can end -- after Stage 0, after Stage 1 on both
    the pass and the fail path, and after the post-pipeline step -- because
    ``st.stop()`` means "this script run is over" and anything not in session
    state by then is lost. Never raises: a diagnostics hand-off that can abort the
    run defeats its own purpose.
    """

    try:
        artifacts = current_extraction_diagnostics().artifacts()
    except Exception:  # noqa: BLE001 - diagnostics must never break the run
        logger.debug("extraction diagnostics could not be published", exc_info=True)
        return {}
    st.session_state[EXTRACTION_DIAGNOSTICS_KEY] = artifacts
    return artifacts


def _last_cleaning_report() -> Dict[str, Any]:
    """The most recent Stage-1 cleaning pass, or ``{}``.

    The boundary needs exactly one fact from it -- whether cleaning discarded
    every process row the model returned -- to tell "the model produced nothing"
    apart from "cleaning removed what it produced". The last recorded pass is the
    right one: Stage 1 records per chunk and then once for the merge, and the
    merge is what the boundary is judging.
    """

    try:
        artifacts = current_extraction_diagnostics().artifacts()
    except Exception:  # noqa: BLE001
        return {}
    passes = _safe_list(_safe_dict(artifacts.get(CLEANING_REPORT_NAME)).get("passes"))
    return _safe_dict(passes[-1]) if passes else {}


#: Audit-round keys that are whole reports or candidate lists. Kept out of the
#: iteration summary and replaced by counts: a round carries up to five
#: candidates, each with its own paths and scores, plus two full gate reports, and
#: this artifact is rewritten once per round.
_BULKY_ROUND_KEYS = ("candidates", "post_patch_gate", "post_settlement_gate")


def _round_summary(round_row: Any) -> Dict[str, Any]:
    """One audit/gap round reduced to a bounded summary."""

    data = _safe_dict(round_row)
    summary = {key: value for key, value in data.items() if key not in _BULKY_ROUND_KEYS}
    for key in _BULKY_ROUND_KEYS:
        value = data.get(key)
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)
        elif isinstance(value, dict):
            summary[f"{key}_ok"] = bool(value.get("ok", False))
            summary[f"{key}_error_count"] = len(_safe_list(value.get("errors")))
    return summary


def _record_audit_round(round_row: Any) -> None:
    """File one completed audit round with the diagnostics recorder."""

    try:
        current_extraction_diagnostics().record_iteration("audit", _round_summary(round_row))
    except Exception:  # noqa: BLE001
        logger.debug("audit round could not be recorded", exc_info=True)


def _record_gap_round(round_row: Any) -> None:
    """File one completed gap-resolution round with the diagnostics recorder."""

    try:
        current_extraction_diagnostics().record_iteration("gap", _round_summary(round_row))
    except Exception:  # noqa: BLE001
        logger.debug("gap round could not be recorded", exc_info=True)


def _record_mapped_failure_snapshot(
    payload: Any,
    *,
    stage: str,
    reason: str,
    report: Any = None,
) -> None:
    """Record what the payload looked like when a post-mapping step failed.

    Counts, hashes and the failing issue codes -- never the payload. A mapped
    payload from a real run is 306 KB to 4.7 MB and is already written elsewhere;
    what is missing after a failure is the small summary that says which stage
    refused it and how much had survived to that point.

    Written only when mapping has actually run, so the *absence* of
    ``mapped_failure_snapshot.json`` remains meaningful: it says the run died
    before the mapper was called.
    """

    try:
        report_dict = _safe_dict(report)
        current_extraction_diagnostics().record_mapped_failure(
            {
                "stage": stage,
                "reason": str(reason)[:2000],
                "payload_hash": diagnostic_payload_hash(payload),
                "entity_counts": diagnostic_entity_counts(payload),
                "process_counts": diagnostic_process_counts(payload),
                "contract_summary": _safe_dict(report_dict.get("summary")),
                "failing_codes": sorted(
                    {
                        str(issue.get("code") or "")
                        for issue in _safe_list(report_dict.get("errors"))
                        if isinstance(issue, dict) and issue.get("code")
                    }
                ),
            }
        )
    except Exception:  # noqa: BLE001
        logger.debug("mapped failure snapshot could not be recorded", exc_info=True)


def _save_pipeline_outputs(
    project_root: Path,
    sbml_bytes: bytes,
    render_ready_bytes: bytes,
    clean_bytes: bytes,
) -> str:
    """Write pathway.sbml, pathway.render_ready.sbml, and pathway.render_ready.clean.sbml to outputs/."""
    out_dir = project_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    if sbml_bytes:
        (out_dir / "pathway.sbml").write_bytes(sbml_bytes)
    if render_ready_bytes:
        (out_dir / "pathway.render_ready.sbml").write_bytes(render_ready_bytes)
    if clean_bytes:
        (out_dir / "pathway.render_ready.clean.sbml").write_bytes(clean_bytes)
    return str(out_dir / "pathway.render_ready.sbml")


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _render_stage_contract_failure(
    failure: StageContractError,
    *,
    operation_label: str,
    download_key: str,
) -> None:
    """Expose a stage-boundary failure without reducing it to one message."""

    report = _safe_dict(failure.report)
    stage = str(report.get("stage") or failure.stage or "unknown_stage")
    effect_on_failure = str(report.get("effect_on_failure") or "unknown")
    summary = _safe_dict(report.get("summary"))

    if stage == "post_extraction":
        boundary = "Stage 2A merged-payload → Stage 2B DB mapping input boundary"
        execution_status = (
            "Stage 2B DB mapping and Stage 4 audit did not run. "
            "The merged extraction/inference payload was rejected before the mapper was called."
        )
    elif stage == "post_mapping":
        boundary = "Stage 2B mapping output → Stage 3 normalization input boundary"
        execution_status = (
            "Stage 2B DB mapping ran, but Stage 3 normalization and Stage 4 audit did not run."
        )
    else:
        boundary = f"{stage} stage boundary"
        execution_status = "The pipeline stopped at this boundary; later stages did not run."

    st.error(f"{operation_label} stopped at the {boundary}.")
    st.info(execution_status)
    st.write(
        {
            "boundary": boundary,
            "stage": stage,
            "effect_on_failure": effect_on_failure,
            "summary": summary,
            "message": str(failure),
        }
    )

    structured_issues: List[Tuple[str, Dict[str, Any]]] = []
    for severity in ("errors", "warnings"):
        for issue in _safe_list(report.get(severity)):
            if isinstance(issue, dict):
                structured_issues.append((severity[:-1], issue))

    st.write(f"Structured issues ({len(structured_issues)})")
    for severity, issue in structured_issues:
        displayed_issue: Dict[str, Any] = {
            "severity": severity,
            "code": issue.get("code", ""),
            "pointer": issue.get("pointer", ""),
            "path": issue.get("path", ""),
            "message": issue.get("message") or issue.get("reason", ""),
        }
        for optional_field in ("bucket", "name"):
            if issue.get(optional_field) not in (None, ""):
                displayed_issue[optional_field] = issue[optional_field]
        st.write(displayed_issue)
        _render_issue_detail(issue.get("detail"))

    st.write("Full stage contract report")
    st.json(report)
    st.download_button(
        "Download stage contract error report",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name="stage_contract_error_report.json",
        mime="application/json",
        key=download_key,
    )


def _render_issue_detail(detail: Any) -> None:
    """Render a failure ``detail`` beneath the issue it explains.

    Collapsed ``st.json`` rather than ``st.expander``: these lists already run
    inside an expander and Streamlit raises on a nested one. The detail arrives
    already bounded by ``t2pw.pipeline.failure_detail``, so it is rendered
    wholesale -- re-truncating here would make the UI disagree with the batch
    artifacts. See ``docs/regression_baseline_2026-07-28.md`` §4.
    """

    data = _safe_dict(detail)
    if not data:
        return
    summary = detail_headline(data)
    if summary:
        st.caption(summary)
    st.json(data, expanded=False)


def _research_flag_rows(issues: Any) -> List[Dict[str, Any]]:
    """Flatten downgraded contract issues into a table the reviewer can scan."""

    rows: List[Dict[str, Any]] = []
    for issue in _safe_list(issues):
        if not isinstance(issue, dict):
            continue
        rows.append(
            {
                "code": str(issue.get("code") or ""),
                "where": str(issue.get("pointer") or issue.get("path") or ""),
                "stage": str(issue.get("stage") or ""),
                # NB: this column has always held the human message, not the
                # structured ``detail`` key -- which is why the value that was
                # actually rejected gets its own "found" column rather than
                # being folded in here.
                "detail": str(issue.get("message") or issue.get("reason") or ""),
                "found": detail_headline(issue.get("detail")) or str(issue.get("found") or ""),
            }
        )
    return rows


def _render_research_mode_panels(artifacts: Any) -> None:
    """Show what research mode relaxed, prominently enough that it is not missed.

    Research mode is fail-open, which is only safe if it is never fail-silent.
    This renders three things in descending order of how much they should worry
    the reviewer: biology/provenance problems that would have aborted a strict
    run, content that strict mode would have deleted, and the PathWhiz format
    rules that were skipped on purpose.
    """

    pa = _safe_dict(artifacts)
    if not is_research(pa.get("export_mode")):
        return

    flags = _safe_list(pa.get("research_review_flags"))
    skipped = _safe_list(pa.get("research_skipped_format_rules"))
    preserved = _safe_list(pa.get("research_normalization_actions"))

    st.subheader("Review flags — research mode")
    if flags:
        st.error(
            f"**{len(flags)} biology/provenance issue(s) need review.** "
            "In strict mode each of these would have stopped the run. The pathway "
            "below was still produced — treat it as a candidate, not a result."
        )
    else:
        st.success(
            "No biology or provenance check failed. Format rules were still "
            "relaxed, so this is a research artifact, not an importable PWML file."
        )
    st.write(
        {
            "biology_provenance_flags": len(flags),
            "format_rules_skipped": len(skipped),
            "content_preserved_that_strict_would_delete": len(preserved),
        }
    )

    if flags:
        with st.expander(f"Biology / provenance flags ({len(flags)})", expanded=True):
            st.table(_research_flag_rows(flags))

    if preserved:
        with st.expander(
            f"Content kept that strict mode would have deleted ({len(preserved)})",
            expanded=False,
        ):
            st.caption(
                "Strict mode deletes these to satisfy the PathWhiz importer or to "
                "pre-empt a gate. Research mode keeps them so you can judge them."
            )
            st.table(
                [
                    {
                        "what": str(action.get("type", "")).replace("research_mode_", ""),
                        "name": str(action.get("name") or action.get("from") or ""),
                        "where": str(action.get("json_pointer") or ""),
                        "why strict drops it": str(action.get("reason") or ""),
                    }
                    for action in preserved
                    if isinstance(action, dict)
                ]
            )

    if skipped:
        with st.expander(f"PathWhiz format rules skipped ({len(skipped)})", expanded=False):
            st.caption(
                "These are importer requirements only — a missing external DB id, "
                "a generated protein_complex wrapper rule. Skipping them says "
                "nothing about whether the biology is right."
            )
            st.table(_research_flag_rows(skipped))

    st.download_button(
        "Download review flags (JSON)",
        data=_json_dump(
            {
                "export_mode": pa.get("export_mode"),
                "biology_provenance_flags": flags,
                "format_rules_skipped": skipped,
                "content_preserved": preserved,
            }
        ),
        file_name="research_mode_review_flags.json",
        mime="application/json",
        key="research_mode_review_flags_download",
    )

    _render_research_citation_report(pa, flags, skipped)


def _render_research_citation_report(
    pa: Dict[str, Any],
    flags: List[Any],
    skipped: List[Any],
) -> None:
    """Render the tiered citation report and its downloadable exports.

    Reads the **pre-merge** synthesized payload (``rag_result.payload``): the
    merged payload has had ``rag_provenance`` / ``source_papers`` /
    ``source_refs`` stripped from every reaction by ``pipeline._clean_processes``,
    so tiering the merged one would report Tier D for everything.

    Imported here rather than at module scope because this is seam S5 — the
    orchestrator is the only place allowed to reach into ``t2pw.rag``, and only
    when RAG actually produced something.
    """

    rag_result = st.session_state.get("rag_result")
    synthesized = getattr(rag_result, "payload", None)
    if not isinstance(synthesized, dict) or not synthesized:
        st.info(
            "No multi-paper synthesis in this run, so there is nothing to cite. "
            "Enable RAG to get a tiered citation report."
        )
        return

    try:
        from t2pw.rag.research_report import build_citation_report

        report = build_citation_report(
            synthesized,
            review_flags=flags,
            format_gaps=skipped,
            rag_result=rag_result,
            mode=RESEARCH,
            # Provenance comes from the pre-merge payload above; identifiers can
            # only come from the mapped one, whose entity rows survive the merge.
            identity_source=pa.get("final_mapped_db") or pa.get("final_mapped"),
        )
    except Exception as exc:  # noqa: BLE001 — the report must never break the run
        st.warning(f"Citation report unavailable: {exc}")
        return

    st.subheader("Citation report — evidence tiers")
    counts = _safe_dict(_safe_dict(report.summary).get("tier_counts"))
    st.write(report.summary)
    if int(counts.get("D", 0) or 0):
        st.error(
            f"{counts['D']} element(s) are **UNSOURCED** — no provenance of any kind. "
            "These are the first thing to check."
        )

    with st.expander("Per-reaction citations", expanded=True):
        if report.lines:
            for line in report.lines:
                st.text(line)
        else:
            st.caption("No reactions carried provenance.")

    with st.expander("Per-entity tiers and supporting passages", expanded=False):
        if report.entities:
            st.table(
                [
                    {
                        "entity": row.get("name", ""),
                        "tier": row.get("tier", ""),
                        "label": row.get("tier_label", ""),
                        "sources": "; ".join(
                            str(_safe_dict(c).get("citation", ""))
                            for c in _safe_list(row.get("sources"))
                        )
                        or "(none)",
                    }
                    for row in report.entities
                ]
            )
        else:
            st.caption("No entities to tier.")

    st.caption(
        "Page numbers are not available in this pipeline and are never invented — "
        "citations give section plus the verbatim passage. A review that cites the "
        "seed paper cannot be told apart from an independent corroboration, because "
        "reference lists are discarded at ingest."
    )

    readable = report.to_text()
    st.download_button(
        "⬇  Download the readable report (TXT) — start here",
        data=readable,
        file_name="research_pathway_report.txt",
        mime="text/plain",
        key="research_report_txt",
        type="primary",
    )
    st.caption(
        "Plain text, top to bottom: every reaction numbered with its substrates, "
        "products, catalyst and its own evidence block, then what to check first."
    )
    with st.expander("Preview the readable report", expanded=False):
        st.text(readable)

    st.caption("Other formats:")
    col_json, col_md, col_csv = st.columns(3)
    col_json.download_button(
        "Pathway + citations (JSON)",
        data=report.to_json(),
        file_name="research_pathway_citations.json",
        mime="application/json",
        key="research_report_json",
        help="Machine-readable, stable key order.",
    )
    col_md.download_button(
        "Review report (Markdown)",
        data=report.to_markdown(),
        file_name="research_pathway_review.md",
        mime="text/markdown",
        key="research_report_md",
        help="Same content, rendered.",
    )
    col_csv.download_button(
        "Element table (CSV)",
        data=report.to_csv(),
        file_name="research_pathway_elements.csv",
        mime="text/csv",
        key="research_report_csv",
        help="One row per element, for spreadsheet triage.",
    )


def _write_reaction_preservation_report(stage: str, payload: Any) -> Optional[Dict[str, Any]]:
    try:
        return write_reaction_preservation_report_if_manifest(
            payload,
            PROJECT_ROOT / "tmp",
            stage=stage,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "stage": stage,
            "error": str(exc),
            "locked_reaction_count": 0,
            "preserved_count": 0,
            "missing_count": 0,
            "changed_count": 0,
            "quarantined_count": 0,
            "details": [],
        }


def _json_dump(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _refinement_qa_issue_groups(qa_report: Any) -> Dict[str, List[Any]]:
    report = _safe_dict(qa_report)
    groups: Dict[str, List[Any]] = {}

    issues = report.get("issues")
    if isinstance(issues, list):
        groups["issues"] = [issue for issue in issues if issue]

    flags = report.get("flags")
    if isinstance(flags, dict):
        for name, values in flags.items():
            if isinstance(values, list):
                non_empty = [value for value in values if value]
                if non_empty:
                    groups[str(name)] = non_empty

    return groups


def _compact_refinement_issue(value: Any) -> str:
    if isinstance(value, dict):
        preferred = [
            "entity",
            "reaction",
            "name",
            "type",
            "reason",
            "message",
            "conflict",
            "assigned_class",
        ]
        parts = [f"{key}: {value[key]}" for key in preferred if value.get(key) not in (None, "", [], {})]
        if parts:
            return "; ".join(parts)
    return str(value)


def _pwml_reference_path(project_root: Path) -> Path:
    ref_candidates = [
        project_root / "reference" / "PW000001.pwml",
        project_root / "reference" / "PW012926.pwml",
    ]
    return next((path for path in ref_candidates if path.exists()), ref_candidates[0])


def _write_reviewed_payload_snapshot(working_json: Dict[str, Any], work_dir: str | Path) -> Path:
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    reviewed_json_path = work_path / "final.mapped.json"
    reviewed_json_path.write_text(
        json.dumps(working_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    post_artifacts = st.session_state.get("post_pipeline_artifacts")
    if isinstance(post_artifacts, dict):
        reviewed_payload = deepcopy(working_json)
        post_artifacts["final_mapped_db"] = reviewed_payload
        post_artifacts["final_export_input"] = deepcopy(reviewed_payload)
        post_artifacts["reviewed_final_mapped_path"] = str(reviewed_json_path)
        st.session_state["post_pipeline_artifacts"] = post_artifacts

    return reviewed_json_path


def _generate_pwml_from_refinement_working_json(work_dir: str | Path) -> Dict[str, Any]:
    working_json = st.session_state.get("refinement_working_json")
    if not isinstance(working_json, dict) or not working_json:
        return {
            "ok": False,
            "error": "No reviewed mapped pathway is available for PWML export.",
            "counts": {},
            "issues": 0,
            "output_path": "",
            "qa": {},
            "grounding_report": {},
        }

    refinement_gate_errors = _safe_list(st.session_state.get("refinement_gate_errors"))
    if refinement_gate_errors:
        return {
            "ok": False,
            "error": (
                "PWML export stopped by pre-export Stage 3 revalidation. "
                "Resolve the refinement gate errors first."
            ),
            "counts": {},
            "issues": len(refinement_gate_errors),
            "output_path": "",
            "qa": {},
            "grounding_report": {},
            "stage3_gate_errors": refinement_gate_errors,
            # Carried even here. This return is reached when the boundary refused
            # OR when Stage 3 failed after it, and the artifacts are the only
            # record of which -- dropping them makes the two indistinguishable.
            **quarantine_session_state(),
        }

    _write_reviewed_payload_snapshot(working_json, work_dir)
    st.session_state.refinement_pwml_ready = True

    project_root = PROJECT_ROOT
    grounding_dict = None
    if bool(st.session_state.get("pwml_grounding", False)):
        grounding_dict = st.session_state.get("pwml_grounding_dict")

    return run_pwml_export(
        working_json,
        pathway_name=str(st.session_state.get("pwml_name") or "Generated Pathway"),
        pathway_description=str(st.session_state.get("pwml_description") or ""),
        pathway_subject=str(st.session_state.get("pwml_subject") or "Metabolic"),
        project_root=project_root,
        ref_path=_pwml_reference_path(project_root),
        vis_width=int(st.session_state.get("pwml_width") or 3200),
        vis_height=int(st.session_state.get("pwml_height") or 1400),
        background_color=str(st.session_state.get("pwml_bg") or "#FFFFFF"),
        grounding_dict=grounding_dict,
        strict_db=bool(st.session_state.get("pwml_strict_db", True)),
        # The boundary already ran, upstream of the Stage 3 check that gated this
        # payload into review. Hand its decisions over rather than making new ones.
        pathway_context=_safe_dict(st.session_state.get("pathway_context")),
        export_mode=coerce_mode(st.session_state.get("export_mode")),
        **quarantine_session_state(),
    )


def _render_review_refine_section(
    *,
    mapping_cache_path: str | Path,
    work_dir: str | Path,
    id_source: str,
    db_config: Dict[str, Any],
) -> bool:
    if st.session_state.get("refinement_working_json") is None:
        return False

    st.header("Review & Refine Pathway")

    if st.session_state.get("refinement_last_error"):
        st.error(st.session_state.refinement_last_error)
    for warning_text in _safe_list(st.session_state.get("refinement_last_warnings")):
        st.warning(warning_text)

    refinement_round = int(st.session_state.get("refinement_round") or 0)
    if refinement_round <= 0:
        st.caption("Round 0: initial mapped output")
    else:
        st.caption(f"Round {refinement_round}: after {refinement_round} refinement rounds")

    graph_bytes = st.session_state.get("refinement_graph_bytes")
    if isinstance(graph_bytes, (bytes, bytearray)) and graph_bytes:
        st.image(graph_bytes, caption="Mapped pathway graph after audit and DB mapping")
        st.download_button(
            "Download mapped graph diagram",
            bytes(graph_bytes),
            file_name="mapped_pathway_graph.png",
            mime="image/png",
            key="download_refinement_graph_png",
        )

    st.subheader("QA")
    qa_report = _safe_dict(st.session_state.get("refinement_qa_report"))
    qa_issue_groups = _refinement_qa_issue_groups(qa_report)
    qa_issue_count = sum(len(values) for values in qa_issue_groups.values())
    if qa_issue_count:
        st.warning(f"{qa_issue_count} QA issues need review.")
        with st.expander("QA Issue Details", expanded=False):
            for group_name, issues in qa_issue_groups.items():
                st.markdown(f"**{group_name.replace('_', ' ').title()}** ({len(issues)})")
                for issue in issues[:25]:
                    st.markdown(f"- {_compact_refinement_issue(issue)}")
                if len(issues) > 25:
                    st.caption(f"{len(issues) - 25} more not shown.")
    else:
        st.success("No QA issues found.")

    with st.expander("Unmapped or Uncertain Entities", expanded=False):
        mapping_misses = _safe_list(st.session_state.get("refinement_mapping_misses"))
        if mapping_misses:
            st.json(mapping_misses)
        else:
            st.success("No unmapped or uncertain entities found.")

    with st.expander("Refinement History", expanded=False):
        history = _safe_list(st.session_state.get("refinement_history"))
        if not history:
            st.info("No refinement rounds have been applied yet.")
        for index, entry in enumerate(history, start=1):
            if not isinstance(entry, dict):
                continue
            round_number = entry.get("round", index)
            user_request = entry.get("user_request") or entry.get("request") or ""
            change_summary = entry.get("change_summary") or entry.get("summary") or ""
            patch_op_count = entry.get("patch_op_count")
            if patch_op_count is None and isinstance(entry.get("patch"), list):
                patch_op_count = len(entry["patch"])
            entities_remapped = _safe_list(entry.get("entities_remapped"))

            st.markdown(f"**Round {round_number}**")
            st.markdown(f"- User request: {user_request or 'None recorded'}")
            st.markdown(f"- Change summary: {change_summary or 'None recorded'}")
            st.markdown(f"- Patch op count: {patch_op_count if patch_op_count is not None else 0}")
            st.markdown(
                "- Entities remapped: "
                + (", ".join(str(entity) for entity in entities_remapped) if entities_remapped else "None")
            )
            for label, key in (
                ("Gate errors", "gate_errors"),
                ("Norm warnings", "norm_warnings"),
                ("Patch apply errors", "patch_apply_errors"),
            ):
                values = _safe_list(entry.get(key))
                if values:
                    st.markdown(f"- {label}:")
                    for value in values[:10]:
                        st.markdown(f"  - {value}")
                    if len(values) > 10:
                        st.caption(f"{len(values) - 10} more {label.lower()} not shown.")

    with st.expander("Debug Downloads", expanded=False):
        reviewed_col, history_col, qa_col, misses_col = st.columns(4)
        reviewed_col.download_button(
            "Reviewed JSON",
            data=json.dumps(st.session_state.refinement_working_json, indent=2, ensure_ascii=False),
            file_name="reviewed_final_mapped.json",
            mime="application/json",
            key="download_reviewed_final_mapped_json",
        )
        history_col.download_button(
            "Refinement History",
            data=json.dumps(st.session_state.refinement_history, indent=2, ensure_ascii=False),
            file_name="refinement_history.json",
            mime="application/json",
            key="download_refinement_history_json",
        )
        qa_col.download_button(
            "QA Report",
            data=json.dumps(st.session_state.refinement_qa_report, indent=2, ensure_ascii=False),
            file_name="refinement_qa_report.json",
            mime="application/json",
            key="download_refinement_qa_report_json",
        )
        misses_col.download_button(
            "Mapping Misses",
            data=json.dumps(st.session_state.refinement_mapping_misses, indent=2, ensure_ascii=False),
            file_name="refinement_mapping_misses.json",
            mime="application/json",
            key="download_refinement_mapping_misses_json",
        )

    st.subheader("AI Edit Chat")
    st.text_area(
        "Message to AI",
        key="refinement_request",
        height=140,
        placeholder="Type the pathway edits to make, then submit them to the AI curator.",
    )

    refine_col, undo_col, pwml_col = st.columns(3)
    if refine_col.button("Submit Changes to AI", key="refinement_submit_ai"):
        refinement_request = str(st.session_state.get("refinement_request") or "").strip()
        if not refinement_request:
            st.warning("Describe the changes you want before submitting to AI.")
            st.stop()

        working_json = st.session_state.get("refinement_working_json")
        graph_bytes = st.session_state.get("refinement_graph_bytes")
        if not isinstance(working_json, dict) or not working_json:
            st.error("No refinement working JSON is available.")
            st.stop()
        if not isinstance(graph_bytes, (bytes, bytearray)) or not graph_bytes:
            st.error("No refinement graph image is available.")
            st.stop()

        with st.spinner("Asking AI for a pathway patch..."):
            ai_result = run_interactive_curator_round(
                working_json=working_json,
                graph_png_bytes=bytes(graph_bytes),
                user_request=refinement_request,
                qa_report=st.session_state.refinement_qa_report,
                mapping_misses=st.session_state.refinement_mapping_misses,
                history=st.session_state.refinement_history,
            )

        if ai_result.get("error"):
            st.session_state.refinement_last_error = str(ai_result["error"])
            st.error(st.session_state.refinement_last_error)
            st.stop()

        patch = ai_result.get("patch") if isinstance(ai_result, dict) else []
        if not patch:
            rationale = str(ai_result.get("rationale", "") if isinstance(ai_result, dict) else "")
            st.session_state.refinement_last_error = None
            st.session_state.refinement_last_warnings = []
            st.info(rationale or "AI returned no patch.")
            st.session_state.refinement_history.append(
                {
                    "round": st.session_state.refinement_round,
                    "user_request": refinement_request,
                    "change_summary": ai_result.get("change_summary") or "No patch was applied.",
                    "rationale": rationale or "No patch was applied.",
                    "patch_op_count": 0,
                    "entities_remapped": [],
                    "gate_errors": [],
                    "norm_warnings": [],
                    "patch_apply_errors": [],
                    "no_patch_applied": True,
                }
            )
            st.stop()

        checkpoint = {
            "round": st.session_state.refinement_round,
            "working_json": copy.deepcopy(st.session_state.refinement_working_json),
            "graph_dict": copy.deepcopy(st.session_state.refinement_graph_dict),
            "graph_bytes": st.session_state.refinement_graph_bytes,
            "qa_report": copy.deepcopy(st.session_state.refinement_qa_report),
            "mapping_report": copy.deepcopy(st.session_state.refinement_mapping_report),
            "mapping_misses": copy.deepcopy(st.session_state.refinement_mapping_misses),
            "history": copy.deepcopy(st.session_state.refinement_history),
            "gate_errors": copy.deepcopy(st.session_state.refinement_gate_errors),
        }
        if not isinstance(st.session_state.get("refinement_checkpoints"), list):
            st.session_state.refinement_checkpoints = []
        st.session_state.refinement_checkpoints.append(checkpoint)
        if len(st.session_state.refinement_checkpoints) > 5:
            st.session_state.refinement_checkpoints.pop(0)

        with st.spinner("Applying patch, remapping IDs, and re-rendering graph..."):
            rerender_result = apply_patch_and_rerender(
                working_json=st.session_state.refinement_working_json,
                patch=patch,
                mapping_cache_path=mapping_cache_path,
                work_dir=work_dir,
                id_source=id_source,
                db_config=db_config,
            )

        st.session_state.refinement_working_json = rerender_result["updated_json"]
        st.session_state.refinement_graph_dict = rerender_result.get("graph") or build_draft_graph(
            rerender_result["updated_json"]
        ).to_dict()
        st.session_state.refinement_graph_bytes = rerender_result["graph_png_bytes"]
        st.session_state.refinement_qa_report = rerender_result["qa_report"]
        st.session_state.refinement_mapping_report = rerender_result["mapping_report"]
        st.session_state.refinement_mapping_misses = rerender_result["mapping_misses"]
        st.session_state.refinement_gate_errors = rerender_result["gate_errors"]
        st.session_state.refinement_round += 1
        st.session_state.refinement_history.append(
            {
                "round": st.session_state.refinement_round,
                "user_request": refinement_request,
                "change_summary": ai_result.get("change_summary", ""),
                "rationale": ai_result.get("rationale", ""),
                "patch_op_count": len(ai_result.get("patch", [])),
                "entities_remapped": rerender_result.get("entities_remapped", []),
                "gate_errors": rerender_result.get("gate_errors", []),
                "norm_warnings": rerender_result.get("norm_warnings", []),
                "patch_apply_errors": rerender_result.get("patch_apply_errors", []),
            }
        )

        warning_messages: List[str] = []
        for label, key in (
            ("Gate errors", "gate_errors"),
            ("Normalization warnings", "norm_warnings"),
            ("Patch apply errors", "patch_apply_errors"),
        ):
            values = _safe_list(rerender_result.get(key))
            if values:
                warning_messages.append(f"{label}: {len(values)} issue(s). Review the refinement history for details.")
        st.session_state.refinement_last_error = None
        st.session_state.refinement_last_warnings = warning_messages
        st.rerun()
    if undo_col.button("Undo Last Refinement", key="refinement_undo_last"):
        if not st.session_state.refinement_checkpoints:
            st.warning("No refinement checkpoint is available to undo.")
            st.stop()

        checkpoint = st.session_state.refinement_checkpoints.pop()
        st.session_state.refinement_round = checkpoint["round"]
        st.session_state.refinement_working_json = checkpoint["working_json"]
        checkpoint_graph_dict = checkpoint.get("graph_dict")
        if not checkpoint_graph_dict and isinstance(checkpoint.get("working_json"), dict):
            checkpoint_graph_dict = build_draft_graph(checkpoint["working_json"]).to_dict()
        st.session_state.refinement_graph_dict = checkpoint_graph_dict
        st.session_state.refinement_graph_bytes = checkpoint["graph_bytes"]
        st.session_state.refinement_qa_report = checkpoint["qa_report"]
        st.session_state.refinement_mapping_report = checkpoint["mapping_report"]
        st.session_state.refinement_mapping_misses = checkpoint["mapping_misses"]
        st.session_state.refinement_history = checkpoint["history"]
        st.session_state.refinement_gate_errors = checkpoint["gate_errors"]
        st.session_state.refinement_last_error = None
        st.success("Last refinement was undone.")
        st.rerun()
    if pwml_col.button("Generate PWML", key="refinement_generate_pwml"):
        try:
            with st.spinner("Building PWML from reviewed pathway JSON..."):
                _pwml_result = _generate_pwml_from_refinement_working_json(work_dir)
        except Exception as exc:  # noqa: BLE001
            _pwml_result = {
                "ok": False,
                "error": str(exc),
                "counts": {},
                "issues": 0,
                "output_path": "",
                "qa": {},
                "grounding_report": {},
            }
        st.session_state["pwml_export_result"] = _pwml_result
        if _pwml_result.get("ok"):
            st.success(f"PWML generated from reviewed JSON: {_pwml_result.get('output_path')}")
        else:
            st.error(f"PWML export failed: {_pwml_result.get('error', 'unknown')}")

    return True


def _json_artifact_entries(
    post_artifacts: Optional[Dict[str, Any]],
    pwml_result: Optional[Dict[str, Any]],
) -> List[Tuple[str, str, Any]]:
    entries: List[Tuple[str, str, Any]] = []
    if isinstance(post_artifacts, dict):
        for label, filename, key in [
            ("Final merged output", "final.json", "final_payload_snapshot"),
            ("Stage 2A extraction + inference input", "pre_mapping_input.json", "pre_mapping_input"),
            ("Stage 2B mapped payload", "stage2.mapped.json", "stage2_payload"),
            ("Stage 2B mapping report", "stage2_mapping_report.json", "stage2_mapping_report"),
            ("Stage 2B runtime schema report", "stage2_runtime_schema_report.json", "stage2_runtime_schema_report"),
            ("Stage 3 normalization input", "pre_normalization_input.json", "pre_normalization_input"),
            ("Post-normalization payload", "pre_normalized_input.json", "pre_normalized_input"),
            ("Final audited", "final.audited.json", "final_audited"),
            ("Final mapped - DB mapping", "final.mapped.json", "final_mapped_db"),
            ("Final export input", "final.export_input.json", "final_export_input"),
            ("Stage 6 mapping report", "mapping_report.json", "mapping_report"),
            ("Enrichment report", "enrichment_report.json", "enrichment_report"),
            ("Post-enrichment runtime schema report", "post_enrichment_runtime_schema_report.json", "post_enrichment_runtime_schema_report"),
            ("Pre-export runtime schema report", "pre_export_runtime_schema_report.json", "pre_export_runtime_schema_report"),
            ("Audit report", "audit_report.json", "audit_report"),
            ("Audit apply report", "audit_apply_report.json", "audit_apply_report"),
            (
                "Quarantined locked reactions after Stage 3",
                "quarantined_locked_reactions.json",
                "locked_reaction_quarantine_artifact",
            ),
        ]:
            value = post_artifacts.get(key)
            if value not in (None, "", [], {}):
                entries.append((label, filename, value))
        if post_artifacts.get("gap_resolution_iterations"):
            entries.append(
                (
                    "Stage 3 resolution iterations",
                    "stage3_resolution_iterations.json",
                    post_artifacts.get("gap_resolution_iterations"),
                )
            )
        preservation_reports = post_artifacts.get("reaction_preservation_reports")
        if isinstance(preservation_reports, dict):
            for stage, report in preservation_reports.items():
                if report not in (None, "", [], {}):
                    entries.append(
                        (
                            f"Reaction preservation {stage}",
                            f"reaction_preservation_{stage}.json",
                            report,
                        )
                    )
    if isinstance(pwml_result, dict):
        for label, filename, key in [
            ("PWML IR", "final.pwml_ir.json", "pwml_ir"),
            ("PWML IR report", "pwml_ir_report.json", "pwml_ir_report"),
            ("PWML IR validation", "pwml_ir_validation.json", "pwml_ir_validation"),
            ("PWML validation report", "pwml_validation_report.json", "validation_report"),
            ("PWML QA", "pwml_qa.json", "qa"),
            ("Reaction preservation before final export", "final_reaction_preservation_report.json", "reaction_preservation_before_final_export"),
            ("Reaction preservation after final export", "reaction_preservation_report_after_final_export.json", "reaction_preservation_after_final_export"),
        ]:
            value = pwml_result.get(key)
            if value not in (None, "", [], {}):
                entries.append((label, filename, value))
    return entries


def render_json_artifact_compare(
    post_artifacts: Optional[Dict[str, Any]],
    pwml_result: Optional[Dict[str, Any]],
    *,
    key_prefix: str,
) -> None:
    entries = _json_artifact_entries(post_artifacts, pwml_result)
    with st.expander("Compare JSON artifacts", expanded=False):
        if not entries:
            st.info("Run audit and DB mapping, then generate PWML from the review panel to populate artifacts here.")
            return
        labels = [entry[0] for entry in entries]
        if len(entries) == 1:
            selected_label = st.selectbox(
                "Artifact",
                labels,
                index=0,
                key=f"{key_prefix}_json_single",
            )
            entry_by_label = {label: (filename, value) for label, filename, value in entries}
            filename, value = entry_by_label[selected_label]
            st.download_button(
                f"Download {filename}",
                _json_dump(value),
                file_name=filename,
                mime="application/json",
                key=f"{key_prefix}_json_single_download",
            )
            st.code(_json_dump(value), language="json")
            return

        default_right = labels.index("PWML IR") if "PWML IR" in labels else min(1, len(labels) - 1)
        st.caption("Use this to inspect final.mapped.json beside the PWML IR or any report.")
        col_left, col_right = st.columns(2)
        left_label = col_left.selectbox(
            "Left artifact",
            labels,
            index=labels.index("Final mapped - DB mapping") if "Final mapped - DB mapping" in labels else 0,
            key=f"{key_prefix}_json_compare_left",
        )
        right_label = col_right.selectbox(
            "Right artifact",
            labels,
            index=default_right,
            key=f"{key_prefix}_json_compare_right",
        )
        entry_by_label = {label: (filename, value) for label, filename, value in entries}
        left_filename, left_value = entry_by_label[left_label]
        right_filename, right_value = entry_by_label[right_label]
        with col_left:
            st.download_button(
                f"Download {left_filename}",
                _json_dump(left_value),
                file_name=left_filename,
                mime="application/json",
                key=f"{key_prefix}_json_compare_left_download",
            )
            st.code(_json_dump(left_value), language="json")
        with col_right:
            st.download_button(
                f"Download {right_filename}",
                _json_dump(right_value),
                file_name=right_filename,
                mime="application/json",
                key=f"{key_prefix}_json_compare_right_download",
            )
            st.code(_json_dump(right_value), language="json")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_objective_score(
    *,
    eval_error_count: int,
    eval_warning_count: int,
    source_error_count: int,
    rejected_count: int,
    accepted_count: int,
    source_patch_count: int,
) -> Tuple[int, int, int, int, int, int]:
    # Lower is better.
    return (
        int(eval_error_count),
        int(eval_warning_count),
        int(source_error_count),
        int(rejected_count),
        -int(accepted_count),
        int(source_patch_count),
    )


def run_libsbml_checker(sbml_bytes: bytes) -> Dict[str, Any]:
    try:
        import libsbml  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"python-libsbml unavailable: {exc}",
            "validation": {
                "check_count": 0,
                "error_count": 0,
                "has_errors": False,
                "messages": [],
            },
        }

    text = ""
    try:
        text = sbml_bytes.decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"Invalid UTF-8 SBML payload: {exc}",
            "validation": {
                "check_count": 0,
                "error_count": 0,
                "has_errors": True,
                "messages": [],
            },
        }

    try:
        doc = libsbml.readSBMLFromString(text)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"libSBML parse failure: {exc}",
            "validation": {
                "check_count": 0,
                "error_count": 0,
                "has_errors": True,
                "messages": [],
            },
        }

    if doc is None:
        return {
            "ok": False,
            "error": "libSBML returned no document.",
            "validation": {
                "check_count": 0,
                "error_count": 0,
                "has_errors": True,
                "messages": [],
            },
        }

    check_count = int(doc.checkConsistency())
    messages: List[Dict[str, Any]] = []
    has_errors = False
    for idx in range(doc.getNumErrors()):
        err = doc.getError(idx)
        entry = {
            "severity": int(err.getSeverity()),
            "category": int(err.getCategory()),
            "message": err.getMessage(),
            "line": int(err.getLine()),
        }
        messages.append(entry)
        if entry["severity"] >= 2:
            has_errors = True

    return {
        "ok": True,
        "error": "",
        "validation": {
            "check_count": check_count,
            "error_count": len(messages),
            "has_errors": has_errors,
            "messages": messages,
        },
    }

def run_post_pipeline_sbml_artifacts(
    final_payload: Dict[str, Any],
    *,
    build_legacy_sbml: bool = False,
    use_llm_audit: bool,
    use_sbml_overwatch: bool,
    default_compartment: str,
    mapping_cache_path: str,
    id_source: str,
    db_host: str,
    db_port: int,
    db_user: str,
    db_password: str,
    db_schema: str,
    audit_max_rounds: int,
    audit_timeout_seconds: int,
    audit_candidate_count: int,
    use_example_retrieval: bool,
    example_index_path: str,
    example_top_k: int,
    use_gap_resolver: bool,
    use_llm_gap_resolver: bool,
    gap_resolver_max_items: int,
    qa_report: Optional[Dict[str, Any]] = None,
    reaction_summary: Optional[str] = None,
    use_stoich_agent: bool = False,
    rag_evidence_context: str = "",
    export_mode: ExportMode = DEFAULT_EXPORT_MODE,
    # The quarantine boundary runs inside this function now (see "THE canonical
    # payload" below), so the three inputs it needs come in with it. They used to
    # be read from session state at the UI call site; defaulted here so a direct
    # caller -- a test, a script -- gets today's behaviour without restating them.
    strict_db: bool = True,
    pathway_context: Optional[Dict[str, Any]] = None,
    outputs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    # ``export_mode`` selects the gate policy for this whole run. "pathwhiz" is
    # the default and behaves exactly as before. "research" runs every check but
    # downgrades FORMAT rules to skipped and BIOLOGY/PROVENANCE failures to
    # non-blocking review flags, and keeps content the strict passes delete.
    research_mode = is_research(export_mode)
    research_flags: List[Dict[str, Any]] = []
    research_skipped: List[Dict[str, Any]] = []

    def _contract(
        validator: Any,
        *args: Any,
        phase: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a stage contract under the active export mode and collect flags.

        A failure here aborts the whole post-pipeline step, so this is the last
        place the *mapped* payload still exists in a frame that knows which stage
        refused it. The snapshot is taken on the way past and the error re-raised
        unchanged, which keeps every existing handler's behaviour identical while
        making sure the run does not die with nothing on disk about the payload
        that died. Post-extraction is excluded: that boundary is diagnosed by
        stage_one_boundary before this function is ever reached, and recording it
        here would overwrite a mapping snapshot with a pre-mapping one.

        ``phase`` names the pipeline boundary the returned report describes, and
        every report leaves here carrying it alongside a schema version. The two
        post-normalization contracts are the reason: one runs before the audit
        rounds and one after, on two different payloads, and until they said so a
        consumer could not tell a superseded verdict from a live one. Defaulted
        from the report's own ``stage`` so the boundaries with only one contract
        each keep a truthful phase without every call site restating it.
        """

        try:
            report = run_stage_contract(validator, *args, mode=export_mode, **kwargs)
        except StageContractError as exc:
            if str(exc.stage or "") != "post_extraction":
                _record_mapped_failure_snapshot(
                    args[0] if args else None,
                    stage=str(exc.stage or getattr(validator, "__name__", "unknown")),
                    reason=str(exc),
                    report=exc.report,
                )
            raise
        if research_mode and isinstance(report, dict):
            research_flags.extend(review_flags(report))
            research_skipped.extend(format_gaps(report))
        if isinstance(report, dict):
            report = stamp_report(
                report,
                phase=phase
                or str(report.get("stage") or getattr(validator, "__name__", "") or "unknown"),
            )
        return report

    project_root = PROJECT_ROOT
    cache_path = Path(mapping_cache_path)
    if not cache_path.is_absolute():
        cache_path = project_root / mapping_cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    db_config = {
        "host": db_host,
        "port": db_port,
        "user": db_user,
        "password": db_password,
        "schema": db_schema,
    }

    temp_root = project_root / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    tmp = temp_root / f"post_pipeline_{uuid4().hex}"
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        input_json = tmp / "final.json"
        audit_report_path = tmp / "audit_report.json"
        audit_patch_path = tmp / "audit_patch.json"
        apply_report_path = tmp / "audit_apply_report.json"
        audited_json = tmp / "final.audited.json"
        stage2_mapped_json = tmp / "stage2.mapped.json"
        stage2_mapping_report_path = tmp / "stage2_mapping_report.json"
        mapped_json = tmp / "final.mapped.json"
        enriched_json = tmp / "final.enriched.json"
        mapping_report_path = tmp / "mapping_report.json"
        enrichment_report_path = tmp / "enrichment_report.json"
        sbml_path = tmp / "pathway.sbml"
        sbml_report_json_path = tmp / "sbml_validation_report.json"
        sbml_report_txt_path = tmp / "sbml_validation_report.txt"
        sbml_overwatch_path = tmp / "sbml_overwatch_report.json"
        gap_resolution_report_path = tmp / "gap_resolution_report.json"
        post_normalization_probe_path = tmp / "post_normalization_probe.json"
        post_transport_attachment_probe_path = tmp / "post_transport_attachment_probe.json"
        post_dedupe_probe_path = tmp / "post_dedupe_probe.json"
        gate_fail_report_path = tmp / "gate_fail_report.json"
        stoich_json_path = tmp / "final.stoich.json"
        stoich_audit_log_path = tmp / "stoich_audit_log.json"

        pre_mapping_input = deepcopy(final_payload)
        post_extraction_contract_report = _contract(validate_post_extraction, pre_mapping_input)
        stage2_result = map_payload(
            deepcopy(pre_mapping_input),
            cache_path=cache_path,
            id_source=id_source,
            db_config=db_config,
            use_cache=True,
            allow_complex_wrapper_creation=False,
            allow_structural_cleanup=False,
            mode=export_mode,
        )
        if not isinstance(stage2_result, dict):
            raise ValueError("Stage 2 mapping returned a malformed result: expected an object.")
        stage2_payload_value = stage2_result.get("payload")
        stage2_report_value = stage2_result.get("report")
        if not isinstance(stage2_payload_value, dict) or not isinstance(stage2_report_value, dict):
            raise ValueError(
                "Stage 2 mapping returned a malformed result: payload and report must be objects."
            )
        stage2_payload = stage2_payload_value
        stage2_mapping_report = stage2_report_value
        stage2_contract_report = _contract(validate_post_mapping, stage2_payload)
        stage2_runtime_schema_report = _safe_dict(
            stage2_contract_report.get("runtime_schema_report")
        )
        stage2_mapped_json.write_text(
            json.dumps(stage2_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        stage2_mapping_report_path.write_text(
            json.dumps(stage2_mapping_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # This is the exact Stage 2 output supplied to Stage 3. The normalizer
        # copies its input, so this object also remains the immutable Stage 2
        # artifact returned to the UI.
        pre_normalization_input = stage2_payload
        normalized_input = stage2_payload
        normalization_report: Dict[str, Any] = {
            "summary": {
                "complexes_created": 0,
                "composites_rewritten": 0,
                "reactions_rewritten": 0,
                "scaffold_split_reactions": 0,
                "entities_moved_out_of_compounds": 0,
                "entities_added_as_compounds": 0,
                "entities_added_as_proteins": 0,
                "catalysts_promoted_to_enzymes": 0,
                "scaffold_inputs_added": 0,
                "scaffold_in_modifiers_count": 0,
                "n_plus_tokens_remaining": 0,
                "complexes_list": [],
                "n_autostate_created": 0,
                "n_entities_assigned_to_autostate": 0,
                "transporters_attached": 0,
                "modifier_refs_canonicalized": 0,
                "modifier_refs_dropped": 0,
                "forbidden_complexes_removed": 0,
                "dedupe_removed_reactions": 0,
                "dedupe_removed_transports": 0,
                "dedupe_removed": 0,
                "dedupe_removed_total": 0,
                "no_op_removed_count": 0,
                "n_same_as_groups": 0,
                "n_aliases_rewritten": 0,
                "n_entities_deduped": 0,
                "n_single_protein_complexes_removed": 0,
                "alias_example_mappings": [],
            },
            "rewrite_map": {},
            "actions": [],
        }
        gate_fail_report: Dict[str, Any] = {}
        post_normalization_probe: Dict[str, Any] = {}
        post_transport_attachment_probe: Dict[str, Any] = {}
        post_dedupe_probe: Dict[str, Any] = {}
        gate_connectivity_summary: Dict[str, Any] = {}
        reaction_preservation_reports: Dict[str, Any] = {}
        post_normalization_contract_report: Dict[str, Any] = {}

        def _checkpoint_probe(
            checkpoint_payload: Dict[str, Any],
            checkpoint_report: Dict[str, Any],
            *,
            gate_snapshot: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            probe_report = deepcopy(checkpoint_report)
            stats = _safe_dict(_safe_dict(gate_snapshot or {}).get("normalization_stats"))
            if not stats:
                try:
                    stats = _safe_dict(compute_normalization_stats(checkpoint_payload, probe_report))
                except Exception:
                    stats = _safe_dict(probe_report.get("summary"))
            graph = _safe_dict(_safe_dict(gate_snapshot or {}).get("connectivity")) or graph_summary(checkpoint_payload)
            return {
                "normalization_stats": stats,
                "graph_summary": graph,
                "payload": deepcopy(checkpoint_payload),
            }

        def _write_normalization_checkpoint(
            name: str,
            checkpoint_payload: Dict[str, Any],
            checkpoint_report: Dict[str, Any],
        ) -> None:
            nonlocal post_normalization_probe, post_transport_attachment_probe, post_dedupe_probe
            if name == "cleanup_disallowed_complexes" and not post_normalization_probe:
                post_normalization_probe = _checkpoint_probe(checkpoint_payload, checkpoint_report)
                post_normalization_probe_path.write_text(
                    json.dumps(post_normalization_probe, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            elif name == "attach_transporters_from_evidence" and not post_transport_attachment_probe:
                post_transport_attachment_probe = _checkpoint_probe(checkpoint_payload, checkpoint_report)
                post_transport_attachment_probe_path.write_text(
                    json.dumps(post_transport_attachment_probe, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            elif name == "run_strict_post_normalization_gates" and not post_dedupe_probe:
                gate_snapshot = _safe_dict(checkpoint_report.get("gate"))
                post_dedupe_probe = _checkpoint_probe(
                    checkpoint_payload,
                    checkpoint_report,
                    gate_snapshot=gate_snapshot,
                )
                post_dedupe_probe_path.write_text(
                    json.dumps(post_dedupe_probe, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        normalization_completed = False
        try:
            normalized_input, normalization_report = normalize_process_payload(
                stage2_payload,
                on_checkpoint=_write_normalization_checkpoint,
                mode=export_mode,
            )
            normalization_completed = True
            gate_details = _safe_dict(normalization_report.get("gate"))
            post_normalization_contract_report = _contract(
                validate_post_normalization,
                normalized_input,
                gate_details,
                phase=PHASE_INITIAL_POST_NORMALIZATION,
            )
            gate_connectivity_summary = _safe_dict(gate_details.get("connectivity"))
            if gate_details and gate_details.get("ok") is False:
                # HISTORY, not a verdict. These errors are handed to the audit at
                # :func:`run_audit` below as the repair list -- the pipeline asks
                # for them to be fixed and then, before this change, filed the run
                # as failed on the very report it had just used as an input. The
                # phase stamp is what lets a consumer tell this apart from the
                # final report; nothing else may read it as an outcome.
                gate_fail_report = stamp_report(
                    {
                        "status": "failed",
                        "stage": "post_normalization_hard_gates",
                        "error": "Hard-gate validation failed after normalization.",
                        "errors": _safe_list(gate_details.get("errors")),
                        "normalization_stats": _safe_dict(gate_details.get("normalization_stats")),
                        "connectivity": gate_connectivity_summary,
                    },
                    phase=PHASE_INITIAL_POST_NORMALIZATION,
                )
                gate_fail_report_path.write_text(
                    json.dumps(gate_fail_report, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as exc:
            gate_fail_report = stamp_report(
                {
                    "status": "failed",
                    "stage": "post_normalization_hard_gates",
                    "error": str(exc),
                },
                phase=PHASE_INITIAL_POST_NORMALIZATION,
            )
            if not post_normalization_probe:
                post_normalization_probe = {
                    "normalization_stats": _safe_dict(normalization_report.get("summary")),
                    "graph_summary": graph_summary(normalized_input),
                    "payload": deepcopy(normalized_input),
                }
                post_normalization_probe_path.write_text(
                    json.dumps(post_normalization_probe, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            if not post_transport_attachment_probe:
                post_transport_attachment_probe = {
                    "normalization_stats": _safe_dict(normalization_report.get("summary")),
                    "graph_summary": graph_summary(normalized_input),
                    "payload": deepcopy(normalized_input),
                }
                post_transport_attachment_probe_path.write_text(
                    json.dumps(post_transport_attachment_probe, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            if not post_dedupe_probe:
                post_dedupe_probe = {
                    "normalization_stats": _safe_dict(normalization_report.get("summary")),
                    "graph_summary": graph_summary(normalized_input),
                    "payload": deepcopy(normalized_input),
                }
                post_dedupe_probe_path.write_text(
                    json.dumps(post_dedupe_probe, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        locked_reaction_quarantine_path = temp_root / "quarantined_locked_reactions.json"
        locked_reaction_quarantine_artifact: Dict[str, Any] = {}
        if normalization_completed:
            locked_reaction_quarantine_artifact = {
                "quarantined_locked_reactions": [
                    deepcopy(row)
                    for row in _safe_list(normalized_input.get("quarantined_locked_reactions"))
                    if isinstance(row, dict)
                ]
            }
            locked_reaction_quarantine_path.write_text(
                json.dumps(locked_reaction_quarantine_artifact, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        preservation_after_normalization = _write_reaction_preservation_report("after_normalization", normalized_input)
        if preservation_after_normalization is not None:
            reaction_preservation_reports["after_normalization"] = preservation_after_normalization
        input_json.write_text(json.dumps(normalized_input, indent=2, ensure_ascii=False), encoding="utf-8")

        audit_iterations: List[Dict[str, Any]] = []
        gap_iterations: List[Dict[str, Any]] = []
        current_input = input_json

        def _payload_file_hash(path: Path) -> str:
            canonical_payload = json.dumps(
                _read_json(path),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            return hashlib.sha1(canonical_payload.encode("utf-8")).hexdigest()

        def _actionable_gap_issue_count(report: Dict[str, Any]) -> int:
            stage3_report = _safe_dict(report.get("stage3"))
            issue_rows: List[Any] = []
            for post_resolution_key in (
                "post_resolution_issues",
                "remaining_issues",
                "unresolved_issues",
            ):
                if post_resolution_key in stage3_report:
                    issue_rows = _safe_list(stage3_report.get(post_resolution_key))
                    break
            else:
                # Gap Resolve currently reports the actionable issues it
                # attempted in `issues`. A following round recollects them
                # from the settled payload, so resolved rows disappear then.
                issue_rows = _safe_list(stage3_report.get("issues"))

            actionable_keys = {
                str(_safe_dict(issue).get("issue_key") or f"issue:{index}")
                for index, issue in enumerate(issue_rows)
                if isinstance(issue, dict)
            }
            actionable_execution_statuses = {
                "error",
                "failed",
                "skipped",
                "unavailable",
                "unmapped",
                "unresolved",
            }
            for index, execution in enumerate(_safe_list(stage3_report.get("executions"))):
                execution_row = _safe_dict(execution)
                status = str(execution_row.get("status") or "").strip().lower()
                if status not in actionable_execution_statuses:
                    continue
                actionable_keys.add(
                    str(execution_row.get("issue_key") or f"execution:{index}")
                )
            return len(actionable_keys)

        seen_hashes = {_payload_file_hash(current_input)}
        current_stage3_gate_errors = _safe_list(gate_fail_report.get("errors"))
        current_round_summary: Optional[str] = reaction_summary  # refreshed each round
        max_rounds = max(1, int(audit_max_rounds))
        timeout_seconds = max(30, int(audit_timeout_seconds))
        max_candidates = max(1, int(audit_candidate_count))
        retrieval_top_k = max(1, int(example_top_k))
        motif_index_data: Dict[str, Any] = {}
        motif_index_error = ""
        resolved_example_index_path = ""
        example_path = Path(example_index_path.strip() or "src/tmp/sbml_motif_index.json")
        if not example_path.is_absolute():
            example_path = project_root / example_path
        resolved_example_index_path = str(example_path)
        use_example_retrieval_effective = bool(use_example_retrieval)
        if example_path.exists():
            use_example_retrieval_effective = True
            try:
                motif_index_data = load_motif_index(example_path)
            except Exception as exc:  # noqa: BLE001
                motif_index_error = str(exc)
        elif use_example_retrieval:
            motif_index_error = f"index_not_found:{example_path}"
        audit_started_at = time.time()
        retry_context_note = ""
        if gate_fail_report:
            gate_errors = _safe_list(gate_fail_report.get("errors"))
            compact_gate_errors = [
                {
                    "path": _safe_dict(error).get("path"),
                    "reason": _safe_dict(error).get("reason"),
                }
                for error in gate_errors
                if isinstance(error, dict)
            ][:20]
            retry_context_note = (
                "Post-normalization gate failures are expected audit inputs. "
                "Repair these issues before export: "
                f"{json.dumps(compact_gate_errors, ensure_ascii=False)}"
            )
        stop_reason = "max_rounds_reached"

        for round_idx in range(1, max_rounds + 1):
            elapsed_before = time.time() - audit_started_at
            if elapsed_before > timeout_seconds:
                stop_reason = "timeout"
                break
            round_input_hash = _payload_file_hash(current_input)

            base_temperature = min(0.65, 0.15 * (round_idx - 1))
            base_max_tokens = min(8000, 3600 + 700 * (round_idx - 1))
            temp_offsets = [0.0, 0.14, 0.28, 0.40, 0.50]
            token_offsets = [0, 500, 900, 1300, 1700]
            candidate_count = 1 if not use_llm_audit else min(max_candidates, len(temp_offsets))
            remaining_seconds = timeout_seconds - elapsed_before
            if use_llm_audit and remaining_seconds < 120:
                candidate_count = 1
            round_candidates: List[Dict[str, Any]] = []
            timed_out_mid_round = False
            retrieval_context = ""
            retrieval_meta = {"selected_count": 0, "top_k": retrieval_top_k}
            if use_example_retrieval_effective and motif_index_data:
                round_payload_for_query = _read_json(current_input)
                query_text = payload_to_query_text(round_payload_for_query, extra=retry_context_note)
                retrieval_context, retrieval_meta = build_retrieval_context(
                    query_text,
                    motif_index_data,
                    top_k=retrieval_top_k,
                    max_chars=3800,
                )
            # S2 — append RAG gap-targeted evidence (if any) to the audit's
            # retrieval_context. Defaulted to "" so with RAG off this is a no-op
            # and retrieval_context stays exactly as built above.
            if rag_evidence_context:
                retrieval_context = "\n\n".join(
                    part for part in (retrieval_context, rag_evidence_context) if part
                )

            for cand_idx in range(candidate_count):
                if (time.time() - audit_started_at) > timeout_seconds:
                    timed_out_mid_round = True
                    break
                cand_temperature = min(0.9, base_temperature + temp_offsets[cand_idx])
                cand_max_tokens = min(10000, base_max_tokens + token_offsets[cand_idx])
                cand_suffix = f"round{round_idx}.cand{cand_idx + 1}"
                cand_audit_report = tmp / f"audit_report.{cand_suffix}.json"
                cand_audit_patch = tmp / f"audit_patch.{cand_suffix}.json"
                cand_apply_report = tmp / f"audit_apply_report.{cand_suffix}.json"
                cand_audited = tmp / f"final.audited.{cand_suffix}.json"
                cand_eval_report = tmp / f"audit_eval_report.{cand_suffix}.json"
                cand_eval_patch = tmp / f"audit_eval_patch.{cand_suffix}.json"

                run_audit(
                    current_input,
                    cand_audit_report,
                    cand_audit_patch,
                    use_llm=use_llm_audit,
                    llm_temperature=cand_temperature,
                    llm_max_tokens=cand_max_tokens,
                    context_note=retry_context_note,
                    retrieval_context=retrieval_context,
                )
                run_apply(
                    current_input,
                    cand_audit_patch,
                    cand_audited,
                    audit_report_path=cand_audit_report,
                    apply_report_path=cand_apply_report,
                )
                run_audit(
                    cand_audited,
                    cand_eval_report,
                    cand_eval_patch,
                    use_llm=False,
                    llm_temperature=0.0,
                    llm_max_tokens=1200,
                    context_note="deterministic post-apply scoring",
                )

                cand_audit = _read_json(cand_audit_report)
                cand_apply = _read_json(cand_apply_report)
                cand_eval = _read_json(cand_eval_report)
                cand_audit_summary = _safe_dict(cand_audit.get("summary"))
                cand_apply_summary = _safe_dict(cand_apply.get("summary"))
                cand_eval_summary = _safe_dict(cand_eval.get("summary"))
                score = _audit_objective_score(
                    eval_error_count=int(cand_eval_summary.get("error_count", 0)),
                    eval_warning_count=int(cand_eval_summary.get("warning_count", 0)),
                    source_error_count=int(cand_audit_summary.get("error_count", 0)),
                    rejected_count=int(cand_apply_summary.get("rejected_count", 0)),
                    accepted_count=committed_change_count(cand_apply),
                    source_patch_count=int(cand_audit_summary.get("patch_count", 0)),
                )

                round_candidates.append(
                    {
                        "index": cand_idx + 1,
                        "temperature": cand_temperature,
                        "max_tokens": cand_max_tokens,
                        "score": list(score),
                        "audit_error_count": int(cand_audit_summary.get("error_count", 0)),
                        "audit_warning_count": int(cand_audit_summary.get("warning_count", 0)),
                        "audit_patch_count": int(cand_audit_summary.get("patch_count", 0)),
                        "accepted_patch_count": committed_change_count(cand_apply),
                        "rejected_patch_count": int(cand_apply_summary.get("rejected_count", 0)),
                        "eval_error_count": int(cand_eval_summary.get("error_count", 0)),
                        "eval_warning_count": int(cand_eval_summary.get("warning_count", 0)),
                        "audit_report_path": str(cand_audit_report),
                        "audit_patch_path": str(cand_audit_patch),
                        "apply_report_path": str(cand_apply_report),
                        "audited_path": str(cand_audited),
                    }
                )

            if not round_candidates:
                fallback_suffix = f"round{round_idx}.fallback"
                cand_audit_report = tmp / f"audit_report.{fallback_suffix}.json"
                cand_audit_patch = tmp / f"audit_patch.{fallback_suffix}.json"
                cand_apply_report = tmp / f"audit_apply_report.{fallback_suffix}.json"
                cand_audited = tmp / f"final.audited.{fallback_suffix}.json"
                cand_eval_report = tmp / f"audit_eval_report.{fallback_suffix}.json"
                cand_eval_patch = tmp / f"audit_eval_patch.{fallback_suffix}.json"
                run_audit(
                    current_input,
                    cand_audit_report,
                    cand_audit_patch,
                    use_llm=False,
                    llm_temperature=0.0,
                    llm_max_tokens=1200,
                    context_note="fallback deterministic audit after timeout/empty candidate set",
                    retrieval_context="",
                )
                run_apply(
                    current_input,
                    cand_audit_patch,
                    cand_audited,
                    audit_report_path=cand_audit_report,
                    apply_report_path=cand_apply_report,
                )
                run_audit(
                    cand_audited,
                    cand_eval_report,
                    cand_eval_patch,
                    use_llm=False,
                    llm_temperature=0.0,
                    llm_max_tokens=1200,
                    context_note="deterministic fallback scoring",
                    retrieval_context="",
                )
                cand_audit = _read_json(cand_audit_report)
                cand_apply = _read_json(cand_apply_report)
                cand_eval = _read_json(cand_eval_report)
                cand_audit_summary = _safe_dict(cand_audit.get("summary"))
                cand_apply_summary = _safe_dict(cand_apply.get("summary"))
                cand_eval_summary = _safe_dict(cand_eval.get("summary"))
                score = _audit_objective_score(
                    eval_error_count=int(cand_eval_summary.get("error_count", 0)),
                    eval_warning_count=int(cand_eval_summary.get("warning_count", 0)),
                    source_error_count=int(cand_audit_summary.get("error_count", 0)),
                    rejected_count=int(cand_apply_summary.get("rejected_count", 0)),
                    accepted_count=committed_change_count(cand_apply),
                    source_patch_count=int(cand_audit_summary.get("patch_count", 0)),
                )
                round_candidates.append(
                    {
                        "index": 1,
                        "temperature": 0.0,
                        "max_tokens": 1200,
                        "score": list(score),
                        "audit_error_count": int(cand_audit_summary.get("error_count", 0)),
                        "audit_warning_count": int(cand_audit_summary.get("warning_count", 0)),
                        "audit_patch_count": int(cand_audit_summary.get("patch_count", 0)),
                        "accepted_patch_count": committed_change_count(cand_apply),
                        "rejected_patch_count": int(cand_apply_summary.get("rejected_count", 0)),
                        "eval_error_count": int(cand_eval_summary.get("error_count", 0)),
                        "eval_warning_count": int(cand_eval_summary.get("warning_count", 0)),
                        "audit_report_path": str(cand_audit_report),
                        "audit_patch_path": str(cand_audit_patch),
                        "apply_report_path": str(cand_apply_report),
                        "audited_path": str(cand_audited),
                    }
                )

            selected_candidate = min(round_candidates, key=lambda c: tuple(c.get("score", [])))
            selected_audit_report = Path(str(selected_candidate["audit_report_path"]))
            selected_audit_patch = Path(str(selected_candidate["audit_patch_path"]))
            selected_apply_report = Path(str(selected_candidate["apply_report_path"]))
            round_audited = Path(str(selected_candidate["audited_path"]))

            shutil.copyfile(selected_audit_report, audit_report_path)
            shutil.copyfile(selected_audit_patch, audit_patch_path)
            shutil.copyfile(selected_apply_report, apply_report_path)
            round_resolved = tmp / f"final.resolved.round{round_idx}.json"
            gap_report_round: Dict[str, Any] = {}
            if (time.time() - audit_started_at) > timeout_seconds:
                timed_out_mid_round = True

            if use_gap_resolver and not timed_out_mid_round:
                gap_temp = min(0.45, 0.05 + 0.08 * (round_idx - 1))
                gap_tokens = min(1400, 700 + 120 * (round_idx - 1))
                gap_report_round = run_gap_resolution(
                    round_audited,
                    round_resolved,
                    gap_resolution_report_path,
                    id_source=id_source,
                    db_config=db_config,
                    use_llm=use_llm_gap_resolver,
                    llm_temperature=gap_temp,
                    llm_max_tokens=gap_tokens,
                    max_items=max(10, int(gap_resolver_max_items)),
                    enable_id_resolution=True,
                    reaction_summary=current_round_summary,
                )
                current_after_round = round_resolved
            else:
                current_after_round = round_audited

            round_audit = _read_json(audit_report_path)
            round_apply = _read_json(apply_report_path)
            summary = _safe_dict(round_audit.get("summary"))
            llm_info = _safe_dict(round_audit.get("llm"))
            apply_summary = _safe_dict(round_apply.get("summary"))
            error_count = int(summary.get("error_count", 0))
            warning_count = int(summary.get("warning_count", 0))
            patch_count = int(summary.get("patch_count", 0))
            # What actually changed the payload. A candidate whose batch rolled
            # back edited nothing, and scoring it as if it had would rank an
            # inert round above one that made a real repair.
            accepted_count = committed_change_count(round_apply)
            rejected_count = int(apply_summary.get("rejected_count", 0))
            audit_payload_hash = _payload_file_hash(round_audited)
            settled_payload_hash = _payload_file_hash(current_after_round)
            audit_payload_changed = audit_payload_hash != round_input_hash
            gap_payload_changed = settled_payload_hash != audit_payload_hash
            settled_payload_changed = settled_payload_hash != round_input_hash
            fresh_gate_report: Dict[str, Any] = {}
            if settled_payload_changed:
                settled_payload = _read_json(current_after_round)
                try:
                    fresh_gate_details = run_strict_post_normalization_gates(
                        settled_payload,
                        enforce_all_proteins_connected=True,
                    )
                    fresh_gate_report = {"ok": True, **fresh_gate_details}
                except GateValidationError as exc:
                    fresh_gate_report = {"ok": False, **_safe_dict(exc.details)}
                fresh_gate_report = stamp_report(fresh_gate_report, phase=PHASE_AUDIT_ROUND)
                # ASSIGNED, not discarded. This contract describes the payload the
                # round settled on; throwing it away left
                # ``post_normalization_contract_report`` describing the pre-audit
                # payload for the rest of the run, so the audit's repairs were
                # invisible to every consumer downstream and the run was reported
                # against a payload that no longer existed. This report is still
                # not a verdict about what shipped -- the remap below moves the
                # payload again -- which is what the phase stamp says.
                post_normalization_contract_report = _contract(
                    validate_post_normalization,
                    settled_payload,
                    fresh_gate_report,
                    phase=PHASE_AUDIT_ROUND,
                )
                current_stage3_gate_errors = _safe_list(fresh_gate_report.get("errors"))
            top_errors = [
                str(_safe_dict(item).get("reason", "")).strip()
                for item in _safe_list(round_audit.get("errors"))
                if isinstance(item, dict) and str(item.get("reason", "")).strip()
            ][:3]

            gap_summary = _safe_dict(gap_report_round.get("summary")) if isinstance(gap_report_round, dict) else {}
            mapped_ids_added = int(gap_summary.get("mapped_ids_added", 0))
            locations_added = int(gap_summary.get("locations_added", 0))
            states_filled = int(gap_summary.get("location_states_filled", 0))
            actionable_gap_issue_count = _actionable_gap_issue_count(gap_report_round)

            repeated_payload = settled_payload_hash in seen_hashes
            seen_hashes.add(settled_payload_hash)
            fresh_gate_errors = _safe_list(fresh_gate_report.get("errors"))
            actionable_issues_remain = bool(
                error_count > 0
                or current_stage3_gate_errors
                or actionable_gap_issue_count > 0
            )

            audit_iterations.append(
                {
                    "round": round_idx,
                    "temperature": float(selected_candidate.get("temperature", base_temperature)),
                    "max_tokens": int(selected_candidate.get("max_tokens", base_max_tokens)),
                    "candidate_count": candidate_count,
                    "selected_candidate_index": int(selected_candidate.get("index", 1)),
                    "selected_score": list(selected_candidate.get("score", [])),
                    "retrieval_selected_count": int(retrieval_meta.get("selected_count", 0)),
                    "retrieval_top_k": int(retrieval_meta.get("top_k", retrieval_top_k)),
                    "error_count": error_count,
                    "warning_count": warning_count,
                    "patch_count": patch_count,
                    "accepted_patch_count": accepted_count,
                    "rejected_patch_count": rejected_count,
                    "llm_ok": bool(llm_info.get("ok", False)),
                    "llm_error": str(llm_info.get("error", "")),
                    "llm_repair_rationale": str(llm_info.get("repair_rationale", "")),
                    "top_errors": top_errors,
                    "gap_mapped_ids_added": mapped_ids_added,
                    "gap_locations_added": locations_added,
                    "gap_location_states_filled": states_filled,
                    "audit_payload_changed": audit_payload_changed,
                    "gap_payload_changed": gap_payload_changed,
                    "settled_payload_changed": settled_payload_changed,
                    "actionable_issues_remain": actionable_issues_remain,
                    "actionable_gap_issue_count": actionable_gap_issue_count,
                    "current_stage3_gate_error_count": len(current_stage3_gate_errors),
                    "payload_repeated": repeated_payload,
                    "post_patch_gate": fresh_gate_report,
                    "post_settlement_gate": fresh_gate_report,
                    "elapsed_seconds": round(time.time() - audit_started_at, 3),
                    "candidates": round_candidates,
                }
            )
            # Filed with the diagnostics recorder as well as the artifacts dict.
            # The artifacts dict only reaches disk if this whole function RETURNS;
            # a run that dies in round 4 of 5 -- on a timeout, a gate, a contract --
            # loses every round it completed. The recorder writes each round as it
            # ends, so the audit history survives the failure it was diagnosing.
            # Bounded on purpose: ``candidates`` and the full gate reports are the
            # bulky keys and are reduced to counts here.
            _record_audit_round(audit_iterations[-1])
            if use_gap_resolver:
                gap_iterations.append(
                    {
                        "round": round_idx,
                        "summary": gap_summary,
                        "db": _safe_dict(gap_report_round.get("db")),
                        "stage3": _safe_dict(gap_report_round.get("stage3")),
                        "payload_changed": gap_payload_changed,
                        "actionable_issue_count": actionable_gap_issue_count,
                    }
                )
                _record_gap_round(gap_iterations[-1])

            # Rebuild reaction summary from this round's settled output so the
            # next gap_resolver call receives an up-to-date picture of the
            # pathway rather than the stale pre-audit string.
            try:
                _round_payload = json.loads(current_after_round.read_text(encoding="utf-8"))
                _round_graph = build_draft_graph(_round_payload)
                _round_qa = generate_qa_report(_round_graph, _round_payload)
                current_round_summary = generate_reaction_summary(_round_graph, _round_qa)
            except Exception:
                pass  # keep previous summary if rebuild fails

            current_input = current_after_round
            if timed_out_mid_round:
                stop_reason = "timeout"
                break
            if (
                fresh_gate_report
                and bool(fresh_gate_report.get("ok", False))
                and not current_stage3_gate_errors
                and error_count == 0
                and actionable_gap_issue_count == 0
            ):
                stop_reason = "clean_post_settlement_gate"
                break
            if not actionable_issues_remain:
                stop_reason = "clean_no_actionable_issues"
                break
            if not settled_payload_changed:
                stop_reason = "stalled_unchanged_payload_with_actionable_issues"
                break
            if repeated_payload:
                stop_reason = "loop_detected_repeated_payload"
                break

            compact_fresh_gate_errors = [
                {
                    "path": _safe_dict(error).get("path"),
                    "reason": _safe_dict(error).get("reason"),
                }
                for error in fresh_gate_errors
                if isinstance(error, dict)
            ][:20]
            retry_context_note = (
                f"Previous attempt unresolved: errors={error_count}, warnings={warning_count}, "
                f"accepted_patches={accepted_count}, gap_payload_changed={gap_payload_changed}. "
                "Prioritize remaining issues: "
                f"{'; '.join(top_errors) if top_errors else 'generic consistency fixes'}. "
                "Fresh post-settlement Stage 3 gate result: "
                f"{json.dumps(compact_fresh_gate_errors, ensure_ascii=False)}"
            )
        else:
            stop_reason = "max_rounds_reached"

        audited_json.write_text(current_input.read_text(encoding="utf-8"), encoding="utf-8")
        post_audit_contract_report = _contract(validate_post_audit, _read_json(audited_json))
        loop_duration = round(time.time() - audit_started_at, 3)
        try:
            preservation_after_audit = _write_reaction_preservation_report(
                "after_audit",
                json.loads(audited_json.read_text(encoding="utf-8")),
            )
            if preservation_after_audit is not None:
                reaction_preservation_reports["after_audit"] = preservation_after_audit
        except Exception:
            pass

        # Rebuild draft graph from the fully-audited payload so the PNG shown
        # in the UI and the reaction summary both reflect the corrected state.
        post_audit_draft_graph_dict: Dict[str, Any] = {}
        post_audit_qa_report: Dict[str, Any] = {}
        post_audit_reaction_summary: str = current_round_summary or ""
        post_audit_png_bytes: bytes = b""
        try:
            _audited_payload = json.loads(audited_json.read_text(encoding="utf-8"))
            _audited_graph = build_draft_graph(_audited_payload)
            post_audit_qa_report = generate_qa_report(_audited_graph, _audited_payload)
            post_audit_reaction_summary = generate_reaction_summary(_audited_graph, post_audit_qa_report)
            post_audit_draft_graph_dict = _audited_graph.to_dict()
        except Exception:
            pass

        # ── Curator step: name fixes, compartment fills, transporter proposals ──
        curator_json = tmp / "final.curated.json"
        curator_report_path = tmp / "curator_report.json"
        curator_report: Dict[str, Any] = {}
        try:
            _meta = _safe_dict(json.loads(audited_json.read_text(encoding="utf-8")).get("metadata", {}))
            curator_report = run_pathway_curator(
                audited_json,
                curator_json,
                curator_report_path,
                reaction_summary=post_audit_reaction_summary or current_round_summary,
                pathway_name=str(_meta.get("pathway_name", "") or ""),
                organism=str(_meta.get("organism", "") or ""),
                llm_temperature=0.2,
                llm_max_tokens=2000,
            )
            # Use curated output as input to mapping only if patches were accepted
            if int(_safe_dict(curator_report.get("summary", {})).get("patches_accepted", 0)) > 0:
                audited_json.write_text(curator_json.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as _cur_exc:
            curator_report = {"error": str(_cur_exc), "summary": {}}

        mapping_result = map_payload(
            json.loads(audited_json.read_text(encoding="utf-8")),
            cache_path=cache_path,
            id_source=id_source,
            db_config=db_config,
            use_cache=False,
            # The synthetic protein_complex wrapper exists only because the
            # PathWhiz importer refuses a bare protein as a reaction enzyme.
            # Research mode does not import, so the wrapping is dropped entirely
            # and enzymes stay plain protein actors. The structural cleanup that
            # exists to keep those wrappers valid goes with it.
            allow_complex_wrapper_creation=not research_mode,
            allow_structural_cleanup=not research_mode,
            mode=export_mode,
        )
        if not isinstance(mapping_result, dict):
            raise ValueError("Stage 6 mapping returned a malformed result: expected an object.")
        mapped_payload_value = mapping_result.get("payload")
        mapping_report_value = mapping_result.get("report")
        if not isinstance(mapped_payload_value, dict) or not isinstance(mapping_report_value, dict):
            raise ValueError(
                "Stage 6 mapping returned a malformed result: payload and report must be objects."
            )
        mapped_payload = mapped_payload_value
        mapping_report = mapping_report_value
        post_remap_contract_report = _contract(validate_post_remap, mapped_payload)
        mapped_json.write_text(json.dumps(mapped_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        mapping_report_path.write_text(json.dumps(mapping_report, indent=2, ensure_ascii=False), encoding="utf-8")
        stoich_audit_log: list = []
        if use_stoich_agent:
            from t2pw.stoich.agent import run_stoich_agent
            mapped_payload, stoich_audit_log = run_stoich_agent(mapped_payload)
            stoich_json_path.write_text(json.dumps(mapped_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            stoich_audit_log_path.write_text(json.dumps(stoich_audit_log, indent=2, ensure_ascii=False), encoding="utf-8")
            mapped_json.write_text(json.dumps(mapped_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        enrichment_cache_path = cache_path.with_name("enrichment_cache.json")
        enrichment_dump_path = project_root / "out" / "enrichment_dump.json"
        enrichment_report: Dict[str, Any] = {}
        sbml_input_path = mapped_json
        try:
            enrichment_report = run_enrichment(
                mapped_json,
                enriched_json,
                enrichment_report_path,
                cache_path=enrichment_cache_path,
                dump_path=enrichment_dump_path,
                qa_report=qa_report,
            )
            sbml_input_path = enriched_json
        except Exception as exc:
            enrichment_report = {
                "summary": {"enrichment_failed": True},
                "error": str(exc),
            }
            sbml_input_path = mapped_json
        final_export_payload = json.loads(sbml_input_path.read_text(encoding="utf-8"))
        post_enrichment_runtime_schema_report = validate_runtime_payload_contract(
            final_export_payload,
            boundary="post_enrichment",
        )
        pre_export_runtime_schema_report = validate_runtime_payload_contract(
            final_export_payload,
            boundary="pre_export",
        )
        preservation_before_final_export = _write_reaction_preservation_report(
            "before_final_export",
            final_export_payload,
        )
        if preservation_before_final_export is not None:
            reaction_preservation_reports["before_final_export"] = preservation_before_final_export

        # ── THE canonical payload ───────────────────────────────────────────
        #
        # Until this block existed there were three candidate "final" payloads
        # and no two consumers agreed on which one they meant:
        #
        #   final_mapped_db          post-remap, PRE-enrichment  -> what the batch
        #                            driver serialized as final_mapped.json
        #   final_export_payload     post-enrichment             -> what SBML was
        #                            built from
        #   final_mapped_quarantined post-quarantine, derived    -> what the
        #                            from final_mapped_db           pre-export
        #                                                           Stage-3 gate
        #                                                           validated
        #
        # So the gate ran on an object that was neither serialized nor exported,
        # and a passing report could not be made authoritative without asserting
        # a pass on something that did not ship. ONE object now: enriched, then
        # quarantined. It is gated here, serialized to final_mapped.json, and
        # handed to the SBML build below -- which is why quarantine moved up
        # from the UI block that used to run it after this function returned.
        canonical_export_payload: Dict[str, Any] = {}
        canonical_payload_hash = ""
        #: Which object the SBML build and ``final_mapped.json`` were made from.
        #: Recorded rather than inferred, because "the exporter consumed the same
        #: payload the gate validated" is exactly the claim that used to be false
        #: and exactly the one a test has to be able to check.
        sbml_input_source = "enriched_pre_quarantine"
        final_stage3_gate_report: Dict[str, Any] = {}
        quarantine_result: Dict[str, Any] = {}
        quarantine_ok = False
        if final_export_payload:
            quarantine_result = run_quarantine_boundary(
                final_export_payload,
                strict_db=bool(strict_db),
                outputs_dir=Path(outputs_dir) if outputs_dir else (project_root / "outputs"),
                pathway_context=pathway_context,
                mode=export_mode,
            )
            quarantine_ok = bool(quarantine_result.get("ok"))
        if quarantine_ok:
            # deepcopy, and the hash taken from the copy that is stored: the
            # report's payload binding is only worth anything if the object the
            # consumer re-hashes is the object the gate saw. Refinement review
            # and the draft-graph builder both take the payload afterwards, and a
            # single in-place mutation would otherwise turn every leg into a hash
            # mismatch -- which, under a "hash differs -> FAIL" rule, fails the
            # whole suite for a bookkeeping reason.
            canonical_export_payload = deepcopy(_safe_dict(quarantine_result.get("payload")))
            canonical_payload_hash = payload_sha256(canonical_export_payload)
            try:
                _final_gate_details = run_strict_post_normalization_gates(
                    deepcopy(canonical_export_payload),
                    # The SAME suite the initial gate ran, on the same terms. This
                    # call used to take the default (False) while the gate whose
                    # failure it is entitled to supersede ran with
                    # ``not is_research(mode)``, so a "pass" here was a pass of a
                    # weaker suite and could not honestly countermand anything.
                    enforce_all_proteins_connected=not research_mode,
                )
                final_stage3_gate_report = {"ok": True, **_final_gate_details}
            except GateValidationError as _final_gate_exc:
                final_stage3_gate_report = {"ok": False, **_safe_dict(_final_gate_exc.details)}
            # Written on the PASSING branch as well as the failing ones. A gate
            # that only ever records failure cannot report success, and the
            # consumer is then left reading a stale failure as the live verdict.
            final_stage3_gate_report = stamp_report(
                {
                    "stage": "final_pre_export_stage3_gates",
                    **final_stage3_gate_report,
                },
                phase=PHASE_FINAL_PRE_EXPORT,
                payload_hash=canonical_payload_hash,
            )
            canonical_json = tmp / "final.canonical.json"
            canonical_json.write_text(
                json.dumps(canonical_export_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # The SBML build below reads this path. Before the reordering it read
            # the enriched, pre-quarantine file -- so the model that was built and
            # the payload that was gated were different objects.
            sbml_input_path = canonical_json
            sbml_input_source = CANONICAL_PAYLOAD_KEY
        # A quarantine refusal leaves no canonical payload, so the final gate does
        # not run and no report is written. That is deliberate: the alternative is
        # to synthesise a verdict for a payload that was never assembled. The
        # artifact set is still stamped, so the consumer fails closed on the
        # missing report rather than reading the refusal as a pass.

        sbml_overwatch_report: Dict[str, Any] = {}
        sbml_diagram_png_bytes = b""
        sbml_diagram_error = ""
        sbml_render_layout_summary: Dict[str, Any] = {}
        sbml_render_ready_sbml_bytes = b""
        sbml_clean_bytes = b""
        sbml_clean_summary: Dict[str, Any] = {}
        sbml_build_report: Dict[str, Any] = {}
        if build_legacy_sbml:
            sbml_build_report = build_sbml(
                sbml_input_path,
                sbml_path,
                sbml_report_json_path,
                sbml_report_txt_path,
                default_compartment_name=default_compartment,
                db_config=db_config,
            )
            if use_sbml_overwatch:
                sbml_overwatch_report = run_sbml_overwatch(
                    sbml_input_path,
                    sbml_path,
                    sbml_report_json_path,
                    sbml_overwatch_path,
                    use_llm=True,
                    llm_max_tokens=3000,
                )
            try:
                render_artifacts = build_render_artifacts(str(sbml_path))
                sbml_diagram_png_bytes = render_artifacts.get("png_bytes", b"")
                sbml_render_layout_summary = _safe_dict(render_artifacts.get("layout_summary"))
                sbml_render_ready_sbml_bytes = render_artifacts.get("render_ready_sbml_bytes", b"")
            except Exception as exc:  # noqa: BLE001
                sbml_diagram_error = str(exc)

            if sbml_render_ready_sbml_bytes:
                try:
                    sbml_clean_bytes, sbml_clean_summary = strip_unmapped(sbml_render_ready_sbml_bytes)
                except Exception:  # noqa: BLE001
                    pass

        return stamp_artifact_set({
            "gate_failed": False,
            # Both of these are HISTORY as of the report lifecycle: they describe
            # the gate run whose failures the audit was then asked to repair, and
            # a consumer that ORs them into the blocking verdict can never let a
            # repair succeed. Retained in full -- ``gate_fail_report.json`` is the
            # audit trail -- and republished under an explicitly historical key so
            # that intent is in the name and not only in a comment.
            "normalization_gate_failed": bool(gate_fail_report),
            "gate_fail_report": gate_fail_report,
            INITIAL_GATE_REPORT_KEY: gate_fail_report,
            # THE verdict: the strict suite re-run on the object below, which is
            # the object serialized to final_mapped.json and built into SBML.
            FINAL_GATE_REPORT_KEY: final_stage3_gate_report,
            CANONICAL_PAYLOAD_KEY: canonical_export_payload,
            "canonical_payload_sha256": canonical_payload_hash,
            "sbml_input_source": sbml_input_source,
            "quarantine_report": _safe_dict(quarantine_result.get("quarantine_report")),
            "quarantine_artifacts": _safe_dict(quarantine_result.get("quarantine_artifacts")),
            "quarantine_coverage": _safe_dict(quarantine_result.get("coverage")),
            "quarantine_ok": quarantine_ok,
            "quarantine_refused": bool(quarantine_result) and not quarantine_ok,
            "quarantine_refusal_reasons": _safe_list(quarantine_result.get("refusal_reasons")),
            "quarantine_review_flags": _safe_list(quarantine_result.get("review_flags")),
            "final_mapped_quarantined": canonical_export_payload,
            "export_mode": coerce_mode(export_mode),
            # Everything research mode downgraded, so the UI can show it loudly.
            # Empty in strict mode, where these findings aborted the run instead.
            "research_review_flags": research_flags,
            "research_skipped_format_rules": research_skipped,
            "research_normalization_actions": [
                action
                for action in _safe_list(normalization_report.get("actions"))
                if isinstance(action, dict)
                and str(action.get("type", "")).startswith("research_mode_")
            ],
            "pre_mapping_input": pre_mapping_input,
            "extraction_inference_payload": pre_mapping_input,
            "post_extraction_contract_report": post_extraction_contract_report,
            "post_extraction_runtime_schema_report": _safe_dict(
                post_extraction_contract_report.get("runtime_schema_report")
            ),
            "stage2_payload": stage2_payload,
            "stage2_mapping_report": stage2_mapping_report,
            "stage2_contract_report": stage2_contract_report,
            "stage2_runtime_schema_report": stage2_runtime_schema_report,
            "pre_normalization_input": pre_normalization_input,
            "pre_normalized_input": normalized_input,
            "post_normalization_payload": normalized_input,
            "pre_normalization_report": normalization_report,
            "post_normalization_contract_report": post_normalization_contract_report,
            "post_normalization_probe": post_normalization_probe,
            "post_transport_attachment_probe": post_transport_attachment_probe,
            "post_dedupe_probe": post_dedupe_probe,
            "connectivity_summary": gate_connectivity_summary or _safe_dict(post_dedupe_probe.get("graph_summary")),
            "audit_report": json.loads(audit_report_path.read_text(encoding="utf-8")),
            "audit_patch": json.loads(audit_patch_path.read_text(encoding="utf-8")),
            "audit_apply_report": json.loads(apply_report_path.read_text(encoding="utf-8")),
            "final_audited": json.loads(audited_json.read_text(encoding="utf-8")),
            "post_audit_contract_report": post_audit_contract_report,
            # ``final_mapped`` IS the canonical payload when there is one, so the
            # key every downstream reader already reaches for names the object
            # that was gated and exported rather than an intermediate. It falls
            # back to the enriched pre-quarantine payload only when quarantine
            # refused, where the diagnostic value of the last complete payload
            # outweighs a key that would otherwise be empty.
            "final_mapped": canonical_export_payload or final_export_payload,
            "final_mapped_db": json.loads(mapped_json.read_text(encoding="utf-8")),
            "final_mapped_enriched": final_export_payload,
            "final_export_input": canonical_export_payload or final_export_payload,
            "mapping_report": mapping_report,
            "post_remap_contract_report": post_remap_contract_report,
            "enrichment_report": enrichment_report,
            "post_enrichment_runtime_schema_report": post_enrichment_runtime_schema_report,
            "pre_export_runtime_schema_report": pre_export_runtime_schema_report,
            "sbml_report_json": json.loads(sbml_report_json_path.read_text(encoding="utf-8"))
            if sbml_report_json_path.exists()
            else {},
            "sbml_report_txt": sbml_report_txt_path.read_text(encoding="utf-8")
            if sbml_report_txt_path.exists()
            else "",
            "sbml_overwatch_report": sbml_overwatch_report,
            "sbml_xml_bytes": sbml_path.read_bytes() if sbml_path.exists() else b"",
            "sbml_diagram_png_bytes": sbml_diagram_png_bytes,
            "sbml_diagram_error": sbml_diagram_error,
            "sbml_render_layout_summary": sbml_render_layout_summary,
            "sbml_render_ready_sbml_bytes": sbml_render_ready_sbml_bytes,
            "sbml_clean_bytes": sbml_clean_bytes,
            "sbml_clean_summary": sbml_clean_summary,
            "sbml_build_report": sbml_build_report,
            "mapping_cache_path": str(cache_path),
            "saved_pathway_sbml_path": _save_pipeline_outputs(
                project_root,
                sbml_path.read_bytes() if sbml_path.exists() else b"",
                sbml_render_ready_sbml_bytes,
                sbml_clean_bytes,
            )
            if build_legacy_sbml
            else "",
            "enrichment_cache_path": str(enrichment_cache_path),
            "enrichment_dump_path": str(enrichment_dump_path),
            "mapping_id_source": id_source,
            "mapping_db_host": db_host,
            "mapping_db_schema": db_schema,
            "example_retrieval_enabled": bool(use_example_retrieval_effective),
            "example_retrieval_requested": bool(use_example_retrieval),
            "example_index_path": resolved_example_index_path,
            "example_index_error": motif_index_error,
            "example_index_entry_count": int(motif_index_data.get("entry_count", 0)) if motif_index_data else 0,
            "post_audit_draft_graph": post_audit_draft_graph_dict,
            "post_audit_qa_report": post_audit_qa_report,
            "post_audit_reaction_summary": post_audit_reaction_summary,
            "post_audit_png_bytes": post_audit_png_bytes,
            "curator_report": curator_report,
            "stoich_audit_log": stoich_audit_log,
            "audit_iterations": audit_iterations,
            "gap_resolution_iterations": gap_iterations,
            "reaction_preservation_reports": reaction_preservation_reports,
            "locked_reaction_quarantine_artifact": locked_reaction_quarantine_artifact,
            "locked_reaction_quarantine_path": str(locked_reaction_quarantine_path),
            "audit_loop_summary": {
                "rounds_executed": len(audit_iterations),
                "max_rounds": max_rounds,
                "timeout_seconds": timeout_seconds,
                "stop_reason": stop_reason,
                "duration_seconds": loop_duration,
            },
        })
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _validate_stage8_export_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Revalidate the Stage 6 payload without repairing or otherwise changing it."""

    try:
        gate_details = run_strict_post_normalization_gates(
            payload,
            enforce_all_proteins_connected=True,
        )
        stage3_gate_report = {"ok": True, **gate_details}
    except GateValidationError as exc:
        stage3_gate_report = {"ok": False, **_safe_dict(exc.details)}

    stage3_contract_report = validate_post_normalization(payload, stage3_gate_report)
    return stage3_gate_report, stage3_contract_report


def run_pwml_export(
    final_payload: Dict[str, Any],
    pathway_name: str,
    pathway_description: str,
    pathway_subject: str,
    project_root: Path,
    ref_path: Path,
    vis_width: int = 3200,
    vis_height: int = 1400,
    background_color: str = "#FFFFFF",
    grounding_dict: Optional[Dict[str, Any]] = None,
    strict_db: bool = True,
    quarantine_report: Optional[Dict[str, Any]] = None,
    quarantine_artifacts: Optional[Dict[str, str]] = None,
    pathway_context: Optional[Dict[str, Any]] = None,
    export_mode: ExportMode = DEFAULT_EXPORT_MODE,
) -> Dict[str, Any]:
    """Export a payload that has ALREADY passed the quarantine boundary.

    ``quarantine_report`` / ``quarantine_artifacts`` are the decisions that
    boundary made, carried in so they ride on every result this function returns.
    This function does not quarantine: :func:`run_quarantine_boundary` runs once,
    upstream of the post-mapping Stage 3 check, and re-running it here would judge
    a *different* payload and overwrite the artifacts describing the graph the
    reviewer approved.

    A caller that reaches the exporter without a boundary decision -- a direct
    call, a script -- runs the boundary itself rather than exporting unquarantined
    material.
    """

    before_export_report: Optional[Dict[str, Any]] = None
    # Bound before the try so the catch-all handler at the bottom can carry them
    # too. An exception raised before the boundary block would otherwise produce
    # the one result that drops the decision, and a caller cannot tell "no
    # quarantine ran" from "quarantine ran and the report was lost".
    quarantine_report = _safe_dict(quarantine_report)
    quarantine_artifacts = dict(quarantine_artifacts or {})
    quarantine_reused = False
    try:
        before_export_report = _write_reaction_preservation_report("before_final_export", final_payload)
        payload = deepcopy(final_payload)
        grounding_report: Dict[str, Any] = {}
        if grounding_dict:
            payload, grounding_report = apply_grounding(payload, grounding_dict)

        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Is the stored decision about THIS payload, under THESE rules? Refinement
        # can have deleted the requested core since the boundary ran, and
        # `apply_grounding` above rewrites identifiers -- either way the report
        # describes a graph that no longer exists, and reusing it would export
        # material nothing admitted. Reuse is therefore gated on the payload hash
        # AND the decision-input hash, not on the report merely existing.
        decision_controls = {
            "mode": export_mode,
            "strict_db": bool(strict_db),
            "pathway_context": pathway_context,
        }
        quarantine_reused = decision_matches(quarantine_report, payload, **decision_controls)
        if not quarantine_reused and quarantine_report and not decision_inputs_match(
            quarantine_report, **decision_controls
        ):
            # Controls moved, not the payload. This is the one mismatch that must
            # NOT be absorbed by re-running: `payload` here is whatever the boundary
            # left behind, and under a strict boundary that is an already reduced
            # graph. Re-judging it in research mode would report a research decision
            # over material a strict pass had already removed -- annotating a graph
            # the reviewer never saw and calling nothing quarantined. The controls
            # are locked for the run; changing one requires a new run.
            changed = sorted(
                key
                for key, value in {
                    "export_mode": coerce_mode(export_mode),
                    "strict_db": bool(strict_db),
                }.items()
                if _safe_dict(quarantine_report.get("decision_inputs")).get(key) != value
            ) or ["requested_pathway_context"]
            return {
                "ok": False,
                "error": (
                    "PWML export stopped: quarantine decision controls changed since "
                    "the boundary ran (" + ", ".join(changed) + "). The stored decision "
                    "was taken under different rules and cannot authorize this export. "
                    "Re-run the pipeline to take a fresh decision."
                ),
                "counts": {},
                "issues": len(changed),
                "refusal_reasons": [f"quarantine_decision_controls_changed:{name}" for name in changed],
                "output_path": "",
                "qa": {},
                "grounding_report": grounding_report,
                "quarantine_report": quarantine_report,
                "quarantine_artifacts": quarantine_artifacts,
                "quarantine_reused": False,
                "reaction_preservation_before_final_export": before_export_report,
            }
        if not quarantine_reused:
            quarantine_result = quarantine_and_close(
                payload,
                strict_db=bool(strict_db),
                pathway_context=pathway_context,
                mode=export_mode,
            )
            quarantine_artifacts = write_quarantine_artifacts(quarantine_result, outputs_dir)
            quarantine_report = quarantine_result.quarantine_report
            if not quarantine_result.ok:
                return {
                    "ok": False,
                    "error": (
                        "PWML export stopped by the quarantine boundary: "
                        + "; ".join(quarantine_result.refusal_reasons)
                    ),
                    "counts": {},
                    "issues": len(quarantine_result.refusal_reasons),
                    "refusal_reasons": list(quarantine_result.refusal_reasons),
                    "output_path": "",
                    "qa": {},
                    "grounding_report": grounding_report,
                    "quarantine_report": quarantine_report,
                    "quarantine_artifacts": quarantine_artifacts,
                    "quarantine_reused": False,
                    "coverage_summary": quarantine_result.coverage,
                    "reaction_preservation_before_final_export": before_export_report,
                }
            payload = quarantine_result.payload

        stage3_gate_report, stage3_contract_report = _validate_stage8_export_payload(payload)
        if not bool(stage3_contract_report.get("ok", False)):
            stage3_gate_path = outputs_dir / "pwml_stage3_gate_report.json"
            stage3_gate_path.write_text(
                json.dumps(
                    {
                        "stage": "pre_export_stage3_revalidation",
                        "mode": "validation_only",
                        "ok": False,
                        "gate_report": stage3_gate_report,
                        "contract_report": stage3_contract_report,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return {
                "ok": False,
                "error": (
                    "PWML export stopped by validation-only pre-export Stage 3 "
                    "revalidation; Stage 8 did not repair the payload."
                ),
                "counts": {},
                "issues": int(_safe_dict(stage3_gate_report.get("summary")).get("error_count", 0))
                or len(_safe_list(stage3_gate_report.get("errors"))),
                "output_path": "",
                "qa": {},
                "grounding_report": grounding_report,
                "stage3_gate_report": stage3_gate_report,
                "stage3_contract_report": stage3_contract_report,
                "stage3_gate_report_path": str(stage3_gate_path),
                "quarantine_report": quarantine_report,
                "quarantine_artifacts": quarantine_artifacts,
                "quarantine_reused": quarantine_reused,
                "reaction_preservation_before_final_export": before_export_report,
            }

        required_gate_path = outputs_dir / "pwml_required_field_gate_report.json"
        gate_payload = deepcopy(payload)
        metadata = gate_payload.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.setdefault("pathway_name", pathway_name)
            metadata.setdefault("name", pathway_name)
            metadata.setdefault("pathway_subject", pathway_subject)
            metadata.setdefault("subject", pathway_subject)
            metadata.setdefault("description", pathway_description)
            metadata.setdefault("width", int(vis_width))
            metadata.setdefault("height", int(vis_height))
        try:
            pre_export_contract = validate_pre_export(gate_payload, strict_db=bool(strict_db))
            required_gate_report = _safe_dict(pre_export_contract.get("pwml_contract_report"))
        except StageContractError as exc:
            pre_export_contract = _safe_dict(exc.report)
            required_gate_report = _safe_dict(pre_export_contract.get("pwml_contract_report"))
            if not required_gate_report:
                required_gate_report = {
                    "ok": False,
                    "errors": _safe_list(pre_export_contract.get("errors")),
                    "warnings": _safe_list(pre_export_contract.get("warnings")),
                    "summary": _safe_dict(pre_export_contract.get("summary")),
                }
        required_gate_report["stage"] = "required_field_gate"
        required_gate_report["pipeline_order"] = [
            "audit_normalize",
            "db_hydration",
            "api_enrichment",
            "llm_gap_resolver_with_tools",
            "required_field_gate",
            "pwml_ir_build",
            "pwml_writer",
            "pwml_qa",
        ]
        required_gate_path.write_text(
            json.dumps(required_gate_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not required_gate_report.get("ok", False):
            return {
                "ok": False,
                "error": "PWML required-field gate failed.",
                "counts": {},
                "issues": int(_safe_dict(required_gate_report.get("summary")).get("error_count", 0)),
                "output_path": "",
                "qa": {},
                "grounding_report": grounding_report,
                "stage3_gate_report": stage3_gate_report,
                "stage3_contract_report": stage3_contract_report,
                "required_gate_report": required_gate_report,
                "required_gate_report_path": str(required_gate_path),
                "quarantine_report": quarantine_report,
                "quarantine_artifacts": quarantine_artifacts,
                "quarantine_reused": quarantine_reused,
                "reaction_preservation_before_final_export": before_export_report,
            }

        pwml_ir, ir_report = build_pwml_ir(
            payload,
            pathway_name=pathway_name,
            pathway_subject=pathway_subject,
            strict_db=bool(strict_db),
            width=int(vis_width),
            height=int(vis_height),
        )
        pwml_ir.setdefault("pathway", {})["description"] = pathway_description
        ir_validation = validate_pwml_ir(pwml_ir)
        blocking_ir_errors = blocking_pwml_ir_errors(ir_report)
        if blocking_ir_errors or ir_validation.get("errors"):
            return {
                "ok": False,
                "error": "PWML IR validation failed.",
                "counts": ir_report.get("counts", {}),
                "issues": len(ir_validation.get("errors", [])),
                "output_path": "",
                "qa": {},
                "grounding_report": grounding_report,
                "stage3_gate_report": stage3_gate_report,
                "stage3_contract_report": stage3_contract_report,
                "required_gate_report": required_gate_report,
                "required_gate_report_path": str(required_gate_path),
                "pwml_ir": pwml_ir,
                "pwml_ir_report": ir_report,
                "pwml_ir_validation": ir_validation,
                "quarantine_report": quarantine_report,
                "quarantine_artifacts": quarantine_artifacts,
                "quarantine_reused": quarantine_reused,
                "reaction_preservation_before_final_export": before_export_report,
            }
        signature = discover_structure_signature(ref_path)
        args = SimpleNamespace(
            name=pathway_name,
            description=pathway_description,
            subject=pathway_subject,
            pw_id="PW000000",
            height=vis_height,
            width=vis_width,
            background_color=background_color,
            ref=str(ref_path),
        )
        builder = DeterministicPwmlBuilder(extraction=pwml_ir, signature=signature, args=args)
        build_result = builder.build()
        tree = etree.ElementTree(build_result.root)
        repaired = repair_tree(tree, signature)
        report = validate_generated_tree(repaired, signature)
        xml_bytes = etree.tostring(
            repaired.getroot(), encoding="utf-8", xml_declaration=True, pretty_print=True
        )
        qa_report = run_pwml_qa(xml_bytes)
        out_path = project_root / "outputs" / "pathway.pwml"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(xml_bytes)
        after_export_report = _write_reaction_preservation_report("after_final_export", payload)
        return {
            "ok": True,
            "counts": build_result.counts,
            "issues": report["issue_count"],
            "output_path": str(out_path),
            "xml_bytes": xml_bytes,
            "validation_report": report,
            "qa": qa_report,
            "grounding_report": grounding_report,
            "stage3_gate_report": stage3_gate_report,
            "stage3_contract_report": stage3_contract_report,
            "required_gate_report": required_gate_report,
            "required_gate_report_path": str(required_gate_path),
            "pwml_ir": pwml_ir,
            "pwml_ir_report": ir_report,
            "pwml_ir_validation": ir_validation,
            "quarantine_report": quarantine_report,
            "quarantine_artifacts": quarantine_artifacts,
            "quarantine_reused": quarantine_reused,
            "reaction_preservation_before_final_export": before_export_report,
            "reaction_preservation_after_final_export": after_export_report,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "counts": {}, "issues": 0,
                "output_path": "", "qa": {}, "grounding_report": {},
                "quarantine_report": quarantine_report,
                "quarantine_artifacts": quarantine_artifacts,
                "quarantine_reused": quarantine_reused,
                "reaction_preservation_before_final_export": before_export_report}


# ── Export mode (OUTSIDE form — governs the whole run AND the later PWML
# export section, so it must be readable from both) ────────────────────────
EXPORT_MODE_LABELS = {
    "Strict PWML": PATHWHIZ,
    "Research (relaxed)": RESEARCH,
}
_export_mode_label = st.radio(
    "Export mode",
    list(EXPORT_MODE_LABELS),
    horizontal=True,
    key="export_mode_radio",
    help=(
        "Strict PWML (default): every check blocks, exactly as before — this is "
        "the mode to use for anything destined for PathBank. "
        "Research (relaxed): PathWhiz *formatting* rules are skipped (no required "
        "external DB identity, no protein_complex wrapper around enzymes, ':' in a "
        "name is fine), and every biology/provenance check still runs but only "
        "FLAGS instead of aborting. Research mode always produces a candidate "
        "pathway with a citation report; it does not produce an importable PWML file."
    ),
)
# .get with a strict default: under a mocked//headless streamlit the widget
# returns a sentinel rather than a label, and an unrecognised value must always
# degrade to the safe mode rather than silently relaxing every gate.
export_mode: ExportMode = EXPORT_MODE_LABELS.get(_export_mode_label, PATHWHIZ)
st.session_state["export_mode"] = export_mode
if is_research(export_mode):
    st.warning(
        "**Research mode — relaxed.** PathWhiz formatting rules are skipped and "
        "biology/provenance problems are reported as review flags instead of "
        "stopping the run. The output is a *candidate* pathway for human review, "
        "not an importable PWML file. Read the Review flags panel before trusting it."
    )

# ── Input mode (OUTSIDE form — triggers immediate re-render on click) ──────
input_mode = st.radio(
    "Input mode",
    ["Paste text", "Upload PDF", "Text + PDF"],
    horizontal=True,
    key="input_mode_radio",
)

# ── PDF controls (OUTSIDE form — file_uploader is banned inside st.form) ───
uses_text_input = input_mode in {"Paste text", "Text + PDF"}
uses_pdf_input = input_mode in {"Upload PDF", "Text + PDF"}
uploaded_pdfs  = []
pdf_page_range = ""
pdf_skip_refs  = True
pdf_ocr        = False

text_entry_count = 1
if uses_text_input:
    text_entry_count = st.number_input(
        "Number of text entries",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="text_entry_count",
    )

if uses_pdf_input:
    uploaded_pdfs = st.file_uploader(
        "Upload scientific PDFs (research papers, pathway descriptions, etc.)",
        type=["pdf"],
        key="pdf_uploads_widget",
        accept_multiple_files=True,
    )
    _c1, _c2, _c3 = st.columns(3)
    pdf_page_range = _c1.text_input(
        "Page range (e.g. 1-20, blank = all)",
        value="",
        key="pdf_page_range",
        help="Leave blank to extract all pages.",
    )
    pdf_skip_refs = _c2.checkbox(
        "Skip References / Acknowledgements",
        value=True,
        key="pdf_skip_refs",
    )
    pdf_ocr = _c3.checkbox(
        "Enable OCR fallback (scanned PDFs)",
        value=False,
        key="pdf_ocr",
        help="Requires tesseract + pytesseract installed.",
    )

# ── Form — only the text area changes; everything else is UNCHANGED ─────────
with st.form("pwml_pipeline"):
    text_entries = []
    if uses_text_input:
        for _idx in range(int(text_entry_count)):
            _label = "Paste pathway description:" if int(text_entry_count) == 1 else f"Paste pathway description {_idx + 1}:"
            text_entries.append(st.text_area(_label, height=220, key=f"pathway_text_{_idx}"))

    user_task_context = st.text_area(
        "Optional extraction focus / task context",
        height=100,
        help="Use this to tell the model what pathway or scope you want extracted. This guides extraction but does not override the source text or validation rules.",
    )

    if uses_pdf_input:
        if uploaded_pdfs:
            _pdf_names = ", ".join(_pdf.name for _pdf in uploaded_pdfs)
            st.info(
                f"PDFs ready: **{_pdf_names}**  "
                f"- configure options above, then click **Run pipeline**."
            )
        else:
            if input_mode == "Upload PDF":
                st.warning("Upload one or more PDFs using the file uploader above, then click **Run pipeline**.")
            else:
                st.caption("Optional: upload one or more PDFs using the file uploader above.")

        
    run_inference = st.checkbox(
        "Run inference/enrichment stage",
        value=True,
        help="Stage 1 always runs. Disable when you only want strict extraction.",
    )

    enable_chunking = st.checkbox(
        "Enable automatic chunking for long inputs",
        value=True,
        help="When enabled, Stage 1 splits long inputs into overlapping chunks before extraction.",
    )

    # ── RAG flag (seam S5) — rendered only when RAG is enabled, so the form is
    # byte-identical to today with RAG_ENABLED off. ──────────────────────────
    if rag_config()["enabled"]:
        rag_incomplete_flag = st.toggle(
            "Multi-paper RAG (novel pathway from multiple papers)",
            value=True,
            help=(
                "ON (default): always run multi-paper RAG — fetches and selects "
                "related papers, retrieves gap-targeted evidence, and synthesizes "
                "one connected, provenance-tagged pathway. "
                "OFF: auto-decide from the scope/gap guardrails — RAG still runs "
                "when the seed pathway is ambiguous (low scope clarity) or "
                "structurally incomplete (dangling reactions / unmapped enzymes / "
                "missing precursors), otherwise it falls through to the standard "
                "single-paper pipeline. "
                "Set RAG_ENABLED=false to disable multi-paper RAG entirely."
            ),
        )
    else:
        rag_incomplete_flag = False

    col_a, col_b, col_c, col_d = st.columns(4)
    extract_attempts = col_a.number_input(
        "Stage 1 auto-repair attempts",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
    )
    infer_attempts = col_b.number_input(
        "Stage 2 auto-repair attempts",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
    )
    temperature = col_c.slider(
        "LLM temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Applied to both stages.",
    )
    infer_rounds = col_d.number_input(
        "Stage 2 QA rounds",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        help="Round 1 is normal inference. Additional rounds include graph QA feedback hints (disconnected entities, missing links).",
    )

    col_tokens_1, col_tokens_2 = st.columns(2)
    extract_tokens = col_tokens_1.number_input(
        "Stage 1 max tokens",
        min_value=500,
        max_value=128000,
        value=24000,
        step=100,
    )
    infer_tokens = col_tokens_2.number_input(
        "Stage 2 max tokens",
        min_value=500,
        max_value=128000,
        value=20000,
        step=100,
    )

    chunk_cols = st.columns(2)
    chunk_size = chunk_cols[0].number_input(
        "Chunk size (approx. words)",
        min_value=200,
        max_value=60000,
        value=10000,
        step=100,
    )
    chunk_overlap = chunk_cols[1].number_input(
        "Chunk overlap (words)",
        min_value=0,
        max_value=20000,
        value=1600,
        step=100,
    )
    st.caption("Runtime scales with: chunks x Stage 2 QA rounds x retry attempts.")

    submit = st.form_submit_button("Run pipeline")

if submit:
    text_parts = [entry.strip() for entry in text_entries if entry.strip()]
    user_task_context = (user_task_context or "").strip() or None

    # PDF extraction runs here — outside the form, so uploaded_pdf is accessible
    if uses_pdf_input:
        if not uploaded_pdfs:
            if input_mode == "Upload PDF":
                st.warning("Please upload one or more PDFs using the file uploader above.")
                st.stop()
            uploaded_pdfs = []

        import tempfile
        import os

        # Parse page range once and apply it to every uploaded PDF.
        _ps, _pe = 1, None
        _pr = (pdf_page_range or "").strip()
        if _pr:
            _parts = _pr.split("-")
            try:
                _ps = int(_parts[0])
                _pe = int(_parts[1]) if len(_parts) > 1 else _ps
            except ValueError:
                st.warning(f"Invalid page range '{_pr}'; extracting all pages.")

        _skip = SKIP_SECTIONS if pdf_skip_refs else set()

        for uploaded_pdf in uploaded_pdfs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as _tmp:
                _tmp.write(uploaded_pdf.read())
                _tmp_path = _tmp.name

            try:
                with st.spinner(f"Extracting text from {uploaded_pdf.name}..."):
                    _pdf = parse_pdf(
                        _tmp_path,
                        page_start=_ps,
                        page_end=_pe,
                        skip_sections=_skip,
                        enable_ocr_fallback=bool(pdf_ocr),
                    )
            finally:
                try:
                    os.unlink(_tmp_path)
                except Exception:
                    pass

            if _pdf["error"]:
                st.error(f"PDF extraction failed for {uploaded_pdf.name}: {_pdf['error']}")
                st.stop()

            if _pdf["text"].strip():
                text_parts.append(_pdf["text"].strip())

            for _w in _pdf.get("warnings", []):
                st.warning(f"{uploaded_pdf.name}: {_w}")

            st.success(
                f"{uploaded_pdf.name}: extracted **{_pdf['pages_used']}** of **{_pdf['total_pages']}** pages "
                f"via **{_pdf['method']}**. "
                f"Sections: {', '.join(_pdf['sections'].keys()) or 'none'}"
            )
            _meta = _pdf.get("metadata", {})
            _mp = [p for p in [
                f"Title: {_meta['title']}"     if _meta.get("title")  else "",
                f"Author(s): {_meta['author']}" if _meta.get("author") else "",
                f"DOI: {_meta['doi']}"          if _meta.get("doi")    else "",
            ] if p]
            if _mp:
                st.caption(" | ".join(_mp))

            with st.expander(f"Preview extracted text from {uploaded_pdf.name} (first 1000 chars)", expanded=False):
                _preview = _pdf["text"][:1000] + ("..." if len(_pdf["text"]) > 1000 else "")
                st.text(_preview)

    text = "\n\n".join(text_parts)
    if not text.strip():
        st.warning("No text to process. Paste text, upload one or more PDFs, or use both.")
        st.stop()

    reset_refinement_state()

    # Opened BEFORE Stage 0, because Stage 0 is one of the boundaries being
    # diagnosed. Everything recorded from here on is flushed to disk as it is
    # recorded, so a run that dies inside Stage 0 has already written
    # stage0_attempts.json by the time the error is rendered.
    _diagnostics_dir = _start_extraction_diagnostics(text)

    # Preprocessing: lightweight context summary to guide extraction and inference
    with st.spinner("Running preprocessor..."):
        # Stage 0 plus its at-most-one bounded-head retry. The retry's guards
        # (skip a deliberate Case C; adopt the second draw only on a strict
        # improvement) live inside the helper because the inline version ran the
        # retry BEFORE the Case-C test below and let it overwrite the context
        # unconditionally — see _run_stage_zero_with_retry for the run that
        # exposed it.
        pathway_context = _run_stage_zero_with_retry(
            text, temperature=temperature, user_task_context=user_task_context
        )
        # A deliberate Case C ambiguous multi-example review also has blank
        # pathway fields, but that is the prompt's guardrail, not a flaky call:
        # it is deterministic and re-running never helps. The branch below
        # raises its own error, so stay quiet here.
        if not _has_usable_context(pathway_context) and not is_ambiguous_multi_example_review_context(
            pathway_context
        ):
            # Stage 0 came back empty. Before disabling RAG, fall back to the scope
            # the user typed in the extraction-focus box — a flaky preprocessor call
            # should never nuke a scope we already have. Show exactly what was
            # inferred so a mis-parse is visible rather than silent.
            seeded_context = _seed_context_from_user_scope(pathway_context, user_task_context)
            if seeded_context is not None:
                pathway_context = seeded_context
                _seed_org = str(seeded_context.get("likely_organism") or "").strip()
                st.info(
                    "Stage 0 returned no usable context, so the pipeline is falling "
                    "back to the scope you provided — pathway "
                    f"\"{seeded_context.get('pathway_name', '')}\""
                    + (f", organism \"{_seed_org}\"" if _seed_org else "")
                    + ". Extraction and multi-paper RAG will use it. Re-run if you'd "
                    "rather the preprocessor try to read the scope from the paper again."
                )
            else:
                st.warning(
                    f"Stage 0 failed: {describe_preprocess_status(pathway_context)}\n\n"
                    "It returned no usable context — no pathway name, "
                    "organism, compounds or proteins, and no scope was provided in the "
                    "extraction-focus box to fall back to. Extraction will be unguided, "
                    "and multi-paper RAG cannot build a literature query, so no papers "
                    "will be fetched. Re-run (usually a transient LLM failure), or type "
                    "the pathway/organism into the focus box so the pipeline can proceed "
                    "without the preprocessor."
                )
        elif preprocess_was_recovered(pathway_context):
            # Not an error: the context is real. But it came from a reply that
            # was cut off mid-JSON, so fields the model never wrote are absent.
            st.warning(
                f"Stage 0 partially recovered: {describe_preprocess_status(pathway_context)}\n\n"
                "The reply was cut off and repaired, so some context fields "
                "(e.g. candidate examples, key compounds) may be missing. Re-run "
                "if the context below looks incomplete."
            )

    _sync_extraction_diagnostics()

    if is_ambiguous_multi_example_review_context(pathway_context):
        candidate_examples = pathway_context.get("candidate_examples", [])
        st.session_state["pipeline_ready"] = False
        st.session_state["pathway_context"] = pathway_context
        st.session_state["pipeline_error"] = {
            "status": "ambiguous_review_scope",
            "message": (
                "multi_example_review detected with no selected_example. "
                "Extraction skipped to prevent mixed-pathway output."
            ),
            "candidate_examples": candidate_examples,
        }
        st.error(
            "Ambiguous review scope: this looks like a multi-example review, but no target example was selected. "
            "Name the example you want in the \"Optional extraction focus / task context\" box above "
            "(one of the candidate example names below works), then re-run. "
            "Re-running without naming a target will produce this same result."
        )
        if candidate_examples:
            st.subheader("Candidate examples")
            st.json(candidate_examples)
        warning = pathway_context.get("warning")
        if isinstance(warning, str) and warning.strip():
            st.warning(warning.strip())
        st.stop()

    # Stage 1: strict extraction with auto-repair
    try:
        with st.spinner("Running Stage 1 extraction..."):
            stage_one, chunk_details = run_stage_one_with_chunking(
                text,
                pathway_context=pathway_context,
                user_task_context=user_task_context,
                artifact_dir=PROJECT_ROOT / "tmp",
                enable_chunking=enable_chunking,
                chunk_word_limit=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                max_attempts=int(extract_attempts),
                temperature=temperature,
                max_tokens=int(extract_tokens),
            )
            # The boundary is settled rather than merely checked: a registry row
            # the extraction referenced but never declared is reconstructed
            # deterministically, and rows the contract rejects get a bounded,
            # localized repair -- neither of which regenerates the pathway. See
            # t2pw.pipeline.stage_one_boundary. When it cannot be settled the
            # payload still comes back and the run reports an INCOMPLETE outcome
            # instead of a pass; nothing is invented to satisfy the contract.
            _boundary = settle_stage_one(
                stage_one,
                mode=coerce_mode(st.session_state.get("export_mode")),
                cleaning_report=_last_cleaning_report(),
            )
            stage_one = _boundary.payload
            st.session_state["stage_one_boundary"] = _boundary.to_summary()
    except PipelineFailure as failure:
        _sync_extraction_diagnostics()
        st.error(f"Extraction failed: {failure}")
        render_attempts("Stage 1 attempts", failure.attempts)
        st.stop()
    except StageContractError as failure:
        _sync_extraction_diagnostics()
        st.error(f"Extraction boundary failed: {failure}")
        st.json(failure.report)
        st.stop()

    _sync_extraction_diagnostics()
    if not _boundary.ok:
        # Explicitly incomplete, not a silent pass and not a fabricated pathway.
        # The artifacts written above are the deliverable for this run.
        st.error(
            "Extraction boundary failed and could not be recovered: "
            f"{_boundary.incomplete_reason}"
        )
        st.info(
            "Localized repair and deterministic registry reconstruction were both "
            "applied; the payload below is as far as the run got. No reaction was "
            "invented to satisfy the contract. The diagnostics for this run "
            f"(stage0_attempts.json, extraction_boundary_report.json, "
            f"cleaning_report.json) are in {_diagnostics_dir}."
        )
        st.json(_boundary.to_summary())
        if _boundary.failure is not None:
            st.json(_boundary.failure.report)
        st.stop()
    if _boundary.reconstruction:
        st.info(
            f"Reconstructed {_boundary.reconstruction.get('shell_count', 0)} unresolved "
            "entity shell(s) for participants named by reactions but missing from the "
            "registry. They carry no identifier and no function; every identity gate "
            "downstream still applies to them."
        )

    # Stage 1 wrote its raw output (incl. out-of-scope reactions) to disk and the
    # locked-reaction manifest above, before this filter runs; the manifest never
    # locked out-of-scope reactions in the first place. Drop them here so Stage 2
    # does not spend effort linking modifiers to reactions that are about to be
    # removed, and so out-of-scope material never reaches export.
    stage_one_in_scope, out_of_scope_removed_reactions = filter_out_of_scope_reactions(stage_one)
    if out_of_scope_removed_reactions:
        st.info(
            f"Removed {len(out_of_scope_removed_reactions)} out-of-scope reaction(s) "
            f"before Stage 2: {', '.join(out_of_scope_removed_reactions)}"
        )

    # ── RAG orchestration (seam S5) — WIRING ONLY; identical path when off ───
    # Guarded on the deploy switch (RAG_ENABLED) ONLY. With RAG_ENABLED off,
    # nothing here runs and user_task_context / the flow stay exactly as today —
    # the standard single-paper pipeline, byte-for-byte. With RAG_ENABLED on, the
    # decision of WHETHER to run the chain is delegated to t2pw.rag triage inside
    # maybe_run_rag (the separation invariant): the UI toggle is passed through as
    # user_flag, so an ON toggle force-runs RAG while an OFF toggle lets
    # should_run_rag auto-decide from the scope/gap guardrails. Only when triage
    # says run does maybe_run_rag run the whole R1–R5 chain and return its
    # outputs; the app then folds the retrieved evidence through the existing S1
    # (user_task_context) and S2 (the audit's retrieval_context, via
    # session_state) params, and hands the synthesized Payload to the
    # post-pipeline path (S3) below.
    rag_result = None
    if rag_config()["enabled"]:
        _rag_seed_graph = build_draft_graph(stage_one_in_scope)
        rag_result = maybe_run_rag(
            pathway_context=pathway_context,
            user_flag=bool(rag_incomplete_flag),
            seed_payload=stage_one_in_scope,
            reports={"qa_graph": generate_qa_report(_rag_seed_graph, stage_one_in_scope)},
            seed_text=text,
        )
        if rag_result is not None:
            st.session_state["rag_result"] = rag_result
            if rag_result.evidence_context:
                user_task_context = "\n\n".join(
                    part
                    for part in (user_task_context, rag_result.evidence_context)
                    if part
                )  # S1
                st.session_state["rag_evidence_context"] = rag_result.evidence_context  # S2

    final_payload = stage_one_in_scope
    stage_two = None
    stage_two_chunks: List[Dict[str, Any]] = []
    stage_two_rounds: List[Dict[str, Any]] = []
    qa_hints = None

    # Stage 2: inference/enrichment + auto-repair
    if run_inference:
        try:
            with st.spinner("Running Stage 2 inference..."):
                stage_two, stage_two_chunks, stage_two_rounds = run_stage_two_with_feedback_loop(
                    text,
                    stage_one_in_scope,
                    chunk_details=chunk_details,
                    pathway_context=pathway_context,
                    user_task_context=user_task_context,
                    qa_rounds=int(infer_rounds),
                    enable_chunking=enable_chunking,
                    chunk_word_limit=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                    max_attempts=int(infer_attempts),
                    temperature=temperature,
                    max_tokens=int(infer_tokens),
                )
        except PipelineFailure as failure:
            st.error(f"Inference stage failed: {failure}")
            label = f"{failure.stage} attempts" if getattr(failure, "stage", None) else "Stage 2 attempts"
            render_attempts(label, failure.attempts)
            st.stop()

        qa_hints = stage_two.get("qa_hints", {}) if isinstance(stage_two, dict) else {}
        final_payload = merge_additions(stage_one_in_scope, stage_two if isinstance(stage_two, dict) else {})

    # S3 — inject R5's synthesized reactions/entities INTO the seed payload rather
    # than REPLACING it. A no-op with RAG off (rag_result is None): final_payload
    # is untouched.
    #
    # RAG's ``to_payload`` emits a thin, reaction-only payload MISSING the rich
    # Stage-1 shape the post-pipeline assumes (no element_locations, no per-reaction
    # compartment, no per-entity mapping_meta, evidence as a LIST, participants as
    # dicts). Replacing the seed with it crashed the downstream audit/DB-mapping
    # with ``'NoneType' object is not subscriptable``. Instead, conform the RAG
    # payload to a Stage-2 additions envelope and MERGE it into the seed with the
    # SAME machinery Stage 2 uses (merge_additions) — so the payload entering the
    # post-pipeline keeps the seed's rich Stage-1 shape (compartments defaulted by
    # the existing default_compartment="cell" path) with RAG reactions slotted in.
    if rag_result is not None and rag_result.synthesized and rag_result.payload:
        from t2pw.rag.conform import conform_rag_additions_for_merge

        seed_reaction_count = _count_reactions(final_payload)
        # The second argument is the MERGE BASE, and it must be `final_payload` --
        # the very object handed to merge_additions on the next line -- because
        # without it the guard inside conform has nothing to compare against and the
        # whole fix is inert.
        #
        # WHY the guard exists. merge_additions dedupes entity rows with
        # _extend_unique, whose signature is json.dumps(row, sort_keys=True). RAG
        # synthesis rebuilds an entity row for every participant of every reaction it
        # resolved, and it resolves the SEED's reactions too (the seed paper is itself
        # indexed, so its own claims can corroborate), so the base's
        # {"name": "LpxA", "class": "protein", "confidence": 1.0, ...} and RAG's
        # {"name": "LpxA", "rag_provenance": {"source_id": "seed_paper", ...}, ...}
        # are two different signatures for one protein and BOTH survive the merge.
        # Measured on runs/2026-07-28_0919 PMC12444477/strict/merged_payload.json:
        # entities.proteins 31 rows for 22 distinct normalized names (all nine seed
        # enzymes doubled), entities.compounds 56 rows for 43. Byte-identical names,
        # so the 2026-07-23 synonym resolver -- which collapses SYNONYMS -- correctly
        # never touched them.
        #
        # WHY `final_payload` and not `stage_one_in_scope`. The base is Stage 1 PLUS
        # Stage 2 whenever inference ran. In that same run 'lipid IV_A precursor',
        # 'Kdo-lipid A precursor' and 'tetra-acylated disaccharide intermediate' are
        # Stage-1 reaction participants that Stage 1 never registered and Stage 2 did;
        # comparing against the Stage-1 seed alone would let those three back in.
        # Passing the object we are about to merge into makes "already registered"
        # mean exactly what merge_additions is about to see.
        #
        # Cost to the research deliverable: none. build_citation_report reads the
        # PRE-MERGE rag_result.payload (research_report.py:332), which conform never
        # mutates, so the dropped rows' source_refs / rag_provenance are still there
        # for the citation report to quote -- only the duplicate ROW in the merged
        # payload goes away. Reconstructed over all 8 delivered merged_payload.json
        # under runs/ that have a RAG side: 279 entity rows -> 220, zero reactions
        # lost, zero new process_normalizer.validate_registry_references errors.
        rag_envelope = conform_rag_additions_for_merge(rag_result.payload, final_payload)
        merged = merge_additions(final_payload, rag_envelope)
        # Preserve the locked-reaction quarantine the old replace-path performed:
        # merge_additions runs apply_post_merge_cleanup WITHOUT the manifest, so
        # run the manifest-aware cleanup on the merged result — a locked Stage-1
        # reaction mangled by synthesis is quarantined for inspection (and the
        # quarantine file written) rather than silently dropped.
        merged, rag_removed_reactions = apply_post_merge_cleanup(
            merged,
            locked_manifest=load_locked_reaction_manifest(
                PROJECT_ROOT / "tmp" / "locked_reaction_manifest.json"
            ),
            quarantine_output_path=PROJECT_ROOT / "tmp" / "quarantined_rag_reactions.json",
        )
        added_reactions = _count_reactions(merged) - seed_reaction_count
        if added_reactions > 0:
            # Differential schema gate at the RAG boundary. The wider pipeline runs
            # this runtime schema in REPORT mode, so the single-paper seed can
            # legitimately carry shapes the strict enforce-mode schema flags (e.g.
            # Stage-2 inference interactions). Holding the merged payload to a
            # stricter bar than the seed it merges into false-positives on those
            # pre-existing shapes. Instead, reject only when the merge INTRODUCES a
            # NEW violation not already present in the seed. Snapshot the seed's
            # error keys BEFORE reassigning final_payload = merged.
            new_error_keys = _post_extraction_new_error_keys(final_payload, merged)
            if new_error_keys:
                # The merge added a genuinely new schema violation — surface it and
                # keep the single-paper extraction rather than pass it downstream.
                try:
                    validate_runtime_payload_contract(
                        merged, boundary="post_extraction", mode="enforce"
                    )
                except StageContractError as failure:
                    st.error(
                        "Multi-paper RAG merge produced a payload that failed the "
                        "post-extraction runtime schema; keeping the single-paper "
                        "extraction rather than passing a malformed payload downstream."
                    )
                    st.json(failure.report)
            else:
                final_payload = merged
                merge_message = (
                    f"Merged {added_reactions} RAG reaction(s) into the single-paper "
                    "extraction."
                )
                if rag_removed_reactions:
                    merge_message += (
                        f" {len(rag_removed_reactions)} synthesized reaction(s) were "
                        f"dropped/quarantined as unusable: "
                        f"{', '.join(rag_removed_reactions)}."
                    )
                st.info(merge_message)
        else:
            st.warning(
                "Multi-paper RAG produced no additional usable reactions (no evidence "
                "retrieved — check that the embeddings endpoint is running and "
                "papers were fetched). Keeping the single-paper extraction."
                + (
                    f" {len(rag_removed_reactions)} synthesized reaction(s) were "
                    f"dropped as unusable: {', '.join(rag_removed_reactions)}."
                    if rag_removed_reactions
                    else ""
                )
            )

    stage2_preservation_report = _write_reaction_preservation_report("after_stage2", final_payload)

    draft_graph, qa_report, reaction_summary = build_and_save_draft_graph(final_payload)
    st.session_state["draft_graph"] = draft_graph.to_dict()
    st.session_state.pop("draft_graph_png_bytes", None)
    st.session_state.pop("draft_graph_render_error", None)
    st.session_state["qa_report"] = qa_report
    st.session_state["reaction_summary"] = reaction_summary

    st.session_state["pipeline_ready"] = True
    st.session_state["run_inference_enabled"] = bool(run_inference)
    st.session_state["pathway_context"] = pathway_context
    st.session_state["user_task_context"] = user_task_context
    st.session_state["stage_one"] = stage_one
    st.session_state["out_of_scope_removed_reactions"] = out_of_scope_removed_reactions
    st.session_state["chunk_details"] = chunk_details
    st.session_state["stage_two"] = stage_two
    st.session_state["stage_two_chunks"] = stage_two_chunks
    st.session_state["stage_two_rounds"] = stage_two_rounds
    st.session_state["qa_hints"] = qa_hints
    st.session_state["final_payload"] = final_payload
    st.session_state["final_payload_snapshot"] = final_payload
    st.session_state["pipeline_source_text"] = text
    st.session_state["reaction_preservation_after_stage2"] = stage2_preservation_report
    st.session_state.pop("post_pipeline_artifacts", None)
    # A new pipeline run voids every quarantine decision from the previous one,
    # in session state AND on disk. Otherwise a second run that dies before the
    # boundary would render the first run's coverage summary for a completely
    # different pathway, and a caller reading the session would find a report
    # authorizing a payload that is no longer loaded.
    reset_quarantine_state(PROJECT_ROOT / "outputs")
    st.session_state["token_stats"] = llm_client_module.get_token_stats()

if st.session_state.get("pipeline_ready"):
    run_inference_from_state = bool(st.session_state.get("run_inference_enabled", False))
    stage_one = st.session_state.get("stage_one", {})
    chunk_details = st.session_state.get("chunk_details", [])
    stage_two = st.session_state.get("stage_two")
    stage_two_chunks = st.session_state.get("stage_two_chunks", [])
    stage_two_rounds = st.session_state.get("stage_two_rounds", [])
    qa_hints = st.session_state.get("qa_hints")
    final_payload = st.session_state.get("final_payload", {})

    _ts = st.session_state.get("token_stats")
    if _ts:
        with st.expander("Token usage & estimated cost", expanded=False):
            _tc1, _tc2, _tc3, _tc4, _tc5 = st.columns(5)
            _tc1.metric("Prompt tokens", f"{_ts['prompt_tokens']:,}")
            _tc2.metric("Completion tokens", f"{_ts['completion_tokens']:,}")
            _tc3.metric("Total tokens", f"{_ts['total_tokens']:,}")
            _tc4.metric("API calls", _ts["api_calls"])
            _tc5.metric("Est. cost (USD)", f"${_ts['estimated_cost_usd']:.4f}")
            st.caption(
                f"Prices: ${llm_client_module._COST_INPUT_PER_M}/1M input, "
                f"${llm_client_module._COST_OUTPUT_PER_M}/1M output. "
                "Override with LLM_COST_INPUT_PER_M / LLM_COST_OUTPUT_PER_M in .env"
            )

    # RAG panels (seam S5) — only when RAG is on and a run produced a result.
    if rag_config()["enabled"] and st.session_state.get("rag_result") is not None:
        render_rag_panels(st.session_state.get("rag_result"))

    st.subheader("Stage 1 - Strict extraction")
    st.caption(f"Stage 1 QA: {qa_summary_line(stage_one)}")
    with st.expander("Stage 1 JSON", expanded=False):
        st.json(stage_one)
        st.download_button(
            "Download Stage 1 JSON",
            json.dumps(stage_one, indent=2),
            file_name="stage1_extract.json",
            mime="application/json",
        )

    out_of_scope_removed_reactions = st.session_state.get("out_of_scope_removed_reactions", [])
    if out_of_scope_removed_reactions:
        st.info(
            f"Removed {len(out_of_scope_removed_reactions)} out-of-scope reaction(s) "
            f"before Stage 2: {', '.join(out_of_scope_removed_reactions)}"
        )

    chunk_count = len(chunk_details)
    if chunk_count > 1:
        st.info(f"Chunked input into {chunk_count} slices (~{int(chunk_size)} words, overlap {int(chunk_overlap)}).")
    for chunk in chunk_details:
        chunk_label = f"Chunk {chunk['chunk_id']} - words {chunk['start_word']}-{chunk['end_word']}"
        with st.expander(chunk_label, expanded=False):
            preview = chunk["text"][:400]
            if len(chunk["text"]) > 400:
                preview += "..."
            st.caption(preview)
            st.markdown("**Chunk output JSON**")
            st.json(chunk["output"])
        render_attempts(f"{chunk_label} attempts", chunk["attempts"])

    if run_inference_from_state and isinstance(stage_two, dict):
        st.subheader("Stage 2 - Inference / enrichment")
        with st.expander("Stage 2 JSON", expanded=False):
            st.json(stage_two)
            if stage_two_rounds:
                st.write("Stage 2 QA rounds", stage_two_rounds)
            st.download_button(
                "Download Stage 2 additions",
                json.dumps(stage_two, indent=2),
                file_name="stage2_additions.json",
                mime="application/json",
            )
            if qa_hints:
                st.write("QA hints", qa_hints)
        chunk_count = len(stage_two_chunks)
        if chunk_count > 1:
            st.info(
                f"Chunked inference into {chunk_count} slices (~{int(chunk_size)} words, overlap {int(chunk_overlap)})."
            )
        for chunk in stage_two_chunks:
            chunk_label = (
                f"Round {chunk.get('qa_round', 1)} - "
                f"Chunk {chunk['chunk_id']} - words {chunk['start_word']}-{chunk['end_word']}"
            )
            with st.expander(chunk_label, expanded=False):
                preview = chunk["text"][:400]
                if len(chunk["text"]) > 400:
                    preview += "..."
                st.caption(preview)
                st.markdown("**Chunk additions JSON**")
                st.json(chunk["output"])
            render_attempts(f"{chunk_label} attempts", chunk["attempts"])

    st.subheader("Final merged output")
    st.caption(f"Final QA: {qa_summary_line(final_payload)}")
    with st.expander("Final merged JSON", expanded=False):
        st.json(final_payload)
        st.download_button(
            "Download merged JSON",
            json.dumps(final_payload, indent=2),
            file_name="pwml_pipeline_output.json",
            mime="application/json",
        )

    st.subheader("Draft Graph")
    draft_graph_dict = st.session_state.get("draft_graph", {})
    if draft_graph_dict:
        dg_meta = draft_graph_dict.get("metadata", {})
        dg_nodes = draft_graph_dict.get("nodes", [])
        dg_edges = draft_graph_dict.get("edges", [])
        orphan_ids = {n["id"] for n in dg_nodes} - {e["source"] for e in dg_edges} - {e["target"] for e in dg_edges}

        dg_col1, dg_col2, dg_col3 = st.columns(3)
        dg_col1.metric("Nodes", len(dg_nodes))
        dg_col2.metric("Edges", len(dg_edges))
        dg_col3.metric("Orphan nodes", len(orphan_ids))

        if orphan_ids:
            st.caption("Orphan nodes (no edges): " + ", ".join(sorted(orphan_ids)))

        with st.expander("Nodes", expanded=False):
            st.dataframe(dg_nodes)
        with st.expander("Edges", expanded=False):
            st.dataframe(dg_edges)
        with st.expander("Raw draft_graph.json", expanded=False):
            st.json(draft_graph_dict)

        st.download_button(
            "Download draft_graph.json",
            json.dumps(draft_graph_dict, indent=2, ensure_ascii=False),
            file_name="draft_graph.json",
            mime="application/json",
        )
        st.info("Graph rendering starts after audit and DB mapping, in the Review & Refine panel.")

    # ------------------------------------------------------------------ QA Report
    st.subheader("QA Report")
    qa_report_data = st.session_state.get("qa_report", {})
    if qa_report_data:
        qa_summary = qa_report_data.get("summary", {})
        qa_flags = qa_report_data.get("flags", {})

        qr_col1, qr_col2 = st.columns(2)
        qr_col1.metric("Total species", qa_summary.get("total_species", 0))
        qr_col2.metric("Total reactions", qa_summary.get("total_reactions", 0))

        FLAG_LABELS: List[Tuple[str, str]] = [
            ("missing_compartments", "Missing compartments"),
            ("missing_modifiers", "Missing modifiers / enzymes"),
            ("possible_complexes", "Possible complexes"),
            ("transport_like_reactions", "Transport-like reactions"),
            ("orphan_nodes", "Orphan nodes (degree 0)"),
            ("missing_ids", "Missing external IDs"),
            ("empty_reactions", "Empty reactions"),
            ("duplicate_species", "Duplicate species"),
            ("inconsistent_class", "Inconsistent entity class"),
        ]

        import pandas as pd  # local import — pandas is already a streamlit dep

        for flag_key, flag_label in FLAG_LABELS:
            items = qa_flags.get(flag_key, [])
            if not items:
                continue
            with st.expander(f"{flag_label} ({len(items)})", expanded=False):
                try:
                    st.dataframe(pd.DataFrame(items), use_container_width=True)
                except Exception:
                    st.json(items)

        empty_flag_keys = [k for k in qa_flags if not qa_flags[k]]
        if empty_flag_keys:
            st.caption("No issues detected for: " + ", ".join(
                lbl for k, lbl in FLAG_LABELS if k in empty_flag_keys
            ))

        st.download_button(
            "Download qa_report.json",
            json.dumps(qa_report_data, indent=2, ensure_ascii=False),
            file_name="qa_report.json",
            mime="application/json",
            key="dl_qa_report",
        )
    else:
        st.info("Run the pipeline to generate a QA report.")

    # ------------------------------------------------------------------ Pathway Summary
    st.subheader("Pathway Summary")
    reaction_summary_text = st.session_state.get("reaction_summary", "")
    if reaction_summary_text:
        st.text_area(
            "Reaction & transport summary (plain text)",
            value=reaction_summary_text,
            height=420,
            disabled=True,
            key="pathway_summary_display",
        )
        st.download_button(
            "Download reaction_summary.txt",
            data=reaction_summary_text,
            file_name="reaction_summary.txt",
            mime="text/plain",
            key="dl_reaction_summary",
        )
    else:
        st.info("Run the pipeline to generate a pathway summary.")

    st.subheader("Post-pipeline audit and DB mapping")
    post_col_a, post_col_b = st.columns(2)
    use_llm_audit = post_col_a.checkbox(
        "Use LLM in audit stage",
        value=True,
        help="Disabling runs deterministic audit rules only.",
        key="post_use_llm_audit",
    )
    use_sbml_overwatch = post_col_a.checkbox(
        "Use SBML semantic overwatch for legacy export",
        value=False,
        help="Only used by the legacy SBML export path.",
        key="post_use_sbml_overwatch",
    )
    use_stoich_agent = post_col_a.checkbox(
        "Stoichiometry agent (fill missing ATP/ADP/NAD etc.)",
        value=False,
        key="post_use_stoich_agent",
    )
    default_compartment = post_col_b.text_input(
        "Default compartment",
        value="cell",
        key="post_default_compartment",
        help="Used when location/state is missing.",
    )
    mapping_cache_text = st.text_input(
        "ID mapping cache path",
        value="data/id_mapping_cache.json",
        key="post_mapping_cache",
        help="Cache file for UniProt/compound mapping lookups.",
    )
    repair_cols = st.columns(3)
    audit_max_rounds = repair_cols[0].number_input(
        "Audit repair max rounds",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
        key="post_audit_max_rounds",
        help="Retry audit/patch cycles until stable or this round limit.",
    )
    audit_timeout_seconds = repair_cols[1].number_input(
        "Audit repair timeout (seconds)",
        min_value=30,
        max_value=1800,
        value=240,
        step=10,
        key="post_audit_timeout_seconds",
        help="Hard timeout for all audit-repair rounds.",
    )
    audit_candidate_count = repair_cols[2].number_input(
        "Audit candidates / round",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        key="post_audit_candidate_count",
        help="Generates multiple LLM repair candidates and picks the best by deterministic score.",
    )
    retrieval_cols = st.columns(3)
    use_example_retrieval = retrieval_cols[0].checkbox(
        "Use SBML motif retrieval during audit",
        value=False,
        key="post_use_example_retrieval",
        help="Injects nearest SBML motif examples into each audit LLM call.",
    )
    example_index_path = retrieval_cols[1].text_input(
        "SBML motif index path",
        value="src/tmp/sbml_motif_index.json",
        key="post_example_index_path",
        help="JSON index built from trusted SBML files.",
    )
    example_top_k = retrieval_cols[2].number_input(
        "SBML motifs top-k",
        min_value=1,
        max_value=8,
        value=3,
        step=1,
        key="post_example_top_k",
        help="How many nearest examples to inject per round.",
    )
    gap_cols = st.columns(2)
    use_gap_resolver = gap_cols[0].checkbox(
        "Use Stage 3 Targeted Resolution (LLM-planned, code-executed)",
        value=True,
        key="post_use_gap_resolver",
        help="Plans DB/API calls with LLM, executes deterministically, then applies selected patches.",
    )
    use_llm_gap_resolver = gap_cols[0].checkbox(
        "Use LLM in Stage 3 planner/selection",
        value=True,
        key="post_use_llm_gap_resolver",
        help="LLM plans query strategy and selects among deterministic DB/API results.",
    )
    gap_resolver_max_items = gap_cols[1].number_input(
        "Gap resolver max entities",
        min_value=10,
        max_value=400,
        value=80,
        step=10,
        key="post_gap_resolver_max_items",
        help="Upper bound for per-round entity resolution workload.",
    )
    id_source_mode = post_col_b.selectbox(
        "ID mapping source",
        options=["hybrid", "db", "api"],
        index=["hybrid", "db", "api"].index((os.getenv("PATHBANK_ID_SOURCE", "hybrid") or "hybrid").strip().lower())
        if (os.getenv("PATHBANK_ID_SOURCE", "hybrid") or "hybrid").strip().lower() in {"hybrid", "db", "api"}
        else 0,
        key="post_mapping_source",
        help="hybrid = PathBank DB first, then API fallback.",
    )
    with st.expander("PathBank DB connection (optional)", expanded=False):
        db_cols = st.columns(2)
        db_host = db_cols[0].text_input("DB host", value=os.getenv("PATHBANK_DB_HOST", ""), key="post_db_host")
        db_port = db_cols[1].number_input(
            "DB port",
            min_value=1,
            max_value=65535,
            value=int(os.getenv("PATHBANK_DB_PORT", "3306") or "3306"),
            step=1,
            key="post_db_port",
        )
        db_user = db_cols[0].text_input("DB user", value=os.getenv("PATHBANK_DB_USER", ""), key="post_db_user")
        db_schema = db_cols[1].text_input("DB schema", value=os.getenv("PATHBANK_DB_SCHEMA", "pathbank"), key="post_db_schema")
        db_password = st.text_input(
            "DB password",
            value=os.getenv("PATHBANK_DB_PASSWORD", ""),
            type="password",
            key="post_db_password",
        )

    post_artifacts = st.session_state.get("post_pipeline_artifacts")
    if isinstance(post_artifacts, dict):
        gate_failed = bool(post_artifacts.get("gate_failed", False))
        normalization_gate_failed = bool(post_artifacts.get("normalization_gate_failed", False))
        audit_summary = post_artifacts.get("audit_report", {}).get("summary", {})
        stage2_mapping_summary = post_artifacts.get("stage2_mapping_report", {}).get("summary", {})
        mapping_summary = post_artifacts.get("mapping_report", {}).get("summary", {})
        enrichment_summary = post_artifacts.get("enrichment_report", {}).get("summary", {})
        sbml_summary = post_artifacts.get("sbml_report_json", {}).get("counts", {})
        sbml_validation = post_artifacts.get("sbml_report_json", {}).get("validation", {})
        sbml_overwatch_summary = post_artifacts.get("sbml_overwatch_report", {}).get("summary", {})
        sbml_layout_summary = _safe_dict(post_artifacts.get("sbml_render_layout_summary"))
        stoich_audit_log = post_artifacts.get("stoich_audit_log") or []
        stoich_additions_made = sum(1 for e in stoich_audit_log if e.get("llm_verdict") == "add")
        stoich_audits_reversed = sum(1 for e in stoich_audit_log if e.get("audit_verdict") == "reversed")

        _stage2_mapped = int(
            stage2_mapping_summary.get(
                "entities_mapped",
                int(stage2_mapping_summary.get("proteins_mapped", 0))
                + int(stage2_mapping_summary.get("compounds_mapped", 0))
                + int(stage2_mapping_summary.get("protein_complexes_mapped", 0)),
            )
        )
        _stage2_total = int(stage2_mapping_summary.get("proteins_total", 0)) + int(
            stage2_mapping_summary.get("compounds_total", 0)
        ) + int(stage2_mapping_summary.get("protein_complexes_total", 0))
        _stage2_ambiguous = int(
            stage2_mapping_summary.get(
                "entities_ambiguous",
                int(stage2_mapping_summary.get("proteins_ambiguous", 0))
                + int(stage2_mapping_summary.get("compounds_ambiguous", 0))
                + int(stage2_mapping_summary.get("protein_complexes_ambiguous", 0)),
            )
        )
        _stage2_unmapped = int(
            stage2_mapping_summary.get(
                "entities_unmapped",
                max(0, _stage2_total - _stage2_mapped - _stage2_ambiguous),
            )
        )
        st.info(
            "Stage 2B mapping completed: "
            f"mapped={_stage2_mapped}, ambiguous={_stage2_ambiguous}, "
            f"unmapped={_stage2_unmapped}, generated_wrappers="
            f"{int(stage2_mapping_summary.get('generated_wrappers_created', 0))}. "
            "Stage 6 remap bypassed cache and may create required wrappers after audit/curation."
        )

        st.write(
            {
                "normalization_stats": _safe_dict(post_artifacts.get("pre_normalization_report")).get("summary", {}),
                "connectivity": _safe_dict(post_artifacts.get("connectivity_summary"))
                or _safe_dict(post_artifacts.get("post_dedupe_probe") or post_artifacts.get("post_normalization_probe")).get("graph_summary", {}),
                "gate_failed": gate_failed,
                # History. The verdict is ``final_stage3_gate_ok`` below: this
                # flag says the first gate run found issues for the audit to
                # repair, which is a different question from whether the payload
                # that shipped passed.
                "normalization_gate_failed": normalization_gate_failed,
                "gate_fail_report": post_artifacts.get("gate_fail_report", {}),
                "final_stage3_gate_ok": _safe_dict(
                    post_artifacts.get(FINAL_GATE_REPORT_KEY)
                ).get("ok"),
                "canonical_payload_sha256": post_artifacts.get("canonical_payload_sha256", ""),
                "audit": audit_summary,
                "stage2_mapping": stage2_mapping_summary,
                "stage6_mapping": mapping_summary,
                "enrichment": enrichment_summary,
                "sbml_counts": sbml_summary,
                "sbml_validation_has_errors": sbml_validation.get("has_errors"),
                "sbml_overwatch": sbml_overwatch_summary,
                "sbml_diagram_generated": bool(post_artifacts.get("sbml_diagram_png_bytes")),
                "sbml_diagram_error": post_artifacts.get("sbml_diagram_error", ""),
                "sbml_geometry_source": sbml_layout_summary.get("geometry_source", ""),
                "sbml_has_drawable_geometry": sbml_layout_summary.get("has_drawable_geometry", False),
                "sbml_location_elements": sbml_layout_summary.get("visible_location_element_count", 0),
                "sbml_edge_count": sbml_layout_summary.get("edge_count", 0),
                "mapping_cache_path": post_artifacts.get("mapping_cache_path"),
                "enrichment_cache_path": post_artifacts.get("enrichment_cache_path"),
                "enrichment_dump_path": post_artifacts.get("enrichment_dump_path"),
                "mapping_id_source": post_artifacts.get("mapping_id_source"),
                "mapping_db_host": post_artifacts.get("mapping_db_host"),
                "mapping_db_schema": post_artifacts.get("mapping_db_schema"),
                "example_retrieval_enabled": post_artifacts.get("example_retrieval_enabled"),
                "example_index_path": post_artifacts.get("example_index_path"),
                "example_index_error": post_artifacts.get("example_index_error"),
                "example_index_entry_count": post_artifacts.get("example_index_entry_count"),
                "audit_loop": post_artifacts.get("audit_loop_summary"),
                "stoich_additions_made": stoich_additions_made,
                "stoich_audits_reversed": stoich_audits_reversed,
            }
        )
        if gate_failed:
            _gfr = _safe_dict(post_artifacts.get("gate_fail_report"))
            st.error(
                f"Hard-gate failure before audit/mapping/SBML: "
                f"{_gfr.get('error', 'unknown error')}"
            )
            _gate_errors = _safe_list(_gfr.get("errors"))
            if _gate_errors:
                with st.expander(f"Gate validation errors ({len(_gate_errors)})", expanded=True):
                    for _ge in _gate_errors:
                        if isinstance(_ge, dict):
                            st.markdown(f"- **`{_ge.get('path', '?')}`**: {_ge.get('reason', _ge)}")
                            _render_issue_detail(_ge.get("detail"))
                        else:
                            st.markdown(f"- {_ge}")
        elif normalization_gate_failed:
            _gfr = _safe_dict(post_artifacts.get("gate_fail_report"))
            _gate_errors = _safe_list(_gfr.get("errors"))
            st.warning(
                f"Post-normalization gate produced {len(_gate_errors)} audit issue(s); "
                "the audit loop ran before export."
            )
        if str(post_artifacts.get("example_index_error", "")).strip():
            st.warning(f"SBML motif retrieval issue: {post_artifacts.get('example_index_error')}")
        if post_artifacts.get("audit_iterations"):
            with st.expander("Audit repair iterations", expanded=False):
                st.write(post_artifacts.get("audit_iterations"))
        if post_artifacts.get("gap_resolution_iterations"):
            with st.expander("Stage 3 resolution iterations", expanded=False):
                st.write(post_artifacts.get("gap_resolution_iterations"))
        if post_artifacts.get("stoich_audit_log"):
            with st.expander("Stoichiometry audit log"):
                st.dataframe(post_artifacts["stoich_audit_log"])
        if sbml_layout_summary:
            st.write("SBML render geometry", sbml_layout_summary)
            if sbml_layout_summary.get("has_drawable_geometry"):
                st.info(
                    "SBML render geometry confirmed: "
                    f"{sbml_layout_summary.get('visible_location_element_count', 0)} visible layout elements "
                    f"({sbml_layout_summary.get('edge_count', 0)} edges, source={sbml_layout_summary.get('geometry_source', 'unknown')})."
                )
            else:
                st.warning("SBML render geometry could not be confirmed from the render-ready SBML.")

        st.download_button(
            "Download pre_mapping_input.json",
            json.dumps(post_artifacts.get("pre_mapping_input", {}), indent=2),
            file_name="pre_mapping_input.json",
            mime="application/json",
            key="dl_pre_mapping_input",
        )
        st.download_button(
            "Download stage2.mapped.json",
            json.dumps(post_artifacts.get("stage2_payload", {}), indent=2),
            file_name="stage2.mapped.json",
            mime="application/json",
            key="dl_stage2_mapped",
        )
        st.download_button(
            "Download stage2_mapping_report.json",
            json.dumps(post_artifacts.get("stage2_mapping_report", {}), indent=2),
            file_name="stage2_mapping_report.json",
            mime="application/json",
            key="dl_stage2_mapping_report",
        )
        st.download_button(
            "Download stage2_runtime_schema_report.json",
            json.dumps(post_artifacts.get("stage2_runtime_schema_report", {}), indent=2),
            file_name="stage2_runtime_schema_report.json",
            mime="application/json",
            key="dl_stage2_runtime_schema_report",
        )
        st.download_button(
            "Download pre_normalization_input.json",
            json.dumps(post_artifacts.get("pre_normalization_input", {}), indent=2),
            file_name="pre_normalization_input.json",
            mime="application/json",
            key="dl_pre_normalization_input",
        )
        st.download_button(
            "Download pre_normalized_input.json",
            json.dumps(post_artifacts.get("pre_normalized_input", {}), indent=2),
            file_name="pre_normalized_input.json",
            mime="application/json",
            key="dl_pre_normalized_input",
        )
        st.download_button(
            "Download pre_normalization_report.json",
            json.dumps(post_artifacts.get("pre_normalization_report", {}), indent=2),
            file_name="pre_normalization_report.json",
            mime="application/json",
            key="dl_pre_normalization_report",
        )
        st.download_button(
            "Download post_normalization_probe.json",
            json.dumps(post_artifacts.get("post_normalization_probe", {}), indent=2),
            file_name="post_normalization_probe.json",
            mime="application/json",
            key="dl_post_normalization_probe",
        )
        st.download_button(
            "Download post_transport_attachment_probe.json",
            json.dumps(post_artifacts.get("post_transport_attachment_probe", {}), indent=2),
            file_name="post_transport_attachment_probe.json",
            mime="application/json",
            key="dl_post_transport_attachment_probe",
        )
        st.download_button(
            "Download post_dedupe_probe.json",
            json.dumps(post_artifacts.get("post_dedupe_probe", {}), indent=2),
            file_name="post_dedupe_probe.json",
            mime="application/json",
            key="dl_post_dedupe_probe",
        )
        if gate_failed or normalization_gate_failed:
            st.download_button(
                "Download gate_fail_report.json",
                json.dumps(post_artifacts.get("gate_fail_report", {}), indent=2),
                file_name="gate_fail_report.json",
                mime="application/json",
                key="dl_gate_fail_report",
            )
        st.download_button(
            "Download audit_report.json",
            json.dumps(post_artifacts["audit_report"], indent=2),
            file_name="audit_report.json",
            mime="application/json",
            key="dl_audit_report",
        )
        st.download_button(
            "Download audit_patch.json",
            json.dumps(post_artifacts["audit_patch"], indent=2),
            file_name="audit_patch.json",
            mime="application/json",
            key="dl_audit_patch",
        )
        st.download_button(
            "Download audit_apply_report.json",
            json.dumps(post_artifacts["audit_apply_report"], indent=2),
            file_name="audit_apply_report.json",
            mime="application/json",
            key="dl_audit_apply",
        )
        if post_artifacts.get("audit_iterations"):
            st.download_button(
                "Download audit_iterations.json",
                json.dumps(post_artifacts["audit_iterations"], indent=2),
                file_name="audit_iterations.json",
                mime="application/json",
                key="dl_audit_iterations",
            )
        if post_artifacts.get("gap_resolution_iterations"):
            st.download_button(
                "Download stage3_resolution_iterations.json",
                json.dumps(post_artifacts["gap_resolution_iterations"], indent=2),
                file_name="stage3_resolution_iterations.json",
                mime="application/json",
                key="dl_gap_resolution_iterations",
            )
        st.download_button(
            "Download final.audited.json",
            json.dumps(post_artifacts["final_audited"], indent=2),
            file_name="final.audited.json",
            mime="application/json",
            key="dl_final_audited",
        )
        st.download_button(
            "Download final.mapped.json",
            json.dumps(post_artifacts.get("final_mapped_db", post_artifacts.get("final_mapped", {})), indent=2),
            file_name="final.mapped.json",
            mime="application/json",
            key="dl_final_mapped",
        )
        if post_artifacts.get("final_export_input") and post_artifacts.get("final_export_input") != post_artifacts.get("final_mapped_db"):
            st.download_button(
                "Download final.export_input.json",
                json.dumps(post_artifacts.get("final_export_input", {}), indent=2),
                file_name="final.export_input.json",
                mime="application/json",
                key="dl_final_export_input",
            )
        st.download_button(
            "Download mapping_report.json",
            json.dumps(post_artifacts["mapping_report"], indent=2),
            file_name="mapping_report.json",
            mime="application/json",
            key="dl_mapping_report",
        )
        st.download_button(
            "Download post_enrichment_runtime_schema_report.json",
            json.dumps(post_artifacts.get("post_enrichment_runtime_schema_report", {}), indent=2),
            file_name="post_enrichment_runtime_schema_report.json",
            mime="application/json",
            key="dl_post_enrichment_runtime_schema_report",
        )
        st.download_button(
            "Download pre_export_runtime_schema_report.json",
            json.dumps(post_artifacts.get("pre_export_runtime_schema_report", {}), indent=2),
            file_name="pre_export_runtime_schema_report.json",
            mime="application/json",
            key="dl_pre_export_runtime_schema_report",
        )
        st.download_button(
            "Download enrichment_report.json",
            json.dumps(post_artifacts.get("enrichment_report", {}), indent=2),
            file_name="enrichment_report.json",
            mime="application/json",
            key="dl_enrichment_report",
        )
        dump_path_value = str(post_artifacts.get("enrichment_dump_path", "") or "").strip()
        if dump_path_value:
            dump_path = Path(dump_path_value)
            if dump_path.exists():
                st.download_button(
                    "Download enrichment_dump.json",
                    dump_path.read_text(encoding="utf-8"),
                    file_name="enrichment_dump.json",
                    mime="application/json",
                    key="dl_enrichment_dump",
                )
    # ── Entity Mapping Debug ─────────────────────────────────────────────────
    with st.expander("Entity Mapping Debug — per-entity API/DB results", expanded=False):
        st.caption(
            "Inspect what each API or DB call returned for every compound and protein, "
            "the candidate list with scores, and which candidate was chosen by the mapper."
        )
        _debug_payload = (post_artifacts or {}).get("final_mapped_db") or (post_artifacts or {}).get("final_mapped")
        _debug_mapping_report = (post_artifacts or {}).get("mapping_report", {})
        _debug_entity_logs = _debug_mapping_report.get("entities", [])
        if not isinstance(_debug_payload, dict):
            st.info("Run the pipeline first to see entity mapping details.")
        else:
            _debug_entities_section = _safe_dict(_debug_payload.get("entities"))
            _debug_compounds = _safe_list(_debug_entities_section.get("compounds"))
            _debug_proteins = _safe_list(_debug_entities_section.get("proteins"))
            _debug_complexes = _safe_list(_debug_entities_section.get("protein_complexes"))

            _debug_items: list = []
            for _di, _dc in enumerate(_debug_compounds):
                if isinstance(_dc, dict) and isinstance(_dc.get("name"), str) and _dc["name"].strip():
                    _debug_items.append(("compound", _di, _dc["name"].strip(), _dc))
            for _di, _dp in enumerate(_debug_proteins):
                if isinstance(_dp, dict) and isinstance(_dp.get("name"), str) and _dp["name"].strip():
                    _debug_items.append(("protein", _di, _dp["name"].strip(), _dp))
            for _di, _dx in enumerate(_debug_complexes):
                if isinstance(_dx, dict) and isinstance(_dx.get("name"), str) and _dx["name"].strip():
                    _debug_items.append(("protein_complex", _di, _dx["name"].strip(), _dx))

            if not _debug_items:
                st.info("No entities found in the mapped payload.")
            else:
                _debug_labels = [f"[{etype}] {ename}" for etype, _, ename, _ in _debug_items]
                _debug_selected = st.selectbox(
                    "Select entity to inspect",
                    _debug_labels,
                    index=0,
                    key="entity_mapping_debug_select",
                )
                _debug_sel_idx = _debug_labels.index(_debug_selected)
                _debug_etype, _debug_eidx, _debug_ename, _debug_entity = _debug_items[_debug_sel_idx]

                _debug_meta = _safe_dict(_debug_entity.get("mapping_meta"))
                _debug_mapped_ids = _safe_dict(_debug_entity.get("mapped_ids"))

                st.markdown(f"**Entity:** `{_debug_ename}` &nbsp; | &nbsp; **Type:** `{_debug_etype}` &nbsp; | &nbsp; **Index:** `{_debug_eidx}`")

                _debug_col1, _debug_col2 = st.columns(2)
                with _debug_col1:
                    st.markdown("##### Final Mapped IDs")
                    if _debug_mapped_ids:
                        st.json(_debug_mapped_ids)
                    else:
                        st.warning("No mapped IDs — entity is unmapped.")
                with _debug_col2:
                    st.markdown("##### Mapping Decision")
                    st.json({
                        "query": _debug_meta.get("query", {}),
                        "provider": _debug_meta.get("provider") or _debug_meta.get("providers", ""),
                        "source": _debug_meta.get("source", ""),
                        "chosen_rule": _debug_meta.get("chosen_rule", ""),
                        "confidence": _debug_meta.get("confidence", 0.0),
                        "route": _debug_meta.get("route", ""),
                        "route_reason": _debug_meta.get("route_reason", ""),
                    })

                _debug_candidates = _safe_list(_debug_meta.get("candidates"))
                st.markdown(f"##### API/DB Candidates Returned ({len(_debug_candidates)} total)")
                if _debug_candidates:
                    st.json(_debug_candidates)
                else:
                    st.info("No candidates were returned by any API or DB lookup.")

                _debug_resolution = _safe_dict(_debug_meta.get("resolution"))
                if _debug_resolution:
                    st.markdown("##### Resolution Details")
                    st.json(_debug_resolution)

                _debug_log_match = [
                    e for e in _debug_entity_logs
                    if e.get("name") == _debug_ename and e.get("entity_type") == _debug_etype
                ]
                if _debug_log_match:
                    st.markdown("##### Mapping Report Log Entry")
                    st.json(_debug_log_match[0])

                for _extra_key in ["matched_alias", "alias_source", "resolved_name", "queries_tried", "literature_aliases"]:
                    if _debug_meta.get(_extra_key):
                        st.markdown(f"**{_extra_key}:** `{_debug_meta[_extra_key]}`")

    # ── DB Gap Resolution ─────────────────────────────────────────────────────
    with st.expander("Resolve unmapped entities via DB", expanded=False):
        st.caption(
            "Runs every unmapped / ambiguous entity from the last pipeline mapping "
            "through the DB lookup primitives and shows what resolves. "
            "Uses the DB connection configured above."
        )

        _gap_db_host = st.session_state.get("post_db_host", os.getenv("PATHBANK_DB_HOST", ""))
        _gap_db_port = int(st.session_state.get("post_db_port", int(os.getenv("PATHBANK_DB_PORT", "3306") or 3306)))
        _gap_db_user = st.session_state.get("post_db_user", os.getenv("PATHBANK_DB_USER", ""))
        _gap_db_pass = st.session_state.get("post_db_password", os.getenv("PATHBANK_DB_PASSWORD", ""))
        _gap_db_schema = st.session_state.get("post_db_schema", os.getenv("PATHBANK_DB_SCHEMA", "pathbank"))

        _gap_mr = (post_artifacts or {}).get("mapping_report", {})
        _gap_entities_log = _gap_mr.get("entities", [])
        _gap_targets = [
            e for e in _gap_entities_log
            if e.get("status") == "unmapped" or e.get("reason") == "ambiguous"
        ]
        _gap_payload = (post_artifacts or {}).get("final_mapped")

        if not _gap_targets:
            st.info("No unmapped or ambiguous entities from the last pipeline run.")
        elif not _gap_db_host or not _gap_db_user:
            st.warning("Configure DB host and user in the PathBank DB connection section above first.")
        else:
            st.write(f"**{len(_gap_targets)} entities to resolve** "
                     f"({sum(1 for e in _gap_targets if e.get('status') == 'unmapped')} unmapped, "
                     f"{sum(1 for e in _gap_targets if e.get('reason') == 'ambiguous')} ambiguous)")

            if st.button("Run DB gap resolution", key="gap_resolve_btn"):
                _gap_resolver = PathBankDbResolver(
                    host=_gap_db_host,
                    port=_gap_db_port,
                    user=_gap_db_user,
                    password=_gap_db_pass,
                    schema=_gap_db_schema,
                )
                if not _gap_resolver.available():
                    st.error(f"pymysql not available: {_gap_resolver.last_error}")
                else:
                    from t2pw.mapping.map_ids import _extract_global_organism
                    _gap_organism = _extract_global_organism(_gap_payload) if isinstance(_gap_payload, dict) else ""
                    try:
                        _gap_out = resolve_mapping_gaps(
                            _gap_payload or {},
                            _gap_mr,
                            _gap_resolver,
                            global_organism=_gap_organism,
                        )
                    finally:
                        _gap_resolver.close()

                    _n_resolved = _gap_out["resolved_count"]
                    _gap_rows = _gap_out["rows"]
                    st.success(f"Resolved **{_n_resolved} / {_gap_out['total']}** entities via DB.")
                    if _gap_resolver.last_error:
                        st.warning(f"DB last_error: `{_gap_resolver.last_error}`")

                    if _gap_rows:
                        import pandas as _pd
                        st.dataframe(
                            _pd.DataFrame(_gap_rows).sort_values("resolved", ascending=False),
                            use_container_width=True,
                        )

                    if _n_resolved > 0:
                        st.download_button(
                            f"Download patched payload ({_n_resolved} newly resolved)",
                            json.dumps(_gap_out["patched_payload"], indent=2, ensure_ascii=False),
                            file_name="final.mapped.gap_resolved.json",
                            mime="application/json",
                            key="dl_gap_resolved_payload",
                        )

    # ── Contract Audit ────────────────────────────────────────────────────────
    # Hidden from UI — set _SHOW_CONTRACT_AUDIT = True to restore.
    _SHOW_CONTRACT_AUDIT = False
    if _SHOW_CONTRACT_AUDIT:
        st.subheader("PWML Contract Audit")
        st.caption(
            "Run this before export to surface every required-field gap in the hydrated payload. "
            "Errors block export; warnings are advisory."
        )
        _contract_audit_target = st.radio(
            "Audit target",
            ["Current payload (pre-IR)", "Last built IR"],
            key="contract_audit_target",
            horizontal=True,
        )
        if st.button("Run Contract Audit", key="run_contract_audit_btn"):
            _audit_input: Optional[Dict[str, Any]] = None
            if _contract_audit_target == "Last built IR":
                _prev_result = st.session_state.get("pwml_export_result")
                _audit_input = _safe_dict(_prev_result).get("pwml_ir") if isinstance(_prev_result, dict) else None
                if not _audit_input:
                    st.warning("No built IR found in session — run PWML export first, or audit the payload instead.")
            else:
                _audit_input = final_payload if isinstance(final_payload, dict) and final_payload else None
                if not _audit_input:
                    st.error("No pipeline payload in session. Run the pipeline first.")
            if _audit_input:
                _contract_report = validate_required_pwml_contract(_audit_input)
                st.session_state["contract_audit_report"] = _contract_report
        _contract_report = st.session_state.get("contract_audit_report")
        if isinstance(_contract_report, dict):
            _ca_summary = _safe_dict(_contract_report.get("summary"))
            _ca_errors = _contract_report.get("errors", [])
            _ca_warnings = _contract_report.get("warnings", [])
            _ca_ok = _contract_report.get("ok", True)
            _ca_checked_as = _ca_summary.get("checked_as", "unknown")
            if _ca_ok and not _ca_warnings:
                st.success(f"Contract audit passed ({_ca_checked_as}): no errors or warnings.")
            elif _ca_ok:
                st.warning(f"Contract audit passed with {len(_ca_warnings)} warning(s) ({_ca_checked_as}).")
            else:
                st.error(f"Contract audit FAILED: {len(_ca_errors)} error(s), {len(_ca_warnings)} warning(s) ({_ca_checked_as}).")
            _audit_cols = st.columns(2)
            with _audit_cols[0]:
                st.markdown(f"**Error codes:** {', '.join(_ca_summary.get('error_codes', [])) or '—'}")
            with _audit_cols[1]:
                st.markdown(f"**Warning codes:** {', '.join(_ca_summary.get('warning_codes', [])) or '—'}")
            if _ca_errors:
                with st.expander(f"Errors ({len(_ca_errors)})", expanded=False):
                    _err_by_code: Dict[str, List[Any]] = {}
                    for _e in _ca_errors:
                        _err_by_code.setdefault(_e.get("code", "unknown"), []).append(_e)
                    for _code, _group in sorted(_err_by_code.items()):
                        st.markdown(f"**{_code}** ({len(_group)})")
                        for _issue in _group:
                            _ptr = _issue.get("pointer", "")
                            _extra = {k: v for k, v in _issue.items() if k not in ("code", "message", "pointer")}
                            _detail = f"`{_ptr}` — " if _ptr else ""
                            _detail += _issue.get("message", "")
                            if _extra:
                                _detail += f"  \n&nbsp;&nbsp;&nbsp;&nbsp;_{', '.join(f'{k}={v}' for k, v in _extra.items())}_"
                            st.markdown(f"- {_detail}")
            if _ca_warnings:
                with st.expander(f"Warnings ({len(_ca_warnings)})", expanded=False):
                    _warn_by_code: Dict[str, List[Any]] = {}
                    for _w in _ca_warnings:
                        _warn_by_code.setdefault(_w.get("code", "unknown"), []).append(_w)
                    for _code, _group in sorted(_warn_by_code.items()):
                        st.markdown(f"**{_code}** ({len(_group)})")
                        for _issue in _group:
                            _ptr = _issue.get("pointer", "")
                            _extra = {k: v for k, v in _issue.items() if k not in ("code", "message", "pointer")}
                            _detail = f"`{_ptr}` — " if _ptr else ""
                            _detail += _issue.get("message", "")
                            if _extra:
                                _detail += f"  \n&nbsp;&nbsp;&nbsp;&nbsp;_{', '.join(f'{k}={v}' for k, v in _extra.items())}_"
                            st.markdown(f"- {_detail}")
            st.download_button(
                "Download contract audit report",
                data=json.dumps(_contract_report, indent=2),
                file_name="contract_audit_report.json",
                mime="application/json",
                key="dl_contract_audit",
            )

    st.divider()
    # ── PWML Export ───────────────────────────────────────────────────────────
    st.subheader("PWML Export")

    _project_root_pwml = PROJECT_ROOT
    _ref_candidates = [
        _project_root_pwml / "reference" / "PW000001.pwml",
        _project_root_pwml / "reference" / "PW012926.pwml",
    ]
    _ref_path_pwml = next((p for p in _ref_candidates if p.exists()), _ref_candidates[0])

    _pwml_cols = st.columns(3)
    _pwml_name = _pwml_cols[0].text_input("Pathway name", value="Generated Pathway", key="pwml_name")
    _pwml_subject = _pwml_cols[1].text_input("Subject", value="Metabolic", key="pwml_subject")
    _pwml_description = _pwml_cols[2].text_input("Description", value="", key="pwml_description")

    _vis_cols = st.columns(3)
    _pwml_width = _vis_cols[0].number_input("Width", min_value=200, max_value=10000, value=3200, step=100, key="pwml_width")
    _pwml_height = _vis_cols[1].number_input("Height", min_value=200, max_value=10000, value=1400, step=100, key="pwml_height")
    _pwml_bg = _vis_cols[2].text_input("Background color", value="#FFFFFF", key="pwml_bg")
    _pwml_strict_db = st.checkbox(
        "Require DB-backed PWML identities",
        value=True,
        key="pwml_strict_db",
        help="When enabled, compounds/proteins/complexes without PathBank/PW IDs stop PWML serialization at the IR stage.",
    )

    _pwml_grounding = st.checkbox("Apply grounding before export", value=False, key="pwml_grounding")
    _pwml_grounding_dict: Optional[Dict[str, Any]] = None
    if _pwml_grounding:
        _pwml_grounding_path = st.text_input(
            "Grounding dictionary path", value="data/grounding_dictionary.example.json",
            key="pwml_grounding_path"
        )
        if st.button("Load grounding dictionary", key="pwml_load_grounding"):
            try:
                _gp = resolve_path(_pwml_grounding_path)
                _pwml_grounding_dict = json.loads(_gp.read_text(encoding="utf-8"))
                if not isinstance(_pwml_grounding_dict, dict):
                    raise ValueError("Grounding dictionary must be a JSON object.")
                st.session_state["pwml_grounding_dict"] = _pwml_grounding_dict
                st.success("Grounding dictionary loaded.")
            except Exception as _ge:
                st.error(f"Grounding load failed: {_ge}")
                st.exception(_ge)  # full traceback (file:line) so the cause isn't a mystery
        _pwml_grounding_dict = st.session_state.get("pwml_grounding_dict")

    if st.button("Run audit and DB mapping", key="pwml_generate_btn"):
        if not isinstance(final_payload, dict) or not final_payload:
            st.error("No pipeline output in session state. Run the pipeline first.")
        else:
            st.session_state.pop("post_pipeline_artifacts", None)
            # A second audit/mapping run produces a new payload version, so the
            # previous run's decision is void before this one starts -- not merely
            # replaced when the boundary happens to be reached. It often is not:
            # mapping can raise, and the handler below catches it. Without the
            # clear, that dead run would leave the earlier decision in session
            # state and its artifacts on disk, authorizing a payload nobody
            # admitted. Cleared HERE, ahead of everything, so the boundary is
            # never the thing that has to remember to invalidate.
            reset_quarantine_state(PROJECT_ROOT / "outputs")
            post_artifacts = {}
            try:
                with st.spinner("Running audit and DB mapping..."):
                    final_payload = propagate_context_organism(
                        final_payload,
                        st.session_state.get("pathway_context"),
                    )
                    artifacts = run_post_pipeline_sbml_artifacts(
                        final_payload,
                        build_legacy_sbml=False,
                        use_llm_audit=bool(use_llm_audit),
                        use_sbml_overwatch=bool(use_sbml_overwatch),
                        default_compartment=(default_compartment or "cell").strip() or "cell",
                        mapping_cache_path=mapping_cache_text.strip() or "id_mapping_cache.json",
                        id_source=(id_source_mode or "hybrid").strip().lower(),
                        db_host=(db_host or "").strip(),
                        db_port=int(db_port),
                        db_user=(db_user or "").strip(),
                        db_password=db_password or "",
                        db_schema=(db_schema or "pathbank").strip() or "pathbank",
                        audit_max_rounds=int(audit_max_rounds),
                        audit_timeout_seconds=int(audit_timeout_seconds),
                        audit_candidate_count=int(audit_candidate_count),
                        use_example_retrieval=bool(use_example_retrieval),
                        example_index_path=(example_index_path or "").strip(),
                        example_top_k=int(example_top_k),
                        use_gap_resolver=bool(use_gap_resolver),
                        use_llm_gap_resolver=bool(use_llm_gap_resolver),
                        gap_resolver_max_items=int(gap_resolver_max_items),
                        qa_report=st.session_state.get("qa_report"),
                        reaction_summary=st.session_state.get("reaction_summary"),
                        use_stoich_agent=bool(use_stoich_agent),
                        rag_evidence_context=(
                            st.session_state.get("rag_evidence_context") or ""
                            if rag_config()["enabled"]
                            else ""
                        ),  # S2 — "" when RAG off, keeping today's path
                        export_mode=coerce_mode(st.session_state.get("export_mode")),
                        # The quarantine boundary runs inside the pipeline now, so
                        # its three inputs travel with the call instead of being
                        # read from session state after the fact.
                        strict_db=bool(st.session_state.get("pwml_strict_db", True)),
                        pathway_context=_safe_dict(st.session_state.get("pathway_context")),
                        outputs_dir=PROJECT_ROOT / "outputs",
                    )
                st.session_state["post_pipeline_artifacts"] = artifacts
                _pa = _safe_dict(artifacts)
                if _pa.get("post_audit_qa_report"):
                    st.session_state["qa_report"] = _pa["post_audit_qa_report"]
                if _pa.get("post_audit_reaction_summary"):
                    st.session_state["reaction_summary"] = _pa["post_audit_reaction_summary"]
                _pa["final_payload_snapshot"] = st.session_state.get("final_payload_snapshot", final_payload)
                if bool(_pa.get("gate_failed", False)):
                    st.warning("Post-pipeline stopped at hard-gate validation. Review gate_fail_report.json.")
                else:
                    # The quarantine boundary and the pre-export Stage 3 gate both
                    # ran INSIDE the pipeline call above, on the canonical payload
                    # -- enriched, then quarantined, then serialized and exported.
                    # They used to run here, after the function returned, which is
                    # why the gate judged an object (quarantine of the PRE-
                    # enrichment payload) that was neither serialized as
                    # final_mapped.json nor handed to the exporter. This block is
                    # now presentation over decisions already made and recorded;
                    # it decides nothing, so a reader never has to ask whether the
                    # UI and the batch driver saw the same verdict.
                    final_mapped_payload = _pa.get(CANONICAL_PAYLOAD_KEY) or _pa.get("final_mapped")
                    mapping_report = _safe_dict(_pa.get("mapping_report"))
                    if isinstance(final_mapped_payload, dict) and final_mapped_payload:
                        _quarantine_flags = _safe_list(_pa.get("quarantine_review_flags"))
                        if _quarantine_flags:
                            # Research mode: the candidate is untouched and the run
                            # continues. These are the findings a strict run would
                            # have acted on, shown so the skip is never invisible.
                            st.warning(
                                f"Quarantine review flags ({len(_quarantine_flags)}): "
                                "research mode kept every process. A strict export "
                                "would have removed the ones listed below."
                            )
                            st.json(_quarantine_flags)
                        if not bool(_pa.get("quarantine_ok")):
                            _refusal_reasons = [
                                str(reason) for reason in _safe_list(_pa.get("quarantine_refusal_reasons"))
                            ]
                            st.session_state.refinement_gate_errors = [
                                {"path": "/processes", "reason": f"quarantine refusal: {reason}"}
                                for reason in _refusal_reasons
                            ]
                            st.error(
                                "Quarantine could not produce a viable requested-pathway core: "
                                + "; ".join(_refusal_reasons)
                            )
                            # Rendered flat rather than inside an expander. A
                            # refusal is the reason the run stopped, so it should
                            # not need a click to read, and st.expander here trips
                            # FragmentThreadState under AppTest once an earlier
                            # app run has executed in the same process.
                            st.caption("Requested-core coverage")
                            st.json(_safe_dict(_pa.get("quarantine_coverage")))
                            st.caption("Strict invariants")
                            st.json(_safe_dict(_pa.get("quarantine_report")).get("strict_invariants"))
                        else:
                            final_gate: Dict[str, Any] = _safe_dict(_pa.get(FINAL_GATE_REPORT_KEY))
                            _stage3_failed = bool(final_gate) and not bool(final_gate.get("ok", False))
                            if _stage3_failed and is_research(coerce_mode(st.session_state.get("export_mode"))):
                                # Research mode fails open through the WHOLE
                                # boundary, not just through quarantine. An
                                # annotate-only quarantine in front of a mode-blind
                                # gate is still a blocked run: a novel enzyme with
                                # no accession fails Stage 3 for the same reason
                                # quarantine flagged it, refinement review never
                                # opens, and the mode that exists for exactly that
                                # pathway delivers nothing. The gate still RAN and
                                # its findings are kept in full below -- what
                                # changes is that they annotate instead of block.
                                # The report itself is already recorded by the
                                # pipeline, on this branch and on the passing one
                                # alike; what research mode adds is the flag
                                # channel, never an edit to the gate's finding.
                                _pa["research_stage3_review_flags"] = _safe_list(final_gate.get("errors"))
                                _pa["research_stage3_failed_open"] = True
                                st.session_state.refinement_gate_errors = []
                                st.warning(
                                    "Research mode: the pre-export Stage 3 revalidation FAILED "
                                    f"({len(_safe_list(final_gate.get('errors')))} error(s)) and was not "
                                    "enforced. The strict gates did NOT pass — this candidate is not "
                                    "PathWhiz-exportable as it stands. Refinement review is open so the "
                                    "findings below can be reviewed against the pathway itself."
                                )
                                # Flat, not in an expander: st.expander here trips
                                # FragmentThreadState under AppTest once an earlier
                                # app run has executed in the same process.
                                st.caption("Stage 3 findings retained as review flags")
                                for _gate_err in _safe_list(final_gate.get("errors")):
                                    if isinstance(_gate_err, dict):
                                        st.markdown(
                                            f"- `{_gate_err.get('path', '')}` — {_gate_err.get('reason', _gate_err)}"
                                        )
                                    else:
                                        st.markdown(f"- {_gate_err}")
                                st.json(final_gate)
                                # The candidate, byte for byte. Research quarantine
                                # returned the payload unchanged, so this is what
                                # came out of mapping and not a reduced graph.
                                initialize_refinement_review_state(final_mapped_payload, mapping_report)
                            elif _stage3_failed:
                                st.session_state.refinement_gate_errors = _safe_list(final_gate.get("errors"))
                                st.error(
                                    "Mapped pathway still fails the pre-export Stage 3 revalidation. "
                                    "Refinement review was not opened."
                                )
                                _gate_errors_for_display = _safe_list(final_gate.get("errors"))
                                with st.expander(
                                    f"Why Stage 3 failed ({len(_gate_errors_for_display)} error(s))",
                                    expanded=True,
                                ):
                                    if _gate_errors_for_display:
                                        for _gate_err in _gate_errors_for_display:
                                            if isinstance(_gate_err, dict):
                                                st.markdown(
                                                    f"- `{_gate_err.get('path', '')}` — {_gate_err.get('reason', _gate_err)}"
                                                )
                                                _render_issue_detail(_gate_err.get("detail"))
                                            else:
                                                st.markdown(f"- {_gate_err}")
                                    else:
                                        st.write("No itemized errors were returned; showing full gate report below.")
                                    st.json(final_gate)
                            else:
                                st.session_state.refinement_gate_errors = []
                                initialize_refinement_review_state(final_mapped_payload, mapping_report)
                                st.info("Mapped pathway is ready for review. PWML generation is paused until approval.")
            except StageContractError as failure:
                _render_stage_contract_failure(
                    failure,
                    operation_label="Audit and DB mapping",
                    download_key="dl_audit_mapping_stage_contract_error",
                )
            except Exception as exc:
                _record_mapped_failure_snapshot(
                    final_payload,
                    stage="post_pipeline",
                    reason=f"{type(exc).__name__}: {exc}",
                )
                st.error(f"Post-pipeline conversion failed: {exc}")
                st.exception(exc)  # full traceback (file:line) so the cause isn't a mystery
            finally:
                # Both the pass and the two failure paths above. The audit and gap
                # rounds completed before an abort are already on disk; this is
                # what carries them, and the mapping snapshot, to the batch driver.
                _sync_extraction_diagnostics()

    # Rendered from session_state on EVERY pass, not inside the Run button's
    # branch: st.download_button triggers a rerun, so a panel rendered only in
    # the click pass would disappear the moment the reviewer downloaded the
    # first file -- and getting it back would mean re-running the whole
    # LLM/DB pipeline. Placed after the Run block so a fresh run renders the
    # artifacts it just stored rather than the previous run's.
    _render_research_mode_panels(st.session_state.get("post_pipeline_artifacts"))

    _refinement_mapping_cache_path = _safe_dict(post_artifacts).get("mapping_cache_path")
    if not _refinement_mapping_cache_path:
        _refinement_cache_text = mapping_cache_text.strip() or "id_mapping_cache.json"
        _refinement_cache_path = Path(_refinement_cache_text)
        if not _refinement_cache_path.is_absolute():
            _refinement_cache_path = PROJECT_ROOT / _refinement_cache_path
        _refinement_mapping_cache_path = str(_refinement_cache_path)
    _refinement_db_config = {
        "host": (db_host or "").strip(),
        "port": int(db_port),
        "user": (db_user or "").strip(),
        "password": db_password or "",
        "schema": (db_schema or "pathbank").strip() or "pathbank",
    }
    if _render_review_refine_section(
        mapping_cache_path=_refinement_mapping_cache_path,
        work_dir=PROJECT_ROOT / "tmp",
        id_source=(id_source_mode or "hybrid").strip().lower(),
        db_config=_refinement_db_config,
    ):
        st.divider()

    _pwml_result = st.session_state.get("pwml_export_result")
    if isinstance(_pwml_result, dict):
        _ir_report = _safe_dict(_pwml_result.get("pwml_ir_report"))
        _ir_validation = _safe_dict(_pwml_result.get("pwml_ir_validation"))
        if _pwml_result.get("ok"):
            st.success(f"Written to: {_pwml_result.get('output_path')}")
            with st.expander("PWML IR report", expanded=False):
                st.write("IR counts", _ir_report.get("counts", {}))
                st.write("IR validation", _ir_validation.get("counts", {}))
                _ir_errors = _safe_list(_ir_report.get("errors"))
                _blocking_ir_errors = blocking_pwml_ir_errors(_ir_report)
                _non_blocking_ir_errors = [
                    e for e in _ir_errors if is_non_blocking_pwml_ir_error(e)
                ]
                if _blocking_ir_errors:
                    st.error("IR errors:\n" + "\n".join(str(e.get("message", e)) for e in _blocking_ir_errors))
                if _non_blocking_ir_errors:
                    st.warning(
                        "Non-blocking PathWhiz DB row matching misses:\n"
                        + "\n".join(str(e.get("message", e)) for e in _non_blocking_ir_errors)
                    )
                if _ir_report.get("warnings"):
                    st.warning("IR warnings:\n" + "\n".join(str(w.get("message", w)) for w in _ir_report["warnings"]))
                unresolved = _safe_dict(_ir_report.get("unresolved"))
                if unresolved:
                    st.write("Unresolved references", unresolved)
            with st.expander("PWML IR JSON", expanded=False):
                _ir_json_str = json.dumps(_pwml_result.get("pwml_ir", {}), indent=2)
                st.code(_ir_json_str, language="json")
                st.download_button(
                    "Download PWML IR JSON",
                    data=_ir_json_str,
                    file_name="final.pwml_ir.json",
                    mime="application/json",
                    key="dl_pwml_ir_inline",
                )
            with st.expander("PWML XML validation and QA", expanded=False):
                st.write("Counts", _pwml_result.get("counts", {}))
                st.write("Structural validation issues", _pwml_result.get("issues", 0))
                _qa = _pwml_result.get("qa", {})
                if _qa:
                    st.write("QA ok", _qa.get("ok"))
                    st.write("QA stats", _qa.get("stats", {}))
                    if _qa.get("errors"):
                        st.error("QA errors:\n" + "\n".join(_qa["errors"]))
                    if _qa.get("warnings"):
                        st.warning("QA warnings:\n" + "\n".join(_qa["warnings"]))
                _gr = _pwml_result.get("grounding_report")
                if _gr:
                    st.write("Grounding report", _gr)
            with st.expander("PWML XML", expanded=False):
                _pwml_xml_str = _pwml_result.get("xml_bytes", b"").decode("utf-8", errors="replace")
                st.code(_pwml_xml_str, language="xml")
            _dl_cols = st.columns(4)
            _dl_cols[0].download_button(
                "Download pathway.pwml", data=_pwml_result["xml_bytes"],
                file_name="pathway.pwml", mime="application/xml", key="dl_pwml"
            )
            _dl_cols[1].download_button(
                "Download PWML IR JSON", data=json.dumps(_pwml_result.get("pwml_ir", {}), indent=2),
                file_name="final.pwml_ir.json", mime="application/json", key="dl_pwml_ir"
            )
            _dl_cols[2].download_button(
                "Download PWML IR report", data=json.dumps(_ir_report, indent=2),
                file_name="pwml_ir_report.json", mime="application/json", key="dl_pwml_ir_report"
            )
            _dl_cols[3].download_button(
                "Download validation report", data=json.dumps(_pwml_result["validation_report"], indent=2),
                file_name="pwml_validation_report.json", mime="application/json", key="dl_pwml_report"
            )
        else:
            st.error(f"PWML export failed: {_pwml_result.get('error', 'unknown')}")
            _gate_report = _safe_dict(_pwml_result.get("required_gate_report"))
            if _gate_report:
                _gate_summary = _safe_dict(_gate_report.get("summary"))
                _gate_errors = _safe_list(_gate_report.get("errors"))
                _gate_warnings = _safe_list(_gate_report.get("warnings"))
                st.error(
                    "Required-field gate errors: "
                    f"{_gate_summary.get('error_count', len(_gate_errors))}; "
                    f"warnings: {_gate_summary.get('warning_count', len(_gate_warnings))}."
                )
                if _gate_errors:
                    with st.expander("Required-field gate errors", expanded=True):
                        for _issue in _gate_errors[:100]:
                            if not isinstance(_issue, dict):
                                continue
                            _ptr = _issue.get("pointer", "")
                            _msg = _issue.get("message", "")
                            st.markdown(f"- `{_ptr}` - {_msg}" if _ptr else f"- {_msg}")
                            _render_issue_detail(_issue.get("detail"))
                st.download_button(
                    "Download required-field gate report",
                    data=json.dumps(_gate_report, indent=2),
                    file_name="pwml_required_field_gate_report.json",
                    mime="application/json",
                    key="dl_pwml_required_gate_failed",
                )
            if _ir_report:
                with st.expander("PWML IR report", expanded=False):
                    st.write("IR counts", _ir_report.get("counts", {}))
                    _ir_errors = _safe_list(_ir_report.get("errors"))
                    _blocking_ir_errors = blocking_pwml_ir_errors(_ir_report)
                    _non_blocking_ir_errors = [
                        e for e in _ir_errors if is_non_blocking_pwml_ir_error(e)
                    ]
                    if _blocking_ir_errors:
                        st.error("IR errors:\n" + "\n".join(str(e.get("message", e)) for e in _blocking_ir_errors))
                    if _non_blocking_ir_errors:
                        st.warning(
                            "Non-blocking PathWhiz DB row matching misses:\n"
                            + "\n".join(str(e.get("message", e)) for e in _non_blocking_ir_errors)
                        )
                    if _ir_report.get("warnings"):
                        st.warning("IR warnings:\n" + "\n".join(str(w.get("message", w)) for w in _ir_report["warnings"]))
                    st.write("Unresolved references", _safe_dict(_ir_report.get("unresolved")))
                st.download_button(
                    "Download PWML IR report",
                    data=json.dumps(_ir_report, indent=2),
                    file_name="pwml_ir_report.json",
                    mime="application/json",
                    key="dl_pwml_ir_report_failed",
                )
            if isinstance(_pwml_result.get("pwml_ir"), dict):
                with st.expander("PWML IR JSON (failed run)", expanded=False):
                    _ir_json_failed_str = json.dumps(_pwml_result.get("pwml_ir", {}), indent=2)
                    st.code(_ir_json_failed_str, language="json")
                st.download_button(
                    "Download PWML IR JSON",
                    data=json.dumps(_pwml_result.get("pwml_ir", {}), indent=2),
                    file_name="final.pwml_ir.json",
                    mime="application/json",
                    key="dl_pwml_ir_failed",
                )

    # ── Final Completeness Audit (optional, non-mutating) ───────────────────
    with st.expander("Final Completeness Audit", expanded=False):
        st.caption(
            "Run a stronger model to verify the final pathway JSON against the "
            "locked reaction manifest and source text. This is read-only and "
            "produces a report only."
        )
        _completeness_enabled = st.checkbox(
            "Enable final completeness audit",
            value=bool(os.getenv("ENABLE_FINAL_COMPLETENESS_AUDIT", "")),
            key="completeness_audit_enabled",
        )
        if st.button("Run Completeness Audit", key="run_completeness_audit_btn"):
            _source_text_for_audit = st.session_state.get("pipeline_source_text", "")
            if not _source_text_for_audit:
                st.warning("No source text available. Run the pipeline first.")
            else:
                _audit_final_json = (
                    st.session_state.get("refinement_working_json")
                    or st.session_state.get("final_payload")
                    or final_payload
                )
                _audit_manifest_dir = PROJECT_ROOT / "tmp"
                _audit_manifest_path = _audit_manifest_dir / "locked_reaction_manifest.json"
                _audit_pres_reports = _safe_dict(
                    _safe_dict(st.session_state.get("post_pipeline_artifacts")).get(
                        "reaction_preservation_reports"
                    )
                )
                _audit_pres_report = (
                    _audit_pres_reports.get("before_final_export")
                    or _audit_pres_reports.get("after_audit")
                )
                with st.spinner("Running final completeness audit..."):
                    try:
                        _completeness_report = run_final_completeness_audit(
                            source_text=_source_text_for_audit,
                            # strip_*: this context is json.dumps'd straight into
                            # the audit prompt, and the Stage 0 diagnostic block
                            # can hold a preview of an untrusted raw model reply.
                            preprocessor_context=strip_preprocess_status(
                                st.session_state.get("pathway_context")
                            ),
                            locked_manifest_path=_audit_manifest_path if _audit_manifest_path.exists() else None,
                            final_pathway_json=_audit_final_json,
                            final_preservation_report=_audit_pres_report,
                            output_path=_audit_manifest_dir,
                            enabled=_completeness_enabled,
                        )
                    except Exception as _ca_exc:
                        _completeness_report = {"enabled": True, "error": str(_ca_exc), "llm_ok": False}
                st.session_state["completeness_audit_report"] = _completeness_report
        _ca_report = st.session_state.get("completeness_audit_report")
        if isinstance(_ca_report, dict):
            if _ca_report.get("skipped"):
                _skip_reason = _ca_report.get("reason", "disabled")
                st.info(f"Completeness audit skipped: {_skip_reason}")
            elif _ca_report.get("llm_ok") is False:
                st.error(f"Completeness audit error: {_ca_report.get('error', 'unknown')}")
            else:
                _oc = _ca_report.get("overall_completeness", "unknown")
                _conf = _ca_report.get("confidence", 0.0)
                _pres = _ca_report.get("preserved_count", 0)
                _locked = _ca_report.get("locked_reaction_count", 0)
                if _oc in ("complete", "mostly_complete"):
                    st.success(f"Completeness: {_oc} (confidence {_conf:.2f}) - {_pres}/{_locked} locked reactions preserved")
                else:
                    st.warning(f"Completeness: {_oc} (confidence {_conf:.2f}) - {_pres}/{_locked} locked reactions preserved")
                _missing = _safe_list(_ca_report.get("missing_locked_reactions"))
                if _missing:
                    st.warning(f"{len(_missing)} missing locked reactions detected.")
                _possible_missed = _safe_list(_ca_report.get("possible_missed_in_scope_reactions"))
                if _possible_missed:
                    st.info(f"{len(_possible_missed)} possible missed in-scope reactions found.")
            st.json(_ca_report)
            st.download_button(
                "Download completeness audit report",
                data=json.dumps(_ca_report, indent=2, ensure_ascii=False),
                file_name="final_completeness_audit.json",
                mime="application/json",
                key="dl_completeness_audit",
            )

    render_pathwhiz_converter_section(llm_client_module)

    st.subheader("Connectivity snapshot")
    stats = graph_summary(final_payload)
    st.write(stats)
    if run_inference_from_state:
        st.write("Connectivity repair hints used for later rounds", build_qa_feedback(final_payload))

    st.subheader("JSON Artifact Viewer")
    _viewer_post_artifacts = st.session_state.get("post_pipeline_artifacts")
    if not isinstance(_viewer_post_artifacts, dict):
        _viewer_post_artifacts = {"final_payload_snapshot": final_payload}
    render_json_artifact_compare(
        _viewer_post_artifacts,
        st.session_state.get("pwml_export_result"),
        key_prefix="bottom_artifact_viewer",
    )

    with st.expander("Legacy SBML Export", expanded=False):
        st.caption("SBML is a legacy export path. Use PWML above for primary output.")
        if st.button("Run legacy SBML export", key="run_legacy_sbml_export_btn"):
            st.session_state.pop("post_pipeline_artifacts", None)
            post_artifacts = {}
            try:
                with st.spinner("Running legacy SBML export..."):
                    legacy_artifacts = run_post_pipeline_sbml_artifacts(
                        final_payload,
                        build_legacy_sbml=True,
                        use_llm_audit=bool(use_llm_audit),
                        use_sbml_overwatch=bool(use_sbml_overwatch),
                        default_compartment=(default_compartment or "cell").strip() or "cell",
                        mapping_cache_path=mapping_cache_text.strip() or "id_mapping_cache.json",
                        id_source=(id_source_mode or "hybrid").strip().lower(),
                        db_host=(db_host or "").strip(),
                        db_port=int(db_port),
                        db_user=(db_user or "").strip(),
                        db_password=db_password or "",
                        db_schema=(db_schema or "pathbank").strip() or "pathbank",
                        audit_max_rounds=int(audit_max_rounds),
                        audit_timeout_seconds=int(audit_timeout_seconds),
                        audit_candidate_count=int(audit_candidate_count),
                        use_example_retrieval=bool(use_example_retrieval),
                        example_index_path=(example_index_path or "").strip(),
                        example_top_k=int(example_top_k),
                        use_gap_resolver=bool(use_gap_resolver),
                        use_llm_gap_resolver=bool(use_llm_gap_resolver),
                        gap_resolver_max_items=int(gap_resolver_max_items),
                        qa_report=st.session_state.get("qa_report"),
                        reaction_summary=st.session_state.get("reaction_summary"),
                        use_stoich_agent=bool(use_stoich_agent),
                    )
                st.session_state["post_pipeline_artifacts"] = legacy_artifacts
                post_artifacts = legacy_artifacts
                st.success("Legacy SBML export completed.")
            except StageContractError as failure:
                _render_stage_contract_failure(
                    failure,
                    operation_label="Legacy SBML export",
                    download_key="dl_legacy_stage_contract_error",
                )
            except Exception as exc:
                st.error(f"Legacy SBML export failed: {exc}")
                st.exception(exc)  # full traceback (file:line) so the cause isn't a mystery
        if not isinstance(post_artifacts, dict):
            post_artifacts = {}
        if post_artifacts.get("sbml_xml_bytes"):
            st.download_button(
                "Download pathway.sbml",
                post_artifacts["sbml_xml_bytes"],
                file_name="pathway.sbml",
                mime="application/xml",
                key="dl_pathway_sbml",
            )
        if post_artifacts.get("sbml_render_ready_sbml_bytes"):
            st.download_button(
                "Download pathway.render_ready.sbml",
                post_artifacts["sbml_render_ready_sbml_bytes"],
                file_name="pathway.render_ready.sbml",
                mime="application/xml",
                key="dl_pathway_render_ready_sbml",
            )
        if post_artifacts.get("sbml_clean_bytes"):
            st.download_button(
                "Download pathway.render_ready.clean.sbml (unmapped entities removed)",
                post_artifacts["sbml_clean_bytes"],
                file_name="pathway.render_ready.clean.sbml",
                mime="application/xml",
                key="dl_pathway_render_ready_clean_sbml",
            )
            clean_summary = post_artifacts.get("sbml_clean_summary", {})
            if clean_summary:
                st.write("Clean SBML removal summary", clean_summary)
        if post_artifacts.get("sbml_diagram_png_bytes"):
            st.image(post_artifacts["sbml_diagram_png_bytes"], caption="Generated SBML diagram")
            st.download_button(
                "Download sbml_diagram.png",
                post_artifacts["sbml_diagram_png_bytes"],
                file_name="sbml_diagram.png",
                mime="image/png",
                key="dl_sbml_diagram_png",
            )
        elif str(post_artifacts.get("sbml_diagram_error", "")).strip():
            st.warning(f"SBML diagram render issue: {post_artifacts.get('sbml_diagram_error')}")
        if post_artifacts.get("sbml_report_json"):
            st.download_button(
                "Download sbml_validation_report.json",
                json.dumps(post_artifacts["sbml_report_json"], indent=2),
                file_name="sbml_validation_report.json",
                mime="application/json",
                key="dl_sbml_json",
            )
        if post_artifacts.get("sbml_report_txt"):
            st.download_button(
                "Download sbml_validation_report.txt",
                post_artifacts["sbml_report_txt"],
                file_name="sbml_validation_report.txt",
                mime="text/plain",
                key="dl_sbml_txt",
            )
        if post_artifacts.get("sbml_overwatch_report"):
            st.download_button(
                "Download sbml_overwatch_report.json",
                json.dumps(post_artifacts["sbml_overwatch_report"], indent=2),
                file_name="sbml_overwatch_report.json",
                mime="application/json",
                key="dl_sbml_overwatch",
            )
        checker_key = "post_pipeline_libsbml_check"
        if post_artifacts.get("sbml_xml_bytes") and st.button("Run libSBML checker on generated SBML", key="run_libsbml_checker_btn"):
            with st.spinner("Running libSBML checker..."):
                st.session_state[checker_key] = run_libsbml_checker(post_artifacts["sbml_xml_bytes"])
        checker_report = st.session_state.get(checker_key)
        if isinstance(checker_report, dict):
            st.write("libSBML checker summary", checker_report.get("validation", {}))
            if str(checker_report.get("error", "")).strip():
                st.error(str(checker_report.get("error", "")))
            st.download_button(
                "Download libsbml_checker_report.json",
                json.dumps(checker_report, indent=2),
                file_name="libsbml_checker_report.json",
                mime="application/json",
                key="dl_libsbml_checker",
            )
