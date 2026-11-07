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
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict
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
    reenter as reenter_extraction_diagnostics,
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

# R1: the post-pipeline/export code lives in t2pw.service.postpipeline; every moved name is
# re-imported here under the same name so streamlit_app.<name> keeps resolving.
from t2pw.service.postpipeline import (
    CanonicalFreezeResult,
    freeze_canonical_payload,
    run_post_pipeline_sbml_artifacts,
    _validate_stage8_export_payload,
    run_pwml_export,
    run_quarantine_boundary,
    graph_summary,
    _safe_dict,
    _safe_list,
    _BULKY_ROUND_KEYS,
    _round_summary,
    _record_audit_round,
    _record_gap_round,
    _record_mapped_failure_snapshot,
    _save_pipeline_outputs,
    _write_reaction_preservation_report,
    _read_json,
    _audit_objective_score,
)

# R2: the RAG orchestration lives in t2pw.service.rag_orchestration; every moved name is
# re-imported here under the same name so streamlit_app.<name> keeps resolving.
from t2pw.service.rag_orchestration import (
    RagOrchestrationResult,
    _count_reactions,
    _post_extraction_error_keys,
    _post_extraction_new_error_keys,
    maybe_run_rag,
    RAG_ROUND_SCRATCH_DIRNAME,
    RAG_LOOP_MAX_ROUNDS_ENV,
    RAG_LOOP_DEADLINE_ENV,
    _MISSING,
    QUARANTINE_SESSION_KEYS,
    rag_loop_max_rounds,
    rag_loop_deadline_seconds,
    rag_round_scratch_dir,
    rag_round_mapping_cache_path,
    rag_round_reports,
    build_rag_round_stages,
    rag_round_merge,
    _rag_round_claims,
    _rag_round_gap_ids,
    _rag_reentry_map,
    run_rag_rounds,
)
from t2pw.service.reporter import StreamlitReporter, install_app_defaults
from t2pw.service.extraction import (
    DIAGNOSTICS_DIRNAME,
    EXTRACTION_DIAGNOSTICS_DIR_KEY,
    EXTRACTION_DIAGNOSTICS_KEY,
    _PREPROCESS_RETRY_CHARS,
    _has_usable_context,
    _run_stage_zero_with_retry as _service_run_stage_zero_with_retry,
    _seed_context_from_user_scope,
    _stage0_context_rank,
    _last_cleaning_report,
    _sync_extraction_diagnostics as _service_sync_extraction_diagnostics,
    assemble_inputs,
    render_attempts as _service_render_attempts,
    run_stage_one,
    run_stage_zero,
)

# R2: a caller that reaches the service through this module without passing a reporter
# or a state mapping keeps today's behaviour: Streamlit calls and st.session_state.
install_app_defaults(reporter=lambda: StreamlitReporter(st), state=lambda: st.session_state)

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


def _run_stage_zero_with_retry(
    text: str,
    *,
    temperature: float,
    user_task_context: Optional[str],
) -> Dict[str, Any]:
    """R3b: moved to ``t2pw.service.extraction``. This name stays, with this module's
    ``preprocess`` passed in, so a patch of ``streamlit_app.preprocess`` still lands."""
    return _service_run_stage_zero_with_retry(
        text, temperature=temperature, user_task_context=user_task_context,
        preprocess_fn=preprocess,
    )


def emit_rag_loop_messages(messages: Any) -> None:
    """Render the loop's collected UI lines. Best-effort UI only."""

    reporter = StreamlitReporter(st)
    for kind, body in _safe_list(messages):
        if kind == "info":
            reporter.info(body)
        elif kind == "warning":
            reporter.warning(body)
        elif kind == "error":
            reporter.error(body)
        elif kind == "json":
            reporter.json(body)


def rag_loop_session_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """The JSON-safe half of the loop record, for session state and diagnostics.

    ``payload`` is deliberately excluded: the graph the loop settled on is
    ``final_payload``, and a second copy under another key would be a second
    authority for the same fact.
    """

    return {
        key: value for key, value in _safe_dict(record).items()
        if key not in {"payload", "messages"}
    }


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
    """R3c: moved to ``t2pw.service.extraction``; the app's Stage-2 and chunk listings keep this name."""
    _service_render_attempts(label, attempts, reporter=StreamlitReporter(st))


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


logger = logging.getLogger(__name__)


def _bounded_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Read an integer UI default from the environment and keep it in range."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d.", name, raw, default)
        return default
    if value < minimum or value > maximum:
        clamped = min(max(value, minimum), maximum)
        logger.warning(
            "Clamping %s=%d to the supported UI range %d..%d.",
            name,
            value,
            minimum,
            maximum,
        )
        return clamped
    return value


def _sync_extraction_diagnostics() -> Dict[str, Any]:
    """R3b: moved to ``t2pw.service.extraction``; the app's remaining call sites keep this name."""
    return _service_sync_extraction_diagnostics(state=st.session_state)


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
            # C-052's ONE rendering entry. It is placed on the post-pipeline side
            # deliberately: that seam persists nothing (its working directory is
            # swept before it returns), so this panel is the operator's only route
            # to its PRE-freeze resolution record. The third export seam's own,
            # POST-freeze record is reachable as a file on disk and is a different
            # document, which is why it is not folded in beside this one.
            #
            # The download name carries no ``pwml`` prefix, so it cannot be
            # confused with the CLI's ``pwml_prefreeze_resolution_report.json`` or
            # with the third seam's ``...streamlit.json``. Three records, three
            # names, one shape.
            (
                "Pre-freeze resolution report",
                "prefreeze_resolution_report.json",
                "prefreeze_resolution_report",
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
        value=_bounded_env_int(
            "OPENROUTER_EXTRACTION_MAX_TOKENS",
            64000,
            minimum=500,
            maximum=128000,
        ),
        step=100,
    )
    infer_tokens = col_tokens_2.number_input(
        "Stage 2 max tokens",
        min_value=500,
        max_value=128000,
        value=_bounded_env_int(
            "OPENROUTER_INFERENCE_MAX_TOKENS",
            64000,
            minimum=500,
            maximum=128000,
        ),
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
    user_task_context = (user_task_context or "").strip() or None

    # R3a: input assembly lives in t2pw.service.extraction.assemble_inputs; each
    # returned failure is one of the st.stop() points that used to be inline here.
    _assembled = assemble_inputs(
        text_entries,
        [(_up.name, _up.read()) for _up in (uploaded_pdfs or [])],
        uses_pdf_input=uses_pdf_input,
        input_mode=input_mode,
        pdf_page_range=pdf_page_range,
        pdf_skip_refs=pdf_skip_refs,
        pdf_ocr=pdf_ocr,
        reporter=StreamlitReporter(st),
    )
    if _assembled.failure is not None:
        st.stop()
    text = _assembled.text

    reset_refinement_state()

    # R3b: Stage 0 lives in t2pw.service.extraction.run_stage_zero: the diagnostics
    # start, the preprocessor and its retry, the seed / recovered messages and the
    # post-Stage-0 diagnostics sync. The ambiguous-review gate below stays here.
    _stage_zero = run_stage_zero(
        text,
        temperature=temperature,
        user_task_context=user_task_context,
        state=st.session_state,
        reporter=StreamlitReporter(st),
        preprocess_fn=preprocess,
    )
    pathway_context = _stage_zero.pathway_context
    _diagnostics_dir = _stage_zero.diagnostics_dir

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

    # R3c: Stage 1 lives in t2pw.service.extraction.run_stage_one: the extraction and
    # its boundary, the three failure exits (each returned, the app stops), the
    # out-of-scope filter and the RAG seed with its user_task_context fold.
    _stage_one = run_stage_one(
        text,
        pathway_context=pathway_context,
        user_task_context=user_task_context,
        diagnostics_dir=_diagnostics_dir,
        enable_chunking=enable_chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        extract_attempts=extract_attempts,
        temperature=temperature,
        extract_tokens=extract_tokens,
        rag_incomplete_flag=rag_incomplete_flag,
        state=st.session_state,
        reporter=StreamlitReporter(st),
    )
    if _stage_one.failure is not None:
        st.stop()
    stage_one = _stage_one.stage_one
    chunk_details = _stage_one.chunk_details
    stage_one_in_scope = _stage_one.stage_one_in_scope
    out_of_scope_removed_reactions = _stage_one.out_of_scope_removed_reactions
    rag_result = _stage_one.rag_result
    user_task_context = _stage_one.user_task_context

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
        # The C-060 gate's advisory phase gates on isinstance(context,
        # PathwayContext) and `pathway_context` here is the Stage-0 DICT, so
        # forwarding it would type-check and do nothing. The factory builds the
        # real frozen context from the one derivation maybe_run_rag also uses. An
        # empty Stage 0 yields PathwayContext(""), which the gate already coerces
        # a missing context to, so no new refusal path appears. `seed_text` is
        # deliberately not supplied -- see the pipeline.py site for why.
        from t2pw.pipeline.entity_admission import pathway_context_from_stage_zero

        final_payload = merge_additions(
            stage_one_in_scope,
            stage_two if isinstance(stage_two, dict) else {},
            pathway_context=pathway_context_from_stage_zero(pathway_context),
            source_text=text,
        )

    # S3 — inject R5's synthesized reactions/entities INTO the seed payload rather
    # than REPLACING it, ONE ROUND AT A TIME. A no-op with RAG off (rag_result is
    # None): final_payload is untouched and t2pw.rag.controller is never imported.
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
    #
    # That conform/merge/cleanup/schema-gate sequence is unchanged; what changed is
    # that it is now the body of ONE ROUND of ``t2pw.rag.controller.run_rag_loop``
    # (``run_rag_rounds`` above) instead of a single straight-line pass. D-008
    # (LOCKED) and PRODUCT_CONTRACT §10: every RAG round must re-enter normalization,
    # mapping, gates, persistence and classification, the loop must be deadline-aware,
    # and it must checkpoint before each round. The controller enforces all of that
    # structurally; this site supplies only ``stages=``, ``round_runner=``,
    # ``deadline=`` and ``checkpoint=``, per the separation invariant at :161.
    #
    # WHY THE CHAIN STAYS AT :5514 AND THE MERGE HAPPENS HERE. Round 0's chain must
    # run BEFORE Stage 2, because its ``evidence_context`` is folded into Stage 2's
    # prompt through S1/S2; the merge must run AFTER it, because the merge base is
    # Stage 1 PLUS Stage 2 whenever inference ran. So round 0 REUSES the result
    # computed there, and only rounds 1..N-1 re-enter the chain — against their own
    # graph and their own recomputed reports.
    #
    # ``RAG_LOOP_MAX_ROUNDS`` defaults to 1, so an unchanged environment runs exactly
    # one round and produces exactly today's merge, today's messages and today's
    # final_payload.
    #
    # THE MERGE STATEMENT STAYS HERE, and the loop calls it. C-060a's wiring test
    # lifts the app's two ``merge_additions`` statements out of this file BY AST and
    # executes the real source bytes, to prove each passes a real PathwayContext
    # (tests/test_c060a_requested_pathway_wiring.py:68-104; both must sit at or
    # above OWNED_FLOOR = 4640, Stage 2's first). Writing the RAG merge inside a
    # helper defined above would put it out of that instrument's reach and disarm
    # C-060a's protection without changing a line of C-060a's test.
    def _rag_merge_site(final_payload, rag_envelope, pathway_context):
        # The RAG leg runs merge_additions a SECOND time, so the same request has
        # to reach the gate on this pass too. Without it the rows RAG adds are
        # screened with no context while the Stage-2 rows were screened with one,
        # and the two passes would disagree about the same species.
        from t2pw.pipeline.entity_admission import pathway_context_from_stage_zero

        merged = merge_additions(
            final_payload,
            rag_envelope,
            pathway_context=pathway_context_from_stage_zero(pathway_context),
        )
        return merged

    if rag_result is not None and rag_result.synthesized and rag_result.payload:
        rag_loop_record = run_rag_rounds(
            final_payload,
            first_result=rag_result,
            merge_site=_rag_merge_site,
            pathway_context=pathway_context,
            user_flag=bool(rag_incomplete_flag),
            seed_text=text,
            export_mode=export_mode,
            strict_db=bool(st.session_state.get("pwml_strict_db", True)),
            reporter=StreamlitReporter(st),
            state=st.session_state,
        )
        final_payload = rag_loop_record["payload"]
        emit_rag_loop_messages(rag_loop_record["messages"])
        # Additive only. ``RagOrchestrationResult`` is a bare SimpleNamespace read
        # through ``getattr(..., default)``, and ``payload`` keeps its meaning — the
        # PRE-MERGE synthesized payload the citation report reads. The loop record is
        # a NEW field beside it, never a redefinition of that one.
        st.session_state["rag_loop"] = rag_loop_session_record(rag_loop_record)
        try:
            rag_result.loop = st.session_state["rag_loop"]
        except Exception:  # noqa: BLE001 - a diagnostic attachment is never fatal
            pass

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
    st.session_state["extraction_diagnostics_recorder"] = current_extraction_diagnostics()
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
            reenter_extraction_diagnostics(st.session_state.get("extraction_diagnostics_recorder"))
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
                        state=st.session_state,
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
                        state=st.session_state,
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

# -- Reaction Evidence (ORCH-742b) --------------------------------------------
# Read-only review of the run's existing artifacts plus a deterministic .txt
# export. Module level, outside the pipeline_ready block, so a saved run
# directory can be reviewed with no run loaded. Presentation only.
from t2pw.app.reaction_evidence_view import render_reaction_evidence  # noqa: E402

render_reaction_evidence()
