import json
import inspect
import os
import re
import hashlib
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
from t2pw.curation.apply_audit_patch import run_apply
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
from t2pw.mapping.map_ids import run_mapping, PathBankDbResolver, resolve_mapping_gaps
from t2pw.sbml.render_pathwhiz_like import build_render_artifacts
from t2pw.pipeline.draft_graph_render import render_draft_graph_to_png_bytes
from t2pw.sbml.strip_unmapped import strip_unmapped
from t2pw.sbml.overwatch import run_sbml_overwatch
from t2pw.sbml.examples import build_retrieval_context, load_motif_index, payload_to_query_text
from t2pw.tools.pathwhiz_converter.ui import render_pathwhiz_converter_section
from t2pw.pipeline.process_normalizer import (
    GateValidationError,
    attach_transporters_from_evidence,
    canonicalize_same_as_aliases,
    cleanup_disallowed_complexes,
    compute_normalization_stats,
    dedupe_processes,
    ensure_autostates,
    normalize_composites,
    normalize_process_actor_schema,
    promote_catalysts,
    prune_disconnected_proteins,
    rewrite_reactions_to_complex_states,
    run_strict_post_normalization_gates,
)
from t2pw.pipeline.pipeline import (
    PipelineFailure,
    build_qa_feedback,
    build_and_save_draft_graph,
    merge_additions,
    propagate_context_organism,
    run_stage_two_with_feedback_loop,
    run_stage_one_with_chunking,
)
from t2pw.pipeline.draft_graph import build_draft_graph
from t2pw.pipeline.qa_graph import generate_qa_report
from t2pw.pipeline.reaction_summary import generate_reaction_summary
from t2pw.pipeline.preprocessor import is_ambiguous_multi_example_review_context, preprocess
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


def reset_refinement_state() -> None:
    for key, default in REFINEMENT_STATE_DEFAULTS.items():
        st.session_state[key] = deepcopy(default)


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
            ("Pre-normalized input", "pre_normalized_input.json", "pre_normalized_input"),
            ("Final audited", "final.audited.json", "final_audited"),
            ("Final mapped - DB mapping", "final.mapped.json", "final_mapped_db"),
            ("Final export input", "final.export_input.json", "final_export_input"),
            ("Mapping report", "mapping_report.json", "mapping_report"),
            ("Enrichment report", "enrichment_report.json", "enrichment_report"),
            ("Audit report", "audit_report.json", "audit_report"),
            ("Audit apply report", "audit_apply_report.json", "audit_apply_report"),
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
    if isinstance(pwml_result, dict):
        for label, filename, key in [
            ("PWML IR", "final.pwml_ir.json", "pwml_ir"),
            ("PWML IR report", "pwml_ir_report.json", "pwml_ir_report"),
            ("PWML IR validation", "pwml_ir_validation.json", "pwml_ir_validation"),
            ("PWML validation report", "pwml_validation_report.json", "validation_report"),
            ("PWML QA", "pwml_qa.json", "qa"),
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

def _norm_text(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def attach_enzymes_from_reaction_evidence(payload, report=None):
    """
    Attach proteins as enzymes if their name appears in the reaction evidence.
    """
    entities = payload.get("entities", {})
    processes = payload.get("processes", {})

    proteins = entities.get("proteins", [])
    reactions = processes.get("reactions", [])

    protein_names = []
    for p in proteins:
        if isinstance(p, dict):
            name = p.get("name")
            if name:
                protein_names.append(name)

    attached = 0

    for rxn in reactions:

        rxn.setdefault("enzymes", [])
        rxn.setdefault("modifiers", [])

        if rxn["enzymes"]:
            continue

        evidence_text = _norm_text(
            (rxn.get("name", "") + " " + rxn.get("evidence", ""))
        )

        matches = []

        for protein in protein_names:
            if _norm_text(protein) in evidence_text:
                matches.append(protein)

        if len(matches) == 1:
            rxn["enzymes"].append(matches[0])
            attached += 1

    if report is not None:
        summary = report.setdefault("summary", {})
        summary["enzymes_attached_from_reaction_evidence"] = attached

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
) -> Dict[str, Any]:
    project_root = PROJECT_ROOT
    cache_path = Path(mapping_cache_path)
    if not cache_path.is_absolute():
        cache_path = project_root / mapping_cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

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

        pre_normalization_input = deepcopy(final_payload)
        normalized_input = deepcopy(final_payload)
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
        try:
            normalize_composites(normalized_input, report=normalization_report)
            rewrite_reactions_to_complex_states(normalized_input, report=normalization_report)
            cleanup_disallowed_complexes(normalized_input, report=normalization_report)
            compute_normalization_stats(normalized_input, normalization_report)
            post_normalization_probe = {
                "normalization_stats": _safe_dict(normalization_report.get("summary")),
                "graph_summary": graph_summary(normalized_input),
                "payload": deepcopy(normalized_input),
            }
            post_normalization_probe_path.write_text(
                json.dumps(post_normalization_probe, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            ensure_autostates(normalized_input, report=normalization_report)
            attach_transporters_from_evidence(normalized_input, report=normalization_report)
            attach_enzymes_from_reaction_evidence(normalized_input, report=normalization_report)
            post_transport_attachment_probe = {
                "normalization_stats": _safe_dict(normalization_report.get("summary")),
                "graph_summary": graph_summary(normalized_input),
                "payload": deepcopy(normalized_input),
            }
            post_transport_attachment_probe_path.write_text(
                json.dumps(post_transport_attachment_probe, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            promote_catalysts(normalized_input, report=normalization_report)
            canonicalize_same_as_aliases(normalized_input, report=normalization_report)
            normalize_process_actor_schema(normalized_input, report=normalization_report)
            dedupe_processes(normalized_input, report=normalization_report)
            prune_disconnected_proteins(normalized_input, report=normalization_report)
            gate_snapshot = run_strict_post_normalization_gates(
                normalized_input,
                report=normalization_report,
                forbidden_complexes=["thyroglobulin:2-aminoacrylic acid"],
                enforce_all_proteins_connected=True,
            )
            gate_connectivity_summary = _safe_dict(gate_snapshot.get("connectivity"))
            post_dedupe_probe = {
                "normalization_stats": _safe_dict(gate_snapshot.get("normalization_stats")),
                "graph_summary": gate_connectivity_summary,
                "payload": deepcopy(normalized_input),
            }
            post_dedupe_probe_path.write_text(
                json.dumps(post_dedupe_probe, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except GateValidationError as exc:
            gate_details = _safe_dict(getattr(exc, "details", {}))
            gate_connectivity_summary = _safe_dict(gate_details.get("connectivity"))
            gate_fail_report = {
                "status": "failed",
                "stage": "post_normalization_hard_gates",
                "error": str(exc),
                "errors": _safe_list(gate_details.get("errors")),
                "normalization_stats": _safe_dict(gate_details.get("normalization_stats")),
                "connectivity": gate_connectivity_summary,
            }
            if not post_dedupe_probe:
                post_dedupe_probe = {
                    "normalization_stats": _safe_dict(gate_details.get("normalization_stats"))
                    or _safe_dict(normalization_report.get("summary")),
                    "graph_summary": gate_connectivity_summary or graph_summary(normalized_input),
                    "payload": deepcopy(normalized_input),
                }
                post_dedupe_probe_path.write_text(
                    json.dumps(post_dedupe_probe, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as exc:
            gate_fail_report = {
                "status": "failed",
                "stage": "post_normalization_hard_gates",
                "error": str(exc),
            }
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
        if gate_fail_report:
            gate_fail_report_path.write_text(json.dumps(gate_fail_report, indent=2, ensure_ascii=False), encoding="utf-8")
            return {
                "gate_failed": True,
                "gate_fail_report": gate_fail_report,
                "pre_normalization_input": pre_normalization_input,
                "pre_normalized_input": normalized_input,
                "pre_normalization_report": normalization_report,
                "post_normalization_probe": post_normalization_probe,
                "post_transport_attachment_probe": post_transport_attachment_probe,
                "post_dedupe_probe": post_dedupe_probe,
                "connectivity_summary": _safe_dict(post_dedupe_probe.get("graph_summary")),
                "audit_report": {"summary": {"error_count": 0, "warning_count": 0, "patch_count": 0}},
                "audit_patch": {},
                "audit_apply_report": {"summary": {"accepted_count": 0, "rejected_count": 0}},
                "final_audited": normalized_input,
                "final_mapped": normalized_input,
                "mapping_report": {"summary": {}},
                "enrichment_report": {"summary": {}},
                "sbml_report_json": {"counts": {}, "validation": {"has_errors": True}},
                "sbml_report_txt": "",
                "sbml_overwatch_report": {},
                "sbml_xml_bytes": b"",
                "sbml_diagram_png_bytes": b"",
                "sbml_diagram_error": "",
                "sbml_render_layout_summary": {},
                "sbml_render_ready_sbml_bytes": b"",
                "sbml_clean_bytes": b"",
                "sbml_clean_summary": {},
                "sbml_build_report": {},
                "mapping_cache_path": str(cache_path),
                "enrichment_cache_path": str(cache_path.with_name("enrichment_cache.json")),
                "enrichment_dump_path": str(project_root / "out" / "enrichment_dump.json"),
                "mapping_id_source": id_source,
                "mapping_db_host": db_host,
                "mapping_db_schema": db_schema,
                "example_retrieval_enabled": False,
                "example_retrieval_requested": bool(use_example_retrieval),
                "example_index_path": "",
                "example_index_error": "",
                "example_index_entry_count": 0,
                "audit_iterations": [],
                "gap_resolution_iterations": [],
                "audit_loop_summary": {
                    "rounds_executed": 0,
                    "max_rounds": 0,
                    "timeout_seconds": 0,
                    "stop_reason": "gate_failed",
                    "duration_seconds": 0,
                },
            }
        input_json.write_text(json.dumps(normalized_input, indent=2, ensure_ascii=False), encoding="utf-8")

        audit_iterations: List[Dict[str, Any]] = []
        gap_iterations: List[Dict[str, Any]] = []
        seen_hashes: set = set()
        current_input = input_json
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
        stop_reason = "max_rounds_reached"

        for round_idx in range(1, max_rounds + 1):
            elapsed_before = time.time() - audit_started_at
            if elapsed_before > timeout_seconds:
                stop_reason = "timeout"
                break

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
                    accepted_count=int(cand_apply_summary.get("accepted_count", 0)),
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
                        "accepted_patch_count": int(cand_apply_summary.get("accepted_count", 0)),
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
                    accepted_count=int(cand_apply_summary.get("accepted_count", 0)),
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
                        "accepted_patch_count": int(cand_apply_summary.get("accepted_count", 0)),
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
                    db_config={
                        "host": db_host,
                        "port": db_port,
                        "user": db_user,
                        "password": db_password,
                        "schema": db_schema,
                    },
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
            accepted_count = int(apply_summary.get("accepted_count", 0))
            rejected_count = int(apply_summary.get("rejected_count", 0))
            top_errors = [
                str(_safe_dict(item).get("reason", "")).strip()
                for item in _safe_list(round_audit.get("errors"))
                if isinstance(item, dict) and str(item.get("reason", "")).strip()
            ][:3]

            gap_summary = _safe_dict(gap_report_round.get("summary")) if isinstance(gap_report_round, dict) else {}
            mapped_ids_added = int(gap_summary.get("mapped_ids_added", 0))
            locations_added = int(gap_summary.get("locations_added", 0))
            states_filled = int(gap_summary.get("location_states_filled", 0))

            payload_hash = hashlib.sha1(current_after_round.read_bytes()).hexdigest()
            repeated_payload = payload_hash in seen_hashes
            seen_hashes.add(payload_hash)

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
                    "payload_repeated": repeated_payload,
                    "elapsed_seconds": round(time.time() - audit_started_at, 3),
                    "candidates": round_candidates,
                }
            )
            if use_gap_resolver:
                gap_iterations.append(
                    {
                        "round": round_idx,
                        "summary": gap_summary,
                        "db": _safe_dict(gap_report_round.get("db")),
                        "stage3": _safe_dict(gap_report_round.get("stage3")),
                    }
                )

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
            if repeated_payload:
                stop_reason = "loop_detected_same_payload"
                break
            if error_count == 0 and accepted_count == 0:
                stop_reason = "clean_no_pending_patch"
                break
            if error_count == 0 and patch_count == 0:
                stop_reason = "clean_no_patch"
                break
            if accepted_count == 0:
                stop_reason = "stalled_no_accepted_patch"
                break

            retry_context_note = (
                f"Previous attempt unresolved: errors={error_count}, warnings={warning_count}, "
                f"accepted_patches={accepted_count}. Prioritize remaining issues: "
                f"{'; '.join(top_errors) if top_errors else 'generic consistency fixes'}."
            )
        else:
            stop_reason = "max_rounds_reached"

        audited_json.write_text(current_input.read_text(encoding="utf-8"), encoding="utf-8")
        loop_duration = round(time.time() - audit_started_at, 3)

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

        run_mapping_params = inspect.signature(run_mapping).parameters
        mapping_kwargs: Dict[str, Any] = {"cache_path": cache_path}
        if "id_source" in run_mapping_params:
            mapping_kwargs["id_source"] = id_source
        if "db_config" in run_mapping_params:
            mapping_kwargs["db_config"] = {
                "host": db_host,
                "port": db_port,
                "user": db_user,
                "password": db_password,
                "schema": db_schema,
            }
        mapping_report = run_mapping(
            audited_json,
            mapped_json,
            mapping_report_path,
            **mapping_kwargs,
        )
        mapped_payload = json.loads(mapped_json.read_text(encoding="utf-8"))
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
                db_config={
                    "host": db_host,
                    "port": db_port,
                    "user": db_user,
                    "password": db_password,
                    "schema": db_schema,
                },
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

        return {
            "gate_failed": False,
            "gate_fail_report": {},
            "pre_normalization_input": pre_normalization_input,
            "pre_normalized_input": normalized_input,
            "pre_normalization_report": normalization_report,
            "post_normalization_probe": post_normalization_probe,
            "post_transport_attachment_probe": post_transport_attachment_probe,
            "post_dedupe_probe": post_dedupe_probe,
            "connectivity_summary": gate_connectivity_summary or _safe_dict(post_dedupe_probe.get("graph_summary")),
            "audit_report": json.loads(audit_report_path.read_text(encoding="utf-8")),
            "audit_patch": json.loads(audit_patch_path.read_text(encoding="utf-8")),
            "audit_apply_report": json.loads(apply_report_path.read_text(encoding="utf-8")),
            "final_audited": json.loads(audited_json.read_text(encoding="utf-8")),
            "final_mapped": json.loads(sbml_input_path.read_text(encoding="utf-8")),
            "final_mapped_db": json.loads(mapped_json.read_text(encoding="utf-8")),
            "final_export_input": json.loads(sbml_input_path.read_text(encoding="utf-8")),
            "mapping_report": mapping_report,
            "enrichment_report": enrichment_report,
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
            "audit_loop_summary": {
                "rounds_executed": len(audit_iterations),
                "max_rounds": max_rounds,
                "timeout_seconds": timeout_seconds,
                "stop_reason": stop_reason,
                "duration_seconds": loop_duration,
            },
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
) -> Dict[str, Any]:
    try:
        payload = deepcopy(final_payload)
        grounding_report: Dict[str, Any] = {}
        if grounding_dict:
            payload, grounding_report = apply_grounding(payload, grounding_dict)
        ensure_autostates(payload)

        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
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
        required_gate_report = validate_required_pwml_contract(gate_payload, strict_db=bool(strict_db))
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
                "required_gate_report": required_gate_report,
                "required_gate_report_path": str(required_gate_path),
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
                "required_gate_report": required_gate_report,
                "required_gate_report_path": str(required_gate_path),
                "pwml_ir": pwml_ir,
                "pwml_ir_report": ir_report,
                "pwml_ir_validation": ir_validation,
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
        return {
            "ok": True,
            "counts": build_result.counts,
            "issues": report["issue_count"],
            "output_path": str(out_path),
            "xml_bytes": xml_bytes,
            "validation_report": report,
            "qa": qa_report,
            "grounding_report": grounding_report,
            "required_gate_report": required_gate_report,
            "required_gate_report_path": str(required_gate_path),
            "pwml_ir": pwml_ir,
            "pwml_ir_report": ir_report,
            "pwml_ir_validation": ir_validation,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "counts": {}, "issues": 0,
                "output_path": "", "qa": {}, "grounding_report": {}}


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

    species_hint = st.text_input(
        "Species (optional)",
        value="",
        key="species_hint_input",
        help=(
            "Organism to attach to biological_states that don't have one "
            "(e.g. 'Homo sapiens', 'Mus musculus', 'Saccharomyces cerevisiae', "
            "'Escherichia coli'). Leave blank if the source text names the organism — "
            "the preprocessor will detect it."
        ),
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

    # Preprocessing: lightweight context summary to guide extraction and inference
    with st.spinner("Running preprocessor..."):
        pathway_context = preprocess(text, temperature=temperature)

    _species_override = (species_hint or "").strip()
    if _species_override:
        if not isinstance(pathway_context, dict):
            pathway_context = {}
        pathway_context["likely_organism"] = _species_override

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
            "Choose one candidate example and rerun extraction."
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
                enable_chunking=enable_chunking,
                chunk_word_limit=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                max_attempts=int(extract_attempts),
                temperature=temperature,
                max_tokens=int(extract_tokens),
            )
    except PipelineFailure as failure:
        st.error(f"Extraction failed: {failure}")
        render_attempts("Stage 1 attempts", failure.attempts)
        st.stop()

    final_payload = stage_one
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
                    stage_one,
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
        final_payload = merge_additions(stage_one, stage_two if isinstance(stage_two, dict) else {})

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
    st.session_state["chunk_details"] = chunk_details
    st.session_state["stage_two"] = stage_two
    st.session_state["stage_two_chunks"] = stage_two_chunks
    st.session_state["stage_two_rounds"] = stage_two_rounds
    st.session_state["qa_hints"] = qa_hints
    st.session_state["final_payload"] = final_payload
    st.session_state["final_payload_snapshot"] = final_payload
    st.session_state.pop("post_pipeline_artifacts", None)
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
        audit_summary = post_artifacts.get("audit_report", {}).get("summary", {})
        mapping_summary = post_artifacts.get("mapping_report", {}).get("summary", {})
        enrichment_summary = post_artifacts.get("enrichment_report", {}).get("summary", {})
        sbml_summary = post_artifacts.get("sbml_report_json", {}).get("counts", {})
        sbml_validation = post_artifacts.get("sbml_report_json", {}).get("validation", {})
        sbml_overwatch_summary = post_artifacts.get("sbml_overwatch_report", {}).get("summary", {})
        sbml_layout_summary = _safe_dict(post_artifacts.get("sbml_render_layout_summary"))
        stoich_audit_log = post_artifacts.get("stoich_audit_log") or []
        stoich_additions_made = sum(1 for e in stoich_audit_log if e.get("llm_verdict") == "add")
        stoich_audits_reversed = sum(1 for e in stoich_audit_log if e.get("audit_verdict") == "reversed")

        st.write(
            {
                "normalization_stats": _safe_dict(post_artifacts.get("pre_normalization_report")).get("summary", {}),
                "connectivity": _safe_dict(post_artifacts.get("connectivity_summary"))
                or _safe_dict(post_artifacts.get("post_dedupe_probe") or post_artifacts.get("post_normalization_probe")).get("graph_summary", {}),
                "gate_failed": gate_failed,
                "gate_fail_report": post_artifacts.get("gate_fail_report", {}),
                "audit": audit_summary,
                "mapping": mapping_summary,
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
            st.error(
                f"Hard-gate failure before audit/mapping/SBML: "
                f"{_safe_dict(post_artifacts.get('gate_fail_report')).get('error', 'unknown error')}"
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
        if gate_failed:
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
        _pwml_grounding_dict = st.session_state.get("pwml_grounding_dict")

    if st.button("Run audit and DB mapping", key="pwml_generate_btn"):
        if not isinstance(final_payload, dict) or not final_payload:
            st.error("No pipeline output in session state. Run the pipeline first.")
        else:
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
                    final_mapped_payload = _pa.get("final_mapped_db") or _pa.get("final_mapped")
                    mapping_report = _safe_dict(_pa.get("mapping_report"))
                    if isinstance(final_mapped_payload, dict) and final_mapped_payload:
                        initialize_refinement_review_state(final_mapped_payload, mapping_report)
                    st.info("Mapped pathway is ready for review. PWML generation is paused until approval.")
            except Exception as exc:
                st.error(f"Post-pipeline conversion failed: {exc}")

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
            except Exception as exc:
                st.error(f"Legacy SBML export failed: {exc}")
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
