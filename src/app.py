import json
import inspect
import os
import re
import hashlib
import time
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

import streamlit as st
from lxml import etree

import llm_client as llm_client_module
from apply_audit_patch import run_apply
from audit_json_llm import run_audit
from enrich_entities import run_enrichment
from grounding import apply_grounding
from gap_resolver import run_gap_resolution
from json_to_sbml import build_sbml
from map_ids import run_mapping
from sbml_render_pathwhiz_like import build_render_artifacts
from draft_graph_render import render_draft_graph_to_png_bytes
from sbml_strip_unmapped import strip_unmapped
from sbml_overwatch import run_sbml_overwatch
from sbml_examples import build_retrieval_context, load_motif_index, payload_to_query_text
from tools.pathwhiz_converter.ui import render_pathwhiz_converter_section
from process_normalizer import (
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
from pipeline import (
    PipelineFailure,
    build_qa_feedback,
    build_and_save_draft_graph,
    merge_additions,
    run_stage_two_with_feedback_loop,
    run_stage_one_with_chunking,
)
from preprocessor import preprocess
from pdf_parser import extract_text_from_pdf, get_pdf_info, parse_pdf, SKIP_SECTIONS
from pwml_validate import discover_structure_signature, repair_tree, validate_generated_tree
from pwml_writer import DeterministicPwmlBuilder
from qa_graph import build_graph, connected_components, degrees, get_entities, node

st.set_page_config(page_title="PWML Multi-Stage Pipeline", layout="wide")
st.title("PWML Extraction -> Inference Pipeline (LM Studio)")


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
    project_root = Path(__file__).resolve().parent.parent
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
) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent
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
                    reaction_summary=st.session_state.get("reaction_summary"),
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
        sbml_overwatch_report: Dict[str, Any] = {}
        if use_sbml_overwatch:
            sbml_overwatch_report = run_sbml_overwatch(
                sbml_input_path,
                sbml_path,
                sbml_report_json_path,
                sbml_overwatch_path,
                use_llm=True,
                llm_max_tokens=3000,
            )
        sbml_diagram_png_bytes = b""
        sbml_diagram_error = ""
        sbml_render_layout_summary: Dict[str, Any] = {}
        sbml_render_ready_sbml_bytes = b""
        sbml_clean_bytes = b""
        sbml_clean_summary: Dict[str, Any] = {}
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
            "mapping_report": mapping_report,
            "enrichment_report": enrichment_report,
            "sbml_report_json": json.loads(sbml_report_json_path.read_text(encoding="utf-8")),
            "sbml_report_txt": sbml_report_txt_path.read_text(encoding="utf-8"),
            "sbml_overwatch_report": sbml_overwatch_report,
            "sbml_xml_bytes": sbml_path.read_bytes(),
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
            ),
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


# ── Input mode (OUTSIDE form — triggers immediate re-render on click) ──────
input_mode = st.radio(
    "Input mode",
    ["Paste text", "Upload PDF"],
    horizontal=True,
    key="input_mode_radio",
)

# ── PDF controls (OUTSIDE form — file_uploader is banned inside st.form) ───
uploaded_pdf   = None
pdf_page_range = ""
pdf_skip_refs  = True
pdf_ocr        = False

if input_mode == "Upload PDF":
    uploaded_pdf = st.file_uploader(
        "Upload a scientific PDF (research paper, pathway description, etc.)",
        type=["pdf"],
        key="pdf_upload_widget",
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
    if input_mode == "Paste text":
        text = st.text_area("Paste pathway description:", height=220)
    else:
        text = ""   # populated after submit from the PDF
        if uploaded_pdf is not None:
            st.info(
                f"PDF ready: **{uploaded_pdf.name}**  "
                f"— configure options above, then click **Run pipeline**."
            )
        else:
            st.warning("Upload a PDF using the file uploader above, then click **Run pipeline**.")

        
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
    # PDF extraction runs here — outside the form, so uploaded_pdf is accessible
    if input_mode == "Upload PDF":
        if uploaded_pdf is None:
            st.warning("Please upload a PDF using the file uploader above.")
            st.stop()

        import tempfile, os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as _tmp:
            _tmp.write(uploaded_pdf.read())
            _tmp_path = _tmp.name

        try:
            # Parse page range
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
            st.error(f"PDF extraction failed: {_pdf['error']}")
            st.stop()

        text = _pdf["text"]

        for _w in _pdf.get("warnings", []):
            st.warning(f"PDF: {_w}")

        st.success(
            f"Extracted **{_pdf['pages_used']}** of **{_pdf['total_pages']}** pages "
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

        with st.expander("Preview extracted text (first 1000 chars)", expanded=False):
            st.text(text[:1000] + ("..." if len(text) > 1000 else ""))

    if not text.strip():
        st.warning("No text to process. Paste text or upload a PDF.")
        st.stop()

    # Preprocessing: lightweight context summary to guide extraction and inference
    with st.spinner("Running preprocessor..."):
        pathway_context = preprocess(text, temperature=temperature)

    # Stage 1: strict extraction with auto-repair
    try:
        with st.spinner("Running Stage 1 extraction..."):
            stage_one, chunk_details = run_stage_one_with_chunking(
                text,
                pathway_context=pathway_context,
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
    try:
        st.session_state["draft_graph_png_bytes"] = render_draft_graph_to_png_bytes(draft_graph.to_dict())
    except Exception as _dg_exc:
        st.session_state["draft_graph_png_bytes"] = b""
        st.session_state["draft_graph_render_error"] = str(_dg_exc)
    st.session_state["qa_report"] = qa_report
    st.session_state["reaction_summary"] = reaction_summary

    st.session_state["pipeline_ready"] = True
    st.session_state["run_inference_enabled"] = bool(run_inference)
    st.session_state["pathway_context"] = pathway_context
    st.session_state["stage_one"] = stage_one
    st.session_state["chunk_details"] = chunk_details
    st.session_state["stage_two"] = stage_two
    st.session_state["stage_two_chunks"] = stage_two_chunks
    st.session_state["stage_two_rounds"] = stage_two_rounds
    st.session_state["qa_hints"] = qa_hints
    st.session_state["final_payload"] = final_payload
    st.session_state.pop("post_pipeline_artifacts", None)

if st.session_state.get("pipeline_ready"):
    run_inference_from_state = bool(st.session_state.get("run_inference_enabled", False))
    stage_one = st.session_state.get("stage_one", {})
    chunk_details = st.session_state.get("chunk_details", [])
    stage_two = st.session_state.get("stage_two")
    stage_two_chunks = st.session_state.get("stage_two_chunks", [])
    stage_two_rounds = st.session_state.get("stage_two_rounds", [])
    qa_hints = st.session_state.get("qa_hints")
    final_payload = st.session_state.get("final_payload", {})

    st.subheader("Stage 1 - Strict extraction")
    st.json(stage_one)
    st.caption(f"Stage 1 QA: {qa_summary_line(stage_one)}")
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
        st.json(stage_two)
        if stage_two_rounds:
            st.write("Stage 2 QA rounds", stage_two_rounds)
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
        st.download_button(
            "Download Stage 2 additions",
            json.dumps(stage_two, indent=2),
            file_name="stage2_additions.json",
            mime="application/json",
        )
        if qa_hints:
            st.write("QA hints", qa_hints)

    st.subheader("Final merged output")
    st.json(final_payload)
    st.caption(f"Final QA: {qa_summary_line(final_payload)}")
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

        if st.button("Render pathway graph", key="btn_render_draft_graph"):
            try:
                st.session_state["draft_graph_png_bytes"] = render_draft_graph_to_png_bytes(draft_graph_dict)
                st.session_state.pop("draft_graph_render_error", None)
            except Exception as _dg_exc:
                st.session_state["draft_graph_png_bytes"] = b""
                st.session_state["draft_graph_render_error"] = str(_dg_exc)

        dg_png = st.session_state.get("draft_graph_png_bytes", b"")
        dg_render_err = st.session_state.get("draft_graph_render_error", "")
        if dg_png:
            st.image(dg_png, caption="Pathway graph (graphviz)")
            st.download_button(
                "Download graph diagram",
                dg_png,
                file_name="pathway_graph.png",
                mime="image/png",
                key="dl_draft_graph_png",
            )
        elif dg_render_err:
            st.warning(f"Graph render failed: {dg_render_err}")

    # ------------------------------------------------------------------ QA Report
    st.subheader("QA Report")
    qa_report_data = st.session_state.get("qa_report", {})
    if qa_report_data:
        qa_summary = qa_report_data.get("summary", {})
        qa_flags = qa_report_data.get("flags", {})

        qr_col1, qr_col2, qr_col3 = st.columns(3)
        qr_col1.metric("Total species", qa_summary.get("total_species", 0))
        qr_col2.metric("Total reactions", qa_summary.get("total_reactions", 0))
        qr_col3.metric(
            "Completeness score",
            f"{qa_summary.get('completeness_score', 0.0):.3f}",
            help="1.0 = no structural issues detected. Lower = more flags.",
        )

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

    st.subheader("Post-pipeline SBML export")
    post_col_a, post_col_b = st.columns(2)
    use_llm_audit = post_col_a.checkbox(
        "Use LLM in audit stage",
        value=True,
        help="Disabling runs deterministic audit rules only.",
        key="post_use_llm_audit",
    )
    use_sbml_overwatch = post_col_a.checkbox(
        "Use SBML semantic overwatch",
        value=True,
        help="Runs deterministic + LLM semantic review on generated SBML.",
        key="post_use_sbml_overwatch",
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
        "Use SBML motif retrieval",
        value=True,
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

    if st.button("Run post-pipeline SBML conversion"):
        try:
            with st.spinner("Running audit, patching, ID mapping, and SBML build..."):
                artifacts = run_post_pipeline_sbml_artifacts(
                    final_payload,
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
                )
            st.session_state["post_pipeline_artifacts"] = artifacts
            if bool(_safe_dict(artifacts).get("gate_failed", False)):
                st.warning("Post-pipeline stopped at hard-gate validation. Review gate_fail_report.json.")
            else:
                st.success("Post-pipeline conversion completed.")
        except Exception as exc:
            st.error(f"Post-pipeline conversion failed: {exc}")

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
            json.dumps(post_artifacts["final_mapped"], indent=2),
            file_name="final.mapped.json",
            mime="application/json",
            key="dl_final_mapped",
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
                with st.expander("Clean SBML removal summary"):
                    st.write(f"Compartments removed: {clean_summary.get('total_removed_compartments', 0)}")
                    st.write(f"Species removed: {clean_summary.get('total_removed_species', 0)}")
                    st.write(f"Reactions removed (no ID): {clean_summary.get('total_removed_reactions', 0)}")
                    st.write(f"Reactions removed (cascade): {clean_summary.get('total_cascade_removed_reactions', 0)}")
                    if clean_summary.get("removed_species"):
                        st.json(clean_summary["removed_species"])
                    if clean_summary.get("removed_reactions") or clean_summary.get("cascade_removed_reactions"):
                        st.json(
                            clean_summary.get("removed_reactions", [])
                            + clean_summary.get("cascade_removed_reactions", [])
                        )
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
        st.download_button(
            "Download sbml_validation_report.json",
            json.dumps(post_artifacts["sbml_report_json"], indent=2),
            file_name="sbml_validation_report.json",
            mime="application/json",
            key="dl_sbml_json",
        )
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

    render_pathwhiz_converter_section(llm_client_module)

    st.subheader("PWML export")
    pwml_col_a, pwml_col_b = st.columns(2)
    ref_path_text = pwml_col_a.text_input("Reference PWML path", value="reference/PW012926.pwml")
    grounding_enabled = pwml_col_b.checkbox("Apply deterministic grounding before PWML export", value=False)

    grounding_path_text = ""
    if grounding_enabled:
        grounding_path_text = st.text_input("Grounding dictionary path", value="data/grounding_dictionary.example.json")

    meta_col_1, meta_col_2, meta_col_3 = st.columns(3)
    pw_id = meta_col_1.text_input("pw-id", value="PW000XYZ")
    named_for_id = meta_col_2.number_input("named-for-id", min_value=1, max_value=99999999, value=123, step=1)
    subject = meta_col_3.text_input("subject", value="Metabolic")

    naming_col_1, naming_col_2 = st.columns(2)
    pathway_name = naming_col_1.text_input("Pathway name", value="Generated Pathway")
    pathway_description = naming_col_2.text_input("Pathway description", value="")

    vis_col_1, vis_col_2, vis_col_3 = st.columns(3)
    vis_width = vis_col_1.number_input("Visualization width", min_value=200, max_value=10000, value=3200, step=100)
    vis_height = vis_col_2.number_input("Visualization height", min_value=200, max_value=10000, value=1400, step=100)
    background_color = vis_col_3.text_input("Background color", value="#FFFFFF")

    if st.button("Generate PWML from final JSON"):
        payload_for_writer = final_payload
        if grounding_enabled:
            try:
                grounding_path = resolve_path(grounding_path_text)
                grounding_dict = json.loads(grounding_path.read_text(encoding="utf-8"))
                if not isinstance(grounding_dict, dict):
                    raise ValueError("Grounding dictionary must be a JSON object.")
                payload_for_writer, grounding_report = apply_grounding(payload_for_writer, grounding_dict)
                st.write("Grounding report", grounding_report)
                st.download_button(
                    "Download grounded JSON",
                    json.dumps(payload_for_writer, indent=2),
                    file_name="grounded_output.json",
                    mime="application/json",
                )
            except Exception as exc:
                st.error(f"Grounding failed: {exc}")
                st.stop()
        try:
            reference_path = resolve_path(ref_path_text)
            signature = discover_structure_signature(reference_path)
            args = SimpleNamespace(
                named_for_id=int(named_for_id),
                name=pathway_name,
                description=pathway_description,
                subject=subject,
                pw_id=pw_id,
                height=int(vis_height),
                width=int(vis_width),
                background_color=background_color,
            )
            builder = DeterministicPwmlBuilder(payload_for_writer, signature, args)
            build = builder.build()
            tree = etree.ElementTree(build.root)
            repaired = repair_tree(tree, signature)
            report = validate_generated_tree(repaired, signature)
            xml_bytes = etree.tostring(
                repaired.getroot(), encoding="utf-8", xml_declaration=True, pretty_print=True
            )
            st.write("PWML generation summary", build.counts)
            st.write("Dummy geometry generated", build.geometry_generated)
            st.write("PWML structural validation", {"ok": report["ok"], "issue_count": report["issue_count"]})
            if not report["ok"]:
                st.json(report["issues"])
            st.download_button(
                "Download PWML",
                data=xml_bytes,
                file_name="out.pwml",
                mime="application/xml",
            )
            st.download_button(
                "Download PWML validation report",
                data=json.dumps(report, indent=2),
                file_name="writer_validation_report.json",
                mime="application/json",
            )
        except Exception as exc:
            st.error(f"PWML generation failed: {exc}")

    st.subheader("Connectivity snapshot")
    stats = graph_summary(final_payload)
    st.write(stats)
    if run_inference_from_state:
        st.write("Connectivity repair hints used for later rounds", build_qa_feedback(final_payload))
