import json
import logging
import hashlib
import time
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, TypedDict
from uuid import uuid4

from lxml import etree

import t2pw.llm.client as llm_client_module
from t2pw.paths import PROJECT_ROOT
from t2pw.curation.apply_audit_patch import committed_change_count, run_apply
from t2pw.curation.audit_json_llm import run_audit
from t2pw.mapping.enrich_entities import run_enrichment
from t2pw.mapping.grounding import apply_grounding
from t2pw.curation.gap_resolver import run_gap_resolution
from t2pw.curation.pathway_curator import run_pathway_curator
from t2pw.sbml.json_to_sbml import build_sbml
from t2pw.mapping.map_ids import map_payload
from t2pw.sbml.render_pathwhiz_like import build_render_artifacts
from t2pw.sbml.strip_unmapped import strip_unmapped
from t2pw.sbml.overwatch import run_sbml_overwatch
from t2pw.sbml.examples import build_retrieval_context, load_motif_index, payload_to_query_text
from t2pw.pipeline.process_normalizer import (
    GateValidationError,
    compute_normalization_stats,
    normalize_process_payload,
    run_strict_post_normalization_gates,
)
from t2pw.pipeline.strict_quarantine import (
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
from t2pw.pipeline.export_mode import (
    DEFAULT_EXPORT_MODE,
    ExportMode,
    coerce_mode,
    format_gaps,
    is_research,
    review_flags,
)
from t2pw.pipeline.extraction_diagnostics import (
    count_entities as diagnostic_entity_counts,
    count_processes as diagnostic_process_counts,
    current as current_extraction_diagnostics,
    payload_hash as diagnostic_payload_hash,
)
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
from t2pw.pipeline.draft_graph import build_draft_graph
from t2pw.pipeline.qa_graph import generate_qa_report
from t2pw.pipeline.reaction_summary import generate_reaction_summary
from t2pw.pipeline.reaction_preservation_validator import (
    write_reaction_preservation_report_if_manifest,
)
from t2pw.pwml.validate import discover_structure_signature, repair_tree, validate_generated_tree
from t2pw.pwml.writer import (
    DeterministicPwmlBuilder,
    blocking_pwml_ir_errors,
)
from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir
from t2pw.pwml.qa import run_pwml_qa
from t2pw.pipeline.qa_graph import build_graph, connected_components, degrees, get_entities, node


logger = logging.getLogger("t2pw.app.streamlit_app")


def run_quarantine_boundary(
    payload: Dict[str, Any],
    *,
    strict_db: bool,
    outputs_dir: Path,
    pathway_context: Optional[Dict[str, Any]] = None,
    mode: ExportMode = DEFAULT_EXPORT_MODE,
    state: Optional[MutableMapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The one place a payload is quarantined. Runs once per PAYLOAD VERSION.

    It sits immediately before the post-mapping Stage 3 revalidation because that
    is the FIRST blocking pre-export decision, not the last: a payload that fails
    there never initializes refinement review, so
    ``t2pw.service.export.run_export`` returns on
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
    # R2: ``state`` is the session mapping this decision is recorded in. The app passes
    # ``st.session_state``; a caller that passes nothing gets a throwaway dict.
    if state is None:
        state = {}
    state["quarantine_report"] = result.quarantine_report
    state["quarantine_artifacts"] = artifacts
    state["quarantine_coverage"] = result.coverage
    state["quarantine_ok"] = bool(result.ok)
    state["quarantine_decision_controls"] = {
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


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


class CanonicalFreezeResult(TypedDict):
    """What :func:`freeze_canonical_payload` hands back. Declared at module level
    and BEFORE that function because this module has no ``from __future__ import
    annotations``, so its return annotation is evaluated eagerly at import time.
    """

    payload: Dict[str, Any]
    payload_hash: str
    final_stage3_gate_report: Dict[str, Any]
    quarantine_result: Dict[str, Any]
    quarantine_ok: bool
    canonical_json_path: Optional[Path]
    sbml_input_source: str


def freeze_canonical_payload(
    final_export_payload: Dict[str, Any],
    *,
    strict_db: bool,
    outputs_dir: Optional[Path],
    pathway_context: Optional[Dict[str, Any]],
    export_mode: ExportMode,
    research_mode: bool,
    tmp: Path,
    state: Optional[MutableMapping[str, Any]] = None,
) -> CanonicalFreezeResult:
    """Quarantine, gate, hash and serialize THE canonical payload. Lifted
    verbatim out of :func:`run_post_pipeline_sbml_artifacts`; it still returns
    the same plain dict, and :class:`CanonicalFreezeResult` only types that dict
    -- it is never called here. So the AST harnesses, which exec this
    ``FunctionDef`` alone under a prepended ``from __future__ import
    annotations``, never evaluate the annotation and need no such class.
    """

    # Function-local on purpose. The module-scope imports are another card's
    # surface, and the AST harnesses exec this ``FunctionDef`` on its own with a
    # hand-built globals dict -- so a module-level import would not be resolvable
    # there, while this one is.
    from t2pw.pipeline.canonical_hash import HASH_SCHEMA_VERSION

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
    canonical_json_path: Optional[Path] = None
    if final_export_payload:
        quarantine_result = run_quarantine_boundary(
            final_export_payload,
            strict_db=bool(strict_db),
            outputs_dir=Path(outputs_dir) if outputs_dir else (PROJECT_ROOT / "outputs"),
            pathway_context=pathway_context,
            mode=export_mode,
            state=state,
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
            # The payload as well as its digest: schema 2's two canonical
            # projections are computed from the OBJECT, and this is the one
            # place that holds the object the gate validated. ``payload_hash``
            # still wins for ``payload_sha256``, so that binding is byte for
            # byte what it was and every historical artifact still verifies.
            payload=canonical_export_payload,
            payload_hash=canonical_payload_hash,
            hash_schema=HASH_SCHEMA_VERSION,
        )
        canonical_json = tmp / "final.canonical.json"
        canonical_json.write_text(
            json.dumps(canonical_export_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # The SBML build below reads this path. Before the reordering it read
        # the enriched, pre-quarantine file -- so the model that was built and
        # the payload that was gated were different objects.
        canonical_json_path = canonical_json
        sbml_input_source = CANONICAL_PAYLOAD_KEY
    # A quarantine refusal leaves no canonical payload, so the final gate does
    # not run and no report is written. That is deliberate: the alternative is
    # to synthesise a verdict for a payload that was never assembled. The
    # artifact set is still stamped, so the consumer fails closed on the
    # missing report rather than reading the refusal as a pass.
    return {
        "payload": canonical_export_payload, "payload_hash": canonical_payload_hash,
        "final_stage3_gate_report": final_stage3_gate_report,
        "quarantine_result": quarantine_result, "quarantine_ok": quarantine_ok,
        "canonical_json_path": canonical_json_path, "sbml_input_source": sbml_input_source,
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
    state: Optional[MutableMapping[str, Any]] = None,
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
            base_max_tokens = llm_client_module.configured_max_tokens()
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
                cand_max_tokens = base_max_tokens
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
                    llm_max_tokens=base_max_tokens,
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
                    llm_max_tokens=base_max_tokens,
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
                    llm_max_tokens=base_max_tokens,
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
                        "max_tokens": base_max_tokens,
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
                gap_tokens = llm_client_module.configured_max_tokens()
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
                llm_max_tokens=llm_client_module.configured_max_tokens(),
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
            merge_identity_duplicates=True,
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
        # DECISIONS.md D-015 (LOCKED): compound name and identity canonicalization
        # is canonical biology and must finish BEFORE the freeze. It used to run
        # only inside ``build_pwml_ir``, downstream of the hash, where
        # PRODUCT_CONTRACT section 5 forbids an exporter to resolve biological
        # content. This mutates ``final_export_payload`` IN PLACE -- the same
        # object every consumer below already holds and the same one the freeze
        # receives -- so nothing downstream can hold a pre-resolution view of it.
        # Import is function-local for the same reason ``freeze_canonical_payload``
        # keeps its own: module scope here is another card's surface.
        from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

        # The report is captured here rather than returned: every key of this
        # function's result dict is built below the freeze seam, which this card
        # does not own. The durable provenance is not in this variable anyway --
        # it is in the payload, which now carries the resolution outcome
        # (``raw_name``, ``synonyms``, ``db_status``) into ``final_mapped.json``.
        # An ambiguous rename or a reference the rename would break raises
        # ``PrefreezeResolutionError`` out of here, leaving the payload untouched.
        prefreeze_resolution_report = run_prefreeze_resolution(
            final_export_payload,
            strict_db=bool(strict_db),
        )
        assert isinstance(prefreeze_resolution_report, dict)
        # Final payload phase 1 (D-W12): flag-guarded database id fill, in place, fails open.
        # Off by default: the helper returns {} and touches nothing. The report stays local.
        from t2pw.service.final_payload import maybe_apply_final_id_fill

        final_payload_fill_report = maybe_apply_final_id_fill(
            final_export_payload, pathway_context, state,
        )
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
        _freeze = freeze_canonical_payload(
            final_export_payload, strict_db=strict_db, outputs_dir=outputs_dir,
            pathway_context=pathway_context, export_mode=export_mode,
            research_mode=research_mode, tmp=tmp, state=state,
        )
        canonical_export_payload = _freeze["payload"]
        canonical_payload_hash = _freeze["payload_hash"]
        sbml_input_source = _freeze["sbml_input_source"]
        final_stage3_gate_report = _freeze["final_stage3_gate_report"]
        quarantine_result = _freeze["quarantine_result"]
        quarantine_ok = _freeze["quarantine_ok"]
        sbml_input_path = _freeze["canonical_json_path"] or sbml_input_path

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
                    llm_max_tokens=llm_client_module.configured_max_tokens(),
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
            # ── C-052 (D-040 §1): ADDITIVE keys, appended. ──────────────────
            #
            # Nothing above is renamed, reordered, re-valued or removed -- the
            # grant is key-ADDITION only, and appending at the end is what makes
            # that mechanically checkable rather than merely claimed.
            #
            # The pre-freeze report was captured above and then DISCARDED, so
            # ``rename_sources_collapsed`` and every compound-resolution
            # diagnostic this seam produced were unreachable from the UI while
            # the CLI (``writer.py``) persisted the same document. These
            # converge on the CLI's OWN key names -- ``prefreeze_resolution_report``
            # and ``prefreeze_review_required``.
            #
            # THE NAMES MATTER, and not for taste. ``batch/driver.py`` ::
            # ``_find_pwml_result`` returns the first ``post_pipeline_artifacts``
            # entry whose key satisfies ``"pwml" in key.lower()`` AS THE PWML
            # EXPORT RESULT. No key of this dict contains ``pwml`` today, so that
            # fallback has never fired from here; a key named
            # ``pwml_prefreeze_resolution_report`` -- the natural "converge on the
            # CLI's filename" instinct -- would make the batch driver report this
            # report as the export on every leg whose ``pwml_export_result`` is
            # empty. The CLI's key carries no ``pwml`` either, so converging on it
            # is safe by construction rather than by luck.
            #
            # The value is the report OBJECT, not a path: this function persists
            # nothing (its ``tmp`` is swept in ``finally`` below) and every other
            # ``*_report`` key here is likewise the document itself. The CLI
            # returns a path under the same name because the CLI writes a file.
            "prefreeze_resolution_report": prefreeze_resolution_report,
            "prefreeze_review_required": _safe_dict(
                prefreeze_resolution_report.get("review_required")
            ),
            # A0-C8's observability half. The canonical path cannot be published
            # as a string a consumer could use: it is built under
            # ``temp_root / f"post_pipeline_{uuid4().hex}"`` and ``rmtree``'d in
            # the ``finally`` below, so the absolute value is both unpredictable
            # and dead by the time anyone reads it. Its NAME and the name of the
            # file actually handed to the SBML build are the two observables that
            # outlive the directory -- and A0-C8 records that ``sbml_input_source``
            # alone is insufficient precisely because it says which *kind* of
            # payload was used and not which *file* was read.
            "canonical_json_path_name": (
                _freeze["canonical_json_path"].name
                if _freeze["canonical_json_path"] is not None
                else ""
            ),
            "sbml_input_path_name": Path(sbml_input_path).name,
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
        # Final payload (D-W12): the paper title replaces the default name, only when the payload carries the key
        # ``final_payload_pathway`` (written under T2PW_FINAL_PAYLOAD). Otherwise the name is returned unchanged.
        from t2pw.service.final_payload import effective_pathway_name

        pathway_name = effective_pathway_name(final_payload, pathway_name)
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

        # ── the pre-freeze sequence, at the THIRD export entry point ────────
        #
        # ``DECISIONS.md`` D-033 (LOCKED) amends D-032 clause 1: there are
        # THREE production export entry points, not two, and this is the third.
        # It is also, measured on this module, the ONLY producer of PWML in the
        # app: ``build_pwml_ir`` appears exactly once in this file -- below --
        # and ``outputs/pathway.pwml`` is written at the bottom of this same
        # function. ``run_post_pipeline_sbml_artifacts`` holds its own seam and
        # never builds an IR, so every byte of Streamlit PWML came out of a
        # payload no canonicalizer had run over.
        #
        # At base that was invisible, because ``ir.build_pwml_ir`` re-resolved
        # compound identity itself, downstream of the freeze, which permanent
        # merge rule 8 and ``PRODUCT_CONTRACT`` §5 forbid. C-051 withdrew that
        # call, so those rows now refuse loudly instead of being repaired
        # silently. This is the withdrawn work put back where D-015 and D-016
        # say it belongs -- upstream of the exporter, on the whole
        # ``PREFREEZE_CANONICALIZERS`` tuple rather than on any one stage, so
        # every present and future canonicalizer reaches this path by
        # construction (D-032 clause 1).
        #
        # WHERE, and why not earlier. The seam sits after the quarantine
        # decision and after both gates -- exactly where the withdrawn ``ir.py``
        # pass used to run -- so nothing this function already decided is judged
        # on different names than it was at base. Above ``decision_matches`` the
        # canonicalization would move the payload before the reuse check and
        # re-quarantine every export; above the gates it would hand them names
        # the base never showed them.
        #
        # Mutates ``payload`` IN PLACE, so ``build_pwml_ir`` below cannot hold a
        # pre-resolution view of it. Function-local import for the same reason
        # the other two seams keep theirs: ``prefreeze_resolution`` is another
        # card's module surface.
        from t2pw.pwml.prefreeze_resolution import (
            PrefreezeResolutionError,
            run_prefreeze_resolution,
        )

        try:
            prefreeze_resolution_report = run_prefreeze_resolution(
                payload,
                strict_db=bool(strict_db),
            )
        except PrefreezeResolutionError as exc:
            # D-015 clause 6: an ambiguous rename, or a reference the rename
            # would break, is a stop condition and the payload is left
            # untouched. Rendered in this function's own failure shape rather
            # than left to the catch-all below, which would bury the code in a
            # string and drop the refusal reason.
            return {
                "ok": False,
                "error": f"Pre-freeze canonicalization failed: {exc.code}: {exc.message}",
                "counts": {},
                "issues": 1,
                "refusal_reasons": [f"prefreeze_resolution_failed:{exc.code}"],
                "output_path": "",
                "qa": {},
                "grounding_report": grounding_report,
                "stage3_gate_report": stage3_gate_report,
                "stage3_contract_report": stage3_contract_report,
                "required_gate_report": required_gate_report,
                "required_gate_report_path": str(required_gate_path),
                "prefreeze_resolution_error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                },
                "quarantine_report": quarantine_report,
                "quarantine_artifacts": quarantine_artifacts,
                "quarantine_reused": quarantine_reused,
                "reaction_preservation_before_final_export": before_export_report,
            }
        # ``report["ok"] is False`` deliberately does NOT abort here. D-029
        # (LOCKED) rules an unreachable PathBank DB a ``review_required``
        # outcome that must not raise by itself, and merge rule 7 keeps an
        # incomplete-but-correct pathway rather than dropping it. Persisting and
        # acting on that flag is the downstream seam's job (D-029's closing
        # paragraph names C-052), so this seam does not invent a reader for it.
        # The durable outcome is on the payload anyway -- ``db_status``,
        # ``raw_name``, ``synonyms`` and the DB-reachability marker
        # ``build_pwml_ir`` reads below.
        assert isinstance(prefreeze_resolution_report, dict)

        # ── C-052 (D-040 §§2-4): the record is PERSISTED, right here ────────
        #
        # WHERE. Immediately after the seam and BEFORE ``build_pwml_ir``. The
        # ``except Exception`` at the bottom of this function wraps everything
        # below, so a failure in the IR build, in the deterministic builder or in
        # the PWML write would otherwise lose the report entirely -- and the
        # export that most needs a resolution record is exactly the one that
        # failed. Written here, the on-disk record survives every return path in
        # this function, which is why none of the failure returns needs a new key.
        #
        # FILENAME. NOT the CLI's. ``writer.py`` writes
        # ``<out_dir>/pwml_prefreeze_resolution_report.json`` with ``--out-dir``
        # defaulting to ``outputs``; run from the project root the two writers
        # would share one file and whichever ran last would win, silently, while
        # a CLI test pins that exact path. Convergence with the CLI is on the
        # record's SHAPE, never on a path two writers share.
        #
        # ``"seam"``. Mandatory, top level, and applied AFTER the spread so the
        # label cannot be shadowed by a future report key of the same name. This
        # call is unambiguously POST-freeze -- it runs on a deepcopy taken after
        # the hash -- whereas ``run_post_pipeline_sbml_artifacts``'s report
        # describes a PRE-freeze canonicalization of a different object.
        # Persisting both under one undifferentiated name would tell the operator
        # they are the same record. This is LABELLING, not repair: it changes
        # nothing about what this seam does or is allowed to do, and the
        # post-freeze placement itself is owned elsewhere.
        prefreeze_report_path = outputs_dir / "pwml_prefreeze_resolution_report.streamlit.json"
        prefreeze_report_path.write_text(
            json.dumps(
                {**prefreeze_resolution_report, "seam": "post_freeze_refinement_reexport"},
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        # D-029, as split by D-040 §8: this seam PERSISTS and SURFACES the
        # verdict. It does not act on it -- no branch here changes whether a PWML
        # is produced -- so merge rules 6, 7 and 8 are untouched by construction.
        prefreeze_review_required = _safe_dict(
            prefreeze_resolution_report.get("review_required")
        )

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
            # C-052: SURFACE the record this seam now persists. The CLI's two
            # key names, carried verbatim, with this function's own established
            # ``<x>_report`` / ``<x>_report_path`` pairing (the same shape
            # ``required_gate_report`` / ``required_gate_report_path`` uses above)
            # so a reader does not have to guess whether a value is a document or
            # a path. Neither name contains ``pwml`` -- see the note on the
            # additive keys of ``run_post_pipeline_sbml_artifacts``.
            "prefreeze_resolution_report": prefreeze_resolution_report,
            "prefreeze_resolution_report_path": str(prefreeze_report_path),
            "prefreeze_review_required": prefreeze_review_required,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "counts": {}, "issues": 0,
                "output_path": "", "qa": {}, "grounding_report": {},
                "quarantine_report": quarantine_report,
                "quarantine_artifacts": quarantine_artifacts,
                "quarantine_reused": quarantine_reused,
                "reaction_preservation_before_final_export": before_export_report}
