"""The "Run audit and DB mapping" handler, moved out of ``streamlit_app.py`` (R4b).

The app keeps the button, the not-a-dict guard, the ``post_pipeline_artifacts`` pop and
the quarantine reset; :func:`run_post_pipeline` is everything after them. Every message
goes through the ``reporter`` the caller passes.

Three things stay in the app and come in as callbacks, because each one runs *inside*
the moved try block and its position there is the behaviour: ``open_review`` (the
13-key refinement review state; it runs before the "ready for review" message, and a
failure in it lands in the generic handler), ``render_issue_detail`` (shared with three
other app sites) and ``render_contract_failure`` (shared with the legacy SBML button; it
runs before the ``finally`` that publishes the diagnostics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, MutableMapping, Optional

import t2pw.config as _config
import t2pw.paths as paths  # PROJECT_ROOT read at call time: tests and the harness patch it
# Module imports, so these are looked up at call time as the app script's per-run
# ``from ... import`` did: a patch of ``postpipeline.run_post_pipeline_sbml_artifacts``
# (the equivalence harness's spy), ``pipeline.propagate_context_organism``,
# ``extraction_diagnostics.reenter``, ``config.rag_config`` or the like reaches the call.
from t2pw.pipeline import export_mode as _export_mode
from t2pw.pipeline import extraction_diagnostics as _diag
from t2pw.pipeline import pipeline as _pipeline
from t2pw.pipeline.gate_reports import CANONICAL_PAYLOAD_KEY, FINAL_GATE_REPORT_KEY
from t2pw.pipeline.stage_contracts import StageContractError
from t2pw.service import extraction as _extraction
from t2pw.service import postpipeline as _postpipeline
from t2pw.service.postpipeline import _safe_dict, _safe_list
from t2pw.service.reporter import Reporter


@dataclass(frozen=True)
class PostRunRequest:
    """The payload and the post-pipeline widget values, raw, as the app reads them. No
    field has a default: the caller states every setting."""

    final_payload: Any
    use_llm_audit: Any
    use_sbml_overwatch: Any
    default_compartment: Any
    mapping_cache_text: Any
    id_source_mode: Any
    db_host: Any
    db_port: Any
    db_user: Any
    db_password: Any = field(repr=False)  # never in a log line or a traceback's repr
    db_schema: Any
    audit_max_rounds: Any
    audit_timeout_seconds: Any
    audit_candidate_count: Any
    use_example_retrieval: Any
    example_index_path: Any
    example_top_k: Any
    use_gap_resolver: Any
    use_llm_gap_resolver: Any
    gap_resolver_max_items: Any
    use_stoich_agent: Any


@dataclass(frozen=True)
class PostRunResult:
    """``final_payload`` is the value the app must rebind: the payload after
    ``propagate_context_organism``, or the one it was given if that call raised.
    ``failure`` is the exception that ended the run, or ``None``; it has already been
    rendered (``render_contract_failure``, or the generic handler's messages)."""

    final_payload: Any = None
    failure: Optional[Exception] = None


def run_post_pipeline(
    request: PostRunRequest,
    *,
    state: MutableMapping[str, Any],
    reporter: Reporter,
    open_review: Callable[[Dict[str, Any], Dict[str, Any]], None],
    render_issue_detail: Callable[[Any], None],
    render_contract_failure: Callable[[StageContractError], None],
) -> PostRunResult:
    """Audit, DB mapping and the presentation of their verdicts, exactly as the app's
    "Run audit and DB mapping" handler ran them after the quarantine reset. Results and
    the artifacts are in ``state``, under the app's session keys."""

    final_payload = request.final_payload
    failure: Optional[Exception] = None
    _diag.reenter(state.get("extraction_diagnostics_recorder"))
    try:
        with reporter.spinner("Running audit and DB mapping..."):
            final_payload = _pipeline.propagate_context_organism(
                final_payload,
                state.get("pathway_context"),
            )
            artifacts = _postpipeline.run_post_pipeline_sbml_artifacts(
                final_payload,
                build_legacy_sbml=False,
                use_llm_audit=bool(request.use_llm_audit),
                use_sbml_overwatch=bool(request.use_sbml_overwatch),
                default_compartment=(request.default_compartment or "cell").strip() or "cell",
                mapping_cache_path=request.mapping_cache_text.strip() or "id_mapping_cache.json",
                id_source=(request.id_source_mode or "hybrid").strip().lower(),
                db_host=(request.db_host or "").strip(),
                db_port=int(request.db_port),
                db_user=(request.db_user or "").strip(),
                db_password=request.db_password or "",
                db_schema=(request.db_schema or "pathbank").strip() or "pathbank",
                audit_max_rounds=int(request.audit_max_rounds),
                audit_timeout_seconds=int(request.audit_timeout_seconds),
                audit_candidate_count=int(request.audit_candidate_count),
                use_example_retrieval=bool(request.use_example_retrieval),
                example_index_path=(request.example_index_path or "").strip(),
                example_top_k=int(request.example_top_k),
                use_gap_resolver=bool(request.use_gap_resolver),
                use_llm_gap_resolver=bool(request.use_llm_gap_resolver),
                gap_resolver_max_items=int(request.gap_resolver_max_items),
                qa_report=state.get("qa_report"),
                reaction_summary=state.get("reaction_summary"),
                use_stoich_agent=bool(request.use_stoich_agent),
                rag_evidence_context=(
                    state.get("rag_evidence_context") or ""
                    if _config.rag_config()["enabled"]
                    else ""
                ),  # S2 — "" when RAG off, keeping today's path
                export_mode=_export_mode.coerce_mode(state.get("export_mode")),
                # The quarantine boundary runs inside the pipeline now, so
                # its three inputs travel with the call instead of being
                # read from session state after the fact.
                strict_db=bool(state.get("pwml_strict_db", True)),
                pathway_context=_safe_dict(state.get("pathway_context")),
                outputs_dir=paths.PROJECT_ROOT / "outputs",
                state=state,
            )
        state["post_pipeline_artifacts"] = artifacts
        _pa = _safe_dict(artifacts)
        if _pa.get("post_audit_qa_report"):
            state["qa_report"] = _pa["post_audit_qa_report"]
        if _pa.get("post_audit_reaction_summary"):
            state["reaction_summary"] = _pa["post_audit_reaction_summary"]
        _pa["final_payload_snapshot"] = state.get("final_payload_snapshot", final_payload)
        if bool(_pa.get("gate_failed", False)):
            reporter.warning("Post-pipeline stopped at hard-gate validation. Review gate_fail_report.json.")
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
                    reporter.warning(
                        f"Quarantine review flags ({len(_quarantine_flags)}): "
                        "research mode kept every process. A strict export "
                        "would have removed the ones listed below."
                    )
                    reporter.json(_quarantine_flags)
                if not bool(_pa.get("quarantine_ok")):
                    _refusal_reasons = [
                        str(reason) for reason in _safe_list(_pa.get("quarantine_refusal_reasons"))
                    ]
                    state["refinement_gate_errors"] = [
                        {"path": "/processes", "reason": f"quarantine refusal: {reason}"}
                        for reason in _refusal_reasons
                    ]
                    reporter.error(
                        "Quarantine could not produce a viable requested-pathway core: "
                        + "; ".join(_refusal_reasons)
                    )
                    # Rendered flat rather than inside an expander. A
                    # refusal is the reason the run stopped, so it should
                    # not need a click to read, and st.expander here trips
                    # FragmentThreadState under AppTest once an earlier
                    # app run has executed in the same process.
                    reporter.caption("Requested-core coverage")
                    reporter.json(_safe_dict(_pa.get("quarantine_coverage")))
                    reporter.caption("Strict invariants")
                    reporter.json(_safe_dict(_pa.get("quarantine_report")).get("strict_invariants"))
                else:
                    final_gate: Dict[str, Any] = _safe_dict(_pa.get(FINAL_GATE_REPORT_KEY))
                    _stage3_failed = bool(final_gate) and not bool(final_gate.get("ok", False))
                    if _stage3_failed and _export_mode.is_research(_export_mode.coerce_mode(state.get("export_mode"))):
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
                        state["refinement_gate_errors"] = []
                        reporter.warning(
                            "Research mode: the pre-export Stage 3 revalidation FAILED "
                            f"({len(_safe_list(final_gate.get('errors')))} error(s)) and was not "
                            "enforced. The strict gates did NOT pass — this candidate is not "
                            "PathWhiz-exportable as it stands. Refinement review is open so the "
                            "findings below can be reviewed against the pathway itself."
                        )
                        # Flat, not in an expander: st.expander here trips
                        # FragmentThreadState under AppTest once an earlier
                        # app run has executed in the same process.
                        reporter.caption("Stage 3 findings retained as review flags")
                        for _gate_err in _safe_list(final_gate.get("errors")):
                            if isinstance(_gate_err, dict):
                                reporter.markdown(
                                    f"- `{_gate_err.get('path', '')}` — {_gate_err.get('reason', _gate_err)}"
                                )
                            else:
                                reporter.markdown(f"- {_gate_err}")
                        reporter.json(final_gate)
                        # The candidate, byte for byte. Research quarantine
                        # returned the payload unchanged, so this is what
                        # came out of mapping and not a reduced graph.
                        open_review(final_mapped_payload, mapping_report)
                    elif _stage3_failed:
                        state["refinement_gate_errors"] = _safe_list(final_gate.get("errors"))
                        reporter.error(
                            "Mapped pathway still fails the pre-export Stage 3 revalidation. "
                            "Refinement review was not opened."
                        )
                        _gate_errors_for_display = _safe_list(final_gate.get("errors"))
                        with reporter.expander(
                            f"Why Stage 3 failed ({len(_gate_errors_for_display)} error(s))",
                            expanded=True,
                        ):
                            if _gate_errors_for_display:
                                for _gate_err in _gate_errors_for_display:
                                    if isinstance(_gate_err, dict):
                                        reporter.markdown(
                                            f"- `{_gate_err.get('path', '')}` — {_gate_err.get('reason', _gate_err)}"
                                        )
                                        render_issue_detail(_gate_err.get("detail"))
                                    else:
                                        reporter.markdown(f"- {_gate_err}")
                            else:
                                reporter.write("No itemized errors were returned; showing full gate report below.")
                            reporter.json(final_gate)
                    else:
                        state["refinement_gate_errors"] = []
                        open_review(final_mapped_payload, mapping_report)
                        reporter.info("Mapped pathway is ready for review. PWML generation is paused until approval.")
    except StageContractError as exc:
        failure = exc
        render_contract_failure(exc)
    except Exception as exc:
        failure = exc
        _postpipeline._record_mapped_failure_snapshot(
            final_payload,
            stage="post_pipeline",
            reason=f"{type(exc).__name__}: {exc}",
        )
        reporter.error(f"Post-pipeline conversion failed: {exc}")
        reporter.exception(exc)  # full traceback (file:line) so the cause isn't a mystery
    finally:
        # Both the pass and the two failure paths above. The audit and gap
        # rounds completed before an abort are already on disk; this is
        # what carries them, and the mapping snapshot, to the batch driver.
        _extraction._sync_extraction_diagnostics(state=state)

    return PostRunResult(final_payload=final_payload, failure=failure)
