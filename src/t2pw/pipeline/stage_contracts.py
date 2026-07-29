from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from t2pw.pipeline.entity_identity import (
    has_protein_external_identity,
    is_generated_complex_wrapper,
    protein_external_identity,
    protein_species_context,
)
from t2pw.pipeline.export_mode import (
    DEFAULT_EXPORT_MODE,
    ExportMode,
    is_research,
    relax_report,
)
from t2pw.pipeline.failure_detail import (
    closest_names,
    detail as build_detail,
    row_digest,
    scrub_detail,
)
from t2pw.pipeline.payload_models import (
    RuntimeSchemaBoundary,
    RuntimeSchemaMode,
    RuntimeSchemaReport,
    RuntimeSchemaValidationError,
    validate_payload_shape,
)


Issue = Dict[str, Any]
Report = Dict[str, Any]

# This is the single live rollout switch for recursive runtime validation.
# Keep report-first behavior until callers deliberately opt into enforcement.
RUNTIME_SCHEMA_MODE: RuntimeSchemaMode = "report"


class StageContractError(ValueError):
    """Raised when a stage boundary contract must abort the pipeline."""

    def __init__(self, stage: str, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.stage = stage
        self.report = dict(report)
        self.errors = list(self.report.get("errors", []))


def validate_runtime_payload_contract(
    payload: Any,
    *,
    boundary: RuntimeSchemaBoundary,
    mode: RuntimeSchemaMode | None = None,
    contract_report: Report | None = None,
) -> RuntimeSchemaReport:
    """Adapt runtime shape validation to existing stage-report semantics.

    Report mode records recursive shape findings as warnings so the additive
    rollout does not change established abort behavior.  Enforce mode raises a
    ``StageContractError`` while preserving the runtime report on the stage
    report.  Issues are deduplicated by stable ``(code, pointer)`` identity.
    """

    effective_mode = mode or RUNTIME_SCHEMA_MODE
    try:
        runtime_report = validate_payload_shape(
            payload,
            boundary=boundary,
            mode=effective_mode,
        )
    except RuntimeSchemaValidationError as exc:
        runtime_report = exc.report
        report = (
            contract_report
            if contract_report is not None
            else _new_report(
                boundary,
                "structural",
                effect_on_failure="abort",
            )
        )
        report["runtime_schema_report"] = runtime_report
        for issue in runtime_report["errors"]:
            _add_issue(report, "errors", issue)
        report["ok"] = False
        report["summary"] = _summary(report)
        raise StageContractError(
            boundary,
            "Runtime payload shape validation failed.",
            report,
        ) from exc

    if contract_report is not None:
        contract_report["runtime_schema_report"] = runtime_report
        for issue in runtime_report["errors"]:
            _add_issue(contract_report, "warnings", issue)
        contract_report["summary"] = _summary(contract_report)
    return runtime_report


def validate_post_extraction(payload: Any) -> Report:
    """Validate the structural contract emitted by extraction."""

    report = _new_report("post_extraction", "structural", effect_on_failure="abort")
    _validate_payload_container(payload, report)
    if report["errors"]:
        _abort(report)

    assert isinstance(payload, dict)
    _validate_named_entities(payload, report)
    _validate_extracted_processes(payload, report)
    return _raise_or_return(report)


def validate_post_mapping(payload: Any) -> Report:
    """Validate the structural contract emitted by ID mapping."""

    report = _new_report("post_mapping", "structural", effect_on_failure="abort")
    _validate_payload_container(payload, report)
    if report["errors"]:
        _abort(report)

    assert isinstance(payload, dict)
    entities = _safe_dict(payload.get("entities"))
    species = entities.get("species")
    if not isinstance(species, list) or not species:
        _add_error(
            report,
            "species_required",
            "Mapped payload must include at least one species row.",
            "/entities/species",
            detail=build_detail(
                found_type=type(species).__name__,
                found_size=len(species) if isinstance(species, (list, dict, str)) else None,
                entity_buckets=sorted(str(name) for name in entities),
            ),
        )

    for bucket, row, idx in _iter_entity_rows(entities):
        if not _text(row.get("name")):
            continue
        pointer = f"/entities/{bucket}/{idx}/mapping_meta"
        mapping_meta = row.get("mapping_meta")
        if not isinstance(mapping_meta, dict):
            _add_error(
                report,
                "entity_missing_mapping_meta",
                "Mapped entity with a name must include a mapping_meta object.",
                pointer,
                bucket=bucket,
                name=row.get("name"),
                # The row shows whether mapping ran and produced nothing, or never
                # touched this entity at all -- a distinction the message cannot
                # make and the one that decides where to look upstream.
                detail=row_digest(row),
            )
            continue

        resolution = mapping_meta.get("resolution")
        if not isinstance(resolution, dict):
            _add_error(
                report,
                "mapping_resolution_required",
                "Mapping metadata must include a resolution object.",
                f"{pointer}/resolution",
                bucket=bucket,
                name=row.get("name"),
                detail=build_detail(
                    found_type=type(resolution).__name__,
                    mapping_meta=mapping_meta,
                ),
            )
            continue

        status = resolution.get("status")
        if not isinstance(status, str) or not status.strip():
            _add_error(
                report,
                "mapping_resolution_status_required",
                "Mapping resolution must include a non-empty status string.",
                f"{pointer}/resolution/status",
                bucket=bucket,
                name=row.get("name"),
                detail=build_detail(
                    found_type=type(status).__name__,
                    found_value=status,
                    resolution=resolution,
                ),
            )
        for field in ("issue", "order_step"):
            value = resolution.get(field)
            if value is not None and not isinstance(value, str):
                _add_error(
                    report,
                    "mapping_resolution_field_invalid",
                    f"Mapping resolution {field} must be a string when present.",
                    f"{pointer}/resolution/{field}",
                    bucket=bucket,
                    name=row.get("name"),
                    field=field,
                    detail=build_detail(
                        found_type=type(value).__name__,
                        found_value=value,
                    ),
                )

    return _raise_or_return(report)


def validate_post_normalization(payload: Any, gate_report: Mapping[str, Any] | None = None) -> Report:
    """
    Validate the post-normalization boundary.

    Structural garbage still aborts. Semantic gate failures are returned as a
    report so the orchestrator can pass them to the audit loop.
    """

    report = _new_report("post_normalization", "semantic", effect_on_failure="feed_audit")
    _validate_payload_container(payload, report)
    if report["errors"]:
        report["contract_type"] = "structural"
        report["effect_on_failure"] = "abort"
        _abort(report)

    gate_report = dict(gate_report or {})
    gate_errors = _issue_list(gate_report.get("errors"))
    _validate_canonical_actor_rows(payload, report)
    if gate_errors or gate_report.get("ok") is False:
        report["ok"] = False
        for issue in gate_errors:
            _add_issue(report, "errors", issue)
        report["gate_report"] = gate_report
        report["summary"] = _summary(report)
    return report


def validate_post_remap(payload: Any) -> Report:
    """Validate generated complex wrappers immediately after Stage 6."""

    report = _new_report("post_remap", "semantic", effect_on_failure="abort")
    _validate_payload_container(payload, report)
    if report["errors"]:
        _abort(report)

    assert isinstance(payload, dict)
    entities = _safe_dict(payload.get("entities"))
    proteins = _safe_list(entities.get("proteins"))
    proteins_by_name = {
        _normalized(row.get("name")): row
        for row in proteins
        if isinstance(row, dict) and _text(row.get("name"))
    }
    proteins_by_identity = {
        protein_external_identity(row).casefold(): row
        for row in proteins
        if isinstance(row, dict) and protein_external_identity(row)
    }

    for cidx, complex_row in enumerate(_safe_list(entities.get("protein_complexes"))):
        if not isinstance(complex_row, dict) or not is_generated_complex_wrapper(complex_row):
            continue
        components = _safe_list(complex_row.get("components"))
        if not components:
            _add_error(
                report,
                "generated_wrapper_missing_components",
                "Generated protein-complex wrapper must declare at least one component.",
                f"/entities/protein_complexes/{cidx}/components",
                detail=row_digest(complex_row),
            )
            continue
        for pidx, component in enumerate(components):
            component_row = component if isinstance(component, dict) else {}
            component_name = component if isinstance(component, str) else next(
                (_text(component_row.get(key)) for key in ("name", "protein", "protein_name", "component", "entity") if _text(component_row.get(key))),
                "",
            )
            declared = proteins_by_name.get(_normalized(component_name))
            component_identity = protein_external_identity(component_row)
            if declared is None and component_identity:
                declared = proteins_by_identity.get(component_identity.casefold())
            pointer = f"/entities/protein_complexes/{cidx}/components/{pidx}"
            if declared is None:
                _add_error(
                    report,
                    "generated_wrapper_component_protein_unresolved",
                    "Generated wrapper component must resolve to a declared protein.",
                    pointer,
                    component=component_name,
                    # Both lookups failed: by normalized name and by external
                    # identity. Which one was even attempted, and what the
                    # registry holds that is close, separates a misspelling from
                    # a protein the extractor never declared.
                    detail=build_detail(
                        component_name=component_name,
                        component_identity=component_identity,
                        component_row=component_row or None,
                        # Suggest DISPLAY names, not the normalized keys, so the
                        # reviewer can paste one straight back into the payload.
                        did_you_mean=closest_names(
                            component_name,
                            (_text(declared_row.get("name")) for declared_row in proteins_by_name.values()),
                        ),
                        declared_protein_count=len(proteins_by_name),
                    ),
                )
                continue
            if not protein_species_context(declared):
                _add_error(
                    report,
                    "generated_wrapper_component_missing_species",
                    "Generated wrapper component protein must include species context.",
                    pointer,
                    protein=declared.get("name"),
                    detail=row_digest(declared, pointer="/entities/proteins (resolved)"),
                )
            if not has_protein_external_identity(declared):
                _add_error(
                    report,
                    "generated_wrapper_component_missing_external_identity",
                    "Generated wrapper component protein must include a UniProt or DrugBank identifier.",
                    pointer,
                    protein=declared.get("name"),
                    detail=row_digest(declared, pointer="/entities/proteins (resolved)"),
                )

    return _raise_or_return(report)


def validate_post_audit(payload: Any) -> Report:
    """Validate that audit returned a structurally usable payload."""

    report = _new_report("post_audit", "structural", effect_on_failure="abort")
    _validate_payload_container(payload, report)
    if report["errors"]:
        _abort(report)

    assert isinstance(payload, dict)
    _validate_named_entities(payload, report)
    return _raise_or_return(report)


def validate_pre_export(payload: Any, *, strict_db: bool = True) -> Report:
    """Validate the final semantic PWML contract before export."""

    report = _new_report("pre_export", "semantic", effect_on_failure="abort")
    _validate_payload_container(payload, report)
    if report["errors"]:
        _abort(report)

    from t2pw.pwml.ir import validate_required_pwml_contract

    pwml_report = validate_required_pwml_contract(payload, strict_db=strict_db)
    report["pwml_contract_report"] = pwml_report
    report["warnings"] = _issue_list(pwml_report.get("warnings"))
    if pwml_report.get("ok") is False:
        report["ok"] = False
        report["errors"] = _issue_list(pwml_report.get("errors"))
        report["summary"] = _summary(report)
        _abort(report, "Pre-export PWML contract failed.")

    report["summary"] = _summary(report)
    return report


def run_stage_contract(
    validator: Any,
    *args: Any,
    mode: ExportMode = DEFAULT_EXPORT_MODE,
    **kwargs: Any,
) -> Report:
    """Run one of the validators above under an export mode.

    In ``pathwhiz`` mode this is a plain call -- the validator body is not
    parameterized at all, which is what makes strict behaviour byte-for-byte
    identical rather than merely intended to be.

    In ``research`` mode the same validator runs unchanged and its findings are
    re-severitied afterwards by :func:`t2pw.pipeline.export_mode.relax_report`:
    FORMAT rules are skipped, BIOLOGY/PROVENANCE findings become non-blocking
    review flags, and the run continues. Structural guards still abort -- there
    is no candidate pathway to review when the payload is not a payload -- so a
    ``StageContractError`` carrying only those is re-raised with the relaxed
    report attached.
    """

    if not is_research(mode):
        return validator(*args, **kwargs)

    try:
        report = validator(*args, **kwargs)
    except StageContractError as exc:
        relaxed = relax_report(exc.report, stage=exc.stage)
        if relaxed.get("errors"):
            raise StageContractError(exc.stage, str(exc), relaxed) from exc
        return relaxed

    return relax_report(report, stage=str(_safe_dict(report).get("stage") or ""))


def _new_report(stage: str, contract_type: str, *, effect_on_failure: str) -> Report:
    return {
        "ok": True,
        "stage": stage,
        "contract_type": contract_type,
        "effect_on_failure": effect_on_failure,
        "errors": [],
        "warnings": [],
        "summary": {"error_count": 0, "warning_count": 0},
    }


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _issue_list(value: Any) -> List[Issue]:
    return [item for item in _safe_list(value) if isinstance(item, dict)]


def _add_error(
    report: Report,
    code: str,
    message: str,
    pointer: str = "",
    detail: Mapping[str, Any] | None = None,
    **extra: Any,
) -> None:
    """Record a blocking issue, optionally with the evidence that produced it.

    ``detail`` carries what the ``message`` cannot: the value actually checked,
    the offending row, the set it failed to match. Bounded by ``scrub_detail``
    on the way in, and omitted when empty so issues without detail keep their
    old shape. See ``docs/regression_baseline_2026-07-28.md`` §4.
    """

    issue: Issue = {"code": code, "message": message}
    if pointer:
        issue["pointer"] = pointer
    issue.update(extra)
    scrubbed = scrub_detail(detail)
    if scrubbed:
        issue["detail"] = scrubbed
    _add_issue(report, "errors", issue)
    report["ok"] = False
    report["summary"] = _summary(report)


def _issue_identity(issue: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(issue.get("code") or ""),
        str(issue.get("pointer") or issue.get("path") or ""),
    )


def _add_issue(report: Report, destination: str, issue: Mapping[str, Any]) -> None:
    candidate = dict(issue)
    identity = _issue_identity(candidate)
    errors = _safe_list(report.get("errors"))
    warnings = _safe_list(report.get("warnings"))

    if destination == "errors":
        if any(_issue_identity(existing) == identity for existing in errors if isinstance(existing, dict)):
            return
        report["warnings"] = [
            existing
            for existing in warnings
            if not isinstance(existing, dict) or _issue_identity(existing) != identity
        ]
        report.setdefault("errors", []).append(candidate)
        return

    if any(
        _issue_identity(existing) == identity
        for existing in [*errors, *warnings]
        if isinstance(existing, dict)
    ):
        return
    report.setdefault("warnings", []).append(candidate)


def _summary(report: Mapping[str, Any]) -> Dict[str, int]:
    return {
        "error_count": len(_safe_list(report.get("errors"))),
        "warning_count": len(_safe_list(report.get("warnings"))),
    }


def _abort(report: Report, message: str | None = None) -> None:
    errors = _safe_list(report.get("errors"))
    first = _safe_dict(errors[0]) if errors else {}
    detail = str(first.get("message") or "Stage contract failed.")
    raise StageContractError(
        str(report.get("stage") or "unknown_stage"),
        message or detail,
        report,
    )


def _raise_or_return(report: Report) -> Report:
    report["summary"] = _summary(report)
    if report["errors"]:
        _abort(report)
    return report


def _validate_payload_container(payload: Any, report: Report) -> None:
    stage = str(report.get("stage") or "")
    if stage in {
        "post_extraction",
        "post_mapping",
        "post_normalization",
        "post_audit",
        "post_remap",
        "post_enrichment",
        "pre_export",
    }:
        validate_runtime_payload_contract(
            payload,
            boundary=stage,  # type: ignore[arg-type]
            contract_report=report,
        )
    if not isinstance(payload, dict):
        _add_error(
            report,
            "invalid_payload",
            "Payload must be a dict.",
            "/",
            detail={"found_type": type(payload).__name__, "found_value": payload},
        )
        return
    # "Payload must include an entities object" is true of a payload that omits
    # the key, one that has it as null, and one that has it as a list -- three
    # different upstream bugs with one message. The found type plus the keys that
    # ARE present distinguishes them without opening the payload by hand.
    for key, code in (("entities", "entities_required"), ("processes", "processes_required")):
        found = payload.get(key)
        if isinstance(found, dict):
            continue
        _add_error(
            report,
            code,
            f"Payload must include a {key} object.",
            f"/{key}",
            detail=build_detail(
                found_type=type(found).__name__,
                key_present=key in payload,
                found_size=len(found) if isinstance(found, (list, dict, str)) else None,
                payload_keys=sorted(str(name) for name in payload),
                found_value=found if not isinstance(found, (list, dict)) else None,
            ),
        )


def _iter_entity_rows(entities: Mapping[str, Any]) -> Iterable[tuple[str, Dict[str, Any], int]]:
    for bucket, rows in entities.items():
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows):
            if isinstance(row, dict):
                yield str(bucket), row, idx


def _validate_named_entities(payload: Mapping[str, Any], report: Report) -> None:
    entities = _safe_dict(payload.get("entities"))
    for bucket, rows in entities.items():
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                _add_error(
                    report,
                    "entity_not_object",
                    "Entity rows must be objects.",
                    f"/entities/{bucket}/{idx}",
                    bucket=bucket,
                    detail=build_detail(found_type=type(row).__name__, found_value=row),
                )
                continue
            if not _text(row.get("name")):
                # A name can be absent, null, blank, or a non-string that _text
                # collapsed to "". The row itself separates those, and its other
                # identity fields (uniprot, species) usually say which upstream
                # stage produced the nameless row.
                _add_error(
                    report,
                    "entity_missing_name",
                    "Every entity row must have a non-empty name.",
                    f"/entities/{bucket}/{idx}/name",
                    bucket=bucket,
                    detail=row_digest(row),
                )


def _validate_extracted_processes(payload: Mapping[str, Any], report: Report) -> None:
    processes = _safe_dict(payload.get("processes"))
    for bucket, rows in processes.items():
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows):
            pointer = f"/processes/{_escape_pointer_token(bucket)}/{idx}"
            if not isinstance(row, dict):
                _add_error(
                    report,
                    "process_not_object",
                    "Process rows must be objects.",
                    pointer,
                    bucket=bucket,
                    expected="object",
                    detail=build_detail(found_type=type(row).__name__, found_value=row),
                )
                continue

            if bucket == "reactions":
                _validate_reaction_structure(row, report, pointer)
            elif bucket == "transports":
                _validate_transport_structure(row, report, pointer)
            elif bucket == "interactions":
                _validate_interaction_structure(row, report, pointer)
            elif bucket == "reaction_coupled_transports":
                _validate_reaction_coupled_transport_structure(row, report, pointer)
            elif bucket == "sub_pathways":
                _validate_sub_pathway_structure(row, report, pointer)


def _validate_reaction_structure(row: Mapping[str, Any], report: Report, pointer: str) -> None:
    if _has_meaningful_process_participants(row.get("inputs")) or _has_meaningful_process_participants(
        row.get("outputs")
    ):
        return
    _add_error(
        report,
        "reaction_missing_participants",
        "Reaction must include at least one usable input or output participant.",
        pointer,
        bucket="reactions",
        required_any=["inputs", "outputs"],
        # "no usable participant" covers an empty list, a list of blanks, and a
        # list of dicts whose name keys are all unrecognised. Only the rows
        # themselves say which, so they decide whether this is an extraction
        # miss or a normalization drop.
        detail=row_digest(row),
    )


def _validate_transport_structure(row: Mapping[str, Any], report: Report, pointer: str) -> None:
    if _is_non_empty_string(row.get("cargo")) or _has_meaningful_elements_with_states(
        row.get("elements_with_states")
    ):
        return
    _add_error(
        report,
        "transport_missing_cargo",
        "Transport must include non-empty cargo or an elements_with_states row with a usable element.",
        pointer,
        bucket="transports",
        required_any=["cargo", "elements_with_states[].element"],
        detail=row_digest(row),
    )


def _validate_interaction_structure(row: Mapping[str, Any], report: Report, pointer: str) -> None:
    has_endpoint_pair = _is_non_empty_string(row.get("entity_1")) and _is_non_empty_string(
        row.get("entity_2")
    )
    if has_endpoint_pair or _has_meaningful_interaction_participants(row.get("participants")):
        return
    _add_error(
        report,
        "interaction_missing_participants",
        "Interaction must include usable entity_1 and entity_2 endpoints or at least one usable participant row.",
        pointer,
        bucket="interactions",
        required_any=[
            ["entity_1", "entity_2"],
            "participants[].{entity,protein,protein_complex,name}",
        ],
        detail=row_digest(row),
    )


def _validate_reaction_coupled_transport_structure(
    row: Mapping[str, Any], report: Report, pointer: str
) -> None:
    has_process_references = _is_non_empty_string(row.get("reaction")) and _is_non_empty_string(
        row.get("transport")
    )
    if has_process_references or _has_meaningful_elements_with_states(row.get("elements_with_states")):
        return
    _add_error(
        report,
        "reaction_coupled_transport_missing_structure",
        "Reaction-coupled transport must include usable reaction and transport references or an elements_with_states row with a usable element.",
        pointer,
        bucket="reaction_coupled_transports",
        required_any=[
            ["reaction", "transport"],
            "elements_with_states[].element",
        ],
        detail=row_digest(row),
    )


def _validate_sub_pathway_structure(row: Mapping[str, Any], report: Report, pointer: str) -> None:
    if (
        _is_non_empty_string(row.get("name"))
        or _has_usable_reference_id(row.get("reference_pathway_id"))
        or _is_non_empty_string(row.get("alttext"))
    ):
        return
    _add_error(
        report,
        "sub_pathway_missing_identity",
        "Sub-pathway must include a non-empty name, reference_pathway_id, or alttext.",
        pointer,
        bucket="sub_pathways",
        required_any=["name", "reference_pathway_id", "alttext"],
        detail=row_digest(row),
    )


def _validate_canonical_actor_rows(payload: Mapping[str, Any], report: Report) -> None:
    processes = _safe_dict(payload.get("processes"))
    actor_slots = (
        ("reactions", "enzymes"),
        ("reactions", "modifiers"),
        ("transports", "transporters"),
        ("interactions", "participants"),
    )
    for process_bucket, actor_field in actor_slots:
        for process_idx, process in enumerate(_safe_list(processes.get(process_bucket))):
            if not isinstance(process, dict):
                continue
            for actor_idx, actor in enumerate(_safe_list(process.get(actor_field))):
                pointer = f"/processes/{process_bucket}/{process_idx}/{actor_field}/{actor_idx}"
                if not isinstance(actor, dict) or not _text(actor.get("entity")) or not _text(actor.get("entity_type")):
                    # One code covers "not a dict", "no entity" and "no
                    # entity_type", and the fix differs for each. The row plus
                    # the two fields as found names which case fired.
                    _add_error(
                        report,
                        "actor_schema_not_canonical",
                        "Normalized actor rows must include non-empty entity and entity_type fields.",
                        pointer,
                        detail=build_detail(
                            found_type=type(actor).__name__,
                            entity=actor.get("entity") if isinstance(actor, dict) else None,
                            entity_type=actor.get("entity_type") if isinstance(actor, dict) else None,
                            actor_row=actor,
                        ),
                    )


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_meaningful_process_participants(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    identity_fields = (
        "name",
        "entity",
        "compound",
        "protein",
        "protein_complex",
        "element",
        "element_collection",
        "nucleic_acid",
    )
    return any(
        _is_non_empty_string(participant)
        or (
            isinstance(participant, dict)
            and any(_is_non_empty_string(participant.get(field)) for field in identity_fields)
        )
        for participant in value
    )


def _has_meaningful_elements_with_states(value: Any) -> bool:
    return isinstance(value, list) and any(
        isinstance(item, dict) and _is_non_empty_string(item.get("element")) for item in value
    )


def _has_meaningful_interaction_participants(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    identity_fields = ("entity", "protein", "protein_complex", "name")
    return any(
        isinstance(participant, dict)
        and any(_is_non_empty_string(participant.get(field)) for field in identity_fields)
        for participant in value
    )


def _has_usable_reference_id(value: Any) -> bool:
    if _is_non_empty_string(value):
        return True
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _escape_pointer_token(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _normalized(value: Any) -> str:
    return " ".join(_text(value).casefold().split())
