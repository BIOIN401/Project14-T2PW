"""Non-mutating runtime shape validation for pipeline payload dictionaries.

The models in this module are an additive runtime companion to
``t2pw.schema``.  They deliberately allow unknown fields because mapping and
enrichment providers attach evolving metadata.  Validation results are never
dumped back into the pipeline, so Pydantic cannot become an implicit
normalization or data-loss stage.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence, TypedDict

from pydantic import BaseModel, ConfigDict, ValidationError


RuntimeSchemaBoundary = Literal[
    "post_extraction",
    "post_mapping",
    "post_normalization",
    "post_audit",
    "post_remap",
    "post_enrichment",
    "pre_export",
]
RuntimeSchemaMode = Literal["report", "enforce"]


class RuntimeSchemaIssue(TypedDict, total=False):
    code: str
    pointer: str
    message: str
    expected: str
    received: str


class RuntimeSchemaSummary(TypedDict):
    error_count: int
    warning_count: int


class RuntimeSchemaReport(TypedDict):
    ok: bool
    boundary: RuntimeSchemaBoundary
    mode: RuntimeSchemaMode
    errors: List[RuntimeSchemaIssue]
    warnings: List[RuntimeSchemaIssue]
    summary: RuntimeSchemaSummary


class RuntimeSchemaValidationError(ValueError):
    """Raised by enforce mode with the same report returned by report mode."""

    def __init__(self, report: RuntimeSchemaReport) -> None:
        self.report = report
        self.errors = list(report["errors"])
        super().__init__(
            f"Runtime payload shape validation failed at {report['boundary']} "
            f"with {report['summary']['error_count']} error(s)."
        )


class _RuntimeModel(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)


class MetadataModel(_RuntimeModel):
    pathway_name: str | None = None
    pathway_description: str | None = None
    pathway_subject: str | None = None
    name: str | None = None
    description: str | None = None
    subject: str | None = None
    organism: str | None = None
    width: int | float | None = None
    height: int | float | None = None


class MappedIdsModel(_RuntimeModel):
    hmdb: str | None = None
    kegg: str | None = None
    chebi: str | None = None
    pubchem: str | None = None
    drugbank: str | None = None
    uniprot: str | None = None
    biocyc: str | None = None
    chemspider: str | None = None
    pwc_id: str | None = None
    pathbank_compound_id: str | int | None = None
    pathbank_protein_id: str | int | None = None
    pathbank_complex_id: str | int | None = None
    pathbank_protein_complex_id: str | int | None = None


class MappingResolutionModel(_RuntimeModel):
    status: str | None = None
    issue: str | None = None
    order_step: str | None = None


class SpeciesReferenceModel(_RuntimeModel):
    name: str | None = None
    taxonomy_id: str | int | None = None
    species_id: str | int | None = None
    pathbank_species_id: str | int | None = None
    confidence: int | float | None = None


class MappingMetaModel(_RuntimeModel):
    query: Dict[str, Any] | None = None
    route: str | None = None
    route_reason: str | None = None
    provider: str | None = None
    providers: List[str] | None = None
    source: str | None = None
    candidates: List[Dict[str, Any]] | None = None
    chosen_rule: str | None = None
    confidence: int | float | None = None
    reviewed: bool | None = None
    resolution: MappingResolutionModel | None = None
    species_resolution: SpeciesReferenceModel | None = None
    issues: List[Dict[str, Any]] | None = None
    pathbank_compound_id: int | None = None
    pathbank_protein_id: int | None = None
    pathbank_complex_id: int | None = None
    pathbank_protein_complex_id: int | None = None
    species_id: int | None = None
    db_gap_resolved: bool | None = None
    fallback_used: bool | None = None
    fallback_reason: str | None = None
    cross_species_placeholder: bool | None = None
    target_organism: str | None = None
    placeholder_target_organisms: List[str] | None = None
    functional_enzyme_name: str | None = None
    sentinel_pathbank_protein_id: int | None = None


class NamedRecordModel(_RuntimeModel):
    name: str


class CommonEntityModel(NamedRecordModel):
    mapped_ids: MappedIdsModel | None = None
    mapping_meta: MappingMetaModel | None = None
    confidence: int | float | None = None
    provenance: str | None = None
    evidence: str | None = None
    source_refs: List[str] | None = None
    aliases: List[str] | None = None
    raw_name: str | None = None
    organism: str | None = None
    species: str | None = None
    species_ref: SpeciesReferenceModel | None = None
    species_id: str | int | None = None
    pathbank_species_id: str | int | None = None
    db_id: str | int | None = None
    db_row: Dict[str, Any] | None = None
    db_match: Dict[str, Any] | None = None


class CellTypeModel(NamedRecordModel):
    ontology_id: str | None = None
    pathwhiz_id: str | int | None = None
    aliases: List[str] | None = None
    mapping_meta: MappingMetaModel | None = None


class SpeciesModel(NamedRecordModel):
    taxonomy_id: str | int | None = None
    classification: str | None = None
    common_name: str | None = None
    pathwhiz_id: str | int | None = None
    pathbank_species_id: str | int | None = None
    aliases: List[str] | None = None
    mapping_meta: MappingMetaModel | None = None


class TissueModel(CellTypeModel):
    pass


class SubcellularLocationModel(CellTypeModel):
    canonical_id: str | None = None
    canonical_name: str | None = None
    go_id: str | None = None
    evidence: str | None = None


class CompoundModel(CommonEntityModel):
    pathbank_compound_id: str | int | None = None
    hmdb_id: str | None = None
    kegg_id: str | None = None
    chebi_id: str | None = None
    pubchem_cid: str | None = None
    drugbank_id: str | None = None
    synonyms: List[str] | None = None


class ElementCollectionModel(CommonEntityModel):
    element_states: List[Dict[str, Any]] | None = None


class NucleicAcidModel(CommonEntityModel):
    pass


class ProteinModel(CommonEntityModel):
    pathbank_protein_id: str | int | None = None
    pw_protein_id: str | int | None = None
    uniprot: str | None = None
    uniprot_id: str | None = None
    drugbank: str | None = None
    drugbank_id: str | None = None
    gene_name: str | None = None
    gene: str | None = None
    ec_number: str | None = None
    ec_numbers: List[str] | None = None


class ProteinComplexComponentModel(_RuntimeModel):
    name: str | None = None
    protein: str | None = None
    protein_name: str | None = None
    component: str | None = None
    entity: str | None = None
    gene: str | None = None
    gene_name: str | None = None
    stoichiometry: int | float | str | None = None
    coefficient: int | float | str | None = None
    pathbank_protein_id: str | int | None = None
    mapped_ids: MappedIdsModel | None = None


class ProteinComplexModel(CommonEntityModel):
    components: List[str | ProteinComplexComponentModel] | None = None
    cofactors: List[str] | None = None
    modifications: List[str] | None = None
    pathbank_complex_id: str | int | None = None
    pathbank_protein_complex_id: str | int | None = None
    issues: List[Dict[str, Any]] | None = None
    generated: bool | None = None
    generation_reason: Literal["single_protein_pathwhiz_wrapper"] | None = None


class BoundModel(NamedRecordModel):
    evidence: str | None = None
    confidence: int | float | None = None
    provenance: str | None = None
    source_refs: List[str] | None = None
    mapping_meta: MappingMetaModel | None = None


class EntitiesModel(_RuntimeModel):
    cell_types: List[CellTypeModel] | None = None
    species: List[SpeciesModel] | None = None
    tissues: List[TissueModel] | None = None
    subcellular_locations: List[SubcellularLocationModel] | None = None
    compounds: List[CompoundModel] | None = None
    element_collections: List[ElementCollectionModel] | None = None
    nucleic_acids: List[NucleicAcidModel] | None = None
    proteins: List[ProteinModel] | None = None
    protein_complexes: List[ProteinComplexModel] | None = None
    bounds: List[BoundModel] | None = None


class BiologicalStateModel(NamedRecordModel):
    species: str | None = None
    cell_type: str | None = None
    tissue: str | None = None
    subcellular_location: str | None = None
    compartment: str | None = None
    compartment_canonical: str | None = None
    evidence: str | None = None
    confidence: int | float | None = None
    provenance: str | None = None
    source_refs: List[str] | None = None
    mapping_meta: MappingMetaModel | None = None
    pathwhiz_id: str | int | None = None
    pwbs_id: str | int | None = None


class ElementLocationModel(_RuntimeModel):
    biological_state: str | None = None
    evidence: str | None = None
    confidence: int | float | None = None
    provenance: str | None = None
    source_refs: List[str] | None = None


class CompoundLocationModel(ElementLocationModel):
    compound: str | None = None


class ElementCollectionLocationModel(ElementLocationModel):
    element_collection: str | None = None


class NucleicAcidLocationModel(ElementLocationModel):
    nucleic_acid: str | None = None


class ProteinLocationModel(ElementLocationModel):
    protein: str | None = None


class ElementLocationsModel(_RuntimeModel):
    compound_locations: List[CompoundLocationModel] | None = None
    element_collection_locations: List[ElementCollectionLocationModel] | None = None
    nucleic_acid_locations: List[NucleicAcidLocationModel] | None = None
    protein_locations: List[ProteinLocationModel] | None = None


class ActorModel(_RuntimeModel):
    entity: str | None = None
    protein: str | None = None
    protein_complex: str | None = None
    name: str | None = None
    entity_type: Literal["protein", "protein_complex", "compound"] | None = None
    role: str | None = None
    biological_state: str | None = None
    evidence: str | None = None
    confidence: int | float | None = None
    provenance: str | None = None
    source_refs: List[str] | None = None


class ProcessParticipantModel(_RuntimeModel):
    name: str | None = None
    entity: str | None = None
    compound: str | None = None
    protein: str | None = None
    protein_complex: str | None = None
    element: str | None = None
    element_collection: str | None = None
    nucleic_acid: str | None = None
    biological_state: str | None = None
    stoichiometry: int | float | str | None = None
    coefficient: int | float | str | None = None
    evidence: str | None = None


ParticipantLike = str | ProcessParticipantModel


class ElementWithStateModel(_RuntimeModel):
    side: Literal["left", "right"] | None = None
    element: str | None = None
    biological_state: str | None = None
    evidence: str | None = None
    confidence: int | float | None = None
    provenance: str | None = None
    source_refs: List[str] | None = None


class ReactionModel(NamedRecordModel):
    inputs: List[ParticipantLike] | None = None
    outputs: List[ParticipantLike] | None = None
    enzymes: List[ActorModel] | None = None
    modifiers: List[ActorModel] | None = None
    biological_state: str | None = None
    evidence: str | None = None
    scope_membership: str | None = None
    confidence: int | float | None = None
    provenance: str | None = None
    source_refs: List[str] | None = None
    spontaneous: bool | None = None


class ReactionCoupledTransportModel(NamedRecordModel):
    reaction: str | None = None
    transport: str | None = None
    modifiers: List[ActorModel] | None = None
    enzymes: List[ActorModel] | None = None
    elements_with_states: List[ElementWithStateModel] | None = None
    evidence: str | None = None
    confidence: int | float | None = None
    provenance: str | None = None
    source_refs: List[str] | None = None


class TransportModel(NamedRecordModel):
    cargo: str | None = None
    from_biological_state: str | None = None
    to_biological_state: str | None = None
    transporters: List[ActorModel] | None = None
    elements_with_states: List[ElementWithStateModel] | None = None
    evidence: str | None = None
    confidence: int | float | None = None
    provenance: str | None = None
    source_refs: List[str] | None = None


class InteractionModel(NamedRecordModel):
    entity_1: str | None = None
    relationship: str | None = None
    entity_2: str | None = None
    participants: List[ActorModel] | None = None
    biological_state: str | None = None
    evidence: str | None = None
    confidence: int | float | None = None
    provenance: str | None = None
    source_refs: List[str] | None = None


class SubPathwayModel(NamedRecordModel):
    sub_pathway_type: str | None = None
    reference_pathway_id: str | int | None = None
    alttext: str | None = None
    evidence: str | None = None
    confidence: int | float | None = None
    provenance: str | None = None
    source_refs: List[str] | None = None


class ProcessesModel(_RuntimeModel):
    reactions: List[ReactionModel] | None = None
    reaction_coupled_transports: List[ReactionCoupledTransportModel] | None = None
    transports: List[TransportModel] | None = None
    interactions: List[InteractionModel] | None = None
    sub_pathways: List[SubPathwayModel] | None = None


class VisualizationModel(NamedRecordModel):
    bound: str | None = None
    compound: str | None = None
    protein: str | None = None
    protein_complex: str | None = None
    nucleic_acid: str | None = None
    element_collection: str | None = None
    reaction: str | None = None
    reaction_coupled_transport: str | None = None
    transport: str | None = None
    interaction: str | None = None
    sub_pathway: str | None = None
    biological_state: str | None = None
    x: int | float | None = None
    y: int | float | None = None
    zindex: int | None = None
    hidden: bool | None = None
    visualization_template_id: str | int | None = None


class VacuousVisualizationsModel(_RuntimeModel):
    vacuous_compound_visualizations: List[VisualizationModel] | None = None
    vacuous_nucleic_acid_visualizations: List[VisualizationModel] | None = None
    vacuous_element_collection_visualizations: List[VisualizationModel] | None = None
    vacuous_protein_visualizations: List[VisualizationModel] | None = None


class VisualizationsModel(_RuntimeModel):
    bounds: List[VisualizationModel] | None = None
    bound_visualizations: List[VisualizationModel] | None = None
    protein_complex_visualizations: List[VisualizationModel] | None = None
    reaction_visualizations: List[VisualizationModel] | None = None
    reaction_coupled_transport_visualizations: List[VisualizationModel] | None = None
    transport_visualizations: List[VisualizationModel] | None = None
    interaction_visualizations: List[VisualizationModel] | None = None
    sub_pathway_visualizations: List[VisualizationModel] | None = None
    edges: List[VisualizationModel] | None = None
    vacuous: VacuousVisualizationsModel | None = None


class PayloadModel(_RuntimeModel):
    metadata: MetadataModel | None = None
    scope_warning: str | None = None
    compartments: List[NamedRecordModel] | None = None
    entities: EntitiesModel
    biological_states: List[BiologicalStateModel] | None = None
    element_locations: ElementLocationsModel | None = None
    processes: ProcessesModel
    visualizations: VisualizationsModel | None = None


_NORMALIZED_BOUNDARIES = {
    "post_normalization",
    "post_audit",
    "post_remap",
    "post_enrichment",
    "pre_export",
}
_UNION_LOCATION_MARKERS = {
    "str",
    "int",
    "float",
    "bool",
    "dict[str,any]",
    "ProcessParticipantModel",
    "ProteinComplexComponentModel",
}


def _escape_pointer_token(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _json_pointer(location: Sequence[Any]) -> str:
    cleaned = [item for item in location if str(item) not in _UNION_LOCATION_MARKERS]
    if not cleaned:
        return "/"
    return "/" + "/".join(_escape_pointer_token(item) for item in cleaned)


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _expected_for_error(error_type: str, context: Mapping[str, Any]) -> str:
    if error_type in {"string_type", "string_sub_type"}:
        return "string"
    if error_type in {"list_type", "tuple_type", "set_type"}:
        return "array"
    if error_type in {"dict_type", "mapping_type", "model_type", "model_attributes_type"}:
        return "object"
    if error_type == "bool_type":
        return "boolean"
    if error_type == "int_type":
        return "integer"
    if error_type in {"float_type", "finite_number"}:
        return "number"
    if error_type == "literal_error":
        return str(context.get("expected") or "allowed literal")
    if error_type == "missing":
        return "required field"
    return "valid value"


def _message_for_error(error_type: str, expected: str) -> str:
    if error_type == "missing":
        return "Required field is missing."
    article = "an" if expected[:1].lower() in "aeiou" else "a"
    return f"Expected {article} {expected}."


def _pydantic_issues(exc: ValidationError) -> List[RuntimeSchemaIssue]:
    issues: List[RuntimeSchemaIssue] = []
    for raw in exc.errors(include_url=False):
        error_type = str(raw.get("type") or "")
        context = raw.get("ctx") if isinstance(raw.get("ctx"), dict) else {}
        expected = _expected_for_error(error_type, context)
        issue: RuntimeSchemaIssue = {
            "code": (
                "runtime_schema_missing_field"
                if error_type == "missing"
                else "runtime_schema_value_error"
                if error_type == "literal_error"
                else "runtime_schema_type_error"
            ),
            "pointer": _json_pointer(raw.get("loc") or ()),
            "message": _message_for_error(error_type, expected),
            "expected": expected,
            "received": _json_type(raw.get("input")),
        }
        issues.append(issue)
    return _dedupe_issues(issues)


def _issue(
    code: str,
    pointer: str,
    message: str,
    *,
    expected: str | None = None,
    received: Any = None,
) -> RuntimeSchemaIssue:
    result: RuntimeSchemaIssue = {"code": code, "pointer": pointer, "message": message}
    if expected:
        result["expected"] = expected
    if received is not None:
        result["received"] = _json_type(received)
    return result


def _dedupe_issues(issues: Iterable[RuntimeSchemaIssue]) -> List[RuntimeSchemaIssue]:
    deduped: List[RuntimeSchemaIssue] = []
    seen: set[tuple[str, str]] = set()
    for issue in issues:
        identity = (str(issue.get("code") or ""), str(issue.get("pointer") or ""))
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(issue)
    return deduped


def _named_entity_rows(payload: Any) -> Iterable[tuple[str, int, Mapping[str, Any]]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("entities"), dict):
        return
    for bucket, rows in payload["entities"].items():
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if isinstance(row, dict) and isinstance(row.get("name"), str) and row["name"].strip():
                yield str(bucket), index, row


def _mapping_boundary_issues(payload: Any) -> List[RuntimeSchemaIssue]:
    issues: List[RuntimeSchemaIssue] = []
    entities = payload.get("entities") if isinstance(payload, dict) else None
    species = entities.get("species") if isinstance(entities, dict) else None
    if not isinstance(species, list) or not species:
        issues.append(
            _issue(
                "species_required",
                "/entities/species",
                "Mapped payload must include at least one species row.",
                expected="non-empty array",
                received=species,
            )
        )
    for bucket, index, row in _named_entity_rows(payload):
        pointer = f"/entities/{_escape_pointer_token(bucket)}/{index}/mapping_meta"
        meta = row.get("mapping_meta")
        if not isinstance(meta, dict):
            issues.append(
                _issue(
                    "entity_missing_mapping_meta",
                    pointer,
                    "Mapped entity with a name is missing mapping_meta.",
                    expected="object",
                    received=meta,
                )
            )
            continue
        resolution = meta.get("resolution")
        if not isinstance(resolution, dict):
            issues.append(
                _issue(
                    "mapping_resolution_required",
                    f"{pointer}/resolution",
                    "Mapping metadata must include a resolution object.",
                    expected="object",
                    received=resolution,
                )
            )
            continue
        status = resolution.get("status")
        if not isinstance(status, str) or not status.strip():
            issues.append(
                _issue(
                    "mapping_resolution_status_required",
                    f"{pointer}/resolution/status",
                    "Mapping resolution must include a non-empty status string.",
                    expected="non-empty string",
                    received=status,
                )
            )
    return issues


def _normalized_actor_issues(payload: Any) -> List[RuntimeSchemaIssue]:
    issues: List[RuntimeSchemaIssue] = []
    processes = payload.get("processes") if isinstance(payload, dict) else None
    if not isinstance(processes, dict):
        return issues
    actor_slots = (
        ("reactions", "enzymes"),
        ("reactions", "modifiers"),
        ("transports", "transporters"),
        ("interactions", "participants"),
    )
    for process_bucket, actor_field in actor_slots:
        rows = processes.get(process_bucket)
        if not isinstance(rows, list):
            continue
        for process_index, process in enumerate(rows):
            if not isinstance(process, dict) or not isinstance(process.get(actor_field), list):
                continue
            for actor_index, actor in enumerate(process[actor_field]):
                pointer = f"/processes/{process_bucket}/{process_index}/{actor_field}/{actor_index}"
                if not isinstance(actor, dict):
                    continue
                entity = actor.get("entity")
                entity_type = actor.get("entity_type")
                if (
                    not isinstance(entity, str)
                    or not entity.strip()
                    or not isinstance(entity_type, str)
                    or not entity_type.strip()
                ):
                    issues.append(
                        _issue(
                            "actor_schema_not_canonical",
                            pointer,
                            "Normalized actor rows must include non-empty entity and entity_type fields.",
                        )
                    )
    return issues


def _generated_wrapper_issues(payload: Any) -> List[RuntimeSchemaIssue]:
    issues: List[RuntimeSchemaIssue] = []
    entities = payload.get("entities") if isinstance(payload, dict) else None
    rows = entities.get("protein_complexes") if isinstance(entities, dict) else None
    if not isinstance(rows, list):
        return issues
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        is_wrapper_shape = row.get("generated") is True or row.get("generation_reason") is not None
        if not is_wrapper_shape:
            continue
        base = f"/entities/protein_complexes/{index}"
        if row.get("generated") is not True:
            issues.append(
                _issue(
                    "generated_wrapper_flag_required",
                    f"{base}/generated",
                    "Generated wrapper rows must set generated to true.",
                    expected="boolean true",
                    received=row.get("generated"),
                )
            )
        if row.get("generation_reason") != "single_protein_pathwhiz_wrapper":
            issues.append(
                _issue(
                    "generated_wrapper_reason_invalid",
                    f"{base}/generation_reason",
                    "Generated wrapper rows must use the supported generation reason.",
                    expected="single_protein_pathwhiz_wrapper",
                    received=row.get("generation_reason"),
                )
            )
        components = row.get("components")
        if not isinstance(components, list) or not components:
            issues.append(
                _issue(
                    "generated_wrapper_missing_components",
                    f"{base}/components",
                    "Generated protein-complex wrapper must declare at least one component.",
                    expected="non-empty array",
                    received=components,
                )
            )
            continue
        for component_index, component in enumerate(components):
            pointer = f"{base}/components/{component_index}"
            if not isinstance(component, dict):
                issues.append(
                    _issue(
                        "generated_wrapper_component_not_structured",
                        pointer,
                        "Generated wrapper components must be structured objects.",
                        expected="object",
                        received=component,
                    )
                )
                continue
            component_name = next(
                (
                    component.get(key).strip()
                    for key in ("name", "protein", "protein_name", "component", "entity")
                    if isinstance(component.get(key), str) and component[key].strip()
                ),
                "",
            )
            if not component_name:
                issues.append(
                    _issue(
                        "generated_wrapper_component_name_required",
                        pointer,
                        "Generated wrapper component objects must include a non-empty protein name.",
                    )
                )
    return issues


def validate_payload_shape(
    payload: Any,
    *,
    boundary: RuntimeSchemaBoundary,
    mode: RuntimeSchemaMode = "report",
) -> RuntimeSchemaReport:
    """Validate the known payload shape without mutating or replacing ``payload``."""

    errors: List[RuntimeSchemaIssue] = []
    try:
        PayloadModel.model_validate(payload, strict=True)
    except ValidationError as exc:
        errors.extend(_pydantic_issues(exc))

    if boundary in {"post_mapping", *list(_NORMALIZED_BOUNDARIES)}:
        errors.extend(_mapping_boundary_issues(payload))
    if boundary in _NORMALIZED_BOUNDARIES:
        errors.extend(_normalized_actor_issues(payload))
    if boundary in {"post_remap", "post_enrichment", "pre_export"}:
        errors.extend(_generated_wrapper_issues(payload))

    errors = _dedupe_issues(errors)
    report: RuntimeSchemaReport = {
        "ok": not errors,
        "boundary": boundary,
        "mode": mode,
        "errors": errors,
        "warnings": [],
        "summary": {"error_count": len(errors), "warning_count": 0},
    }
    if mode == "enforce" and errors:
        raise RuntimeSchemaValidationError(report)
    return report


__all__ = [
    "RuntimeSchemaBoundary",
    "RuntimeSchemaIssue",
    "RuntimeSchemaMode",
    "RuntimeSchemaReport",
    "RuntimeSchemaValidationError",
    "validate_payload_shape",
]
