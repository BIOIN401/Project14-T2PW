"""Typed payload contracts for the T2PW pipeline.

These types document the JSON shapes exchanged between extraction, mapping,
normalization, audit, and export. They intentionally do not validate or mutate
payloads at runtime.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict, Union


JsonValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]

PayloadEntityType = Literal[
    "compound",
    "protein",
    "protein_complex",
    "nucleic_acid",
    "element_collection",
]
PayloadActorEntityType = Literal["protein", "protein_complex", "compound"]
PayloadActorRole = Literal[
    "catalyst",
    "inhibitor",
    "activator",
    "transporter",
    "participant",
]
PayloadProvenance = Literal["extracted", "inferred", "curated", "enriched"]
PayloadReactionScope = Literal[
    "core",
    "anaplerotic",
    "cataplerotic",
    "auxiliary",
    "out_of_scope",
]
PayloadReactionClass = Literal[
    "biochemical_reaction",
    "transport_reaction",
    "interaction",
    "subpathway",
    "activation",
    "inhibition",
    "binding",
]
PayloadInteractionClass = Literal["activation", "inhibition", "binding", "interaction"]

PayloadClassField = TypedDict("PayloadClassField", {"class": str}, total=False)
PayloadCompoundClassField = TypedDict(
    "PayloadCompoundClassField",
    {"class": Literal["compound", "cofactor", "ion", "element_collection"]},
    total=False,
)
PayloadElementCollectionClassField = TypedDict(
    "PayloadElementCollectionClassField",
    {"class": Literal["element_collection"]},
    total=False,
)
PayloadNucleicAcidClassField = TypedDict(
    "PayloadNucleicAcidClassField",
    {"class": Literal["nucleic_acid"]},
    total=False,
)
PayloadProteinClassField = TypedDict(
    "PayloadProteinClassField",
    {"class": Literal["protein", "transporter"]},
    total=False,
)
PayloadProteinComplexClassField = TypedDict(
    "PayloadProteinComplexClassField",
    {"class": Literal["protein_complex"]},
    total=False,
)
PayloadReactionClassField = TypedDict(
    "PayloadReactionClassField",
    {"class": PayloadReactionClass},
    total=False,
)
PayloadTransportReactionClassField = TypedDict(
    "PayloadTransportReactionClassField",
    {"class": Literal["transport_reaction"]},
    total=False,
)
PayloadInteractionClassField = TypedDict(
    "PayloadInteractionClassField",
    {"class": PayloadInteractionClass},
    total=False,
)


class PayloadMappedIds(TypedDict, total=False):
    """External and PathBank identifiers attached by mapping stages."""

    hmdb: str
    kegg: str
    chebi: str
    pubchem: str
    drugbank: str
    uniprot: str
    biocyc: str
    chemspider: str
    pwc_id: str
    pathbank_compound_id: Union[str, int]
    pathbank_protein_id: Union[str, int]
    pathbank_complex_id: Union[str, int]
    pathbank_protein_complex_id: Union[str, int]


class PayloadMappingResolution(TypedDict, total=False):
    status: str
    issue: str
    order_step: str


class PayloadMappingMeta(TypedDict, total=False):
    """Mapping metadata written onto entity rows."""

    query: Dict[str, Any]
    route: str
    route_reason: str
    providers: List[str]
    source: str
    candidates: List[Dict[str, Any]]
    chosen_rule: str
    confidence: float
    resolution: PayloadMappingResolution
    pathbank_compound_id: int
    pathbank_protein_id: int
    pathbank_complex_id: int
    pathbank_protein_complex_id: int
    species_id: int
    db_gap_resolved: bool
    fallback_used: bool
    fallback_reason: str
    cross_species_placeholder: bool
    target_organism: str
    placeholder_target_organisms: List[str]
    functional_enzyme_name: str
    sentinel_pathbank_protein_id: int


class PayloadNamedRecord(TypedDict):
    name: str


class PayloadCommonRecord(PayloadNamedRecord, PayloadClassField, total=False):
    """Common optional fields used by extracted, mapped, and curated rows."""

    mapped_ids: PayloadMappedIds
    mapping_meta: PayloadMappingMeta
    confidence: float
    provenance: PayloadProvenance
    evidence: str
    source_refs: List[str]
    aliases: List[str]
    raw_name: str
    db_status: str
    db_id: Union[str, int]
    db_row: Dict[str, Any]
    db_match: Dict[str, Any]
    chosen_rule: str
    organism: str
    species: str
    species_id: Union[str, int]
    pathbank_species_id: Union[str, int]


class PayloadCellType(PayloadNamedRecord, total=False):
    ontology_id: str
    pathwhiz_id: Union[str, int]
    aliases: List[str]


class PayloadSpecies(PayloadNamedRecord, total=False):
    taxonomy_id: str
    classification: str
    common_name: str
    pathwhiz_id: Union[str, int]
    pathbank_species_id: Union[str, int]
    aliases: List[str]
    mapping_meta: PayloadMappingMeta


class PayloadTissue(PayloadNamedRecord, total=False):
    ontology_id: str
    pathwhiz_id: Union[str, int]
    aliases: List[str]


class PayloadCompartment(PayloadNamedRecord, total=False):
    canonical_name: str
    go_id: str


class PayloadSubcellularLocation(PayloadNamedRecord, total=False):
    canonical_id: str
    ontology_id: str
    go_id: str
    evidence: str
    pathwhiz_id: Union[str, int]
    aliases: List[str]
    mapping_meta: PayloadMappingMeta


class PayloadCompound(PayloadCommonRecord, PayloadCompoundClassField, total=False):
    pathbank_compound_id: Union[str, int]
    hmdb_id: str
    kegg_id: str
    chebi_id: str
    pubchem_cid: str
    drugbank_id: str
    cas: str
    biocyc_id: str
    chemspider_id: str
    moldb_inchi: str
    moldb_inchikey: str
    short_name: str
    synonyms: List[str]
    type: str
    compound_class: str
    compound_type: str


class PayloadElementCollection(PayloadCommonRecord, PayloadElementCollectionClassField, total=False):
    element_states: List[Dict[str, Any]]


class PayloadNucleicAcid(PayloadCommonRecord, PayloadNucleicAcidClassField, total=False):
    pass


class PayloadProtein(PayloadCommonRecord, PayloadProteinClassField, total=False):
    pathbank_protein_id: Union[str, int]
    pw_protein_id: Union[str, int]
    uniprot: str
    uniprot_id: str
    drugbank: str
    drugbank_id: str
    gene_name: str
    gene: str
    ec_number: str
    ec_numbers: List[str]


class PayloadProteinComplexComponent(TypedDict, total=False):
    name: str
    protein: str
    protein_name: str
    component: str
    entity: str
    gene: str
    gene_name: str
    stoichiometry: Union[int, str]
    coefficient: Union[int, str]
    mapped_ids: PayloadMappedIds


PayloadProteinComplexComponentLike = Union[str, PayloadProteinComplexComponent]


class PayloadProteinComplex(PayloadCommonRecord, PayloadProteinComplexClassField, total=False):
    components: List[PayloadProteinComplexComponentLike]
    cofactors: List[str]
    modifications: List[str]
    pathbank_complex_id: Union[str, int]
    pathbank_protein_complex_id: Union[str, int]
    issues: List[Dict[str, Any]]
    generated: bool
    generation_reason: Literal["single_protein_pathwhiz_wrapper"]


class PayloadBound(PayloadNamedRecord, total=False):
    evidence: str
    confidence: float
    provenance: PayloadProvenance
    source_refs: List[str]


class PayloadEntities(TypedDict, total=False):
    cell_types: List[PayloadCellType]
    species: List[PayloadSpecies]
    tissues: List[PayloadTissue]
    subcellular_locations: List[PayloadSubcellularLocation]
    compounds: List[PayloadCompound]
    element_collections: List[PayloadElementCollection]
    nucleic_acids: List[PayloadNucleicAcid]
    proteins: List[PayloadProtein]
    protein_complexes: List[PayloadProteinComplex]
    bounds: List[PayloadBound]


class PayloadBiologicalState(PayloadNamedRecord, total=False):
    species: str
    cell_type: str
    tissue: str
    subcellular_location: str
    compartment: str
    compartment_canonical: str
    evidence: str
    confidence: float
    provenance: PayloadProvenance
    source_refs: List[str]
    mapping_meta: PayloadMappingMeta
    pathwhiz_id: Union[str, int]
    pwbs_id: Union[str, int]


class PayloadElementLocation(TypedDict, total=False):
    biological_state: str
    evidence: str
    confidence: float
    provenance: PayloadProvenance
    source_refs: List[str]


class PayloadCompoundLocation(PayloadElementLocation, total=False):
    compound: str


class PayloadElementCollectionLocation(PayloadElementLocation, total=False):
    element_collection: str


class PayloadNucleicAcidLocation(PayloadElementLocation, total=False):
    nucleic_acid: str


class PayloadProteinLocation(PayloadElementLocation, total=False):
    protein: str


class PayloadElementLocations(TypedDict, total=False):
    compound_locations: List[PayloadCompoundLocation]
    element_collection_locations: List[PayloadElementCollectionLocation]
    nucleic_acid_locations: List[PayloadNucleicAcidLocation]
    protein_locations: List[PayloadProteinLocation]


class PayloadReactionActor(TypedDict, total=False):
    """Actor row used by reaction modifiers/enzymes and related process slots.

    ``entity`` is the canonical name field. ``protein`` and ``protein_complex``
    remain documented for backwards-compatible payloads that have not yet been
    normalized.
    """

    entity: str
    protein: str
    protein_complex: str
    name: str
    entity_type: PayloadActorEntityType
    role: PayloadActorRole
    biological_state: str
    evidence: str
    confidence: float
    provenance: PayloadProvenance
    source_refs: List[str]


class PayloadProcessParticipant(TypedDict, total=False):
    name: str
    entity: str
    compound: str
    protein: str
    protein_complex: str
    element: str
    element_collection: str
    nucleic_acid: str
    biological_state: str
    stoichiometry: Union[int, float, str]
    coefficient: Union[int, float, str]
    evidence: str


PayloadProcessParticipantLike = Union[str, PayloadProcessParticipant]


class PayloadElementWithState(TypedDict, total=False):
    side: Literal["left", "right"]
    element: str
    biological_state: str
    evidence: str
    confidence: float
    provenance: PayloadProvenance
    source_refs: List[str]


class PayloadReaction(PayloadNamedRecord, PayloadReactionClassField, total=False):
    inputs: List[PayloadProcessParticipantLike]
    outputs: List[PayloadProcessParticipantLike]
    enzymes: List[PayloadReactionActor]
    modifiers: List[PayloadReactionActor]
    biological_state: str
    evidence: str
    scope_membership: PayloadReactionScope
    confidence: float
    provenance: PayloadProvenance
    source_refs: List[str]
    spontaneous: bool


class PayloadReactionCoupledTransport(PayloadNamedRecord, PayloadTransportReactionClassField, total=False):
    reaction: str
    transport: str
    modifiers: List[PayloadReactionActor]
    enzymes: List[PayloadReactionActor]
    elements_with_states: List[PayloadElementWithState]
    evidence: str
    confidence: float
    provenance: PayloadProvenance
    source_refs: List[str]


class PayloadTransport(PayloadNamedRecord, PayloadTransportReactionClassField, total=False):
    cargo: str
    from_biological_state: str
    to_biological_state: str
    transporters: List[PayloadReactionActor]
    elements_with_states: List[PayloadElementWithState]
    evidence: str
    confidence: float
    provenance: PayloadProvenance
    source_refs: List[str]


class PayloadInteraction(PayloadNamedRecord, PayloadInteractionClassField, total=False):
    entity_1: str
    relationship: str
    entity_2: str
    participants: List[PayloadReactionActor]
    biological_state: str
    evidence: str
    confidence: float
    provenance: PayloadProvenance
    source_refs: List[str]


class PayloadSubPathway(PayloadNamedRecord, total=False):
    sub_pathway_type: str
    reference_pathway_id: Union[str, int]
    alttext: str
    evidence: str
    confidence: float
    provenance: PayloadProvenance
    source_refs: List[str]


class PayloadProcesses(TypedDict, total=False):
    reactions: List[PayloadReaction]
    reaction_coupled_transports: List[PayloadReactionCoupledTransport]
    transports: List[PayloadTransport]
    interactions: List[PayloadInteraction]
    sub_pathways: List[PayloadSubPathway]


class PayloadVisualization(PayloadNamedRecord, total=False):
    bound: str
    compound: str
    protein: str
    protein_complex: str
    nucleic_acid: str
    element_collection: str
    reaction: str
    reaction_coupled_transport: str
    transport: str
    interaction: str
    sub_pathway: str
    biological_state: str
    evidence: str
    confidence: float
    provenance: PayloadProvenance
    source_refs: List[str]
    x: Union[int, float]
    y: Union[int, float]
    zindex: int
    hidden: bool
    visualization_template_id: Union[str, int]


class PayloadVacuousVisualizations(TypedDict, total=False):
    vacuous_compound_visualizations: List[PayloadVisualization]
    vacuous_nucleic_acid_visualizations: List[PayloadVisualization]
    vacuous_element_collection_visualizations: List[PayloadVisualization]
    vacuous_protein_visualizations: List[PayloadVisualization]


class PayloadVisualizations(TypedDict, total=False):
    bounds: List[PayloadVisualization]
    bound_visualizations: List[PayloadVisualization]
    protein_complex_visualizations: List[PayloadVisualization]
    reaction_visualizations: List[PayloadVisualization]
    reaction_coupled_transport_visualizations: List[PayloadVisualization]
    transport_visualizations: List[PayloadVisualization]
    interaction_visualizations: List[PayloadVisualization]
    sub_pathway_visualizations: List[PayloadVisualization]
    edges: List[PayloadVisualization]
    vacuous: PayloadVacuousVisualizations


class Payload(TypedDict, total=False):
    scope_warning: str
    compartments: List[PayloadCompartment]
    entities: PayloadEntities
    biological_states: List[PayloadBiologicalState]
    element_locations: PayloadElementLocations
    processes: PayloadProcesses
    visualizations: PayloadVisualizations


class PayloadAdditions(TypedDict, total=False):
    entities: PayloadEntities
    biological_states: List[PayloadBiologicalState]
    element_locations: PayloadElementLocations
    processes: PayloadProcesses


class PayloadQaHints(TypedDict, total=False):
    intended_effect: str
    expected_changes: List[str]


class PayloadInferenceResult(TypedDict, total=False):
    additions: PayloadAdditions
    qa_hints: PayloadQaHints


__all__ = [
    "JsonValue",
    "Payload",
    "PayloadActorEntityType",
    "PayloadActorRole",
    "PayloadAdditions",
    "PayloadBiologicalState",
    "PayloadBound",
    "PayloadCellType",
    "PayloadClassField",
    "PayloadCommonRecord",
    "PayloadCompartment",
    "PayloadCompound",
    "PayloadCompoundClassField",
    "PayloadCompoundLocation",
    "PayloadElementCollection",
    "PayloadElementCollectionClassField",
    "PayloadElementCollectionLocation",
    "PayloadElementLocation",
    "PayloadElementLocations",
    "PayloadElementWithState",
    "PayloadEntities",
    "PayloadEntityType",
    "PayloadInferenceResult",
    "PayloadInteraction",
    "PayloadInteractionClass",
    "PayloadInteractionClassField",
    "PayloadMappedIds",
    "PayloadMappingMeta",
    "PayloadMappingResolution",
    "PayloadNamedRecord",
    "PayloadNucleicAcid",
    "PayloadNucleicAcidClassField",
    "PayloadNucleicAcidLocation",
    "PayloadProcessParticipant",
    "PayloadProcessParticipantLike",
    "PayloadProcesses",
    "PayloadProtein",
    "PayloadProteinClassField",
    "PayloadProteinComplex",
    "PayloadProteinComplexClassField",
    "PayloadProteinComplexComponent",
    "PayloadProteinComplexComponentLike",
    "PayloadProteinLocation",
    "PayloadProvenance",
    "PayloadQaHints",
    "PayloadReaction",
    "PayloadReactionActor",
    "PayloadReactionClass",
    "PayloadReactionClassField",
    "PayloadReactionCoupledTransport",
    "PayloadReactionScope",
    "PayloadSpecies",
    "PayloadSubPathway",
    "PayloadSubcellularLocation",
    "PayloadTissue",
    "PayloadTransport",
    "PayloadTransportReactionClassField",
    "PayloadVacuousVisualizations",
    "PayloadVisualization",
    "PayloadVisualizations",
]
