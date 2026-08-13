"""Two versioned canonical projections of a pipeline payload (D-001).

``canonical_graph_sha256`` covers biological content only -- EXPORTERS bind to it,
so lineage churn must NOT move it. ``canonical_payload_sha256`` covers the whole
saved ``final_mapped.json``, lineage and evidence included -- PERSISTED-ARTIFACT
INTEGRITY binds to it, so lineage churn MUST move it. Both drop
:data:`HASH_AND_STAMP_FIELDS` from their own input: a hash is never an input to
itself. Schema 1 is ``gate_reports.payload_sha256`` and is never redefined here.

:data:`GRAPH_FIELDS` and :data:`GRAPH_SECTIONS` are an ALLOWLIST, never a denylist,
and the projection descends only through keys they name. A denylist would admit
every future field into the graph hash by default, so "the biology did not change"
would become a claim about fields nobody chose. The accepted cost is the opposite
failure -- a new biological field stays invisible to the graph hash until it is
named here -- and D-001 takes that trade deliberately.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, FrozenSet, List, Mapping

#: The split-hash schema. Historical artifacts stay schema 1 and keep its meaning.
HASH_SCHEMA_VERSION = 2
HASH_SCHEMA_VERSION_LEGACY = 1
SUPPORTED_HASH_SCHEMA_VERSIONS = frozenset({HASH_SCHEMA_VERSION_LEGACY, HASH_SCHEMA_VERSION})
HASH_SCHEMA_VERSION_KEY = "hash_schema_version"
CANONICAL_GRAPH_SHA256_KEY = "canonical_graph_sha256"
CANONICAL_PAYLOAD_SHA256_KEY = "canonical_payload_sha256"

#: Dropped from BOTH projections' input: a hash may not be an input to itself, and
#: a stamp describes the artifact rather than the biology inside it.
HASH_AND_STAMP_FIELDS: FrozenSet[str] = frozenset(
    "payload_sha256 canonical_graph_sha256 canonical_payload_sha256 hash_schema_version "
    "report_schema_version phase artifact_set_version".split())

#: Every field PRODUCT_CONTRACT section 5 requires to stay equivalent, and no other.
#: A key absent here is dropped wherever it appears, and the projection never
#: descends into it -- which is what keeps lineage, evidence, confidence,
#: provenance, rag_provenance, rag_confidence, source_papers, source_refs,
#: mapping_meta, candidates, synonyms, aliases, quarantine and policy reports,
#: scope labels and internal ids out of the graph hash along with their contents.
GRAPH_FIELDS: FrozenSet[str] = frozenset((
    # name and class
    "name class type canonical_name common_name classification compound_class "
    "compound_type short_name "
    # external identifiers, on the record and under mapped_ids
    "mapped_ids biocyc biocyc_id canonical_id cas chebi chebi_id chemspider "
    "chemspider_id drugbank drugbank_id ec_number ec_numbers gene gene_name go_id "
    "hmdb hmdb_id kegg kegg_id ontology_id pathbank_complex_id pathbank_compound_id "
    "pathbank_protein_complex_id pathbank_protein_id pathbank_species_id pathwhiz_id "
    "pubchem pubchem_cid pw_protein_id pwbs_id pwc_id uniprot uniprot_id "
    # the rest of the identity keys ir.py consumes (its bucket key lists, its
    # component list and its uniprot/drugbank lists). Named here because
    # graph_projection may only admit a key this allowlist names, and an
    # identifier the exporter binds an entity's identity to is exactly the kind
    # of key that must not stay invisible. No committed artifact carries any of
    # them at any depth, so naming them moves no existing graph hash.
    "drugbank-id pathbank_cell_type_id pathbank_element_collection_id "
    "pathbank_nucleic_acid_id pathbank_subcellular_location_id pathbank_tissue_id "
    "protein_id pw_cell_type_id pw_complex_id pw_compound_id pw_element_collection_id "
    "pw_nucleic_acid_id pw_species_id pw_subcellular_location_id pw_tissue_id "
    "uniprot-id "
    # organism context
    "organism species species_id species_name species_ref taxonomy_id "
    # complexes and their components
    "components cofactors modifications "
    # cellular locations
    "biological_state cell_type compartment compartment_canonical element_states "
    "location side subcellular_location tissue "
    # process -> entity references, with role and stoichiometry
    "compound element element_collection entity entity_type nucleic_acid protein "
    "protein_complex role stoichiometry coefficient participants elements_with_states "
    # reactions, transports and interactions
    "inputs outputs enzymes modifiers transporters cargo direction reversible "
    "spontaneous from_biological_state to_biological_state entity_1 entity_2 "
    "interaction_type relationship reaction transport reference_pathway_id "
    "sub_pathway_type"
).split())

#: Payload sections that hold buckets of records, and the buckets each may hold. A
#: section or bucket unnamed here never reaches the graph hash.
GRAPH_SECTIONS: Dict[str, FrozenSet[str]] = {
    "entities": frozenset(
        "bounds cell_types compounds element_collections nucleic_acids proteins "
        "protein_complexes species subcellular_locations tissues".split()),
    "element_locations": frozenset(
        "compound_locations element_collection_locations nucleic_acid_locations "
        "protein_locations".split()),
    "processes": frozenset(
        "interactions reaction_coupled_transports reactions sub_pathways "
        "transports".split()),
}


def _keep(value: Any, fields: FrozenSet[str] = GRAPH_FIELDS) -> Any:
    """``value`` reduced to the keys ``fields`` names, recursively."""
    if isinstance(value, (list, tuple)):
        return [_keep(item, fields) for item in value]
    if not isinstance(value, Mapping):
        return value  # a participant given as a bare name is biology; keep it
    return {key: _keep(value[key]) for key in value if key in fields}


def _records(value: Any) -> List[Any]:
    return [_keep(row) for row in (value if isinstance(value, (list, tuple)) else [])]


def strip_hash_fields(payload: Any) -> Dict[str, Any]:
    """``payload`` without the hash and stamp keys it carries about itself."""
    source = payload if isinstance(payload, Mapping) else {}
    return {k: v for k, v in source.items() if k not in HASH_AND_STAMP_FIELDS}


def graph_projection(payload: Any) -> Dict[str, Any]:
    """The biological graph of ``payload``, and nothing else.

    ``ir.py`` settles an entity's EXPORTED identity through a four-tier fallback
    (``ir._first_nonempty``): the record, then ``mapping_meta``, then
    ``mapped_ids``, then the FIRST ``mapping_meta.candidates`` entry. :func:`_keep`
    reaches tiers 1 and 3 and neither of the other two, so on 49 committed rows
    the identifier actually exported could change without moving this hash. Tiers
    2 and 4 are qualified back in here (A0-C1) and NOTHING else from them is: one
    value per ordered key list, never the candidate list, its length, its order,
    its scores, its other entries, or a losing key on the winning list.

    So a reorder moves this hash exactly when it moves WHICH value ``ir.py``
    consumes -- which is when the exported identity moved -- and a ``score`` edit,
    or a reshuffle that leaves the consumed value in front, never does.

    The value is merged under its own key, so the hash sees the identity that is
    exported rather than the tier it was found at, and only where :func:`_keep`
    could not already see it -- every record carrying its own identifier still
    projects byte for byte as before. :data:`GRAPH_FIELDS` still decides
    admission: a key it does not name stays out of the graph here too.
    """
    #: ``ir.py``'s ordered identity key lists, by the bucket each is applied to
    #: (``entity_specs``/``component_specs`` via ``_db_id``, ``:2184``/``:2187``
    #: uniprot and drugbank, ``:545`` compound xrefs). MIRRORED rather than
    #: imported: exporters BIND to ``canonical_graph_sha256``, so what the hash
    #: covers may not be defined by the exporter it is checked against. Order is
    #: load-bearing -- ``_first_nonempty`` scans a whole list at one tier before
    #: dropping to the next, so the first key holding a value at the winning tier
    #: is the consumed one and every other key on that list is not.
    uniprot = ("uniprot", "uniprot_id", "uniprot-id")
    protein_db = ("pathbank_protein_id", "pw_protein_id", "pathwhiz_id")
    per_bucket = {
        "cell_types": (("pathbank_cell_type_id", "pw_cell_type_id", "pathwhiz_id"),),
        "compounds": (("pathbank_compound_id", "pw_compound_id", "pathwhiz_id"),
                      ("hmdb_id", "hmdb"), ("kegg_id", "kegg"),
                      ("pubchem_cid", "pubchem"), ("pwc_id",)),
        "element_collections": (("pathbank_element_collection_id",
                                 "pw_element_collection_id", "pathwhiz_id"),),
        "nucleic_acids": (("pathbank_nucleic_acid_id", "pw_nucleic_acid_id",
                           "pathwhiz_id"),),
        "protein_complexes": (("pathbank_complex_id", "pathbank_protein_complex_id",
                               "pw_complex_id", "pathwhiz_id"),),
        "proteins": (protein_db, uniprot,
                     ("drugbank", "drugbank_id", "drugbank-id")),
        "species": (("pathbank_species_id", "pw_species_id", "pathwhiz_id"),),
        "subcellular_locations": (("pathbank_subcellular_location_id",
                                   "pw_subcellular_location_id", "pathwhiz_id"),),
        "tissues": (("pathbank_tissue_id", "pw_tissue_id", "pathwhiz_id"),),
    }
    #: The one nested container ``ir.py`` resolves identity inside: a protein
    #: complex's components, at ``ir.py:1204`` and ``:1212``. No other bucket has
    #: an identity call site below its own record.
    per_component = {"protein_complexes": ("components",
                                           (protein_db + ("protein_id",), uniprot))}

    def consumed(row: Any, key_lists: Any) -> Dict[str, Any]:
        """The identity values ``ir._first_nonempty`` takes from a tier ``_keep``
        cannot see -- and only those. A list that wins at tier 1 or 3 is already
        projected, so it contributes nothing and is not re-stated here."""
        if not isinstance(row, Mapping):
            return {}
        meta = row.get("mapping_meta")
        meta = meta if isinstance(meta, Mapping) else {}
        mapped = row.get("mapped_ids")
        mapped = mapped if isinstance(mapped, Mapping) else {}
        candidates = meta.get("candidates")
        first = candidates[0] if isinstance(candidates, (list, tuple)) and candidates else {}
        first = first if isinstance(first, Mapping) else {}
        found: Dict[str, Any] = {}
        for keys in key_lists:
            for tier, hidden in ((row, False), (meta, True), (mapped, False), (first, True)):
                key = next((k for k in keys
                            if k in tier and tier[k] not in (None, "")), None)
                if key is None:
                    continue  # this tier holds no value for this list; drop to the next
                if hidden and key in GRAPH_FIELDS:
                    found[key] = _keep(tier[key])
                break  # the first tier with a value is the one ir.py consumes
        return found

    def qualified(rows: Any, bucket: str) -> List[Any]:
        """:func:`_records` with each row's consumed fallback identity merged in."""
        kept = _records(rows)
        key_lists = per_bucket.get(bucket, ())
        nested_key, nested_lists = per_component.get(bucket, ("", ()))
        if not (key_lists or nested_lists):
            return kept
        for row, record in zip(rows if isinstance(rows, (list, tuple)) else [], kept):
            if not isinstance(record, dict):
                continue
            record.update(consumed(row, key_lists))
            inner = record.get(nested_key) if nested_lists else None
            outer = row.get(nested_key) if isinstance(row, Mapping) else None
            if isinstance(inner, list) and isinstance(outer, (list, tuple)):
                for source_component, projected in zip(outer, inner):
                    if isinstance(projected, dict):
                        projected.update(consumed(source_component, nested_lists))
        return kept

    source = strip_hash_fields(payload)
    graph: Dict[str, Any] = {
        "metadata": _keep(source.get("metadata"), frozenset({"organism"})),
        "biological_states": _records(source.get("biological_states")),
    }
    for section, buckets in GRAPH_SECTIONS.items():
        node = source.get(section)
        node = node if isinstance(node, Mapping) else {}
        graph[section] = {bucket: qualified(node.get(bucket), bucket) for bucket in buckets}
    return graph


def _digest(value: Any) -> str:
    """The schema-1 serialization, reused so both schemas agree byte for byte."""
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_graph_sha256(payload: Any) -> str:
    """Biological content only. Exporters bind to this; lineage never moves it."""
    return _digest(graph_projection(payload))


def canonical_payload_sha256(payload: Any) -> str:
    """The whole saved artifact, evidence included. Integrity binds to this.

    Equals ``payload_sha256`` for a payload carrying no stamp of its own, which is
    what keeps every historical binding reproducible.
    """
    return _digest(strip_hash_fields(payload))
