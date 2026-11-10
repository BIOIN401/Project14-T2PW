"""Clean-model dataclasses (P-14..P-25).

Every value is a ``Field(value, prov)``; every payload key without a typed home lands in ``extras`` so
``roundtrip.to_payload(build(p)) == p``. ``UNSET`` + label ``unknown`` means "the payload is silent":
nothing is invented, and ``to_raw`` skips such fields. ``frozen=True`` marks values that came from the
frozen payload: the online layer must never overwrite them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional

LABELS = ("paper", "db", "inferred", "default", "unknown")


class _Unset:
    def __repr__(self) -> str:
        return "UNSET"

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self


UNSET = _Unset()


@dataclass
class Prov:
    label: str = "unknown"  # one of LABELS
    source: Optional[str] = None
    accession: Optional[str] = None
    retrieved: Optional[str] = None  # ISO date, db labels only
    match_type: Optional[str] = None
    quote: Optional[str] = None
    location: Optional[str] = None  # section / page in the paper
    frozen: bool = False


@dataclass
class Field:
    value: Any = UNSET
    prov: Prov = field(default_factory=Prov)

    @property
    def absent(self) -> bool:
        return self.value is UNSET


def F() -> Any:
    return field(default_factory=Field)


def frozen(value: Any, label: str, **kw: Any) -> Field:
    return Field(copy.deepcopy(value), Prov(label=label, frozen=True, **kw))


def entity_label(raw: Dict[str, Any]) -> str:
    """paper / inferred / unknown for a payload row, from its provenance key and lineage origins."""
    origins = {x.get("origin") for x in (raw.get("provenance_lineage") or []) if isinstance(x, dict)}
    prov = raw.get("provenance")
    if "paper_stated" in origins or prov in ("extracted", "paper"):
        return "paper"
    if prov == "inferred" or origins:
        return "inferred"
    return "unknown"


def _is_convention(v: Any) -> bool:
    return v is False or v is None or v == "" or (v == 1 and not isinstance(v, bool))


@dataclass
class Node:
    """Base: KEYS maps payload key -> attribute; LISTS maps payload key -> (attribute, item class)."""

    extras: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    KEYS: ClassVar[Dict[str, str]] = {}
    LISTS: ClassVar[Dict[str, Any]] = {}
    IDS: ClassVar[FrozenSet[str]] = frozenset()  # payload keys that are database ids -> label db
    META: ClassVar[FrozenSet[str]] = frozenset()  # pipeline bookkeeping -> label inferred
    EVID: ClassVar[FrozenSet[str]] = frozenset()  # verbatim paper quotes
    DEFAULTS: ClassVar[FrozenSet[str]] = frozenset()  # convention values -> label default

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]):
        n = cls()
        lab = entity_label(raw)
        meta = raw.get("mapping_meta") if isinstance(raw.get("mapping_meta"), dict) else {}
        for k, v in raw.items():
            if k in cls.LISTS and isinstance(v, list):
                attr, item = cls.LISTS[k]
                setattr(n, attr, [item.from_raw(x) for x in v])
            elif k in cls.KEYS:
                if k in cls.IDS:
                    f = frozen(v, "db", source=meta.get("provider") or meta.get("source"), accession=str(v),
                               match_type=raw.get("chosen_rule") or meta.get("chosen_rule"))
                elif k in cls.EVID:
                    f = frozen(v, "paper", quote=v if isinstance(v, str) else None)
                elif k in cls.META:
                    f = frozen(v, "inferred")
                elif k in cls.DEFAULTS:
                    f = frozen(v, "default" if _is_convention(v) else "inferred")
                else:
                    f = frozen(v, lab)
                setattr(n, cls.KEYS[k], f)
            else:
                n.extras[k] = copy.deepcopy(v)
        return n

    def to_raw(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, attr in self.KEYS.items():
            f = getattr(self, attr)
            if not f.absent:
                out[k] = copy.deepcopy(f.value)
        for k, (attr, _) in self.LISTS.items():
            items = getattr(self, attr)
            if items is not None:
                out[k] = [i.to_raw() for i in items]
        for k, v in self.extras.items():
            out[k] = copy.deepcopy(v)
        return out


@dataclass
class Ref(Node):
    """A reference to an entity from a reaction, complex, transport or interaction (bare string or dict in the payload)."""

    entity: Field = F()
    entity_type: Field = F()
    role: Field = F()
    confidence: Field = F()
    provenance: Field = F()
    evidence: Field = F()
    stoichiometry: Field = F()  # unknown unless the payload carries it (bare strings never do)
    location: Field = F()  # location ref; filled only when the payload implies it, labelled inferred
    bare: bool = False
    entity_key: Optional[str] = None
    unknown: bool = False  # PathBank "Unknown" sentinel: an unknown member, not an entity
    KEYS = {"entity_type": "entity_type", "role": "role", "confidence": "confidence", "provenance": "provenance",
            "evidence": "evidence", "stoichiometry": "stoichiometry"}
    META = frozenset({"confidence", "provenance"})
    EVID = frozenset({"evidence"})
    ENTITY_KEYS: ClassVar[tuple] = ("entity", "name", "protein", "protein_complex")

    @classmethod
    def from_raw(cls, raw: Any):
        if not isinstance(raw, dict):  # bare string (or any other scalar): kept as-is
            n = cls(bare=True)
            n.entity = frozen(raw, "paper")
            return n
        ek = next((k for k in cls.ENTITY_KEYS if isinstance(raw.get(k), str)), None)
        n = super().from_raw({k: v for k, v in raw.items() if k != ek})
        n.entity_key = ek
        if ek is not None:
            n.entity = frozen(raw[ek], entity_label(raw))
        if not n.stoichiometry.absent:  # a payload number is a pipeline value, not a paper statement
            n.stoichiometry.prov.label = "default" if n.stoichiometry.value in (1, None) else "inferred"
        return n

    def to_raw(self) -> Any:
        if self.bare:
            return copy.deepcopy(self.entity.value)
        out = super().to_raw()
        if self.entity_key is not None:
            out[self.entity_key] = copy.deepcopy(self.entity.value)
        return out


@dataclass
class Species(Node):
    name: Field = F()
    taxonomy_id: Field = F()
    common_name: Field = F()
    classification: Field = F()
    lineage: Field = F()
    mapping: Field = F()
    KEYS = {"name": "name", "taxonomy_id": "taxonomy_id", "common_name": "common_name",
            "classification": "classification", "provenance_lineage": "lineage", "mapping_meta": "mapping"}
    IDS = frozenset({"taxonomy_id"})
    META = frozenset({"provenance_lineage", "mapping_meta"})


@dataclass
class Location(Node):
    """Compartment. go_id / parent / kind are slots for the online layer; the payload has none of them."""

    name: Field = F()
    canonical_id: Field = F()
    evidence: Field = F()
    lineage: Field = F()
    mapping: Field = F()
    go_id: Field = F()
    parent: Field = F()
    kind: Field = F()  # membrane / lumen
    KEYS = {"name": "name", "canonical_id": "canonical_id", "evidence": "evidence",
            "provenance_lineage": "lineage", "mapping_meta": "mapping"}
    META = frozenset({"provenance_lineage", "mapping_meta"})
    EVID = frozenset({"evidence"})


@dataclass
class State(Node):
    """A biological state = species + location (+ cell type / tissue): the payload's unit of location."""

    name: Field = F()
    species: Field = F()
    location: Field = F()
    cell_type: Field = F()
    tissue: Field = F()
    evidence: Field = F()
    lineage: Field = F()
    KEYS = {"name": "name", "species": "species", "subcellular_location": "location", "cell_type": "cell_type",
            "tissue": "tissue", "evidence": "evidence", "provenance_lineage": "lineage"}
    META = frozenset({"provenance_lineage"})
    EVID = frozenset({"evidence"})


@dataclass
class Placement(Node):
    """Entity-in-state row from element_locations (location is its own row, never part of the identity)."""

    entity: Field = F()
    state: Field = F()
    evidence: Field = F()
    lineage: Field = F()
    entity_key: Optional[str] = None
    KEYS = {"biological_state": "state", "evidence": "evidence", "provenance_lineage": "lineage"}
    META = frozenset({"provenance_lineage"})
    EVID = frozenset({"evidence"})
    ENTITY_KEYS: ClassVar[tuple] = ("compound", "protein", "nucleic_acid", "element_collection")

    @classmethod
    def from_raw(cls, raw):
        ek = next((k for k in cls.ENTITY_KEYS if k in raw), None)
        n = super().from_raw({k: v for k, v in raw.items() if k != ek})
        n.entity_key = ek
        if ek is not None:
            n.entity = frozen(raw[ek], entity_label(raw))
        return n

    def to_raw(self):
        out = super().to_raw()
        if self.entity_key is not None:
            out[self.entity_key] = copy.deepcopy(self.entity.value)
        return out


@dataclass
class Compound(Node):
    name: Field = F()
    synonyms: Field = F()
    short_name: Field = F()
    chebi_id: Field = F()
    kegg_id: Field = F()
    hmdb_id: Field = F()
    pubchem_cid: Field = F()
    cas: Field = F()
    drugbank_id: Field = F()
    inchikey: Field = F()  # slots for the online layer
    formula: Field = F()
    smiles: Field = F()
    confidence: Field = F()
    provenance: Field = F()
    db_status: Field = F()
    chosen_rule: Field = F()
    lineage: Field = F()
    mapping: Field = F()
    KEYS = {"name": "name", "synonyms": "synonyms", "short_name": "short_name", "chebi_id": "chebi_id",
            "kegg_id": "kegg_id", "hmdb_id": "hmdb_id", "pubchem_cid": "pubchem_cid", "cas": "cas",
            "drugbank_id": "drugbank_id", "confidence": "confidence", "provenance": "provenance",
            "db_status": "db_status", "chosen_rule": "chosen_rule", "provenance_lineage": "lineage",
            "mapping_meta": "mapping"}
    IDS = frozenset({"chebi_id", "kegg_id", "hmdb_id", "pubchem_cid", "cas", "drugbank_id"})
    META = frozenset({"confidence", "provenance", "db_status", "chosen_rule", "provenance_lineage", "mapping_meta"})


@dataclass
class Protein(Node):
    name: Field = F()
    synonyms: Field = F()
    gene_name: Field = F()
    organism: Field = F()
    taxonomy_id: Field = F()
    uniprot: Field = F()
    ec_number: Field = F()
    ec_numbers: Field = F()
    identity_status: Field = F()  # mapping status
    confidence: Field = F()
    provenance: Field = F()
    lineage: Field = F()
    mapping: Field = F()
    inferred_suggestion: Field = F()  # proposal only, never an id (P-20)
    KEYS = {"name": "name", "synonyms": "synonyms", "gene_name": "gene_name", "organism": "organism",
            "taxonomy_id": "taxonomy_id", "uniprot": "uniprot", "ec_number": "ec_number", "ec_numbers": "ec_numbers",
            "identity_status": "identity_status", "confidence": "confidence", "provenance": "provenance",
            "provenance_lineage": "lineage", "mapping_meta": "mapping"}
    IDS = frozenset({"uniprot", "ec_number", "ec_numbers", "taxonomy_id"})
    META = frozenset({"identity_status", "confidence", "provenance", "provenance_lineage", "mapping_meta"})


@dataclass
class NucleicAcid(Node):
    name: Field = F()
    synonyms: Field = F()
    confidence: Field = F()
    provenance: Field = F()
    lineage: Field = F()
    mapping: Field = F()
    KEYS = {"name": "name", "synonyms": "synonyms", "confidence": "confidence", "provenance": "provenance",
            "provenance_lineage": "lineage", "mapping_meta": "mapping"}
    META = frozenset({"confidence", "provenance", "provenance_lineage", "mapping_meta"})


@dataclass
class Complex(Node):
    """Members are entity refs; the complex itself never carries a member's accession."""

    name: Field = F()
    members: Optional[List[Ref]] = None
    evidence: Field = F()
    organism: Field = F()
    taxonomy_id: Field = F()
    generated: Field = F()
    identity_status: Field = F()
    confidence: Field = F()
    provenance: Field = F()
    lineage: Field = F()
    mapping: Field = F()
    KEYS = {"name": "name", "evidence": "evidence", "organism": "organism", "taxonomy_id": "taxonomy_id",
            "generated": "generated", "identity_status": "identity_status", "confidence": "confidence",
            "provenance": "provenance", "provenance_lineage": "lineage", "mapping_meta": "mapping"}
    LISTS = {"components": ("members", Ref)}
    IDS = frozenset({"taxonomy_id"})
    META = frozenset({"identity_status", "confidence", "provenance", "provenance_lineage", "mapping_meta", "generated"})
    EVID = frozenset({"evidence"})


@dataclass
class Reaction(Node):
    """Silent fields (direction, reversible, rhea, ...) stay unknown; nothing is invented."""

    name: Field = F()
    reactants: Optional[List[Ref]] = None
    products: Optional[List[Ref]] = None
    catalysts: Optional[List[Ref]] = None
    modifiers: Optional[List[Ref]] = None
    evidence: Field = F()
    state: Field = F()
    spontaneous: Field = F()
    locked_reaction_id: Field = F()
    scope_membership: Field = F()
    lineage: Field = F()
    direction: Field = F()
    reversible: Field = F()
    rhea_id: Field = F()
    ec_number: Field = F()
    kegg_reaction_id: Field = F()
    go_mf: Field = F()
    step_index: Field = F()
    KEYS = {"name": "name", "evidence": "evidence", "biological_state": "state", "spontaneous": "spontaneous",
            "locked_reaction_id": "locked_reaction_id", "scope_membership": "scope_membership",
            "provenance_lineage": "lineage"}
    LISTS = {"inputs": ("reactants", Ref), "outputs": ("products", Ref), "enzymes": ("catalysts", Ref),
             "modifiers": ("modifiers", Ref)}
    META = frozenset({"locked_reaction_id", "scope_membership", "provenance_lineage"})
    EVID = frozenset({"evidence"})
    DEFAULTS = frozenset({"spontaneous"})


@dataclass
class Transport(Node):
    name: Field = F()
    cargo: Field = F()
    from_state: Field = F()
    to_state: Field = F()
    from_location: Field = F()
    to_location: Field = F()
    transporters: Optional[List[Ref]] = None
    evidence: Field = F()
    lineage: Field = F()
    mechanism: Field = F()
    KEYS = {"name": "name", "cargo": "cargo", "from_biological_state": "from_state", "to_biological_state": "to_state",
            "evidence": "evidence", "provenance_lineage": "lineage"}
    LISTS = {"transporters": ("transporters", Ref)}
    META = frozenset({"provenance_lineage"})
    EVID = frozenset({"evidence"})


@dataclass
class Interaction(Node):
    source: Field = F()
    target: Field = F()
    name: Field = F()
    relationship: Field = F()  # raw text, kept verbatim
    kind: Field = F()  # controlled vocabulary (vocab.py)
    sign: Field = F()  # positive / negative / unsigned
    sbo: Field = F()  # SBO / MI slot
    evidence: Field = F()
    state: Field = F()
    lineage: Field = F()
    endpoints_declared: Optional[bool] = None
    KEYS = {"entity_1": "source", "entity_2": "target", "name": "name", "relationship": "relationship",
            "evidence": "evidence", "biological_state": "state", "provenance_lineage": "lineage"}
    META = frozenset({"provenance_lineage"})
    EVID = frozenset({"evidence"})


@dataclass
class Removed:
    """What was quarantined or removed, and why (a view; the payload keys it came from stay in extras)."""

    kind: str
    name: Optional[str]
    reason: Optional[str]
    source: str
    detail: Any = None


@dataclass
class MappingRecord:
    """Mapping rule and flag behind one entity's ids."""

    kind: str
    name: Optional[str]
    rule: Optional[str]
    flag: Optional[str]
    status: Optional[str]
    source: Optional[str]


@dataclass
class Pathway:
    title: Field = F()
    short_name: Field = F()
    pmid: Field = F()
    pmcid: Field = F()
    doi: Field = F()
    authors: Field = F()
    year: Field = F()
    organism: Field = F()
    taxon_id: Field = F()
    pathway_class: Field = F()
    tool: Field = F()
    tool_version: Field = F()
    run_date: Field = F()
    mode: Field = F()
    licences: List[Field] = field(default_factory=list)  # attribution entries per database used
    extras: Dict[str, Any] = field(default_factory=dict)


# payload key -> (CleanModel attribute, class)
ENTITY_LISTS = {"species": ("species", Species), "subcellular_locations": ("locations", Location),
                "compounds": ("compounds", Compound), "proteins": ("proteins", Protein),
                "protein_complexes": ("complexes", Complex), "nucleic_acids": ("nucleic_acids", NucleicAcid)}
PROCESS_LISTS = {"reactions": ("reactions", Reaction), "transports": ("transports", Transport),
                 "interactions": ("interactions", Interaction)}
PLACEMENT_KEYS = ("compound_locations", "protein_locations", "nucleic_acid_locations", "element_collection_locations")
CONTAINERS = ("entities", "element_locations", "processes")


@dataclass
class CleanModel:
    pathway: Pathway = field(default_factory=Pathway)
    species: Optional[List[Species]] = None
    locations: Optional[List[Location]] = None
    compounds: Optional[List[Compound]] = None
    proteins: Optional[List[Protein]] = None
    complexes: Optional[List[Complex]] = None
    nucleic_acids: Optional[List[NucleicAcid]] = None
    states: Optional[List[State]] = None
    placements: Dict[str, List[Placement]] = field(default_factory=dict)  # element_locations key -> rows
    reactions: Optional[List[Reaction]] = None
    transports: Optional[List[Transport]] = None
    interactions: Optional[List[Interaction]] = None
    removed: List[Removed] = field(default_factory=list)
    mapping_records: List[MappingRecord] = field(default_factory=list)
    db_layer: Dict[str, Any] = field(default_factory=dict)  # online fill: items / suggestions / conflicts / refused; never in the payload
    containers: List[str] = field(default_factory=list)  # which of CONTAINERS the payload had
    extras: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {"top": {}, "entities": {}, "element_locations": {}, "processes": {}})
