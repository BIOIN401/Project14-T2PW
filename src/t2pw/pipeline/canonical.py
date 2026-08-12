"""Biological-equivalence comparator for the canonical JSON, PWML and SBML graphs.

``PRODUCT_CONTRACT`` section 5 requires that reloading ``final_mapped.json`` and exporting
it again produces a BIOLOGICALLY EQUIVALENT pathway. :func:`biological_equivalence` decides
that: it PARSES and NORMALIZES all three documents into one format-independent graph and
compares the twelve section-5 dimensions in :data:`DIMENSIONS`.

D-007 (LOCKED) -- a hash is not evidence. No hash-to-hash comparison, no self-comparison and
no raw-text comparison decides anything here; nothing in this module hashes anything.

This is a COMPARATOR. It reports differences and never repairs, resolves or normalizes a
real difference away, never mutates its inputs, performs NO network or database lookup
(section 8), is wired into no caller (C-052 does that), and imports nothing from
``t2pw.rag`` (the RAG -> core separation invariant).

ORGANISM -- COMPARED, NEVER CANONICALIZED (D-016, D-021 section 2). Dimension 11 stays in
scope across all three formats and is compared by DETERMINISTIC LEXICAL NORMALIZATION ONLY
(:func:`normalize_text`: case, whitespace, punctuation). There is deliberately no synonym,
alias or taxonomy table: ``Homo sapiens`` against ``human`` is REPORTED, not resolved.
Species canonicalization is a pre-freeze unit owned solely by C-045, planning-only and
undispatched. CONSEQUENCE, RECORDED NOT SOLVED: T-102 therefore cannot fully pass from
C-052 alone while C-045 stays planning-only -- this comparator will correctly report an
organism mismatch only C-045 can eliminate. That is EXPECTED, is attributable to C-045, and
is never grounds to narrow this comparator's scope.

ABSENT IS NEVER EQUAL. A dimension a format genuinely cannot express is reported as
:data:`NotRepresentable` and never counted as agreement. A value a format merely failed to
carry is reported as a DIFFERENCE, not defaulted into agreement -- so direction and
reversibility are read AS STATED and the PWML exporter's "absent means Right" default is
NOT adopted here, because letting an exporter's default define equivalence is exactly the
post-freeze reinterpretation section 5 forbids.

Every projection is sorted; no set iteration order, clock or randomness reaches the output,
so two runs on the same inputs return the same report.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

FORMAT_JSON, FORMAT_PWML, FORMAT_SBML = "json", "pwml", "sbml"
FORMATS: Tuple[str, ...] = (FORMAT_JSON, FORMAT_PWML, FORMAT_SBML)

#: The PRODUCT_CONTRACT section 5 must-remain-equivalent dimensions, in contract order.
DIMENSIONS: Tuple[str, ...] = (
    "reactions", "reactants_and_products", "directionality_and_reversibility",
    "stoichiometry", "enzymes_and_modifiers", "transports_and_cargo",
    "entities_and_identifiers", "complexes_and_components", "cellular_locations",
    "process_entity_references", "organism_context", "no_broken_references")

VERDICT_EQUIVALENT, VERDICT_NOT_EQUIVALENT, VERDICT_INCOMPLETE = (
    "equivalent", "not_equivalent", "incomplete")
STATUS_EQUIVALENT, STATUS_DIFFERENT, STATUS_NOT_REPRESENTABLE = (
    "equivalent", "different", "not_representable")


class NotRepresentable:
    """A dimension the format genuinely cannot express. NEVER counted as agreement."""

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __repr__(self) -> str:
        return "<not_representable>"


class _Absent:
    """A value the document did not carry. Unequal to everything, by design."""

    def __repr__(self) -> str:
        return "<absent>"


ABSENT = _Absent()

#: SBML L3V1 core models a process only as ``<reaction>``, defined by stoichiometric
#: transformation. A non-stoichiometric regulatory interaction ("heme inhibits ALAS2") has
#: no core construct, and encoding one as a reaction would invent a transformation that does
#: not occur -- so it is not representable rather than merely absent.
SBML_NO_INTERACTIONS = NotRepresentable(
    "SBML L3V1 core has no construct for a non-stoichiometric regulatory interaction; its "
    "only process type is <reaction>, defined by stoichiometric transformation")

#: SBML L3V1 core has no organism element. Organism is expressible only as a MIRIAM taxonomy
#: IDENTIFIER (``urn:miriam:taxonomy:9606``), which this module does compare -- but the
#: organism NAME has no carrier, and inventing one from the identifier would require the
#: taxonomy table D-016 reserves to C-045.
SBML_NO_ORGANISM_NAME = NotRepresentable(
    "SBML L3V1 core has no organism name element; organism is expressible only as a MIRIAM "
    "taxonomy identifier, which this comparison does cover")


@dataclass(frozen=True)
class Difference:
    """One reported difference, in one dimension, between one pair of formats."""

    dimension: str
    formats: Tuple[str, str]
    key: str
    values: Tuple[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {"dimension": self.dimension, "formats": list(self.formats),
                "key": self.key, "values": list(self.values)}


@dataclass(frozen=True)
class Gap:
    """One dimension, or sub-key, one format genuinely cannot express."""

    dimension: str
    format: str
    key: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"dimension": self.dimension, "format": self.format, "key": self.key,
                "reason": self.reason}


@dataclass(frozen=True)
class EquivalenceReport:
    """The verdict, scoped to ``formats_compared``, and every finding behind it."""

    verdict: str
    formats_compared: Tuple[str, ...]
    dimension_status: Mapping[str, str]
    differences: Tuple[Difference, ...]
    not_representable: Tuple[Gap, ...]

    @property
    def equivalent(self) -> bool:
        return self.verdict == VERDICT_EQUIVALENT

    def differences_for(self, dimension: str) -> Tuple[Difference, ...]:
        return tuple(d for d in self.differences if d.dimension == dimension)

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "formats_compared": list(self.formats_compared),
                "dimension_status": dict(self.dimension_status),
                "differences": [d.to_dict() for d in self.differences],
                "not_representable": [g.to_dict() for g in self.not_representable]}


# --------------------------------------------------------------------------- normalization

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^0-9a-z\s]+")


def normalize_text(value: Any) -> str:
    """``value`` case-folded, punctuation and whitespace collapsed. LEXICAL ONLY.

    This is the whole of the organism comparison rule (D-016): it resolves
    ``  Mitochondrial  Matrix `` against ``mitochondrial matrix`` and deliberately does NOT
    resolve ``Homo sapiens`` against ``human``.
    """
    if value is None or isinstance(value, (NotRepresentable, _Absent)):
        return ""
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", str(value).strip().casefold())).strip()


#: Source key -> canonical database. A SCHEMA spelling map across the three formats
#: (``chebi-id`` / ``chebi`` / ``urn:miriam:chebi``), never a biological synonym table.
#: Taxonomy is deliberately absent: organism is dimension 11, not an entity identifier.
_ID_KEYS: Dict[str, str] = {s: db for db, spellings in (
    ("hmdb", "hmdb hmdb_id hmdb-id"), ("kegg", "kegg kegg_id kegg-id kegg.compound"),
    ("chebi", "chebi chebi_id chebi-id"), ("cas", "cas cas_id cas-id"),
    ("biocyc", "biocyc biocyc_id biocyc-id"),
    ("chemspider", "chemspider chemspider_id chemspider-id"),
    ("drugbank", "drugbank drugbank_id drugbank-id"),
    ("pubchem", "pubchem pubchem_cid pubchem-cid pubchem.compound"),
    ("uniprot", "uniprot uniprot_id uniprot-id"), ("gene", "gene gene_name gene-name"),
    ("ec", "ec ec_number ec-number"),
) for s in spellings.split()}


def _norm_id(database: str, value: Any) -> str:
    """``CHEBI:17627`` and ``17627`` agree; a redundant database prefix is lexical noise.

    A repeated identifier -- a JSON list, or a PWML ``<ec-numbers>`` container -- reduces to
    its values sorted and comma-joined, so the two spellings of one set still agree.
    """
    if isinstance(value, (list, tuple, set)):
        return ",".join(sorted(_norm_id(database, item) for item in value))
    text = str(value).strip()
    if text.casefold().startswith(database.casefold() + ":"):
        text = text[len(database) + 1:]
    return text.strip().casefold()


def _collect_ids(source: Any, into: Dict[str, str]) -> None:
    if isinstance(source, Mapping):
        for key, value in source.items():
            database = _ID_KEYS.get(str(key).strip().casefold())
            if database and value not in (None, "", [], {}):
                into[database] = _norm_id(database, value)


def _entity_key(kind: str, name: Any) -> str:
    return "{0}|{1}".format(kind, normalize_text(name))


def _number(value: Any, default: Any = 1) -> Any:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return int(number) if number == int(number) else number


def _text(value: Any) -> Optional[str]:
    text = "" if value is None else str(value).strip()
    return text or None


# --------------------------------------------------------------------------- graph model

_KINDS = ("compound", "protein", "protein_complex", "nucleic_acid", "element_collection")


@dataclass
class _Graph:
    """One document reduced to biology, with every format-specific spelling removed."""

    fmt: str
    organism_names: List[str] = field(default_factory=list)
    organism_taxa: List[str] = field(default_factory=list)
    entities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    complexes: Dict[str, List[Tuple[str, Any]]] = field(default_factory=dict)
    locations: List[str] = field(default_factory=list)
    states: Dict[str, Dict[str, str]] = field(default_factory=dict)
    entity_locations: List[Tuple[str, str]] = field(default_factory=list)
    reactions: List[Dict[str, Any]] = field(default_factory=list)
    transports: List[Dict[str, Any]] = field(default_factory=list)
    interactions: Any = field(default_factory=list)
    broken: List[str] = field(default_factory=list)

    def add_entity(self, kind: str, name: Any, ids: Optional[Dict[str, str]] = None) -> str:
        key = _entity_key(kind, name)
        self.entities.setdefault(key, {"kind": kind, "ids": {}})["ids"].update(ids or {})
        return key

    def resolve(self, name: Any, hint: Optional[str], where: str) -> str:
        """The entity key ``name`` denotes, recording a broken reference (dimension 12)."""
        norm = normalize_text(name)
        if not norm:
            self.broken.append("{0}: empty entity reference".format(where))
            return "unresolved|"
        if hint and "{0}|{1}".format(hint, norm) in self.entities:
            return "{0}|{1}".format(hint, norm)
        for kind in _KINDS:
            if "{0}|{1}".format(kind, norm) in self.entities:
                return "{0}|{1}".format(kind, norm)
        self.broken.append("{0}: unresolved {1} '{2}'".format(where, hint or "entity", norm))
        return "{0}|{1}".format(hint or "unresolved", norm)

    def resolve_state(self, name: Any, where: str) -> str:
        norm = normalize_text(name)
        if norm and norm not in self.states:
            self.broken.append("{0}: unresolved biological state '{1}'".format(where, norm))
        return norm


# --------------------------------------------------------------------------- input loading


def _load_json(source: Any) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    if isinstance(source, Path):
        source = source.read_bytes()
    payload = json.loads(source.decode("utf-8") if isinstance(source, bytes) else source)
    if not isinstance(payload, Mapping):
        raise ValueError("canonical JSON payload must be a JSON object")
    return payload


def _xml_root(source: Any) -> ElementTree.Element:
    if isinstance(source, ElementTree.Element):
        return source
    if isinstance(source, Path):
        source = source.read_bytes()
    return ElementTree.fromstring(source.encode("utf-8") if isinstance(source, str) else source)


def _tag(element: ElementTree.Element) -> str:
    """The local name, so an XML namespace never reaches a biological comparison."""
    return element.tag.rsplit("}", 1)[-1]


def _children(parent: Optional[ElementTree.Element], name: str) -> List[ElementTree.Element]:
    return [] if parent is None else [c for c in parent if _tag(c) == name]


def _section(parent: Optional[ElementTree.Element], name: str) -> Optional[ElementTree.Element]:
    found = _children(parent, name)
    return found[0] if found else None


def _field(parent: ElementTree.Element, name: str) -> Optional[str]:
    node = _section(parent, name)
    return None if node is None or node.get("nil") == "true" else _text(node.text)


def _descend(root: ElementTree.Element, name: str) -> Optional[ElementTree.Element]:
    return next((e for e in root.iter() if _tag(e) == name), None)


def _list_of(parent: Optional[ElementTree.Element], name: str) -> List[ElementTree.Element]:
    node = _section(parent, name)
    return list(node) if node is not None else []


# --------------------------------------------------------------------------- JSON parser

_JSON_BUCKETS = {"compounds": "compound", "proteins": "protein",
                 "protein_complexes": "protein_complex", "nucleic_acids": "nucleic_acid",
                 "element_collections": "element_collection"}
_JSON_LOCATIONS = {"compound_locations": "compound", "protein_locations": "protein",
                   "nucleic_acid_locations": "nucleic_acid",
                   "element_collection_locations": "element_collection"}
_PARTICIPANT_NAME_KEYS = ("entity", "compound", "protein", "protein_complex", "nucleic_acid",
                          "element_collection", "name")


def _rows(node: Any, key: str) -> List[Mapping[str, Any]]:
    rows = node.get(key) if isinstance(node, Mapping) else None
    return [r for r in rows if isinstance(r, Mapping)] if isinstance(rows, list) else []


def _participants(value: Any) -> List[Tuple[Any, Optional[str], Any]]:
    """``["glycine"]`` and ``[{"compound": "glycine", "stoichiometry": 2}]`` alike."""
    out: List[Tuple[Any, Optional[str], Any]] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, Mapping):
            name = next((item[k] for k in _PARTICIPANT_NAME_KEYS if item.get(k)), None)
            hint = normalize_text(item.get("entity_type")).replace(" ", "_") or None
            out.append((name, hint, _number(item.get("stoichiometry",
                                                     item.get("coefficient", 1)))))
        elif item not in (None, ""):
            out.append((item, None, 1))
    return out


def _parse_json(payload: Mapping[str, Any]) -> _Graph:
    graph = _Graph(FORMAT_JSON)
    entities = payload.get("entities")
    for row in _rows(entities, "species"):
        if normalize_text(row.get("name")):
            graph.organism_names.append(normalize_text(row["name"]))
        if _text(row.get("taxonomy_id")):
            graph.organism_taxa.append(_norm_id("taxonomy", row["taxonomy_id"]))
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping) and normalize_text(metadata.get("organism")):
        graph.organism_names.append(normalize_text(metadata["organism"]))

    for row in _rows(entities, "subcellular_locations"):
        if normalize_text(row.get("name")):
            graph.locations.append(normalize_text(row["name"]))
    for bucket, kind in _JSON_BUCKETS.items():
        for row in _rows(entities, bucket):
            ids: Dict[str, str] = {}
            _collect_ids(row.get("mapped_ids"), ids)
            _collect_ids(row, ids)
            graph.add_entity(kind, row.get("name"), ids)

    for row in _rows(entities, "protein_complexes"):
        components = []
        for member in row.get("components") if isinstance(row.get("components"), list) else []:
            name = member.get("name") if isinstance(member, Mapping) else member
            stoich = _number(member.get("stoichiometry", 1)) if isinstance(member, Mapping) else 1
            if name:
                components.append((_entity_key("protein", name), stoich))
        graph.complexes[_entity_key("protein_complex", row.get("name"))] = components

    # A state records only its subcellular location: the organism it names is dimension 11
    # and is compared there, so recording it here too would count one fact twice.
    for row in payload.get("biological_states") or []:
        if isinstance(row, Mapping):
            graph.states[normalize_text(row.get("name"))] = {
                "location": normalize_text(row.get("subcellular_location"))}
    for bucket, kind in _JSON_LOCATIONS.items():
        for row in _rows(payload.get("element_locations"), bucket):
            where = "/element_locations/" + bucket
            name = next((row[k] for k in (kind, "entity", "name") if row.get(k)), None)
            graph.entity_locations.append((graph.resolve(name, kind, where),
                                           graph.resolve_state(row.get("biological_state"), where)))

    processes = payload.get("processes")
    for index, row in enumerate(_rows(processes, "reactions")):
        where = "/processes/reactions/{0}".format(index)
        reversible = row.get("reversible")
        spontaneous = row.get("spontaneous")
        graph.reactions.append({
            "left": [(graph.resolve(n, h, where), s) for n, h, s in _participants(row.get("inputs"))],
            "right": [(graph.resolve(n, h, where), s) for n, h, s in _participants(row.get("outputs"))],
            "direction": normalize_text(row.get("direction")) or None,
            "reversible": reversible if isinstance(reversible, bool) else None,
            "spontaneous": spontaneous if isinstance(spontaneous, bool) else None,
            "enzymes": [graph.resolve(n, h, where) for n, h, _ in _participants(row.get("enzymes"))],
            "modifiers": [graph.resolve(n, h, where) for n, h, _ in _participants(row.get("modifiers"))],
            "state": graph.resolve_state(row.get("biological_state"), where)})

    for index, row in enumerate(_rows(processes, "transports")):
        where = "/processes/transports/{0}".format(index)
        cargo = row.get("cargo")
        graph.transports.append({
            "cargo": [(graph.resolve(n, h, where), s) for n, h, s in
                      _participants(cargo if isinstance(cargo, list) else [cargo])],
            "from": graph.resolve_state(row.get("from_biological_state"), where),
            "to": graph.resolve_state(row.get("to_biological_state"), where),
            "transporters": [graph.resolve(n, h, where) for n, h, _ in
                             _participants(row.get("transporters"))]})

    graph.interactions = []
    for index, row in enumerate(_rows(processes, "interactions")):
        where = "/processes/interactions/{0}".format(index)
        graph.interactions.append({
            "left": graph.resolve(row.get("entity_1"), None, where),
            "right": graph.resolve(row.get("entity_2"), None, where),
            "type": normalize_text(row.get("interaction_type") or row.get("relationship"))})
    return graph


# --------------------------------------------------------------------------- PWML parser

_PWML_SECTIONS = {"compounds": ("compound", "compound"), "proteins": ("protein", "protein"),
                  "protein-complexes": ("protein-complex", "protein_complex"),
                  "nucleic-acids": ("nucleic-acid", "nucleic_acid"),
                  "element-collections": ("element-collection", "element_collection")}
_PWML_LOCATIONS = {"compound-locations": ("compound-location", "compound-id", "compound"),
                   "protein-locations": ("protein-location", "protein-id", "protein"),
                   "nucleic-acid-locations": ("nucleic-acid-location", "nucleic-acid-id",
                                              "nucleic_acid"),
                   "element-collection-locations": ("element-collection-location",
                                                    "element-collection-id",
                                                    "element_collection")}
_PWML_ELEMENT_TYPES = {"compound": "compound", "protein": "protein",
                       "proteincomplex": "protein_complex", "nucleicacid": "nucleic_acid",
                       "elementcollection": "element_collection"}
_DIRECTIONS = {"right": "right", "left": "left", "both": "both"}
_BOOLEANS = {"true": True, "false": False}


def _pwml_ids(element: ElementTree.Element) -> Dict[str, str]:
    """Identifiers on a PWML entity, including any held in a list CONTAINER.

    Some identifiers are written as a container of repeats --
    ``<ec-numbers><ec-number>2.3.1.37</ec-number></ec-numbers>`` (`pwml/writer.py:380`
    and `:1219`) -- so reading direct children alone loses every value inside one, and the
    comparator would then report a difference between two documents that both carry it.
    """
    ids: Dict[str, str] = {}
    for child in element:
        if child.get("nil") == "true":
            continue
        database = _ID_KEYS.get(_tag(child).casefold())
        if database is not None:
            if _text(child.text):
                ids[database] = _norm_id(database, child.text)
            continue
        grouped: Dict[str, List[str]] = {}
        for nested in child:
            nested_db = _ID_KEYS.get(_tag(nested).casefold())
            if nested_db and nested.get("nil") != "true" and _text(nested.text):
                grouped.setdefault(nested_db, []).append(nested.text)
        for nested_db, values in grouped.items():
            ids[nested_db] = _norm_id(nested_db, values)
    return ids


def _parse_pwml(root: ElementTree.Element) -> _Graph:
    graph = _Graph(FORMAT_PWML)
    view = _descend(root, "pathway-visualization")
    if view is None:
        raise ValueError("PWML document has no <pathway-visualization> element")

    for row in _children(_section(view, "species"), "species"):
        if _field(row, "name"):
            graph.organism_names.append(normalize_text(_field(row, "name")))
        if _field(row, "taxonomy-id"):
            graph.organism_taxa.append(_norm_id("taxonomy", _field(row, "taxonomy-id")))

    locations: Dict[str, str] = {}
    for row in _children(_section(view, "subcellular-locations"), "subcellular-location"):
        locations[_field(row, "id") or ""] = normalize_text(_field(row, "name"))
        if normalize_text(_field(row, "name")):
            graph.locations.append(normalize_text(_field(row, "name")))

    by_id: Dict[Tuple[str, str], str] = {}
    for section, (item, kind) in _PWML_SECTIONS.items():
        for row in _children(_section(view, section), item):
            by_id[(kind, _field(row, "id") or "")] = graph.add_entity(
                kind, _field(row, "name"), _pwml_ids(row))

    def lookup(kind: str, ref: Optional[str], where: str) -> str:
        key = by_id.get((kind, ref or ""))
        if key is None:
            graph.broken.append("{0}: unresolved {1} id {2}".format(where, kind, ref))
            return "unresolved|{0}".format(ref or "")
        return key

    for row in _children(_section(view, "protein-complexes"), "protein-complex"):
        where = "/protein-complexes/" + (normalize_text(_field(row, "name")) or "?")
        graph.complexes[_entity_key("protein_complex", _field(row, "name"))] = [
            (lookup("protein", _field(m, "protein-id"), where),
             _number(_field(m, "stoichiometry")))
            for m in _children(_section(row, "protein_complex-proteins"),
                               "protein-complex-protein")]

    states: Dict[str, str] = {}
    for row in _children(_section(view, "biological-states"), "biological-state"):
        name = normalize_text(_field(row, "name"))
        states[_field(row, "id") or ""] = name
        graph.states[name] = {
            "location": locations.get(_field(row, "subcellular-location-id") or "", "")}
    for section, (item, ref, kind) in _PWML_LOCATIONS.items():
        for row in _children(_section(view, section), item):
            graph.entity_locations.append((
                lookup(kind, _field(row, ref), "/" + section),
                states.get(_field(row, "biological-state-id") or "", "")))

    def element_of(node: ElementTree.Element, where: str) -> str:
        kind = _PWML_ELEMENT_TYPES.get(
            normalize_text(_field(node, "element-type")).replace(" ", ""), "")
        return lookup(kind, _field(node, "element-id"), where)

    def enzyme_of(node: ElementTree.Element, where: str) -> Optional[str]:
        for tag, kind in (("protein-complex-id", "protein_complex"), ("protein-id", "protein")):
            if _field(node, tag):
                return lookup(kind, _field(node, tag), where)
        return None

    # A reaction's biological state is carried on its visualization record. The surrounding
    # coordinates are layout and are dropped; the state itself is biology and is not.
    reaction_states = {
        _field(v, "reaction-id") or "": states.get(_field(v, "biological-state-id") or "", "")
        for v in _children(_section(view, "reaction-visualizations"), "reaction-visualization")}

    for index, row in enumerate(_children(_section(view, "reactions"), "reaction")):
        where = "/reactions/{0}".format(index)
        direction = _DIRECTIONS.get(normalize_text(_field(row, "direction")))
        enzymes = [e for e in (enzyme_of(n, where) for n in _children(
            _section(row, "reaction-enzymes"), "reaction-enzyme")) if e is not None]
        sides = {side: [(element_of(n, where), _number(_field(n, "stoichiometry")))
                        for n in _children(_section(row, tag + "s"), tag)]
                 for side, tag in (("left", "reaction-left-element"),
                                   ("right", "reaction-right-element"))}
        graph.reactions.append({
            "left": sides["left"], "right": sides["right"], "direction": direction,
            # PWML carries no separate reversible flag: <direction>Both</direction> IS how
            # the schema states reversibility. A spelling equivalence, not an inference.
            "reversible": None if direction is None else direction == "both",
            "spontaneous": _BOOLEANS.get(normalize_text(_field(row, "spontaneous"))),
            "enzymes": enzymes, "modifiers": enzymes,
            "state": reaction_states.get(_field(row, "id") or "", "")})

    for index, row in enumerate(_children(_section(view, "transports"), "transport")):
        where = "/transports/{0}".format(index)
        cargo, source, target = [], "", ""
        for node in _children(_section(row, "transport-elements"), "transport-element"):
            cargo.append((element_of(node, where), _number(_field(node, "stoichiometry"))))
            left = states.get(_field(node, "left-biological-state-id") or "", "")
            right = states.get(_field(node, "right-biological-state-id") or "", "")
            reverse = _DIRECTIONS.get(normalize_text(_field(node, "direction"))) == "left"
            source, target = (right, left) if reverse else (left, right)
        graph.transports.append({"cargo": cargo, "from": source, "to": target, "transporters": [
            e for e in (enzyme_of(n, where) for n in _children(
                _section(row, "transport-transporters"), "transport-transporter"))
            if e is not None]})

    # The PWML <interaction> element carries a name and a type but no participant elements.
    # Absent participants are reported as a DIFFERENCE against the format that carries them,
    # never as agreement and never as a format limitation this repository cannot demonstrate.
    graph.interactions = [{"left": ABSENT, "right": ABSENT,
                           "type": normalize_text(_field(row, "interaction-type"))}
                          for row in _children(_section(view, "interactions"), "interaction")]
    return graph


# --------------------------------------------------------------------------- SBML parser

_SBML_TYPES = {"compound": "compound", "protein": "protein",
               "protein_complex": "protein_complex", "nucleic_acid": "nucleic_acid",
               "element_collection": "element_collection"}
_MIRIAM = re.compile(r"urn:miriam:([A-Za-z0-9.\-]+):(.+)$")


def _miriam(element: ElementTree.Element) -> List[Tuple[str, str]]:
    """Every ``urn:miriam:<db>:<id>`` RDF resource under ``element``, database-normalized."""
    found = []
    for node in element.iter():
        for name, value in node.attrib.items():
            match = (_MIRIAM.match(str(value).strip())
                     if name.rsplit("}", 1)[-1] == "resource" else None)
            if match:
                found.append((match.group(1).casefold(), match.group(2)))
    return found


def _pathwhiz_attr(element: ElementTree.Element, tag: str, name: str) -> str:
    for node in element.iter():
        if _tag(node) == tag:
            for key, value in node.attrib.items():
                if key.rsplit("}", 1)[-1] == name:
                    return (_text(value) or "").casefold()
    return ""


def _parse_sbml(root: ElementTree.Element) -> _Graph:
    graph = _Graph(FORMAT_SBML)
    model = root if _tag(root) == "model" else _descend(root, "model")
    if model is None:
        raise ValueError("SBML document has no <model> element")

    # Organism IS representable in SBML, through the standard MIRIAM taxonomy annotation
    # (urn:miriam:taxonomy:<id>) -- the same mechanism this exporter already uses for UniProt
    # and ChEBI. Its absence is therefore a real loss to REPORT, not a format limitation.
    graph.organism_names = SBML_NO_ORGANISM_NAME
    for database, value in _miriam(model):
        if database in ("taxonomy", "taxon"):
            graph.organism_taxa.append(_norm_id("taxonomy", value))

    compartments: Dict[str, str] = {}
    for row in _list_of(model, "listOfCompartments"):
        name = normalize_text(row.get("name") or row.get("id"))
        compartments[row.get("id") or ""] = name
        if name:
            graph.locations.append(name)
            graph.states[name] = {"location": name}

    by_id: Dict[str, str] = {}
    members: Dict[str, List[str]] = {}
    for row in _list_of(model, "listOfSpecies"):
        ids = {db: _norm_id(db, value) for db, value in
               ((_ID_KEYS.get(d), v) for d, v in _miriam(row)) if db}
        key = graph.add_entity(_SBML_TYPES.get(_pathwhiz_attr(row, "species", "species_type"),
                                               "compound"), row.get("name") or row.get("id"), ids)
        by_id[row.get("id") or ""] = key
        graph.entity_locations.append((key, compartments.get(row.get("compartment") or "", "")))
        if graph.entities[key]["kind"] == "protein_complex":
            members[key] = [_text(n.text) or "" for n in row.iter() if _tag(n) == "protein"]

    for key, listed in members.items():
        counted: Dict[str, int] = {}
        for member in listed:
            counted[member] = counted.get(member, 0) + 1
        graph.complexes[key] = []
        for member in sorted(counted):
            resolved = by_id.get(member)
            if resolved is None:
                graph.broken.append("/listOfSpecies/{0}: unresolved component {1}".format(
                    key, member))
                resolved = "unresolved|{0}".format(member)
            graph.complexes[key].append((resolved, counted[member]))

    def species_ref(node: ElementTree.Element, where: str) -> Tuple[str, Any]:
        key = by_id.get(node.get("species") or "")
        if key is None:
            graph.broken.append("{0}: unresolved species {1}".format(where, node.get("species")))
            key = "unresolved|{0}".format(node.get("species") or "")
        return key, _number(node.get("stoichiometry"))

    graph.interactions = SBML_NO_INTERACTIONS
    for index, row in enumerate(_list_of(model, "listOfReactions")):
        where = "/listOfReactions/{0}".format(index)
        left = [species_ref(n, where) for n in _list_of(row, "listOfReactants")]
        right = [species_ref(n, where) for n in _list_of(row, "listOfProducts")]
        modifiers = [species_ref(n, where)[0] for n in _list_of(row, "listOfModifiers")]
        reversible = _BOOLEANS.get((row.get("reversible") or "").casefold())
        if _pathwhiz_attr(row, "reaction", "reaction_type") == "transport":
            # SBML binds a species to exactly one compartment, so one species object cannot
            # state both a from-state and a to-state. The pair is reported ABSENT against the
            # formats that do carry it structurally.
            graph.transports.append({"cargo": left or right, "from": ABSENT, "to": ABSENT,
                                     "transporters": modifiers})
            continue
        graph.reactions.append({
            "left": left, "right": right, "reversible": reversible, "spontaneous": None,
            "direction": None if reversible is None else ("both" if reversible else "right"),
            "enzymes": modifiers, "modifiers": modifiers,
            "state": compartments.get(row.get("compartment") or "", "")})
    return graph


_PARSERS = {FORMAT_JSON: _parse_json, FORMAT_PWML: _parse_pwml, FORMAT_SBML: _parse_sbml}
_LOADERS = {FORMAT_JSON: _load_json, FORMAT_PWML: _xml_root, FORMAT_SBML: _xml_root}


# --------------------------------------------------------------------------- projection


def _keyed(records: Sequence[Mapping[str, Any]], parts: str) -> Dict[str, Mapping[str, Any]]:
    """Key each process by participant identity -- deterministic and format-independent.

    Identity is SIDE-INDEPENDENT, so moving a participant from reactant to product keeps the
    key and stays visible under ``reactants_and_products`` instead of silently becoming a
    different reaction. A repeated key is suffixed rather than dropped.
    """
    out: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        key = " + ".join(sorted(
            k for part in parts.split() for k, _ in record[part]))
        if key in out:
            key = "{0} #{1}".format(key, 1 + sum(1 for k in out if k.split(" #")[0] == key))
        out[key] = record
    return out


def _refs(record: Mapping[str, Any]) -> Tuple[str, ...]:
    tagged = [("reactant", k) for k, _ in record["left"]]
    tagged += [("product", k) for k, _ in record["right"]]
    tagged += [("enzyme", k) for k in record["enzymes"]]
    tagged += [("modifier", k) for k in record["modifiers"]]
    if record.get("state"):
        tagged.append(("biological_state", record["state"]))
    return tuple(sorted("{0}={1}".format(role, key) for role, key in tagged))


def _project(graph: _Graph) -> Dict[str, Any]:
    reactions = _keyed(graph.reactions, "left right")
    transports = _keyed(graph.transports, "cargo")
    interactions = graph.interactions
    if not isinstance(interactions, NotRepresentable):
        interactions = {"{0} {1} {2}".format(r["left"], r["type"], r["right"]):
                        ("left={0}".format(r["left"]), "right={0}".format(r["right"]))
                        for r in interactions}
    entity_locations: Dict[str, List[str]] = {}
    for entity, state in graph.entity_locations:
        entity_locations.setdefault(entity, []).append(state)
    return {
        "reactions": tuple(sorted(reactions)),
        "reactants_and_products": {
            k: {s: tuple(sorted(e for e, _ in v[s])) for s in ("left", "right")}
            for k, v in reactions.items()},
        "directionality_and_reversibility": {
            k: {f: v[f] for f in ("direction", "reversible", "spontaneous")}
            for k, v in reactions.items()},
        "stoichiometry": {k: {s: tuple(sorted(v[s])) for s in ("left", "right")}
                          for k, v in reactions.items()},
        "enzymes_and_modifiers": {k: {f: tuple(sorted(v[f])) for f in ("enzymes", "modifiers")}
                                  for k, v in reactions.items()},
        "transports_and_cargo": {
            k: {"cargo": tuple(sorted(v["cargo"])), "from": v["from"], "to": v["to"],
                "transporters": tuple(sorted(v["transporters"]))} for k, v in transports.items()},
        "entities_and_identifiers": {k: tuple(sorted(v["ids"].items()))
                                     for k, v in graph.entities.items()},
        "complexes_and_components": {k: tuple(sorted(v)) for k, v in graph.complexes.items()},
        "cellular_locations": {
            "subcellular_locations": tuple(sorted(set(graph.locations))),
            "biological_states": {k: dict(sorted(v.items())) for k, v in graph.states.items()},
            "entity_locations": {k: tuple(sorted(set(v))) for k, v in entity_locations.items()}},
        "process_entity_references": {
            "reactions": {k: _refs(v) for k, v in reactions.items()},
            "transports": {k: tuple(sorted(["cargo={0}".format(c) for c, _ in v["cargo"]]
                                           + ["transporter={0}".format(t)
                                              for t in v["transporters"]]))
                           for k, v in transports.items()},
            "interactions": interactions},
        "organism_context": {
            "names": (graph.organism_names if isinstance(graph.organism_names, NotRepresentable)
                      else tuple(sorted(set(graph.organism_names)))),
            "taxonomy_ids": tuple(sorted(set(graph.organism_taxa)))},
        "no_broken_references": tuple(sorted(set(graph.broken))),
    }


# --------------------------------------------------------------------------- comparison


def _display(value: Any) -> str:
    if isinstance(value, (NotRepresentable, _Absent)):
        return repr(value)
    text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    return text if len(text) <= 300 else text[:297] + "..."


def _walk(dimension: str, pair: Tuple[str, str], left: Any, right: Any, path: Tuple[str, ...],
          diffs: List[Difference], gaps: List[Gap]) -> None:
    key = "/".join(path) or "-"
    if isinstance(left, NotRepresentable) or isinstance(right, NotRepresentable):
        other = right if isinstance(left, NotRepresentable) else left
        if not isinstance(other, NotRepresentable) and not other:
            return  # nothing to express: an empty set is not a lost dimension
        for value, fmt in ((left, pair[0]), (right, pair[1])):
            if isinstance(value, NotRepresentable):
                gaps.append(Gap(dimension, fmt, key, value.reason))
        return  # the gap is recorded; it is NEVER recorded as agreement
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for sub in sorted(set(left) | set(right), key=str):
            _walk(dimension, pair, left.get(sub, ABSENT), right.get(sub, ABSENT),
                  path + (str(sub),), diffs, gaps)
    elif left is not right and left != right:
        diffs.append(Difference(dimension, pair, key, (_display(left), _display(right))))


def _compare(projections: Mapping[str, Mapping[str, Any]]
             ) -> Tuple[List[Difference], List[Gap]]:
    order = [f for f in FORMATS if f in projections]
    diffs: List[Difference] = []
    gaps: List[Gap] = []
    for dimension in DIMENSIONS:
        if dimension == "no_broken_references":
            # An INTRA-format integrity dimension: two documents that share a broken
            # reference are not in agreement, they are both wrong. Each is reported against
            # the format that carries it.
            for fmt in order:
                diffs.extend(Difference(dimension, (fmt, fmt), broken,
                                        ("<broken reference>", "<resolves>"))
                             for broken in projections[fmt][dimension])
            continue
        for index, left in enumerate(order):
            for right in order[index + 1:]:
                _walk(dimension, (left, right), projections[left][dimension],
                      projections[right][dimension], (), diffs, gaps)
    return diffs, gaps


def biological_equivalence(json_payload: Any = None, pwml: Any = None,
                           sbml: Any = None) -> EquivalenceReport:
    """Decide whether the supplied documents describe the same biology.

    Each argument is a parsed payload, an XML ``Element``, a ``Path`` to read, or the
    document text/bytes. ``None`` means "not supplied" and that format is left out of
    ``formats_compared``. At least TWO formats are required: comparing one document against
    itself is not evidence of anything (D-007).

    The verdict is scoped to ``formats_compared``. ``equivalent`` only when every comparable
    dimension agrees across every supplied pair AND nothing was unrepresentable;
    ``incomplete`` when the comparable dimensions agree but at least one could not be
    compared; ``not_equivalent`` when any difference was found.
    """
    supplied = [(fmt, source) for fmt, source in
                ((FORMAT_JSON, json_payload), (FORMAT_PWML, pwml), (FORMAT_SBML, sbml))
                if source is not None]
    if len(supplied) < 2:
        raise ValueError("biological equivalence needs at least two of json/pwml/sbml; a "
                         "document compared against itself proves nothing (D-007)")

    projections = {fmt: _project(_PARSERS[fmt](_LOADERS[fmt](source)))
                   for fmt, source in supplied}
    diffs, gaps = _compare(projections)
    status = {d: STATUS_DIFFERENT if any(x.dimension == d for x in diffs)
              else STATUS_NOT_REPRESENTABLE if any(g.dimension == d for g in gaps)
              else STATUS_EQUIVALENT for d in DIMENSIONS}
    return EquivalenceReport(
        verdict=(VERDICT_NOT_EQUIVALENT if diffs else
                 VERDICT_INCOMPLETE if gaps else VERDICT_EQUIVALENT),
        formats_compared=tuple(fmt for fmt, _ in supplied),
        dimension_status=status,
        differences=tuple(sorted(diffs, key=lambda d: (d.dimension, d.formats, d.key, d.values))),
        # One gap per (dimension, format, key): the same gap surfaces once per format pair.
        not_representable=tuple(sorted(
            {(g.dimension, g.format, g.key): g for g in gaps}.values(),
            key=lambda g: (g.dimension, g.format, g.key))))
