"""Within-run entity normalization and alias tracking.

What this module does
---------------------
Folds ``ATP``, ``atp`` and ``adenosine 5'-triphosphate`` into one research
entity *for the duration of one run*, remembering every surface form that was
seen and which paper contributed it.

What it explicitly does NOT do
------------------------------
It never contacts ChEBI, HMDB, KEGG, PubChem, UniProt or any other biochemical
identifier service, and a normalized name is **never** presented as an external
identifier or as database verification. Version 1 admits on paper evidence
alone; see :mod:`t2pw.mining.resolver` for the deferred interface where real
database validation will later attach.

Because the merge is local and lexical it can be wrong. Every merge is therefore
recorded with its reason so an auditor can see that ``"OPDA"`` was folded into
``"12-oxophytodienoic acid"`` because a paper wrote the two together, not
because a model felt they matched.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from t2pw.mining.ids import entity_id

#: Entity types the research graph distinguishes. ``enzyme``/``protein``/``gene``
#: never carry chemical connectivity -- see :mod:`t2pw.mining.graph`.
METABOLITE = "metabolite"
ENZYME = "enzyme"
PROTEIN = "protein"
GENE = "gene"
COFACTOR = "cofactor"
UNKNOWN = "unknown"

ENTITY_TYPES = (METABOLITE, ENZYME, PROTEIN, GENE, COFACTOR, UNKNOWN)

#: Types that may act as chemical participants (substrate/product).
CHEMICAL_TYPES = frozenset({METABOLITE, COFACTOR, UNKNOWN})

#: Types that are annotations on a reaction, never chemical bridges.
CATALYTIC_TYPES = frozenset({ENZYME, PROTEIN, GENE})

_WS = re.compile(r"\s+")
_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "ω": "omega", "μ": "mu",
}
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_PRIMES = dict.fromkeys(map(ord, "′″ʹ’"), "'")

#: Trailing charge notation. Stripped for comparison only; the surface form that
#: carried it is kept as an alias, so "NAD+" is never rewritten to "NAD".
_CHARGE_TAIL = re.compile(r"(?<=[a-z0-9])\s*\(?\d*[+-]\)?$", re.IGNORECASE)

#: Well-known cofactors. Used ONLY to type an entity, never to merge names and
#: never to justify connectivity: two reactions sharing ATP are not thereby a
#: pathway (see ``graph.chemically_connected``'s cofactor rule).
_COFACTOR_NAMES = frozenset(
    {
        "atp", "adp", "amp", "gtp", "gdp", "nad", "nadh", "nad+", "nadp", "nadph",
        "nadp+", "fad", "fadh2", "fmn", "coa", "coenzyme a", "acetyl-coa",
        "s-adenosylmethionine", "sam", "h2o", "water", "o2", "oxygen", "co2",
        "carbon dioxide", "nh3", "ammonia", "phosphate", "pi", "ppi",
        "pyrophosphate", "h+", "proton", "h2o2",
    }
)

_ENZYME_MARKERS = (
    "ase", "synthase", "synthetase", "reductase", "oxidase", "transferase",
    "kinase", "dehydrogenase", "hydrolase", "isomerase", "lyase", "ligase",
    "cyclase", "peroxidase", "mutase", "carboxylase", "decarboxylase",
    "cytochrome p450", "p450",
)


def normalize_name(text: Any) -> str:
    """Lowercased comparison key. Whitespace, dashes, primes and greek folded."""
    raw = str(text or "")
    raw = raw.translate(_DASHES).translate(_PRIMES)
    for greek, latin in _GREEK.items():
        raw = raw.replace(greek, latin)
    raw = _WS.sub(" ", raw).strip().lower()
    raw = raw.strip(" .,;:")
    return raw


def comparison_key(text: Any) -> str:
    """:func:`normalize_name` plus charge-tail and separator folding.

    This is the key two surface forms must share to be considered the same
    entity. ``NAD+`` and ``NAD`` share it; ``ATP`` and ``ADP`` do not.
    """
    norm = normalize_name(text)
    if not norm:
        return ""
    stripped = _CHARGE_TAIL.sub("", norm).strip()
    base = stripped or norm
    base = base.replace("_", "-").replace(" - ", "-")
    base = _WS.sub(" ", base).strip()
    return base


def looks_like_enzyme(name: Any) -> bool:
    norm = normalize_name(name)
    if not norm:
        return False
    return any(norm.endswith(m) or m in norm for m in _ENZYME_MARKERS)


def is_cofactor(name: Any) -> bool:
    return comparison_key(name) in {comparison_key(c) for c in _COFACTOR_NAMES}


def infer_entity_type(name: Any, declared: str = "") -> str:
    """Type an entity from its declared role first, then from its name.

    A declared type always wins: the extraction schema asks the model to say
    whether a participant is a metabolite or a protein, and second-guessing a
    declared answer with a suffix heuristic is how ``synthase`` ends up a
    metabolite.
    """
    dec = normalize_name(declared)
    if dec in ENTITY_TYPES:
        if dec == METABOLITE and is_cofactor(name):
            return COFACTOR
        return dec
    if dec in {"compound", "chemical", "small molecule", "substrate", "product"}:
        return COFACTOR if is_cofactor(name) else METABOLITE
    if dec in {"enzyme", "catalyst"}:
        return ENZYME
    if dec in {"protein", "polypeptide", "complex"}:
        return PROTEIN
    if dec in {"gene", "locus", "orf"}:
        return GENE
    if is_cofactor(name):
        return COFACTOR
    if looks_like_enzyme(name):
        return ENZYME
    return METABOLITE if name else UNKNOWN


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
@dataclass
class ResearchEntity:
    """One normalized entity in the research graph.

    ``external_refs`` is always the deferred-resolver shape in version 1 --
    ``{"status": "not_evaluated"}``. It exists so that a later, separately
    authorized database-validation layer has a place to write without a schema
    migration, and so that no reader can mistake a normalized *name* for a
    resolved *identifier*.
    """

    entity_id: str
    name: str
    key: str
    entity_type: str = UNKNOWN
    aliases: List[str] = field(default_factory=list)
    #: alias surface form -> the reason it was folded in.
    alias_provenance: Dict[str, str] = field(default_factory=dict)
    evidence_ids: List[str] = field(default_factory=list)
    paper_ids: List[str] = field(default_factory=list)
    first_round: int = 0
    last_updated_round: int = 0
    external_refs: Dict[str, Any] = field(default_factory=lambda: {"status": "not_evaluated"})

    @property
    def is_chemical(self) -> bool:
        return self.entity_type in CHEMICAL_TYPES

    @property
    def is_catalytic(self) -> bool:
        return self.entity_type in CATALYTIC_TYPES

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EntityRegistry:
    """Within-run entity table. Local, lexical, and auditable.

    Merging rule: two surface forms are the same entity when they share a
    :func:`comparison_key`, or when a caller explicitly declares an alias with a
    stated reason (``link_alias``). Nothing merges on semantic similarity, on
    embedding distance, or because a model said the names "seem related".
    """

    def __init__(self) -> None:
        self._by_id: Dict[str, ResearchEntity] = {}
        self._key_to_id: Dict[str, str] = {}

    # -- lookup -------------------------------------------------------------
    def get(self, entity: str) -> Optional[ResearchEntity]:
        if entity in self._by_id:
            return self._by_id[entity]
        found = self._key_to_id.get(comparison_key(entity))
        return self._by_id.get(found) if found else None

    def resolve(self, name: Any) -> Optional[str]:
        """Entity id for a surface form, or ``None`` if unseen."""
        return self._key_to_id.get(comparison_key(name))

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, item: object) -> bool:
        return self.get(str(item)) is not None

    def all(self) -> List[ResearchEntity]:
        return sorted(self._by_id.values(), key=lambda e: (e.entity_type, e.key))

    # -- mutation -----------------------------------------------------------
    def upsert(
        self,
        name: Any,
        *,
        declared_type: str = "",
        evidence_ids: Optional[Iterable[str]] = None,
        paper_ids: Optional[Iterable[str]] = None,
        round_index: int = 0,
    ) -> ResearchEntity:
        """Register a surface form, creating or extending its entity."""
        surface = _WS.sub(" ", str(name or "")).strip()
        if not surface:
            raise ValueError("cannot register an empty entity name")
        key = comparison_key(surface)
        etype = infer_entity_type(surface, declared_type)

        existing_id = self._key_to_id.get(key)
        if existing_id is None:
            eid = entity_id(key, etype)
            entity = ResearchEntity(
                entity_id=eid,
                name=surface,
                key=key,
                entity_type=etype,
                aliases=[surface],
                alias_provenance={surface: "first surface form observed"},
                first_round=int(round_index),
                last_updated_round=int(round_index),
            )
            self._by_id[eid] = entity
            self._key_to_id[key] = eid
        else:
            entity = self._by_id[existing_id]
            if surface not in entity.aliases:
                entity.aliases.append(surface)
                entity.alias_provenance[surface] = f"same normalized key {key!r}"
            # A declared type upgrades UNKNOWN but never silently overwrites a
            # already-declared type: a disagreement is information, not a fix.
            if entity.entity_type == UNKNOWN and etype != UNKNOWN:
                entity.entity_type = etype
            entity.last_updated_round = max(entity.last_updated_round, int(round_index))

        for ev in evidence_ids or []:
            if ev and ev not in entity.evidence_ids:
                entity.evidence_ids.append(str(ev))
        for pid in paper_ids or []:
            if pid and pid not in entity.paper_ids:
                entity.paper_ids.append(str(pid))
        return entity

    def link_alias(self, existing: Any, alias: Any, reason: str) -> Optional[ResearchEntity]:
        """Declare ``alias`` another surface form of ``existing``, with a reason.

        Used when a paper states an equivalence in so many words ("12-OPDA
        (12-oxophytodienoic acid)"). The reason is persisted; an unexplained
        alias link is not possible through this API.
        """
        entity = self.get(str(existing))
        if entity is None:
            return None
        surface = _WS.sub(" ", str(alias or "")).strip()
        if not surface:
            return entity
        key = comparison_key(surface)
        other_id = self._key_to_id.get(key)
        if other_id and other_id != entity.entity_id:
            # Two distinct entities are being declared equivalent: fold the
            # younger one in, keeping every alias and every evidence pointer.
            other = self._by_id.pop(other_id)
            for a in other.aliases:
                if a not in entity.aliases:
                    entity.aliases.append(a)
                    entity.alias_provenance[a] = reason or "declared alias"
            for ev in other.evidence_ids:
                if ev not in entity.evidence_ids:
                    entity.evidence_ids.append(ev)
            for pid in other.paper_ids:
                if pid not in entity.paper_ids:
                    entity.paper_ids.append(pid)
            for k, v in list(self._key_to_id.items()):
                if v == other_id:
                    self._key_to_id[k] = entity.entity_id
        if surface not in entity.aliases:
            entity.aliases.append(surface)
            entity.alias_provenance[surface] = reason or "declared alias"
        self._key_to_id[key] = entity.entity_id
        return entity

    def to_dict(self) -> Dict[str, Any]:
        return {e.entity_id: e.to_dict() for e in self.all()}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EntityRegistry":
        reg = cls()
        for payload in (data or {}).values():
            entity = ResearchEntity(
                entity_id=str(payload.get("entity_id") or ""),
                name=str(payload.get("name") or ""),
                key=str(payload.get("key") or ""),
                entity_type=str(payload.get("entity_type") or UNKNOWN),
                aliases=[str(a) for a in payload.get("aliases") or []],
                alias_provenance={str(k): str(v) for k, v in (payload.get("alias_provenance") or {}).items()},
                evidence_ids=[str(a) for a in payload.get("evidence_ids") or []],
                paper_ids=[str(a) for a in payload.get("paper_ids") or []],
                first_round=int(payload.get("first_round") or 0),
                last_updated_round=int(payload.get("last_updated_round") or 0),
                external_refs=dict(payload.get("external_refs") or {"status": "not_evaluated"}),
            )
            if not entity.entity_id:
                continue
            reg._by_id[entity.entity_id] = entity
            reg._key_to_id[entity.key] = entity.entity_id
            for alias in entity.aliases:
                reg._key_to_id.setdefault(comparison_key(alias), entity.entity_id)
        return reg


def participant_signature(
    registry: EntityRegistry, names: Iterable[Any]
) -> Tuple[str, ...]:
    """Order-insensitive tuple of comparison keys for a reaction side."""
    keys = set()
    for name in names:
        entity = registry.get(str(name))
        keys.add(entity.key if entity else comparison_key(name))
    return tuple(sorted(k for k in keys if k))


__all__ = [
    "METABOLITE",
    "ENZYME",
    "PROTEIN",
    "GENE",
    "COFACTOR",
    "UNKNOWN",
    "ENTITY_TYPES",
    "CHEMICAL_TYPES",
    "CATALYTIC_TYPES",
    "ResearchEntity",
    "EntityRegistry",
    "normalize_name",
    "comparison_key",
    "infer_entity_type",
    "is_cofactor",
    "looks_like_enzyme",
    "participant_signature",
]
