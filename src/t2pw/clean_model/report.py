"""Counts over a CleanModel."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict

from .model import CleanModel


@dataclass
class ModelReport:
    counts: Dict[str, int] = field(default_factory=dict)
    interactions_by_kind: Dict[str, int] = field(default_factory=dict)
    interactions_by_sign: Dict[str, int] = field(default_factory=dict)
    raw_spellings: int = 0
    complexes_with_unknown_member: int = 0
    complexes_all_unknown: int = 0
    reactions_with_quote: int = 0
    transports_with_quote: int = 0
    interactions_with_quote: int = 0
    undeclared_interaction_endpoints: int = 0
    pathway: Dict[str, bool] = field(default_factory=dict)


def _n(x: Any) -> int:
    return len(x or [])


def _quoted(items: Any) -> int:
    return sum(1 for i in items or [] if not i.evidence.absent and isinstance(i.evidence.value, str) and i.evidence.value.strip())


def summary(m: CleanModel) -> ModelReport:
    r = ModelReport()
    r.counts = {k: _n(getattr(m, k)) for k in ("species", "locations", "compounds", "proteins", "complexes",
                                                "nucleic_acids", "states", "reactions", "transports", "interactions")}
    r.counts["placements"] = sum(len(v) for v in m.placements.values())
    r.counts["removed"] = len(m.removed)
    r.counts["mapping_records"] = len(m.mapping_records)
    ixs = m.interactions or []
    r.interactions_by_kind = dict(Counter(i.kind.value for i in ixs))
    r.interactions_by_sign = dict(Counter(i.sign.value for i in ixs))
    r.raw_spellings = len({i.relationship.value for i in ixs if not i.relationship.absent})
    r.complexes_with_unknown_member = sum(1 for c in m.complexes or [] if "unknown_member" in c.flags)
    r.complexes_all_unknown = sum(1 for c in m.complexes or [] if "all_members_unknown" in c.flags)
    r.reactions_with_quote, r.transports_with_quote, r.interactions_with_quote = (
        _quoted(m.reactions), _quoted(m.transports), _quoted(ixs))
    r.undeclared_interaction_endpoints = sum(1 for i in ixs if i.endpoints_declared is False)
    pw = m.pathway
    r.pathway = {k: not getattr(pw, k).absent for k in ("title", "pmid", "pmcid", "doi", "organism", "year")}
    return r
