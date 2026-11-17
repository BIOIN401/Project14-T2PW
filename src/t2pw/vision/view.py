"""Scoped views of a big payload (docs/vision/PLAN.md section 2.2).

``view(big, scope)`` returns a payload in the final_mapped schema. It is reference-closed: every
entity and state name a kept row references, and that resolved in the base, still resolves.
With an empty extension, ``view(big, DEFAULT)`` and ``view(big, FULL)`` are a deep copy of the base.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from t2pw.vision.payload import STATES, iter_sections, keyed_rows, norm, refs, section_rows


@dataclass(frozen=True)
class ScopeSpec:
    paper_only: bool = False
    max_hops: Optional[int] = 1


PAPER_ONLY = ScopeSpec(paper_only=True, max_hops=None)
DEFAULT = ScopeSpec(paper_only=False, max_hops=1)
FULL = ScopeSpec(paper_only=False, max_hops=None)

# Entity sections the paper-only view prunes; context sections (species, locations, cell types,
# tissues, compartments) pass through so states stay closed.
PRUNED_ENTITY_SECTIONS = ("entities.compounds", "entities.proteins", "entities.protein_complexes",
                          "entities.nucleic_acids", "entities.element_collections")


def more_detailed(spec: ScopeSpec) -> ScopeSpec:
    if spec.paper_only:
        return DEFAULT
    return FULL if spec.max_hops is None else ScopeSpec(False, spec.max_hops + 1)


def less_detailed(spec: ScopeSpec) -> ScopeSpec:
    if spec.paper_only or (spec.max_hops is not None and spec.max_hops <= 1):
        return PAPER_ONLY
    return DEFAULT if spec.max_hops is None else ScopeSpec(False, spec.max_hops - 1)


def _is_paper(records: List[Dict[str, Any]]) -> bool:
    return any(isinstance(r, dict) and r.get("kind") == "paper" for r in records)


def _name(row: Any) -> str:
    return norm(row.get("name")) if isinstance(row, dict) else ""


def _paper_only(big: Dict[str, Any]) -> Dict[str, Any]:
    prov = big.get("provenance") or {}
    rows = [(s, k, r) for s, k, r, is_base in keyed_rows(big) if is_base]
    keep = {k for s, k, r in rows if s.startswith("processes.") and _is_paper(prov.get(k, []))}
    needed = {norm(n) for s, k, r in rows if k in keep for n in refs(s, r)[0]}
    changed = True
    while changed:  # a kept complex needs its components
        changed = False
        for s, k, r in rows:
            if k in keep or not s.startswith("entities."):
                continue
            if s not in PRUNED_ENTITY_SECTIONS or _name(r) in needed or _is_paper(prov.get(k, [])):
                keep.add(k)
                new = {norm(n) for n in refs(s, r)[0]} - needed
                changed = changed or bool(new)
                needed |= new
    base_ents = {_name(r) for s, k, r in rows if s.startswith("entities.")}
    kept_ents = {_name(r) for s, k, r in rows if s.startswith("entities.") and k in keep}
    for s, k, r in rows:
        if s.startswith("element_locations."):
            ents = [norm(n) for n in refs(s, r)[0]]
            if all(e in kept_ents or e not in base_ents for e in ents):
                keep.add(k)
    # a state stays when a kept row uses it, no base row used it, or it carries its own paper record
    users = [(k, refs(s, r)[1]) for s, k, r in rows if s.startswith(("processes.", "element_locations."))]
    used = {norm(n) for k, states in users for n in states}
    kept_used = {norm(n) for k, states in users if k in keep for n in states}
    keep |= {k for s, k, r in rows if s == STATES
             and (_name(r) in kept_used or _name(r) not in used or _is_paper(prov.get(k, [])))}

    out = copy.deepcopy(big["base"])
    keys = defaultdict(list)
    for s, k, r in rows:
        keys[s].append(k)
    for s, section in iter_sections(out):
        section[:] = [r for r, k in zip(section, keys[s]) if k in keep]
    # a dropped reaction's name leaves reaction_order unless a kept reaction carries it
    # (the rule of service/final_payload_merge.py, keyed by the shared reference key)
    rx = [(r, k) for s, k, r in rows if s == "processes.reactions" and isinstance(r, dict)]
    gone = {_name(r) for r, k in rx if k not in keep} - {_name(r) for r, k in rx if k in keep}
    if gone and isinstance(out.get("reaction_order"), list):
        out["reaction_order"] = [x for x in out["reaction_order"] if not (isinstance(x, str) and norm(x) in gone)]
    return out


def view(big: Dict[str, Any], scope: ScopeSpec = DEFAULT) -> Dict[str, Any]:
    """The payload a scope shows: paper-only prunes the base; otherwise base + extension rows with
    hop <= max_hops (all of them when max_hops is None) plus the extension entities/states they need."""
    if scope.paper_only:
        return _paper_only(big)
    out = copy.deepcopy(big["base"])
    ext_rows = [(s, k, r) for s, k, r, is_base in keyed_rows(big) if not is_base]
    if not ext_rows:
        return out
    hops = big.get("hops") or {}
    keep = {k for s, k, r in ext_rows
            if scope.max_hops is None or (hops.get(k) is not None and hops[k] <= scope.max_hops)}
    changed = True
    while changed:  # closure: kept rows pull in the extension entities and states they name
        changed = False
        ents, states = set(), set()
        for s, k, r in ext_rows:
            if k in keep:
                e, st, _ = refs(s, r)
                ents |= {norm(n) for n in e}
                states |= {norm(n) for n in st}
        for s, k, r in ext_rows:
            if k not in keep and ((s.startswith("entities.") and _name(r) in ents)
                                  or (s == STATES and _name(r) in states)):
                keep.add(k)
                changed = True
    for s, k, r in ext_rows:
        if k in keep:
            section_rows(out, s, create=True).append(copy.deepcopy(r))
    return out


def view_stats(big: Dict[str, Any], scope: ScopeSpec = DEFAULT) -> Dict[str, Dict[str, int]]:
    """Rows per section: total (base + extension), kept by the scope, dropped."""
    total = Counter(s for s, _, _, _ in keyed_rows(big))
    kept = Counter({s: len(rows) for s, rows in iter_sections(view(big, scope))})
    return {s: {"total": total[s], "kept": kept[s], "dropped": total[s] - kept[s]} for s in total}


def unresolved_refs(payload: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """(section, 'entity' | 'state', name) for every reference that names no row in ``payload``."""
    sections = list(iter_sections(payload))
    ents = {_name(r) for s, rows in sections if s.startswith("entities.") for r in rows}
    states = {_name(r) for s, rows in sections if s == STATES for r in rows}
    out = []
    for s, rows in sections:
        for r in rows:
            e, st, _ = refs(s, r)
            out += [(s, "entity", n) for n in e if norm(n) not in ents]
            out += [(s, "state", n) for n in st if norm(n) not in states]
    return out
