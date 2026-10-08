"""Frontier generation: what the next round should go and look for.

This is the part that is genuinely different from the production RAG controller.
That controller is *gap-driven*: it detects a predefined set of gaps in a pathway
it is trying to complete and retrieves evidence to fill them. Mining is
*expansion-driven*: every round derives its next questions from what the last
round just discovered, so the network grows outward instead of being filled in.

Frontier kinds are exploration strategies, not evidence rules. Freeing the
exploration is the point; the evidence requirements downstream are unchanged.

The frontier is capped (``max_frontier_size``), and a quarter of its slots are kept
for ``missing_species_evidence`` items: in pilot-003 the cap cut 137 of 194 items,
every species item among them, so target-species evidence was never searched for.
:func:`cap_frontier` applies the cap and the reservation wherever the cap applies.

One rule constrains generation: a frontier item may use enzyme names, gene
symbols, pathway names, synonyms and paper terminology *to find papers*. Those
shared terms never create a biochemical connection -- only
:func:`t2pw.mining.graph.connection_between` does that, and it cannot see them.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from t2pw.mining.claims import ClaimStore
from t2pw.mining.config import MiningConfig, StartingInput
from t2pw.mining.graph import ResearchGraph
from t2pw.mining.ids import frontier_id as make_frontier_id
from t2pw.mining.models import (
    FRONTIER_BRANCH,
    FRONTIER_BRIDGE,
    FRONTIER_COFACTOR,
    FRONTIER_COMPARTMENT,
    FRONTIER_CONFLICT,
    FRONTIER_DIRECTION,
    FRONTIER_DOWNSTREAM,
    FRONTIER_ENZYME,
    FRONTIER_PATHWAY_REFERENCE,
    FRONTIER_SEED,
    FRONTIER_SPECIES,
    FRONTIER_UPSTREAM,
    FrontierItem,
    GraphDelta,
    STATUS_CANDIDATE,
    STATUS_INCOMPLETE,
)
from t2pw.mining.normalize import CHEMICAL_TYPES, comparison_key, is_cofactor
from t2pw.mining.queries import rescue_terms

#: Default priorities. Higher runs first. A bridge query can turn a quarantined
#: component into pathway, and a conflict query can settle a visible
#: disagreement, so both outrank ordinary expansion.
PRIORITY = {
    FRONTIER_SEED: 10.0,
    FRONTIER_BRIDGE: 9.0,
    FRONTIER_CONFLICT: 8.0,
    FRONTIER_DOWNSTREAM: 7.0,
    FRONTIER_UPSTREAM: 6.5,
    FRONTIER_SPECIES: 6.0,
    FRONTIER_BRANCH: 5.0,
    FRONTIER_ENZYME: 4.5,
    FRONTIER_PATHWAY_REFERENCE: 4.0,
    FRONTIER_DIRECTION: 3.0,
    FRONTIER_COMPARTMENT: 2.5,
    FRONTIER_COFACTOR: 1.0,
}

#: The share of the frontier's slots kept for ``missing_species_evidence`` items
#: whenever that many exist (ORCH-745 F-3). Those items are the only ones that ask
#: for target-species evidence for a held lead.
SPECIES_RESERVE = 0.25


def seed_frontier(config: MiningConfig) -> List[FrontierItem]:
    """Round 1's frontier, built from the declared starting inputs.

    A run with a topic but no seeds still gets one item -- the topic itself --
    so a topic-only run has somewhere to start.
    """
    items: List[FrontierItem] = []

    def add(target: str, reason: str, kind: str = FRONTIER_SEED) -> None:
        text = str(target or "").strip()
        if not text:
            return
        items.append(
            FrontierItem(
                frontier_item_id=make_frontier_id(kind, text, 0),
                kind=kind,
                target=text,
                reason=reason,
                origin_round=0,
                priority=PRIORITY.get(kind, 1.0),
            )
        )

    for seed in config.seed_entities:
        add(seed.value or seed.raw, f"user-supplied {seed.kind.replace('_', ' ')}")
    if config.topic:
        add(config.topic, "run topic")
    if not items:
        add(config.target_species, "no topic or seed supplied; species-only start")
    return items


def _entity_names(graph: ResearchGraph, entity_ids: Iterable[str]) -> List[str]:
    names: List[str] = []
    for eid in entity_ids:
        entity = graph.registry.get(eid)
        if entity is not None and entity.name not in names:
            names.append(entity.name)
    return names


def _chemical(graph: ResearchGraph, entity_id: str) -> bool:
    entity = graph.registry.get(entity_id)
    if entity is None:
        return False
    return entity.entity_type in CHEMICAL_TYPES and not is_cofactor(entity.name)


def generate(
    *,
    graph: ResearchGraph,
    store: ClaimStore,
    delta: GraphDelta,
    config: MiningConfig,
    round_index: int,
    previous: Optional[Sequence[FrontierItem]] = None,
    exhausted_ids: Optional[Set[str]] = None,
    cut: Optional[List[FrontierItem]] = None,
) -> List[FrontierItem]:
    """Derive the next round's frontier from what this round discovered.

    Returns a deduplicated list ordered and capped at ``budgets.max_frontier_size``
    by :func:`cap_frontier`; the items the cap drops are appended to ``cut``. Items
    already tried and exhausted are not regenerated.
    """
    spent = set(exhausted_ids or set())
    for item in previous or []:
        if item.exhausted:
            spent.add(item.frontier_item_id)

    out: Dict[str, FrontierItem] = {}

    def add(
        kind: str,
        target: str,
        reason: str,
        *,
        entity_ids: Optional[Sequence[str]] = None,
        component_id: str = "",
        claim_identity: str = "",
        priority_bonus: float = 0.0,
        relevance: int = 0,
    ) -> None:
        text = str(target or "").strip()
        if not text:
            return
        fid = make_frontier_id(kind, text, 0)  # round-independent: same question, same id
        if fid in spent or fid in out:
            return
        out[fid] = FrontierItem(
            frontier_item_id=fid,
            kind=kind,
            target=text,
            reason=reason,
            origin_round=int(round_index),
            priority=PRIORITY.get(kind, 1.0) + priority_bonus,
            component_id=component_id,
            entity_ids=list(entity_ids or []),
            claim_identity=claim_identity,
            relevance=relevance,
        )

    touched_ids = set(delta.added_reaction_ids) | set(delta.updated_reaction_ids) | set(
        delta.promoted_reaction_ids
    ) | set(delta.added_candidate_reaction_ids)
    touched = [graph.reactions[r] for r in sorted(touched_ids) if r in graph.reactions]

    # -- 1. bridges: the highest-value question a mining run can ask ---------
    primary_chem = {
        e for r in graph.primary_reactions for e in r.chemical_ids() if _chemical(graph, e)
    }
    for component in graph.candidate_components:
        comp_chem: List[str] = []
        for rid in component.reaction_ids:
            reaction = graph.reactions.get(rid)
            if reaction is None:
                continue
            comp_chem.extend(e for e in reaction.chemical_ids() if _chemical(graph, e))
        comp_names = _entity_names(graph, dict.fromkeys(comp_chem))
        primary_names = _entity_names(graph, sorted(primary_chem))
        if not comp_names:
            continue
        anchor = comp_names[0]
        other = primary_names[0] if primary_names else config.topic or config.target_species
        add(
            FRONTIER_BRIDGE,
            f"{anchor} | {other}",
            (
                f"candidate component {component.component_id} is paper-supported but not "
                "chemically connected to the primary graph; look for a biochemical step "
                "linking them"
            ),
            entity_ids=list(dict.fromkeys(comp_chem))[:6],
            component_id=component.component_id,
            priority_bonus=min(2.0, 0.2 * len(component.reaction_ids)),
        )

    # -- 2. conflicts --------------------------------------------------------
    for conflict in store.conflicts.values():
        if conflict.resolved or len(conflict.paper_ids) < 2:
            continue
        reaction = graph.by_identity(conflict.claim_identity)
        label = reaction.label() if reaction is not None else conflict.claim_identity
        add(
            FRONTIER_CONFLICT,
            f"{conflict.attribute}: {label}",
            (
                f"independent papers disagree about {conflict.attribute}; retrieve further "
                "evidence rather than choosing a side"
            ),
            claim_identity=conflict.claim_identity,
        )

    # -- 3. downstream / upstream from the touched reactions -----------------
    consumed: Set[str] = set()
    produced: Set[str] = set()
    for reaction in graph.reactions.values():
        consumed.update(reaction.substrate_ids)
        produced.update(reaction.product_ids)

    for reaction in touched:
        for eid in reaction.product_ids:
            if not _chemical(graph, eid) or eid in consumed:
                continue
            name = _entity_names(graph, [eid])
            if name:
                add(
                    FRONTIER_DOWNSTREAM,
                    name[0],
                    f"newly supported product of {reaction.label()} with no known consuming "
                    "reaction; expand downstream",
                    entity_ids=[eid],
                )
        for eid in reaction.substrate_ids:
            if not _chemical(graph, eid) or eid in produced:
                continue
            name = _entity_names(graph, [eid])
            if name:
                add(
                    FRONTIER_UPSTREAM,
                    name[0],
                    f"substrate of {reaction.label()} with no supported precursor reaction; "
                    "expand upstream",
                    entity_ids=[eid],
                )

    # -- 4. branch reactions on newly discovered metabolites ----------------
    for eid in delta.added_entity_ids:
        entity = graph.registry.get(eid)
        if entity is None:
            continue
        if entity.entity_type in CHEMICAL_TYPES and not is_cofactor(entity.name):
            add(
                FRONTIER_BRANCH,
                entity.name,
                "newly discovered metabolite; look for other reactions it participates in",
                entity_ids=[eid],
            )
        elif entity.is_catalytic:
            add(
                FRONTIER_ENZYME,
                entity.name,
                "newly discovered enzyme; look for other reactions it catalyses "
                "(a shared enzyme is a search term, never a chemical connection)",
                entity_ids=[eid],
            )
        elif is_cofactor(entity.name):
            add(
                FRONTIER_COFACTOR,
                entity.name,
                "cofactor / co-substrate seen in a new reaction; investigate its role",
                entity_ids=[eid],
            )

    # -- 5. explicit pathway references found in paper text -----------------
    for reaction in touched:
        for pathway in reaction.pathway_names:
            add(
                FRONTIER_PATHWAY_REFERENCE,
                pathway,
                "pathway named in the supporting paper; retrieve its other steps "
                "(a shared pathway name is a search term, never a chemical connection)",
            )

    # -- 6. missing attributes on admitted reactions ------------------------
    for reaction in touched:
        if reaction.direction in ("", "unspecified") and reaction.reversible is None:
            add(
                FRONTIER_DIRECTION,
                reaction.label(),
                "direction and reversibility are unsupported; retrieve evidence for them",
                claim_identity=reaction.claim_identity,
            )
        if not reaction.compartment:
            add(
                FRONTIER_COMPARTMENT,
                reaction.label(),
                "no compartment evidence; retrieve subcellular localization",
                claim_identity=reaction.claim_identity,
            )

    # -- 7. claims held back for species or completeness reasons ------------
    target_keys = {comparison_key(config.target_species),
                   comparison_key(config.target_species_normalized)} - {""}
    run_keys = {comparison_key(t) for t in rescue_terms(config, graph.reactions.values())}

    def on_route(identity: str, claims: Sequence[Any]) -> int:
        """One point each: a sibling claim names the target species; a participant or
        enzyme is a seed or a participant of a target-supported reaction; the lead's
        component shares a non-cofactor chemical with the primary graph."""
        score = int(any(k in comparison_key(c.species_text) for c in claims for k in target_keys))
        names = {comparison_key(n) for c in claims
                 for n in (*c.substrate_names, *c.product_names, c.enzyme)} - {""}
        score += int(bool(names & run_keys))
        reaction = graph.by_identity(identity)
        if reaction is not None:
            component = graph.components.get(reaction.component_id)
            rids = component.reaction_ids if component else [reaction.reaction_id]
            score += int(any(graph.reactions[r].chemical_ids() & primary_chem
                             for r in rids if r in graph.reactions))
        return score

    for identity, history in store.histories.items():
        current = history.current
        if current is None or current.status not in {STATUS_CANDIDATE, STATUS_INCOMPLETE}:
            continue
        siblings = store.siblings(identity)
        if not siblings:
            continue
        claim = siblings[0]
        label = f"{claim.subject} -> {claim.object}".strip(" ->")
        if current.status == STATUS_CANDIDATE:
            add(
                FRONTIER_SPECIES,
                label or identity,
                (
                    f"held as a lead because species state is {claim.species_state!r}; "
                    f"retrieve direct {config.target_species} evidence "
                    "(an ortholog never transfers a reaction between species)"
                ),
                claim_identity=identity,
                relevance=on_route(identity, siblings),
            )
        else:
            add(
                FRONTIER_UPSTREAM,
                label or identity,
                "claim is incomplete; retrieve the missing substrates or products",
                claim_identity=identity,
            )

    kept, dropped = cap_frontier(out.values(), config.budgets.max_frontier_size)
    if cut is not None:
        cut.extend(dropped)
    return kept


def _rank(item: FrontierItem) -> Tuple[Any, ...]:
    return (-item.priority, item.kind, item.target.lower(), item.frontier_item_id)


def cap_frontier(
    items: Iterable[FrontierItem], cap: int
) -> Tuple[List[FrontierItem], List[FrontierItem]]:
    """Order ``items`` and keep at most ``cap`` of them. Returns ``(kept, cut)``.

    ``ceil(SPECIES_RESERVE x cap)`` slots go to ``missing_species_evidence`` items,
    the most on-route first (``FrontierItem.relevance``), whenever that many exist;
    the rest are filled in the priority order. A reserved item takes the EARLIER of
    its priority-order place and the next fourth place (4th, 8th, ...): the order is
    the order a round asks in, and a round whose discovery or processing budget runs
    out after a few items would otherwise never reach one, while an item that already
    outranks the items after it must never be pushed behind them. With no species
    item this is the plain priority order and cut. ``cap`` 0 means no cap.
    """
    ordered = sorted(items, key=_rank)
    cap = max(1, int(cap or len(ordered) or 1))
    species = sorted((i for i in ordered if i.kind == FRONTIER_SPECIES),
                     key=lambda i: (-i.relevance, *_rank(i)))
    reserved = species[: math.ceil(SPECIES_RESERVE * cap)]
    taken = {i.frontier_item_id for i in reserved}
    rest = [i for i in ordered if i.frontier_item_id not in taken][: cap - len(reserved)]
    stride = round(1 / SPECIES_RESERVE)
    kept: List[FrontierItem] = []
    for item in rest:
        while reserved and (len(kept) % stride == stride - 1 or _rank(reserved[0]) <= _rank(item)):
            kept.append(reserved.pop(0))
        kept.append(item)
    kept.extend(reserved)
    kept_ids = {i.frontier_item_id for i in kept}
    return kept, [i for i in ordered if i.frontier_item_id not in kept_ids]


def merge_frontier(
    current: Sequence[FrontierItem],
    generated: Sequence[FrontierItem],
    *,
    cap: int,
    cut: Optional[List[FrontierItem]] = None,
) -> List[FrontierItem]:
    """Carry unexhausted items forward and fold in the new ones, capped by
    :func:`cap_frontier`. The items the cap drops are appended to ``cut``."""
    merged: Dict[str, FrontierItem] = {}
    for item in list(current) + list(generated):
        if item.exhausted:
            continue
        existing = merged.get(item.frontier_item_id)
        if existing is None or item.priority > existing.priority:
            merged[item.frontier_item_id] = item
    kept, dropped = cap_frontier(merged.values(), cap)
    if cut is not None:
        cut.extend(dropped)
    return kept


__all__ = ["PRIORITY", "SPECIES_RESERVE", "seed_frontier", "generate", "cap_frontier",
           "merge_frontier"]
