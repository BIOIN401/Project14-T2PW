"""The research graph: one primary pathway graph plus quarantined candidates.

Two graph classes, one rule
---------------------------
The **primary graph** holds paper-supported, target-species reactions that are
*chemically* connected to the rest of the primary network. Everything else that
is relevant but not chemically connected lives in a **candidate subgraph**,
visibly quarantined, with its provenance intact, eligible for a later bridge.

What counts as a chemical connection
------------------------------------
A shared *chemical participant* in compatible roles -- a product of one reaction
appearing as a substrate of another (sequential), two reactions consuming the
same metabolite (branch), two reactions producing it (convergent).

What never counts, no matter how suggestive:

* a shared enzyme, protein or gene
* a shared pathway name
* a shared paper
* similar vocabulary or semantic similarity
* co-occurrence in a passage
* a shared disease or phenotype
* a model asserting that two reactions "seem related"

:func:`connection_between` is the only function in this package that can answer
"are these two reactions connected", and it reads nothing but chemical
participants and their roles.

Cofactors
---------
By default a shared cofactor (ATP, water, NAD+, ...) does **not** by itself make
a connection. Nearly every reaction touches ATP or water, so admitting cofactor
bridges would collapse the whole literature into one component and make the
primary/candidate distinction meaningless. This is a conservative, documented
default and is switchable per graph (``allow_cofactor_bridges=True``) for
experiments that want it; the setting is recorded in the run manifest.

Reactions are first-class
-------------------------
A reaction is its own node with substrate and product edges -- never a pairwise
metabolite edge. A 3-substrate / 2-product reaction serializes as one reaction
node with five incident edges, so no viewer can read it as a chain of pairs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from t2pw.mining.ids import component_id as make_component_id
from t2pw.mining.ids import reaction_id as make_reaction_id
from t2pw.mining.ids import sha256_text
from t2pw.mining.normalize import (
    CHEMICAL_TYPES,
    EntityRegistry,
    comparison_key,
    is_cofactor,
)

GRAPH_PRIMARY = "primary"
GRAPH_CANDIDATE = "candidate"

CONNECTION_SEQUENTIAL = "product_to_substrate"
CONNECTION_BRANCH = "shared_substrate"
CONNECTION_CONVERGENT = "shared_product"

#: Ordered strongest-first. A sequential link is the canonical pathway step; a
#: branch or convergence is a genuine metabolic junction but a weaker claim about
#: flow, so it is labelled differently wherever it is displayed.
CONNECTION_KINDS = (CONNECTION_SEQUENTIAL, CONNECTION_BRANCH, CONNECTION_CONVERGENT)


@dataclass
class ConnectionEvidence:
    """Why two reactions are considered chemically connected."""

    kind: str
    via_entity_id: str
    via_entity_name: str
    left_reaction_id: str
    right_reaction_id: str
    is_cofactor: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReactionHistoryEvent:
    """One thing that happened to a reaction, kept forever."""

    round_index: int
    event: str
    detail: str = ""
    at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchReaction:
    """One paper-supported reaction node in the research graph.

    ``graph_status`` and ``component_id`` are *placement*, not truth: a reaction
    that moves from a candidate component into the primary graph is the same
    reaction with the same evidence, and its ``history`` says when and why it
    moved. Nothing is rewritten on promotion.
    """

    reaction_id: str
    claim_identity: str
    substrate_ids: List[str] = field(default_factory=list)
    product_ids: List[str] = field(default_factory=list)
    substrate_names: List[str] = field(default_factory=list)
    product_names: List[str] = field(default_factory=list)
    enzyme_names: List[str] = field(default_factory=list)
    enzyme_entity_ids: List[str] = field(default_factory=list)
    gene: str = ""
    direction: str = "unspecified"
    reversible: Optional[bool] = None
    compartment: str = ""
    conditions: str = ""
    tissue: str = ""
    species_text: str = ""
    species_state: str = ""
    pathway_names: List[str] = field(default_factory=list)

    graph_status: str = GRAPH_CANDIDATE
    component_id: str = ""
    discovery_round: int = 0
    promoted_round: int = -1

    claim_ids: List[str] = field(default_factory=list)
    paper_ids: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    conflict_ids: List[str] = field(default_factory=list)
    decision_status: str = ""
    decision_reason: str = ""
    #: Did the admission policy admit this reaction to the research graph proper?
    #: A quarantined reaction (wrong/ambiguous species, hypothesis-only,
    #: incomplete) is placed in a labelled quarantine component and can NEVER
    #: reach the primary graph by chemistry alone -- chemical connectivity is a
    #: placement rule, not an admission override.
    admitted: bool = False
    #: Why it is quarantined. "" for an admitted reaction.
    quarantine_reason: str = ""
    hypothesis: bool = False
    #: Deferred external-database validation. Always this shape in version 1.
    external_validation: Dict[str, Any] = field(
        default_factory=lambda: {"status": "not_evaluated"}
    )
    history: List[ReactionHistoryEvent] = field(default_factory=list)

    @property
    def in_primary(self) -> bool:
        return self.graph_status == GRAPH_PRIMARY

    @property
    def independent_paper_count(self) -> int:
        return len({p for p in self.paper_ids if p})

    @property
    def multi_paper_supported(self) -> bool:
        return self.independent_paper_count > 1

    def chemical_ids(self) -> Set[str]:
        return set(self.substrate_ids) | set(self.product_ids)

    def note(self, round_index: int, event: str, detail: str = "", at: str = "") -> None:
        self.history.append(
            ReactionHistoryEvent(round_index=int(round_index), event=event, detail=detail, at=at)
        )

    def label(self) -> str:
        left = " + ".join(self.substrate_names) or "?"
        right = " + ".join(self.product_names) or "?"
        arrow = "<=>" if self.reversible else "->"
        return f"{left} {arrow} {right}"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["history"] = [h.to_dict() for h in self.history]
        payload["label"] = self.label()
        payload["multi_paper_supported"] = self.multi_paper_supported
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResearchReaction":
        payload = {k: v for k, v in dict(data or {}).items()
                   if k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload["history"] = [
            ReactionHistoryEvent(**{k: v for k, v in dict(h).items()
                                    if k in ReactionHistoryEvent.__dataclass_fields__})  # type: ignore[attr-defined]
            for h in (data or {}).get("history") or []
        ]
        return cls(**payload)


@dataclass
class CandidateSubgraph:
    """A quarantined component: relevant, paper-supported, not yet connected.

    ``promotion_history`` survives promotion. A component that was promoted in
    round 6 still says it was discovered in round 2 and quarantined until a
    bridge arrived, because that provenance is the point.
    """

    component_id: str
    reaction_ids: List[str] = field(default_factory=list)
    created_round: int = 0
    promoted_round: int = -1
    promoted: bool = False
    relevance_score: float = 0.0
    rationale: str = ""
    #: ``""`` for an ordinary disconnected-but-admitted component; otherwise the
    #: admission reason that quarantined it (species state, hypothesis,
    #: incomplete). A quarantine component is never promoted by a bridge.
    quarantine_reason: str = ""
    promotion_history: List[Dict[str, Any]] = field(default_factory=list)
    bridge_reaction_ids: List[str] = field(default_factory=list)
    bridge_paper_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateSubgraph":
        payload = {k: v for k, v in dict(data or {}).items()
                   if k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**payload)


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------
def connection_between(
    left: ResearchReaction,
    right: ResearchReaction,
    *,
    registry: Optional[EntityRegistry] = None,
    allow_cofactor_bridges: bool = False,
) -> Optional[ConnectionEvidence]:
    """Chemical connection between two reactions, or ``None``.

    Reads substrate and product entity ids ONLY. It does not receive, and
    therefore cannot consult, enzymes, papers, pathway names, text or embeddings
    -- the isolation is structural, not a matter of discipline.
    """
    if left.reaction_id == right.reaction_id:
        return None

    def named(entity_id_: str) -> str:
        if registry is not None:
            entity = registry.get(entity_id_)
            if entity is not None:
                return entity.name
        return entity_id_

    def usable(entity_id_: str) -> bool:
        if allow_cofactor_bridges:
            return True
        if registry is None:
            return True
        entity = registry.get(entity_id_)
        if entity is None:
            return True
        if entity.entity_type not in CHEMICAL_TYPES:
            # An enzyme/protein/gene can never be a chemical bridge even if some
            # upstream code put it on a participant list.
            return False
        return not is_cofactor(entity.name)

    left_sub, left_prod = set(left.substrate_ids), set(left.product_ids)
    right_sub, right_prod = set(right.substrate_ids), set(right.product_ids)

    # A reversible reaction can supply either side, so its roles are unioned for
    # the sequential test only. Branch/convergent tests stay role-literal.
    left_out = left_prod | (left_sub if left.reversible else set())
    right_in = right_sub | (right_prod if right.reversible else set())
    right_out = right_prod | (right_sub if right.reversible else set())
    left_in = left_sub | (left_prod if left.reversible else set())

    for kind, shared in (
        (CONNECTION_SEQUENTIAL, (left_out & right_in) | (right_out & left_in)),
        (CONNECTION_BRANCH, left_sub & right_sub),
        (CONNECTION_CONVERGENT, left_prod & right_prod),
    ):
        usable_shared = sorted(e for e in shared if e and usable(e))
        if usable_shared:
            via = usable_shared[0]
            return ConnectionEvidence(
                kind=kind,
                via_entity_id=via,
                via_entity_name=named(via),
                left_reaction_id=left.reaction_id,
                right_reaction_id=right.reaction_id,
                is_cofactor=bool(registry and (e := registry.get(via)) and is_cofactor(e.name)),
            )
    return None


def _topic_relevance(reaction: ResearchReaction, topic_terms: Sequence[str]) -> float:
    """Fraction of topic terms this reaction's own text mentions, in [0, 1].

    Used ONLY to rank components for the primary-component choice. It never
    creates connectivity and never admits a reaction.
    """
    terms = [t for t in topic_terms if t]
    if not terms:
        return 0.0
    haystack = " ".join(
        [
            *reaction.substrate_names,
            *reaction.product_names,
            *reaction.enzyme_names,
            *reaction.pathway_names,
        ]
    ).lower()
    hit = sum(1 for t in terms if t in haystack)
    return hit / float(len(terms))


def topic_terms(topic: str) -> List[str]:
    """Lowercased content tokens of a topic string, for relevance ranking."""
    stop = {"the", "of", "and", "in", "a", "an", "for", "to", "biosynthesis_of"}
    raw = comparison_key(topic)
    return [t for t in raw.replace("-", " ").split() if t and t not in stop and len(t) > 2]


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------
class ResearchGraph:
    """Primary graph + candidate subgraphs, with a full placement history.

    Invariants enforced here rather than trusted:

    * a reaction is in the primary graph only if it is chemically connected to
      another primary reaction -- or it is the seed of the primary component;
    * a candidate component is promoted only when a *chemical* bridge exists,
      and the bridging reaction ids are recorded;
    * nothing is ever deleted; placement changes append history.
    """

    def __init__(
        self,
        *,
        registry: Optional[EntityRegistry] = None,
        topic: str = "",
        allow_cofactor_bridges: bool = False,
    ) -> None:
        self.registry = registry if registry is not None else EntityRegistry()
        self.topic = str(topic or "")
        self.allow_cofactor_bridges = bool(allow_cofactor_bridges)
        self.reactions: Dict[str, ResearchReaction] = {}
        self.components: Dict[str, CandidateSubgraph] = {}
        #: Set once the primary component has been chosen; "" until then.
        self.primary_component_id: str = ""
        self.primary_selection_rationale: str = ""
        self.connections: List[ConnectionEvidence] = []

    # -- queries ------------------------------------------------------------
    @property
    def primary_reactions(self) -> List[ResearchReaction]:
        return [r for r in self._ordered() if r.in_primary]

    @property
    def candidate_reactions(self) -> List[ResearchReaction]:
        return [r for r in self._ordered() if not r.in_primary]

    @property
    def candidate_components(self) -> List[CandidateSubgraph]:
        return [
            c
            for cid, c in sorted(self.components.items())
            if cid != self.primary_component_id and not c.promoted
        ]

    def _ordered(self) -> List[ResearchReaction]:
        return [self.reactions[k] for k in sorted(self.reactions)]

    def by_identity(self, identity: str) -> Optional[ResearchReaction]:
        for reaction in self._ordered():
            if reaction.claim_identity == identity:
                return reaction
        return None

    def connections_for(self, reaction_id: str) -> List[ConnectionEvidence]:
        return [
            c
            for c in self.connections
            if reaction_id in (c.left_reaction_id, c.right_reaction_id)
        ]

    def graph_hash(self) -> str:
        """Content hash of placement. Two identical graphs hash identically."""
        payload = [
            [r.reaction_id, r.graph_status, r.component_id, sorted(r.paper_ids)]
            for r in self._ordered()
        ]
        return sha256_text(json.dumps(payload, sort_keys=True))[:32]

    # -- mutation -----------------------------------------------------------
    def add_reaction(self, reaction: ResearchReaction, *, round_index: int, at: str = "") -> ResearchReaction:
        """Insert or merge a reaction and place it by CHEMISTRY alone.

        Returns the stored reaction, which may be an existing one that this call
        merged new evidence into (the deduplication path: same normalized
        identity, independent supporting evidence preserved).
        """
        existing = self.by_identity(reaction.claim_identity)
        if existing is not None:
            merged = self._merge_into(existing, reaction, round_index=round_index, at=at)
            if reaction.admitted and not merged.admitted:
                # Reconsideration: a claim that was quarantined has now been
                # admitted on new evidence. Lift it out of quarantine and place
                # it properly -- without erasing the quarantine history.
                old_component = merged.component_id
                merged.admitted = True
                merged.quarantine_reason = ""
                merged.decision_status = reaction.decision_status
                merged.decision_reason = reaction.decision_reason
                merged.species_state = reaction.species_state or merged.species_state
                component = self.components.get(old_component)
                if component is not None and merged.reaction_id in component.reaction_ids:
                    component.reaction_ids.remove(merged.reaction_id)
                    if not component.reaction_ids:
                        self.components.pop(old_component, None)
                merged.component_id = ""
                merged.note(
                    round_index,
                    "quarantine_lifted",
                    f"left {old_component}: {reaction.decision_reason}",
                    at,
                )
                self._place(merged, round_index=round_index, at=at)
            return merged

        if not reaction.reaction_id:
            reaction.reaction_id = make_reaction_id(reaction.claim_identity)
        reaction.discovery_round = int(round_index)
        self.reactions[reaction.reaction_id] = reaction
        reaction.note(round_index, "discovered", reaction.label(), at)
        self._place(reaction, round_index=round_index, at=at)
        return reaction

    def _merge_into(
        self,
        existing: ResearchReaction,
        incoming: ResearchReaction,
        *,
        round_index: int,
        at: str = "",
    ) -> ResearchReaction:
        """Fold a duplicate normalized claim in, keeping independent evidence.

        Deduplication never discards a supporting source. Two papers reporting
        the same chemistry produce one reaction with two paper ids, which is
        exactly what "reactions supported by multiple papers" later counts.
        """
        added_paper = False
        for pid in incoming.paper_ids:
            if pid and pid not in existing.paper_ids:
                existing.paper_ids.append(pid)
                added_paper = True
        for cid in incoming.claim_ids:
            if cid and cid not in existing.claim_ids:
                existing.claim_ids.append(cid)
        for eid in incoming.evidence_ids:
            if eid and eid not in existing.evidence_ids:
                existing.evidence_ids.append(eid)
        for name in incoming.enzyme_names:
            if name and name not in existing.enzyme_names:
                existing.enzyme_names.append(name)
        for eid in incoming.enzyme_entity_ids:
            if eid and eid not in existing.enzyme_entity_ids:
                existing.enzyme_entity_ids.append(eid)
        for name in incoming.pathway_names:
            if name and name not in existing.pathway_names:
                existing.pathway_names.append(name)
        if not existing.compartment and incoming.compartment:
            existing.compartment = incoming.compartment
        if existing.reversible is None and incoming.reversible is not None:
            existing.reversible = incoming.reversible
        if existing.direction in ("", "unspecified") and incoming.direction not in ("", "unspecified"):
            existing.direction = incoming.direction
        existing.note(
            round_index,
            "evidence_added" if added_paper else "duplicate_merged",
            f"paper(s) {', '.join(incoming.paper_ids) or '-'}",
            at,
        )
        return existing

    def _place(self, reaction: ResearchReaction, *, round_index: int, at: str = "") -> None:
        """Assign a reaction to the primary graph or a candidate component.

        Chemistry decides *where* an admitted reaction sits. It never decides
        *whether* a reaction is admitted: a quarantined reaction that happens to
        share a metabolite with the primary graph stays quarantined, because the
        thing keeping it out is its species state (or its hypothesis status, or
        its incompleteness), and a shared metabolite says nothing about any of
        those.
        """
        links = [
            link
            for other in self._ordered()
            if other.reaction_id != reaction.reaction_id
            and (link := connection_between(
                reaction,
                other,
                registry=self.registry,
                allow_cofactor_bridges=self.allow_cofactor_bridges,
            )) is not None
        ]
        for link in links:
            if link.to_dict() not in [c.to_dict() for c in self.connections]:
                self.connections.append(link)

        neighbours = [
            self.reactions[
                link.right_reaction_id
                if link.left_reaction_id == reaction.reaction_id
                else link.left_reaction_id
            ]
            for link in links
        ]

        if not reaction.admitted:
            self._quarantine(reaction, neighbours, round_index=round_index, at=at)
            return

        # An admitted reaction groups only with other ADMITTED reactions; a
        # quarantined neighbour is not a component it may join.
        neighbours = [n for n in neighbours if n.admitted]

        primary_neighbour = next((n for n in neighbours if n.in_primary), None)
        if primary_neighbour is not None and self.primary_component_id:
            self._attach_primary(reaction, round_index=round_index, at=at,
                                 detail=f"chemically connected to {primary_neighbour.reaction_id}")
            self._promote_now_connected(round_index=round_index, at=at, bridge=reaction)
            return

        # Join an existing candidate component through a chemical link, else
        # start a new one.
        neighbour_components = sorted({n.component_id for n in neighbours if n.component_id})
        if neighbour_components:
            target = neighbour_components[0]
            reaction.component_id = target
            reaction.graph_status = (
                GRAPH_PRIMARY if target == self.primary_component_id else GRAPH_CANDIDATE
            )
            component = self.components[target]
            if reaction.reaction_id not in component.reaction_ids:
                component.reaction_ids.append(reaction.reaction_id)
            reaction.note(round_index, "joined_component", target, at)
            # Two previously separate candidate components can now be one.
            for other_cid in neighbour_components[1:]:
                self._merge_components(target, other_cid, round_index=round_index, at=at)
            if reaction.graph_status == GRAPH_PRIMARY:
                reaction.promoted_round = int(round_index)
            self._promote_now_connected(round_index=round_index, at=at, bridge=reaction)
            return

        cid = make_component_id(reaction.claim_identity)
        while cid in self.components:
            cid = make_component_id(cid + "!")
        self.components[cid] = CandidateSubgraph(
            component_id=cid,
            reaction_ids=[reaction.reaction_id],
            created_round=int(round_index),
            rationale="new disconnected component at discovery",
        )
        reaction.component_id = cid
        reaction.graph_status = GRAPH_CANDIDATE
        reaction.note(round_index, "quarantined", f"component {cid}", at)

        # The very first component of a run becomes the primary graph only once
        # the run explicitly selects it (select_primary_component); until then
        # everything is a candidate, which is the honest state.

    def _quarantine(
        self,
        reaction: ResearchReaction,
        neighbours: Sequence[ResearchReaction],
        *,
        round_index: int,
        at: str = "",
    ) -> None:
        """Place a non-admitted reaction in a labelled quarantine component.

        Quarantine components group by *reason*, so a wrong-species lead and a
        hypothesis-only proposal never blur into one another, and neither can be
        promoted by a bridge -- a bridge answers "is it connected", not "is it
        the right species" or "is it an observation".
        """
        reason = reaction.quarantine_reason or reaction.decision_status or "quarantined"
        kin = [
            n
            for n in neighbours
            if not n.admitted
            and (n.quarantine_reason or n.decision_status or "quarantined") == reason
            and n.component_id
        ]
        if kin:
            target = sorted({n.component_id for n in kin})[0]
        else:
            target = make_component_id(f"quarantine::{reason}::{reaction.claim_identity}")
            while target in self.components:
                target = make_component_id(target + "!")
            self.components[target] = CandidateSubgraph(
                component_id=target,
                created_round=int(round_index),
                rationale=f"quarantined at discovery: {reason}",
                quarantine_reason=reason,
            )
        component = self.components[target]
        component.quarantine_reason = component.quarantine_reason or reason
        if reaction.reaction_id not in component.reaction_ids:
            component.reaction_ids.append(reaction.reaction_id)
        reaction.component_id = target
        reaction.graph_status = GRAPH_CANDIDATE
        reaction.note(
            round_index,
            "quarantined",
            f"component {target}: {reaction.decision_reason or reason}",
            at,
        )

    def _attach_primary(
        self, reaction: ResearchReaction, *, round_index: int, at: str, detail: str
    ) -> None:
        reaction.component_id = self.primary_component_id
        was_candidate = reaction.graph_status != GRAPH_PRIMARY
        reaction.graph_status = GRAPH_PRIMARY
        if was_candidate:
            reaction.promoted_round = int(round_index)
        component = self.components.get(self.primary_component_id)
        if component and reaction.reaction_id not in component.reaction_ids:
            component.reaction_ids.append(reaction.reaction_id)
        reaction.note(round_index, "admitted_to_primary", detail, at)

    def _merge_components(
        self, keep: str, drop: str, *, round_index: int, at: str = ""
    ) -> None:
        if keep == drop or drop not in self.components:
            return
        source = self.components.pop(drop)
        target = self.components[keep]
        for rid in source.reaction_ids:
            if rid not in target.reaction_ids:
                target.reaction_ids.append(rid)
            reaction = self.reactions.get(rid)
            if reaction is not None:
                reaction.component_id = keep
                reaction.note(round_index, "component_merged", f"{drop} -> {keep}", at)
        target.promotion_history.append(
            {"round": int(round_index), "event": "absorbed_component", "component_id": drop}
        )

    # -- promotion ----------------------------------------------------------
    def _promote_now_connected(
        self, *, round_index: int, at: str = "", bridge: Optional[ResearchReaction] = None
    ) -> List[str]:
        """Promote every candidate component now chemically joined to primary.

        Runs to a fixed point: promoting component A can connect component B
        through A, and B is promoted in the same pass. The bridging reaction is
        recorded on every component it promoted.
        """
        if not self.primary_component_id:
            return []
        promoted: List[str] = []
        changed = True
        while changed:
            changed = False
            primary_ids = {r.reaction_id for r in self.primary_reactions}
            for cid in sorted(self.components):
                if cid == self.primary_component_id or self.components[cid].promoted:
                    continue
                component = self.components[cid]
                if component.quarantine_reason:
                    # A bridge answers "is it connected", never "is it admissible".
                    continue
                link = None
                for rid in sorted(component.reaction_ids):
                    reaction = self.reactions.get(rid)
                    if reaction is None or not reaction.admitted:
                        continue
                    for pid in sorted(primary_ids):
                        found = connection_between(
                            reaction,
                            self.reactions[pid],
                            registry=self.registry,
                            allow_cofactor_bridges=self.allow_cofactor_bridges,
                        )
                        if found is not None:
                            link = found
                            break
                    if link is not None:
                        break
                if link is None:
                    continue
                bridge_id = bridge.reaction_id if bridge is not None else link.right_reaction_id
                bridge_papers = sorted(set(bridge.paper_ids)) if bridge is not None else []
                component.promoted = True
                component.promoted_round = int(round_index)
                component.bridge_reaction_ids = sorted({bridge_id, link.left_reaction_id, link.right_reaction_id})
                component.bridge_paper_ids = bridge_papers
                component.promotion_history.append(
                    {
                        "round": int(round_index),
                        "event": "promoted_to_primary",
                        "bridge_reaction_id": bridge_id,
                        "bridge_kind": link.kind,
                        "via_entity": link.via_entity_name,
                        "bridge_paper_ids": bridge_papers,
                    }
                )
                for rid in list(component.reaction_ids):
                    reaction = self.reactions.get(rid)
                    if reaction is None:
                        continue
                    self._attach_primary(
                        reaction,
                        round_index=round_index,
                        at=at,
                        detail=(
                            f"candidate component {cid} promoted via {link.kind} on "
                            f"{link.via_entity_name!r} (bridge {bridge_id})"
                        ),
                    )
                promoted.append(cid)
                changed = True
        return promoted

    def select_primary_component(
        self, *, round_index: int, at: str = "", topic: str = ""
    ) -> Optional[str]:
        """Choose the primary component by evidence-weighted topic relevance.

        Deterministic policy, recorded in ``primary_selection_rationale``:

        ``score = sum over reactions of (distinct supporting papers) x (1 + topic relevance)``

        Ties break on: more reactions, then more distinct papers, then the
        lexicographically smallest component id. No connection is fabricated to
        merge components -- the losers stay candidates.
        """
        if self.primary_component_id or not self.components:
            return self.primary_component_id or None
        terms = topic_terms(topic or self.topic)

        scored: List[Tuple[float, int, int, str]] = []
        for cid in sorted(self.components):
            component = self.components[cid]
            if component.quarantine_reason:
                # A quarantine component is never eligible to BE the primary graph.
                continue
            reactions = [
                self.reactions[r]
                for r in component.reaction_ids
                if r in self.reactions and self.reactions[r].admitted
            ]
            if not reactions:
                continue
            papers: Set[str] = set()
            score = 0.0
            for reaction in reactions:
                papers.update(p for p in reaction.paper_ids if p)
                score += reaction.independent_paper_count * (1.0 + _topic_relevance(reaction, terms))
            component.relevance_score = round(score, 4)
            scored.append((score, len(reactions), len(papers), cid))

        if not scored:
            return None
        scored.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
        best = scored[0]
        self.primary_component_id = best[3]
        chosen = self.components[best[3]]
        chosen.rationale = (
            f"selected as the primary component: evidence-weighted topic relevance "
            f"{best[0]:.4f} over {best[1]} reaction(s) and {best[2]} distinct paper(s); "
            f"{len(scored) - 1} other component(s) retained as candidates"
        )
        self.primary_selection_rationale = chosen.rationale
        for rid in chosen.reaction_ids:
            reaction = self.reactions.get(rid)
            if reaction is not None:
                reaction.graph_status = GRAPH_PRIMARY
                reaction.note(round_index, "primary_component_selected", chosen.rationale, at)
        self._promote_now_connected(round_index=round_index, at=at)
        return self.primary_component_id

    def rebuild_connections(self) -> None:
        """Recompute every pairwise connection. Idempotent; used after resume."""
        self.connections = []
        ordered = self._ordered()
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                link = connection_between(
                    left,
                    right,
                    registry=self.registry,
                    allow_cofactor_bridges=self.allow_cofactor_bridges,
                )
                if link is not None:
                    self.connections.append(link)

    # -- serialization ------------------------------------------------------
    def primary_to_dict(self) -> Dict[str, Any]:
        primary = self.primary_reactions
        primary_ids = {r.reaction_id for r in primary}
        return {
            "component_id": self.primary_component_id,
            "selection_rationale": self.primary_selection_rationale,
            "allow_cofactor_bridges": self.allow_cofactor_bridges,
            "reaction_count": len(primary),
            "entity_ids": sorted({e for r in primary for e in r.chemical_ids()}),
            "reactions": [r.to_dict() for r in primary],
            "connections": [
                c.to_dict()
                for c in self.connections
                if c.left_reaction_id in primary_ids and c.right_reaction_id in primary_ids
            ],
            "graph_hash": self.graph_hash(),
        }

    def candidates_to_dict(self) -> Dict[str, Any]:
        out: List[Dict[str, Any]] = []
        for component in sorted(self.components.values(), key=lambda c: c.component_id):
            if component.component_id == self.primary_component_id:
                continue
            reactions = [
                self.reactions[r].to_dict()
                for r in sorted(component.reaction_ids)
                if r in self.reactions
            ]
            payload = component.to_dict()
            payload["reactions"] = reactions
            out.append(payload)
        return {"components": out, "count": len(out)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "allow_cofactor_bridges": self.allow_cofactor_bridges,
            "primary_component_id": self.primary_component_id,
            "primary_selection_rationale": self.primary_selection_rationale,
            "reactions": {r.reaction_id: r.to_dict() for r in self._ordered()},
            "components": {c.component_id: c.to_dict() for c in self.components.values()},
            "entities": self.registry.to_dict(),
            "connections": [c.to_dict() for c in self.connections],
            "graph_hash": self.graph_hash(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResearchGraph":
        payload = dict(data or {})
        graph = cls(
            registry=EntityRegistry.from_dict(payload.get("entities")),
            topic=str(payload.get("topic") or ""),
            allow_cofactor_bridges=bool(payload.get("allow_cofactor_bridges")),
        )
        for rid, row in (payload.get("reactions") or {}).items():
            graph.reactions[str(rid)] = ResearchReaction.from_dict(row)
        for cid, row in (payload.get("components") or {}).items():
            graph.components[str(cid)] = CandidateSubgraph.from_dict(row)
        graph.primary_component_id = str(payload.get("primary_component_id") or "")
        graph.primary_selection_rationale = str(payload.get("primary_selection_rationale") or "")
        graph.rebuild_connections()
        return graph


def load_starting_graph(
    payload: Mapping[str, Any], *, registry: Optional[EntityRegistry] = None
) -> ResearchGraph:
    """Adopt an existing pathway / research graph as a READ-ONLY starting network.

    Accepts either a previously exported mining graph (``reactions`` keyed by id)
    or a plain pathway payload with a ``reactions`` list of
    ``{name, inputs, outputs, enzymes, biological_state}`` -- the shape the T2PW
    pipeline uses. Nothing here produces PWML, calls the PWML exporter, or runs a
    PathWhiz importer gate; the graph is structure and provenance only.
    """
    data = dict(payload or {})
    if isinstance(data.get("reactions"), dict) and "entities" in data:
        return ResearchGraph.from_dict(data)

    graph = ResearchGraph(registry=registry or EntityRegistry(), topic=str(data.get("topic") or ""))
    rows = data.get("reactions") or []
    source_label = str(data.get("source") or "supplied_starting_graph")
    for index, row in enumerate(rows):
        row = dict(row or {})
        inputs = [str(x) for x in (row.get("inputs") or row.get("substrates") or [])]
        outputs = [str(x) for x in (row.get("outputs") or row.get("products") or [])]
        if not inputs and not outputs:
            continue
        enzymes: List[str] = []
        for enz in row.get("enzymes") or []:
            if isinstance(enz, Mapping):
                name = str(enz.get("protein") or enz.get("name") or "").strip()
            else:
                name = str(enz or "").strip()
            if name:
                enzymes.append(name)
        sub_entities = [graph.registry.upsert(n, declared_type="metabolite", round_index=0) for n in inputs]
        prod_entities = [graph.registry.upsert(n, declared_type="metabolite", round_index=0) for n in outputs]
        enz_entities = [graph.registry.upsert(n, declared_type="enzyme", round_index=0) for n in enzymes]
        from t2pw.mining.models import reaction_identity

        identity = reaction_identity(inputs, outputs, species=str(data.get("species") or ""))
        reaction = ResearchReaction(
            reaction_id=make_reaction_id(identity),
            claim_identity=identity,
            substrate_ids=[e.entity_id for e in sub_entities],
            product_ids=[e.entity_id for e in prod_entities],
            substrate_names=inputs,
            product_names=outputs,
            enzyme_names=enzymes,
            enzyme_entity_ids=[e.entity_id for e in enz_entities],
            compartment=str(row.get("biological_state") or row.get("compartment") or ""),
            species_text=str(data.get("species") or ""),
            species_state="",
            decision_status="starting_graph",
            decision_reason=f"supplied as a read-only starting network ({source_label})",
            admitted=True,
        )
        reaction.note(0, "starting_graph_input", f"row {index} of {source_label}")
        graph.add_reaction(reaction, round_index=0)
    graph.select_primary_component(round_index=0, topic=graph.topic)
    return graph


__all__ = [
    "GRAPH_PRIMARY",
    "GRAPH_CANDIDATE",
    "CONNECTION_KINDS",
    "CONNECTION_SEQUENTIAL",
    "CONNECTION_BRANCH",
    "CONNECTION_CONVERGENT",
    "ConnectionEvidence",
    "ReactionHistoryEvent",
    "ResearchReaction",
    "CandidateSubgraph",
    "ResearchGraph",
    "connection_between",
    "load_starting_graph",
    "topic_terms",
]
