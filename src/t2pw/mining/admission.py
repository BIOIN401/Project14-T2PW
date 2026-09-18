"""Admission policy: from an extracted claim to a decision and a graph node.

This is the only place a claim becomes a reaction. It is deliberately separate
from :mod:`t2pw.mining.graph`, which decides *placement* by chemistry alone --
admission decides *whether the biology is defensible*, placement decides *where
it connects*. Neither can do the other's job.

The ordered gates
-----------------
1. **Extraction integrity.** An empty, truncated, unparseable or
   partially-repaired completion is rejected outright. This is the hard stop for
   the known empty/truncated-completion failure mode: a repaired fragment is a
   diagnostic record and can never become a reaction.
2. **Paper evidence.** No evidence span, no claim. Version 1's admission basis
   *is* paper evidence -- and nothing else is required, in particular no external
   database identifier.
3. **Over-claiming.** A material field carrying a value with no evidence span is
   STRIPPED, not believed, and the strip is recorded. A passage supporting
   substrate and product but not an enzyme yields a reaction with no enzyme.
4. **Chemical completeness.** No substrates or no products -> ``incomplete``,
   retained and reconsiderable.
5. **Hypothesis.** A reaction the paper proposes rather than reports ->
   ``hypothesis_only``, quarantined and labelled.
6. **Species guardrail.** Anything but ``target_species_supported`` -> a labelled
   lead in quarantine. Never an established target-species reaction.
7. **Conflict.** A cross-paper disagreement about a material attribute ->
   ``conflicting``: still admitted to the graph, with the disagreement attached
   and the disputed attribute left unset rather than silently picked.
8. Otherwise ``accepted``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from t2pw.mining.claims import ClaimStore
from t2pw.mining.graph import ResearchReaction
from t2pw.mining.ids import reaction_id as make_reaction_id
from t2pw.mining.models import (
    DIRECTION_UNSPECIFIED,
    ClaimDecision,
    EvidenceSpan,
    ExtractedClaim,
    PaperRecord,
    STATUS_ACCEPTED,
    STATUS_CANDIDATE,
    STATUS_CONFLICTING,
    STATUS_HYPOTHESIS_ONLY,
    STATUS_INCOMPLETE,
    STATUS_REJECTED,
)
from t2pw.mining.normalize import EntityRegistry
from t2pw.mining.species import (
    SPECIES_TARGET_SUPPORTED,
    SPECIES_STATES,
)

#: Decisions that place a reaction in the research graph. ``conflicting`` is
#: here on purpose: a disagreement about an enzyme does not unmake the chemistry,
#: and hiding the reaction would hide the conflict with it.
ADMITTED_STATUSES = frozenset({STATUS_ACCEPTED, STATUS_CONFLICTING})

#: Decisions that keep a claim quarantined but fully preserved.
QUARANTINED_STATUSES = frozenset(
    {STATUS_CANDIDATE, STATUS_HYPOTHESIS_ONLY, STATUS_INCOMPLETE}
)

EVIDENCE_LEVEL_MULTI_PAPER = "multiple_independent_papers"
EVIDENCE_LEVEL_SINGLE_PAPER = "single_paper"
EVIDENCE_LEVEL_REVIEW_ONLY = "review_article_only"
EVIDENCE_LEVEL_NONE = "no_paper_evidence"

#: Material fields that may be stripped when unsupported, rather than rejecting
#: the whole claim. Substrates/products are NOT here -- they are the reaction.
STRIPPABLE_FIELDS = ("enzyme", "protein", "gene", "compartment", "direction", "reversible")


@dataclass
class AdmissionOutcome:
    """The result of judging one claim."""

    decision: ClaimDecision
    reaction: Optional[ResearchReaction] = None
    stripped_fields: List[str] = field(default_factory=list)
    conflict_ids: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def admitted(self) -> bool:
        return self.decision.status in ADMITTED_STATUSES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "reaction_id": self.reaction.reaction_id if self.reaction else "",
            "stripped_fields": list(self.stripped_fields),
            "conflict_ids": list(self.conflict_ids),
            "notes": list(self.notes),
        }


def evidence_level(
    *, paper_ids: Sequence[str], papers: Mapping[str, PaperRecord]
) -> str:
    """Classify support strength from the *papers*, not from model confidence."""
    distinct = [p for p in dict.fromkeys(paper_ids) if p]
    if not distinct:
        return EVIDENCE_LEVEL_NONE
    if len(distinct) > 1:
        return EVIDENCE_LEVEL_MULTI_PAPER
    record = papers.get(distinct[0])
    if record is not None and record.evidence_type == "review":
        return EVIDENCE_LEVEL_REVIEW_ONLY
    return EVIDENCE_LEVEL_SINGLE_PAPER


def _strip_unsupported(claim: ExtractedClaim) -> List[str]:
    """Remove material fields the evidence does not support. Records what went.

    This is the enforcement of "do not let the model create a stronger claim than
    its evidence supports", and it happens BEFORE any decision so a stripped
    enzyme cannot influence the verdict it helped produce.
    """
    stripped: List[str] = []
    for name in STRIPPABLE_FIELDS:
        current = getattr(claim, name, None)
        has_value = (
            current is not None
            and str(current).strip() != ""
            and not (name == "direction" and current == DIRECTION_UNSPECIFIED)
        )
        if not has_value:
            continue
        if claim.evidence_for(name):
            continue
        stripped.append(name)
        if name == "reversible":
            claim.reversible = None
        elif name == "direction":
            claim.direction = DIRECTION_UNSPECIFIED
        else:
            setattr(claim, name, "")
    return stripped


def build_reaction(
    claim: ExtractedClaim,
    *,
    registry: EntityRegistry,
    round_index: int,
    paper_ids: Sequence[str],
    evidence_ids: Sequence[str],
    claim_ids: Sequence[str],
    status: str,
    reason: str,
    admitted: bool,
    quarantine_reason: str = "",
) -> ResearchReaction:
    """Turn an admitted claim into a reaction node, registering its entities.

    Enzymes are registered as entities too, but their ids go into
    ``enzyme_entity_ids`` -- NOT into ``substrate_ids``/``product_ids``. That
    separation is what makes enzyme-only connectivity structurally impossible:
    :func:`~t2pw.mining.graph.connection_between` never reads the enzyme list.
    """
    substrates = [
        registry.upsert(
            p.name,
            declared_type=p.declared_type or "metabolite",
            evidence_ids=p.evidence_ids or evidence_ids,
            paper_ids=paper_ids,
            round_index=round_index,
        )
        for p in claim.substrates
    ]
    products = [
        registry.upsert(
            p.name,
            declared_type=p.declared_type or "metabolite",
            evidence_ids=p.evidence_ids or evidence_ids,
            paper_ids=paper_ids,
            round_index=round_index,
        )
        for p in claim.products
    ]
    enzyme_names = [n for n in (claim.enzyme, claim.protein) if str(n or "").strip()]
    enzymes = [
        registry.upsert(
            name,
            declared_type="enzyme",
            evidence_ids=claim.evidence_for("enzyme") or claim.evidence_for("protein"),
            paper_ids=paper_ids,
            round_index=round_index,
        )
        for name in enzyme_names
    ]

    return ResearchReaction(
        reaction_id=make_reaction_id(claim.claim_identity),
        claim_identity=claim.claim_identity,
        substrate_ids=[e.entity_id for e in substrates],
        product_ids=[e.entity_id for e in products],
        substrate_names=[p.name for p in claim.substrates],
        product_names=[p.name for p in claim.products],
        enzyme_names=enzyme_names,
        enzyme_entity_ids=[e.entity_id for e in enzymes],
        gene=claim.gene,
        direction=claim.direction,
        reversible=claim.reversible,
        compartment=claim.compartment,
        conditions=claim.conditions,
        tissue=claim.tissue,
        species_text=claim.species_text,
        species_state=claim.species_state,
        pathway_names=list(claim.pathway_names),
        discovery_round=int(round_index),
        claim_ids=list(dict.fromkeys(claim_ids)),
        paper_ids=list(dict.fromkeys(paper_ids)),
        evidence_ids=list(dict.fromkeys(evidence_ids)),
        decision_status=status,
        decision_reason=reason,
        admitted=bool(admitted),
        quarantine_reason=("" if admitted else (quarantine_reason or status)),
        hypothesis=bool(claim.hypothesis),
    )


def clear_disputed_attributes(reaction: ResearchReaction, conflicts: Sequence[Any]) -> List[str]:
    """Blank every attribute independent papers disagree about, on the graph node.

    Applied to the STORED reaction after a merge, not only to the freshly built
    one: a reaction admitted in round 1 with one enzyme attribution, contradicted
    in round 2, must stop displaying the round-1 value. Both positions stay in
    ``conflicts.json`` and on the claims -- what is removed is the graph node's
    silent assertion that one of them is the answer.
    """
    cleared: List[str] = []
    for conflict in conflicts:
        if len(getattr(conflict, "paper_ids", []) or []) < 2:
            continue
        attribute = getattr(conflict, "attribute", "")
        if attribute == "enzyme":
            if reaction.enzyme_names or reaction.enzyme_entity_ids:
                reaction.enzyme_names = []
                reaction.enzyme_entity_ids = []
                cleared.append("enzyme")
        elif attribute == "direction" and reaction.direction not in ("", DIRECTION_UNSPECIFIED):
            reaction.direction = DIRECTION_UNSPECIFIED
            cleared.append("direction")
        elif attribute == "reversible" and reaction.reversible is not None:
            reaction.reversible = None
            cleared.append("reversible")
        elif attribute == "compartment" and reaction.compartment:
            reaction.compartment = ""
            cleared.append("compartment")
    return cleared


def judge(
    claim: ExtractedClaim,
    *,
    store: ClaimStore,
    registry: EntityRegistry,
    papers: Mapping[str, PaperRecord],
    evidence: Mapping[str, EvidenceSpan],
    round_index: int,
    decided_at: str = "",
) -> AdmissionOutcome:
    """Apply the ordered gates to one claim and return the decision.

    ``claim`` must already be registered in ``store`` (so its siblings, and any
    independent support, are visible to the evidence-level and conflict tests).
    """
    notes: List[str] = []
    identity = claim.claim_identity
    previous = store.history(identity)
    supersedes = previous.current.decision_id if previous and previous.current else ""

    def decide(status: str, reason: str, target: str = "none", level: str = "",
               detail: Optional[Dict[str, Any]] = None) -> ClaimDecision:
        return ClaimDecision.build(
            claim=claim,
            status=status,
            reason=reason,
            round_index=round_index,
            target=target,
            evidence_level=level,
            supersedes=supersedes,
            decided_at=decided_at,
            detail=detail or {},
            supporting_paper_ids=store.supporting_papers(identity) or [claim.paper_id],
        )

    # -- gate 1: extraction integrity ---------------------------------------
    if not claim.admissible_extraction:
        return AdmissionOutcome(
            decision=decide(
                STATUS_REJECTED,
                f"extraction status {claim.extraction_status!r}: an empty, truncated, "
                "unparseable or partially-repaired completion can never become a reaction",
                detail={"extraction_status": claim.extraction_status},
            ),
            notes=["rejected before any biological test; this is a model-output failure"],
        )

    # -- gate 2: paper evidence ---------------------------------------------
    resolvable = [eid for eid in claim.evidence_ids if eid in evidence]
    if not resolvable:
        return AdmissionOutcome(
            decision=decide(
                STATUS_REJECTED,
                "no resolvable evidence span; paper evidence is the admission basis "
                "in version 1 and there is none",
                level=EVIDENCE_LEVEL_NONE,
            ),
        )
    missing_papers = [
        eid for eid in resolvable if evidence[eid].paper_id not in papers
    ]
    if missing_papers:
        return AdmissionOutcome(
            decision=decide(
                STATUS_REJECTED,
                f"evidence {missing_papers[0]} points at a paper this run has no record of; "
                "the provenance chain does not close",
            ),
        )

    # -- gate 3: over-claiming ----------------------------------------------
    stripped = _strip_unsupported(claim)
    if stripped:
        notes.append(
            "stripped unsupported field(s) rather than believing them: " + ", ".join(stripped)
        )

    paper_ids = store.supporting_papers(identity) or [claim.paper_id]
    all_evidence = store.evidence_ids(identity) or resolvable
    claim_ids = [c.claim_id for c in store.siblings(identity)] or [claim.claim_id]
    level = evidence_level(paper_ids=paper_ids, papers=papers)

    # -- gate 4: chemical completeness --------------------------------------
    if not claim.substrates or not claim.products:
        missing = "substrates" if not claim.substrates else "products"
        return AdmissionOutcome(
            decision=decide(
                STATUS_INCOMPLETE,
                f"no {missing} supported by the evidence; retained and reconsiderable "
                "if a later round supplies them",
                target="none",
                level=level,
                detail={"missing": missing},
            ),
            stripped_fields=stripped,
            notes=notes,
        )

    # -- gate 5: hypothesis --------------------------------------------------
    if claim.hypothesis:
        reaction = build_reaction(
            claim,
            registry=registry,
            round_index=round_index,
            paper_ids=paper_ids,
            evidence_ids=all_evidence,
            claim_ids=claim_ids,
            status=STATUS_HYPOTHESIS_ONLY,
            reason="the paper proposes this reaction rather than reporting it",
            admitted=False,
            quarantine_reason="hypothesis_only",
        )
        return AdmissionOutcome(
            decision=decide(
                STATUS_HYPOTHESIS_ONLY,
                "the source presents this as a proposal or model, not an observed result; "
                "retained as a labelled hypothesis, never as an established reaction",
                target="none",
                level=level,
            ),
            reaction=reaction,
            stripped_fields=stripped,
            notes=notes,
        )

    # -- gate 6: species guardrail ------------------------------------------
    if claim.species_state not in SPECIES_STATES:
        notes.append(f"unrecognised species state {claim.species_state!r}; treated as not stated")
    if claim.species_state != SPECIES_TARGET_SUPPORTED:
        reaction = build_reaction(
            claim,
            registry=registry,
            round_index=round_index,
            paper_ids=paper_ids,
            evidence_ids=all_evidence,
            claim_ids=claim_ids,
            status=STATUS_CANDIDATE,
            reason=f"species state {claim.species_state}: {claim.species_reason}",
            admitted=False,
            quarantine_reason=f"species:{claim.species_state}",
        )
        return AdmissionOutcome(
            decision=decide(
                STATUS_CANDIDATE,
                f"species state {claim.species_state!r} is not target-species-supported "
                f"({claim.species_reason or 'no species support recorded'}); retained as a "
                "labelled lead, not as an established target-species reaction",
                target="none",
                level=level,
                detail={"species_state": claim.species_state},
            ),
            reaction=reaction,
            stripped_fields=stripped,
            notes=notes,
        )

    # -- gate 7: conflict ----------------------------------------------------
    conflicts = store.detect_conflicts(identity, round_index)
    cross_paper = [c for c in conflicts if len(c.paper_ids) > 1]
    status = STATUS_CONFLICTING if cross_paper else STATUS_ACCEPTED
    if cross_paper:
        attributes = sorted({c.attribute for c in cross_paper})
        reason = (
            "admitted with an unresolved disagreement: independent papers report different "
            f"values for {', '.join(attributes)}. Both positions are retained; the disputed "
            "attribute is left unset rather than silently chosen."
        )
        for attribute in attributes:
            if attribute == "reversible":
                claim.reversible = None
            elif attribute == "direction":
                claim.direction = DIRECTION_UNSPECIFIED
            elif attribute == "enzyme":
                claim.enzyme = ""
                claim.protein = ""
            else:
                setattr(claim, attribute, "")
        notes.append(
            "disputed attribute(s) cleared on the graph node; every position stays in "
            "conflicts.json and on the claims"
        )
    else:
        reason = (
            f"paper-supported, target-species reaction with {level.replace('_', ' ')}; "
            "no external database lookup was required or performed"
        )

    reaction = build_reaction(
        claim,
        registry=registry,
        round_index=round_index,
        paper_ids=paper_ids,
        evidence_ids=all_evidence,
        claim_ids=claim_ids,
        status=status,
        reason=reason,
        admitted=True,
    )
    reaction.conflict_ids = sorted({c.conflict_id for c in conflicts})

    return AdmissionOutcome(
        decision=decide(status, reason, target="graph", level=level),
        reaction=reaction,
        stripped_fields=stripped,
        conflict_ids=sorted({c.conflict_id for c in conflicts}),
        notes=notes,
    )


__all__ = [
    "ADMITTED_STATUSES",
    "QUARANTINED_STATUSES",
    "STRIPPABLE_FIELDS",
    "EVIDENCE_LEVEL_MULTI_PAPER",
    "EVIDENCE_LEVEL_SINGLE_PAPER",
    "EVIDENCE_LEVEL_REVIEW_ONLY",
    "EVIDENCE_LEVEL_NONE",
    "AdmissionOutcome",
    "evidence_level",
    "build_reaction",
    "clear_disputed_attributes",
    "judge",
]
