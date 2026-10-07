"""Claim storage: deduplication, conflicts, and a decision history that never
forgets.

Deduplication rule
------------------
Two claims with the same :func:`~t2pw.mining.models.reaction_identity` describe
the same chemistry in the same species. They collapse into one *reaction* but
**never** into one *citation*: each keeps its own claim record, its own evidence
spans and its own paper, so "supported by three independent papers" stays a
countable fact rather than a lost one.

Conflict rule
-------------
Same identity, different value for a material attribute (enzyme, direction,
reversibility, compartment) = a :class:`~t2pw.mining.models.ConflictRecord`. Both
positions are kept and displayed. Nothing picks a winner, and a conflict does not
delete either claim.

Reconsideration rule
--------------------
Unlike a closed rejection ledger, a decision here is versioned, not final. A
claim rejected as ``incomplete`` in round 2 is re-decided when round 4 brings
the missing evidence, and the earlier decision stays readable in the chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from t2pw.mining.ids import conflict_id as make_conflict_id
from t2pw.mining.models import (
    ClaimDecision,
    ConflictRecord,
    DIRECTION_UNSPECIFIED,
    ExtractedClaim,
)
from t2pw.mining.normalize import comparison_key
from t2pw.mining.species import (
    SPECIES_AMBIGUOUS,
    SPECIES_IN_VITRO,
    SPECIES_INFERRED_ONLY,
    SPECIES_MULTIPLE,
    SPECIES_NON_TARGET,
    SPECIES_NOT_STATED,
    SPECIES_TARGET_SUPPORTED,
    aggregate_species,
)

#: Attributes compared across papers for the same reaction identity. Substrates,
#: products and species are excluded because they are part of the identity --
#: a difference there is a different reaction, not a conflict about one.
CONFLICT_ATTRIBUTES = ("enzyme", "direction", "reversible", "compartment")


#: ``extraction.ATTRIBUTION_MODEL_CITATION``, spelled out so this module does not
#: import the extraction stack.
_MODEL_CITATION = "model_citation"

#: Strongest reason against admission first, as in
#: :func:`t2pw.mining.species.aggregate_species`. Used only for a claim that
#: carries no sentence reading to recompute from.
_SPECIES_PRECEDENCE = (
    SPECIES_INFERRED_ONLY,
    SPECIES_IN_VITRO,
    SPECIES_NOT_STATED,
    SPECIES_MULTIPLE,
    SPECIES_NON_TARGET,
    SPECIES_AMBIGUOUS,
    SPECIES_TARGET_SUPPORTED,
)


def _weakest(*states: str) -> str:
    """The state with the strongest reason against admission (unknown ranks first)."""
    present = [s for s in states if s]
    return min(present, key=lambda s: _SPECIES_PRECEDENCE.index(s) if s in _SPECIES_PRECEDENCE else -1,
               default="")


def _recompute_species(stored: ExtractedClaim, incoming: ExtractedClaim) -> None:
    """The same paper, the same identity, two batches: decide the species state on
    the COMBINED evidence (ORCH-745 F-6). Base copied the later batch's state over
    the earlier one, so an in-vitro-only batch read after the knockout batch
    demoted the reaction, and the result depended on batch order.

    The graded sentences, the model-cited species spans and the flags are unioned
    (same paper only: a claim from another paper is never merged here) and the
    aggregation re-run, so ``register(A); register(B)`` == ``register(B); register(A)``.
    A side with no sentence reading (a claim built outside extraction, or restored
    from a run older than schema 1.2.0) has nothing to union: its state can only
    weaken the result, whichever side it is on.
    """
    readings = [dict(c.species_support) for c in (stored, incoming) if c.species_support]
    unread = [(c.species_state, c.species_reason) for c in (stored, incoming) if not c.species_support]
    if readings:
        rows = {(r.get("evidence_id"), r.get("sentence"), r.get("grade")): dict(r)
                for reading in readings for r in reading.get("sentences") or []}
        sentences = [rows[k] for k in sorted(rows, key=lambda k: (str(k[0]), int(k[1] or 0), str(k[2])))]
        merged = {**readings[-1], **readings[0],
                  "sentences": sentences,
                  "species_evidence_ids": sorted({e for r in readings for e in r.get("species_evidence_ids") or []}),
                  "species_span_ok": any(r.get("species_span_ok") for r in readings),
                  "claimed_inferred": any(r.get("claimed_inferred") for r in readings)}
        # ponytail: a claim in the "unspecified:<state>" namespace keeps its identity
        # even if the union moves its state; both states are non-admissible.
        verdict = aggregate_species(
            target_species_normalized=str(merged.get("target_species") or ""),
            observed=[stored.species_text] if stored.species_text else [],
            species_evidence_ids=merged["species_evidence_ids"],
            sentences=sentences,
            species_span_ok=merged["species_span_ok"],
            claimed_inferred=merged["claimed_inferred"],
        )
        merged["grade"] = verdict.grade
        stored.species_support = merged
        state, reason = verdict.state, verdict.reason
    else:
        state, reason = stored.species_state, stored.species_reason
    weakest = _weakest(state, *(s for s, _ in unread))
    if weakest and weakest != state:
        state, reason = next((s, r) for s, r in unread if s == weakest)
    stored.species_state, stored.species_reason = state, reason


def _attr_value(claim: ExtractedClaim, attribute: str) -> str:
    if attribute == "enzyme":
        return comparison_key(claim.enzyme or claim.protein)
    if attribute == "direction":
        value = claim.direction
        return "" if value in ("", DIRECTION_UNSPECIFIED) else comparison_key(value)
    if attribute == "reversible":
        return "" if claim.reversible is None else ("true" if claim.reversible else "false")
    if attribute == "compartment":
        return comparison_key(claim.compartment)
    return ""


@dataclass
class ClaimHistory:
    """The full life of one reaction identity, in order."""

    claim_identity: str
    claim_ids: List[str] = field(default_factory=list)
    decisions: List[ClaimDecision] = field(default_factory=list)
    first_seen_round: int = 0
    last_updated_round: int = 0

    @property
    def current(self) -> Optional[ClaimDecision]:
        return self.decisions[-1] if self.decisions else None

    @property
    def current_status(self) -> str:
        return self.current.status if self.current else ""

    def narrative(self) -> List[str]:
        """Human-readable history: the sentences a reviewer actually wants."""
        lines: List[str] = []
        for decision in self.decisions:
            target = "" if decision.target == "none" else f" -> {decision.target}"
            lines.append(
                f"round {decision.round_index}: {decision.status}{target} "
                f"({decision.reason})"
            )
        return lines

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_identity": self.claim_identity,
            "claim_ids": list(self.claim_ids),
            "first_seen_round": self.first_seen_round,
            "last_updated_round": self.last_updated_round,
            "current_status": self.current_status,
            "decisions": [d.to_dict() for d in self.decisions],
            "narrative": self.narrative(),
        }


class ClaimStore:
    """Every claim, decision and conflict a run produced. Append-mostly."""

    def __init__(self) -> None:
        self.claims: Dict[str, ExtractedClaim] = {}
        self.by_identity: Dict[str, List[str]] = {}
        self.histories: Dict[str, ClaimHistory] = {}
        self.conflicts: Dict[str, ConflictRecord] = {}

    # -- registration -------------------------------------------------------
    def register(self, claim: ExtractedClaim) -> Tuple[ExtractedClaim, bool]:
        """Store ``claim``. Returns ``(stored_claim, is_new_independent_support)``.

        A repeat of the *same* claim from the *same* paper folds its evidence
        into the existing record (``is_new`` False). A claim with the same
        identity from a *different* paper is stored separately (``is_new`` True)
        -- that is the independent support the report counts.
        """
        identity = claim.claim_identity
        siblings = [self.claims[cid] for cid in self.by_identity.get(identity, [])]

        same_paper = next((s for s in siblings if s.paper_id == claim.paper_id), None)
        if same_paper is not None:
            for eid in claim.evidence_ids:
                if eid not in same_paper.evidence_ids:
                    same_paper.evidence_ids.append(eid)
            for fname, ids in claim.field_evidence.items():
                bucket = same_paper.field_evidence.setdefault(fname, [])
                for eid in ids:
                    if eid not in bucket:
                        bucket.append(eid)
            # A field is model-cited if either pass cited it: the evidence ids are
            # unioned above, so the label must not stay "verbatim" (IT2 N3/N4).
            for fname, how in claim.field_attribution.items():
                if how == _MODEL_CITATION or fname not in same_paper.field_attribution:
                    same_paper.field_attribution[fname] = how
            # A field the earlier pass could not support may now be supported.
            for fname in ("enzyme", "protein", "gene", "compartment", "conditions", "tissue"):
                if not getattr(same_paper, fname, "") and getattr(claim, fname, ""):
                    setattr(same_paper, fname, getattr(claim, fname))
            if same_paper.reversible is None and claim.reversible is not None:
                same_paper.reversible = claim.reversible
            if same_paper.direction in ("", DIRECTION_UNSPECIFIED) and claim.direction not in ("", DIRECTION_UNSPECIFIED):
                same_paper.direction = claim.direction
            same_paper.species_text = same_paper.species_text or claim.species_text
            _recompute_species(same_paper, claim)
            same_paper.last_updated_round = max(
                same_paper.last_updated_round, claim.last_updated_round, claim.first_seen_round
            )
            if same_paper.claim_id != claim.claim_id:
                if claim.claim_id not in same_paper.equivalent_claim_ids:
                    same_paper.equivalent_claim_ids.append(claim.claim_id)
            return same_paper, False

        self.claims[claim.claim_id] = claim
        bucket = self.by_identity.setdefault(identity, [])
        if claim.claim_id not in bucket:
            bucket.append(claim.claim_id)
        for sibling in siblings:
            if claim.claim_id not in sibling.equivalent_claim_ids:
                sibling.equivalent_claim_ids.append(claim.claim_id)
            if sibling.claim_id not in claim.equivalent_claim_ids:
                claim.equivalent_claim_ids.append(sibling.claim_id)

        history = self.histories.get(identity)
        if history is None:
            history = ClaimHistory(
                claim_identity=identity,
                first_seen_round=claim.first_seen_round,
                last_updated_round=claim.first_seen_round,
            )
            self.histories[identity] = history
        if claim.claim_id not in history.claim_ids:
            history.claim_ids.append(claim.claim_id)
        history.last_updated_round = max(history.last_updated_round, claim.first_seen_round)
        return claim, bool(siblings)

    # -- queries ------------------------------------------------------------
    def siblings(self, identity: str) -> List[ExtractedClaim]:
        return [self.claims[cid] for cid in self.by_identity.get(identity, []) if cid in self.claims]

    def supporting_papers(self, identity: str) -> List[str]:
        seen: List[str] = []
        for claim in self.siblings(identity):
            if claim.paper_id and claim.paper_id not in seen:
                seen.append(claim.paper_id)
        return seen

    def evidence_ids(self, identity: str) -> List[str]:
        out: List[str] = []
        for claim in self.siblings(identity):
            for eid in claim.evidence_ids:
                if eid not in out:
                    out.append(eid)
        return out

    def history(self, identity: str) -> Optional[ClaimHistory]:
        return self.histories.get(identity)

    def current_status(self, identity: str) -> str:
        history = self.histories.get(identity)
        return history.current_status if history else ""

    def reconsiderable(self, identity: str) -> bool:
        """Is this identity open to a new decision if new evidence arrives?

        Everything except an already-accepted claim is. An accepted claim is not
        re-decided downward by this mechanism -- new contradicting evidence
        becomes a conflict record instead, which stays visible.
        """
        history = self.histories.get(identity)
        if history is None or history.current is None:
            return True
        return history.current.reconsiderable

    # -- decisions ----------------------------------------------------------
    def record_decision(self, decision: ClaimDecision) -> ClaimDecision:
        history = self.histories.setdefault(
            decision.claim_identity,
            ClaimHistory(
                claim_identity=decision.claim_identity,
                first_seen_round=decision.round_index,
            ),
        )
        previous = history.current
        if previous is not None and not decision.supersedes:
            decision.supersedes = previous.decision_id
        if previous is not None and previous.decision_id == decision.decision_id:
            # Identical re-decision in the same round: nothing changed.
            return previous
        history.decisions.append(decision)
        history.last_updated_round = max(history.last_updated_round, decision.round_index)
        return decision

    def decisions(self) -> List[ClaimDecision]:
        out: List[ClaimDecision] = []
        for identity in sorted(self.histories):
            out.extend(self.histories[identity].decisions)
        return out

    # -- conflicts ----------------------------------------------------------
    def detect_conflicts(self, identity: str, round_index: int) -> List[ConflictRecord]:
        """Compare every sibling claim attribute-by-attribute.

        A conflict needs two claims from *different papers* asserting different
        non-empty values. One paper disagreeing with itself across two passages
        is recorded too (it is a real ambiguity) but is labelled as
        ``same_paper``.
        """
        siblings = self.siblings(identity)
        found: List[ConflictRecord] = []
        if len(siblings) < 2:
            return found

        for attribute in CONFLICT_ATTRIBUTES:
            positions: Dict[str, Dict[str, Any]] = {}
            for claim in siblings:
                value = _attr_value(claim, attribute)
                if not value:
                    continue
                bucket = positions.setdefault(
                    value,
                    {"value": value, "claim_ids": [], "paper_ids": [], "evidence_ids": []},
                )
                if claim.claim_id not in bucket["claim_ids"]:
                    bucket["claim_ids"].append(claim.claim_id)
                if claim.paper_id and claim.paper_id not in bucket["paper_ids"]:
                    bucket["paper_ids"].append(claim.paper_id)
                for eid in claim.evidence_for(attribute) or claim.evidence_ids:
                    if eid not in bucket["evidence_ids"]:
                        bucket["evidence_ids"].append(eid)
            if len(positions) < 2:
                continue

            cid = make_conflict_id(attribute, identity)
            all_papers = sorted({p for b in positions.values() for p in b["paper_ids"]})
            record = self.conflicts.get(cid)
            if record is None:
                record = ConflictRecord(
                    conflict_id=cid,
                    claim_identity=identity,
                    attribute=attribute,
                    first_seen_round=int(round_index),
                )
                self.conflicts[cid] = record
            record.positions = [positions[k] for k in sorted(positions)]
            record.claim_ids = sorted({c for b in positions.values() for c in b["claim_ids"]})
            record.paper_ids = all_papers
            record.last_updated_round = int(round_index)
            record.resolution_note = (
                "same_paper: one paper reports two different values"
                if len(all_papers) < 2
                else "cross_paper: independent papers disagree"
            )
            for claim in siblings:
                if cid not in claim.conflicts_with:
                    claim.conflicts_with.append(cid)
            found.append(record)
        return found

    def conflicts_for(self, identity: str) -> List[ConflictRecord]:
        return [c for c in self.conflicts.values() if c.claim_identity == identity]

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claims": {cid: c.to_dict() for cid, c in sorted(self.claims.items())},
            "by_identity": {k: list(v) for k, v in sorted(self.by_identity.items())},
            "histories": {k: h.to_dict() for k, h in sorted(self.histories.items())},
            "conflicts": {k: c.to_dict() for k, c in sorted(self.conflicts.items())},
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "ClaimStore":
        store = cls()
        payload = dict(data or {})
        for cid, row in (payload.get("claims") or {}).items():
            store.claims[str(cid)] = ExtractedClaim.from_dict(row)
        for identity, ids in (payload.get("by_identity") or {}).items():
            store.by_identity[str(identity)] = [str(i) for i in ids or []]
        for identity, row in (payload.get("histories") or {}).items():
            history = ClaimHistory(
                claim_identity=str(identity),
                claim_ids=[str(i) for i in (row or {}).get("claim_ids") or []],
                first_seen_round=int((row or {}).get("first_seen_round") or 0),
                last_updated_round=int((row or {}).get("last_updated_round") or 0),
            )
            history.decisions = [
                ClaimDecision.from_dict(d) for d in (row or {}).get("decisions") or []
            ]
            store.histories[str(identity)] = history
        for cid, row in (payload.get("conflicts") or {}).items():
            store.conflicts[str(cid)] = ConflictRecord.from_dict(row)
        return store


__all__ = ["CONFLICT_ATTRIBUTES", "ClaimHistory", "ClaimStore"]
