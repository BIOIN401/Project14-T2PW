"""Protein identity verification for RAG typed-gap resolutions.

An accession lifted out of a sentence is a *claim*, not an identity. This module
is the seam between that claim and the repository's real verification ladder,
``t2pw.mapping.map_ids.verify_real_protein_identity`` — the same one the mapping
stage uses, with the same rungs (identifier resolution, entity type, candidate
evidence, species, name gate, score, margin) and the same fail-closed behaviour.

Why an adapter rather than a predicate
--------------------------------------
The ladder does not judge a string; it judges a *shipped identifier against local
resolver evidence*. It needs candidate rows that describe the accession, the
organism the run asked for, and the resolver's score and margin. Nothing in the
RAG chain produces those — retrieval returns passages, not resolver candidates —
so the honest wiring is:

* if a caller can supply candidate evidence (a resolver, a cache, a test), this
  adapter runs the real ladder over it and reports its verdict;
* if it cannot, the adapter reports :data:`REASON_NO_LOCAL_EVIDENCE` and verifies
  nothing. The gap stays open, the protein keeps no ``mapped_ids``, and the
  existing PathBank Unknown-protein fallback carries it honestly downstream.

The failure mode this exists to prevent is the tempting one: a verifier that
returns ``True`` because the accession matched the UniProt grammar. Well-formed
is not verified — ``P12345`` is well-formed and belongs to somebody. Writing it
into ``mapped_ids`` on that basis claims a confirmation nobody made, and it does
so in the one field the exporter trusts absolutely.

Offline: importing this module pulls no network and no DB. The heavy
``t2pw.mapping.map_ids`` import is deferred to the call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = [
    "IdentityCheck",
    "IdentityVerdict",
    "build_identity_verifier",
    "fail_closed_verifier",
    "REASON_NO_LOCAL_EVIDENCE",
    "REASON_LADDER_UNAVAILABLE",
    "OUTCOME_VERIFIED",
    "OUTCOME_NO_RESOLVER",
    "OUTCOME_RESOLVER_REJECTED",
]

#: No resolver candidates were available for this protein, so the ladder had
#: nothing to judge. Distinct from "the ladder judged it and said no".
REASON_NO_LOCAL_EVIDENCE = "no_local_resolver_evidence"
#: The mapping ladder itself could not be imported or ran into an error.
REASON_LADDER_UNAVAILABLE = "identity_ladder_unavailable"

# --- machine-readable outcome codes -----------------------------------------
# The caller needs to tell "nothing judged this claim" from "the ladder judged it
# and refused", and those are diagnostics that must not be recovered by matching
# the human-readable ``reason``. Each is set explicitly at the point the verdict
# is constructed, so it is a fact recorded by the code that knows it rather than
# one inferred downstream from prose.
OUTCOME_VERIFIED = "verified"
#: No verifier, no candidate provider, no candidates, or an unusable ladder — in
#: every case the claim was never judged.
OUTCOME_NO_RESOLVER = "unverified_no_resolver"
#: A wired resolver ran the full ladder and refused: missing evidence for THIS
#: accession, wrong entity type, wrong species, an implausible name, a score below
#: the floor, or too thin a margin over a rival.
OUTCOME_RESOLVER_REJECTED = "rejected_by_identity_resolver"


@dataclass
class IdentityVerdict:
    """What the ladder decided, and on what evidence.

    ``verified`` is the only thing that may write ``mapped_ids``. Everything else
    is diagnostics — but the diagnostics are the point: an unverified identity has
    to say *why*, or "the gap is still open" is indistinguishable from a bug.
    """

    verified: bool
    reason: str = ""
    #: One of :data:`OUTCOME_VERIFIED` / :data:`OUTCOME_NO_RESOLVER` /
    #: :data:`OUTCOME_RESOLVER_REJECTED`. Set at construction, never derived from
    #: ``reason``.
    outcome: str = OUTCOME_RESOLVER_REJECTED
    entity_name: str = ""
    organism: str = ""
    identity: str = ""
    score: float = -1.0
    margin: float = -1.0
    checks: Dict[str, Any] = field(default_factory=dict)
    candidates_considered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": bool(self.verified),
            "reason": self.reason,
            "outcome": self.outcome,
            "entity_name": self.entity_name,
            "organism": self.organism,
            "identity": self.identity,
            "score": self.score,
            "margin": self.margin,
            "checks": dict(self.checks),
            "candidates_considered": self.candidates_considered,
        }


@dataclass(frozen=True)
class IdentityCheck:
    """What a verifier call returns: a boolean that also says which branch it took.

    Truthy exactly when the identity is verified, so every existing
    ``bool(verifier(...))`` caller is unaffected; ``identity_outcome`` is what lets
    the diagnostics distinguish "no resolver was available" from "a resolver ran
    and refused" without reading the prose reason.
    """

    verified: bool
    identity_outcome: str
    verdict: Optional[IdentityVerdict] = None

    def __bool__(self) -> bool:
        return bool(self.verified)


#: A ``name -> [candidate row]`` callable. Candidate rows are the resolver shape
#: ``map_ids`` produces (``uniprot`` / ``name`` / ``organism`` / ``score`` / ...).
CandidateProvider = Callable[[str], Sequence[Dict[str, Any]]]


def build_identity_verifier(
    candidate_provider: Optional[CandidateProvider] = None,
    *,
    organism: str = "",
    record: Optional[List[Dict[str, Any]]] = None,
) -> Callable[[str, Dict[str, Any]], "IdentityCheck"]:
    """Return the ``(name, mapped_ids) -> IdentityCheck`` verifier synthesis expects.

    ``candidate_provider`` supplies the local resolver evidence the ladder needs.
    Pass ``None`` (the production default today) and the verifier is fail-closed:
    it always returns ``False`` with :data:`REASON_NO_LOCAL_EVIDENCE`, so a gap is
    never closed on evidence nobody produced.

    ``record``, when given, receives one :meth:`IdentityVerdict.to_dict` per call
    — which is how a run can report "we tried, here is what the ladder said"
    rather than silently declining.

    The returned callable yields an :class:`IdentityCheck`: truthy exactly when
    verified, and carrying the machine-readable outcome code so the caller never
    has to recover the branch from the reason text.
    """

    def _verify(name: str, mapped_ids: Dict[str, Any]) -> "IdentityCheck":
        verdict = verify_identity(
            name, mapped_ids, candidate_provider=candidate_provider, organism=organism
        )
        if record is not None:
            record.append(verdict.to_dict())
        return IdentityCheck(
            verified=bool(verdict.verified),
            identity_outcome=verdict.outcome,
            verdict=verdict,
        )

    return _verify


def fail_closed_verifier(
    record: Optional[List[Dict[str, Any]]] = None,
) -> Callable[[str, Dict[str, Any]], "IdentityCheck"]:
    """The explicit production default: verify nothing, say so, close no gap."""
    return build_identity_verifier(None, record=record)


def verify_identity(
    entity_name: str,
    mapped_ids: Dict[str, Any],
    *,
    candidate_provider: Optional[CandidateProvider] = None,
    organism: str = "",
) -> IdentityVerdict:
    """Run the repository's real identity ladder over local resolver evidence."""
    name = str(entity_name or "").strip()
    ids = dict(mapped_ids or {})
    if candidate_provider is None:
        return IdentityVerdict(
            verified=False,
            reason=REASON_NO_LOCAL_EVIDENCE,
            outcome=OUTCOME_NO_RESOLVER,
            entity_name=name,
            organism=organism,
            checks={
                "candidate_evidence": "no resolver is wired into this run, so the "
                "ladder has nothing to judge; the protein keeps the Unknown-protein "
                "fallback"
            },
        )
    try:
        candidates = list(candidate_provider(name) or [])
    except Exception:  # noqa: BLE001 - a failing provider is missing evidence
        candidates = []
    if not candidates:
        # A provider that returns nothing for this protein is the same fact as no
        # provider at all: there was no evidence for the ladder to weigh.
        return IdentityVerdict(
            verified=False,
            reason=REASON_NO_LOCAL_EVIDENCE,
            outcome=OUTCOME_NO_RESOLVER,
            entity_name=name,
            organism=organism,
            candidates_considered=0,
        )
    try:
        from t2pw.mapping.map_ids import verify_real_protein_identity
    except Exception:  # pragma: no cover - defensive; mapping is a sibling
        return IdentityVerdict(
            verified=False,
            reason=REASON_LADDER_UNAVAILABLE,
            outcome=OUTCOME_NO_RESOLVER,
            entity_name=name,
            organism=organism,
            candidates_considered=len(candidates),
        )
    try:
        raw = verify_real_protein_identity(
            name,
            candidates=candidates,
            mapped_ids=ids,
            organism=organism,
        )
    except Exception as exc:  # noqa: BLE001 - a ladder error is not a pass
        return IdentityVerdict(
            verified=False,
            reason=f"{REASON_LADDER_UNAVAILABLE}: {type(exc).__name__}",
            outcome=OUTCOME_NO_RESOLVER,
            entity_name=name,
            organism=organism,
            candidates_considered=len(candidates),
        )
    verified = bool(raw.get("verified"))
    return IdentityVerdict(
        verified=verified,
        reason=str(raw.get("reason") or ""),
        # The ladder RAN. Whatever it decided is a judgement about this claim.
        outcome=OUTCOME_VERIFIED if verified else OUTCOME_RESOLVER_REJECTED,
        entity_name=name,
        organism=str(raw.get("organism") or organism),
        identity=str(raw.get("identity") or ""),
        score=float(raw.get("score", -1.0) or -1.0),
        margin=float(raw.get("margin", -1.0) or -1.0),
        checks=dict(raw.get("checks") or {}),
        candidates_considered=len(candidates),
    )
