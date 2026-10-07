"""The mining data model.

Six layers are kept deliberately separate, and nothing collapses them into one
opaque model response:

1. **factual paper evidence** -- :class:`PaperRecord`, :class:`EvidenceSpan`
2. **normalized entity representation** -- :mod:`t2pw.mining.normalize`
3. **model interpretation** -- :class:`ExtractedClaim`
4. **admission decision** -- :class:`ClaimDecision`
5. **graph connectivity** -- :mod:`t2pw.mining.graph`
6. **exploratory hypothesis** -- the ``hypothesis`` flag on a claim, and the
   ``hypothesis_only`` decision status

The provenance chain every accepted reaction must satisfy is::

    reaction -> claim -> evidence span -> paper -> source identifier / file hash

:func:`provenance_chain_complete` is the machine check for exactly that.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from t2pw.mining.config import ARTIFACT_SCHEMA_VERSION
from t2pw.mining.ids import claim_id, decision_id, stable_join
from t2pw.mining.normalize import comparison_key
from t2pw.mining.species import SPECIES_NOT_STATED

# ---------------------------------------------------------------------------
# Claim / decision vocabularies
# ---------------------------------------------------------------------------
CLAIM_REACTION = "reaction"
CLAIM_ENZYME_ASSOCIATION = "enzyme_association"
CLAIM_SPECIES_SUPPORT = "species_support"
CLAIM_COMPARTMENT = "compartment"
CLAIM_DIRECTION = "direction"
CLAIM_PATHWAY_MEMBERSHIP = "pathway_membership"

CLAIM_TYPES = (
    CLAIM_REACTION,
    CLAIM_ENZYME_ASSOCIATION,
    CLAIM_SPECIES_SUPPORT,
    CLAIM_COMPARTMENT,
    CLAIM_DIRECTION,
    CLAIM_PATHWAY_MEMBERSHIP,
)

STATUS_ACCEPTED = "accepted"
STATUS_CANDIDATE = "candidate"
STATUS_CONFLICTING = "conflicting"
STATUS_INCOMPLETE = "incomplete"
STATUS_REJECTED = "rejected"
STATUS_HYPOTHESIS_ONLY = "hypothesis_only"

DECISION_STATUSES = (
    STATUS_ACCEPTED,
    STATUS_CANDIDATE,
    STATUS_CONFLICTING,
    STATUS_INCOMPLETE,
    STATUS_REJECTED,
    STATUS_HYPOTHESIS_ONLY,
)

#: Statuses that keep a claim reconsiderable in a later round. Unlike a closed
#: rejection ledger, nothing here is permanent: a claim rejected in round 2 can
#: be re-decided in round 5 when new evidence arrives.
RECONSIDERABLE_STATUSES = frozenset(
    {STATUS_CANDIDATE, STATUS_CONFLICTING, STATUS_INCOMPLETE, STATUS_REJECTED, STATUS_HYPOTHESIS_ONLY}
)

EXTRACTION_OK = "ok"
EXTRACTION_EMPTY = "empty_completion"
EXTRACTION_TRUNCATED = "truncated_completion"
EXTRACTION_SCHEMA_ERROR = "schema_error"
EXTRACTION_PARSE_ERROR = "parse_error"
EXTRACTION_PARTIAL_REPAIR = "partial_repair_diagnostic_only"

#: Extraction statuses that may NEVER reach the accepted graph. A repaired or
#: truncated completion is a diagnostic record, not a reaction.
NON_ADMISSIBLE_EXTRACTION_STATUSES = frozenset(
    {EXTRACTION_EMPTY, EXTRACTION_TRUNCATED, EXTRACTION_SCHEMA_ERROR,
     EXTRACTION_PARSE_ERROR, EXTRACTION_PARTIAL_REPAIR}
)

EVIDENCE_PRIMARY = "primary"
EVIDENCE_REVIEW = "review"
EVIDENCE_PREPRINT = "preprint"
EVIDENCE_UNKNOWN = "unknown"

DIRECTION_FORWARD = "forward"
DIRECTION_REVERSE = "reverse"
DIRECTION_UNSPECIFIED = "unspecified"

_WS = re.compile(r"\s+")


def _clean(value: Any) -> str:
    return _WS.sub(" ", str(value if value is not None else "")).strip()


def _str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        text = _clean(value)
        return [text] if text else []
    out: List[str] = []
    for item in value:
        text = _clean(item)
        if text and text not in out:
            out.append(text)
    return out


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------
@dataclass
class PaperRecord:
    """Everything known about one source paper, as it was known.

    Every field is optional except ``paper_id`` because real literature metadata
    is patchy; what matters is that an absent field reads as absent rather than
    as an invented value. ``screening_outcome`` and the inclusion/exclusion
    reasons are kept on the paper so a run can show the papers it retrieved and
    then EXCLUDED, filtered apart from accepted evidence rather than mixed in.
    """

    paper_id: str
    pmid: str = ""
    pmcid: str = ""
    doi: str = ""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    article_type: str = ""
    url: str = ""
    acquisition_source: str = ""
    cache_ref: str = ""
    file_sha256: str = ""
    file_name: str = ""
    retrieved_at: str = ""
    screening_outcome: str = ""
    screening_score: float = 0.0
    inclusion_reasons: List[str] = field(default_factory=list)
    exclusion_reasons: List[str] = field(default_factory=list)
    evidence_type: str = EVIDENCE_UNKNOWN
    #: The rule that decided ``evidence_type``, quoting what it matched, so an
    #: auditor can see why a paper is a review. "" when nothing decided it.
    evidence_type_basis: str = ""
    discovered_round: int = 0
    processed_round: int = -1
    abstract: str = ""
    full_text_chars: int = 0
    observed_organisms: List[str] = field(default_factory=list)
    observed_pathways: List[str] = field(default_factory=list)
    #: The frontier item / query that led here. "" for a user-supplied paper.
    discovered_via_query_id: str = ""
    discovered_via_frontier_id: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def has_source_identifier(self) -> bool:
        """The last link of the provenance chain: does this paper resolve?

        A PMID, PMCID, DOI, URL or uploaded-file hash. Without at least one of
        these an "evidence chain" ends at a title, which is not auditable.
        """
        return bool(self.pmid or self.pmcid or self.doi or self.url or self.file_sha256)

    @property
    def included(self) -> bool:
        return not self.exclusion_reasons

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PaperRecord":
        payload = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
@dataclass
class EvidenceSpan:
    """One quoted passage, with enough locator detail to find it again.

    ``text`` is the EXACT supporting text -- not a paraphrase and not a model
    summary. An extraction that cannot quote the passage it relied on does not
    get to create an evidence span.
    """

    evidence_id: str
    paper_id: str
    text: str
    section: str = ""
    page: Optional[int] = None
    paragraph: Optional[int] = None
    locator: str = ""
    chunk_id: str = ""
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    context_ref: str = ""
    extraction_round: int = 0
    retrieval_query_id: str = ""
    frontier_item_id: str = ""
    retrieval_score: float = 0.0
    retrieved_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceSpan":
        payload = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------
@dataclass
class ReactionParticipant:
    """One substrate or product, with its own evidence."""

    name: str
    declared_type: str = ""
    stoichiometry: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    #: How the participant got its evidence: ``model_citation`` (the model cited a
    #: span it was shown) or ``verbatim_cited_span`` (the name occurs verbatim in a
    #: span the claim cites). An unsupported participant is not kept at all.
    attribution: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedClaim:
    """What the model read out of one passage, before any admission decision.

    The separation that matters: ``field_evidence`` attaches evidence ids to
    *individual fields*. A passage that supports substrate and product but says
    nothing about an enzyme produces a claim whose ``enzyme`` is empty and whose
    ``field_evidence`` has no ``enzyme`` key -- the model does not get to invent
    one to fill the schema, and :meth:`unsupported_fields` catches it if it
    tries.
    """

    claim_id: str
    claim_identity: str
    claim_type: str = CLAIM_REACTION
    paper_id: str = ""

    subject: str = ""
    relationship: str = ""
    object: str = ""

    species_text: str = ""
    species_state: str = SPECIES_NOT_STATED
    species_reason: str = ""

    substrates: List[ReactionParticipant] = field(default_factory=list)
    products: List[ReactionParticipant] = field(default_factory=list)
    enzyme: str = ""
    protein: str = ""
    gene: str = ""
    direction: str = DIRECTION_UNSPECIFIED
    reversible: Optional[bool] = None
    compartment: str = ""
    conditions: str = ""
    tissue: str = ""
    pathway_names: List[str] = field(default_factory=list)

    #: field name -> evidence ids supporting THAT field specifically.
    field_evidence: Dict[str, List[str]] = field(default_factory=dict)
    #: field name -> how its evidence was attached: ``model_citation`` (the model
    #: cited the span) or ``verbatim_cited_span`` (the value occurs verbatim in
    #: one of this claim's own cited participant spans; for ``tissue``, in any span
    #: the claim cites).
    field_attribution: Dict[str, str] = field(default_factory=dict)
    evidence_ids: List[str] = field(default_factory=list)
    #: The sentence-level species reading (:func:`t2pw.mining.species.grade_sentences`):
    #: ``grade`` (``in_planta`` / ``review`` / ``in_vitro`` / ``inference`` / ``host`` /
    #: ``none``), the graded supporting ``sentences``, and what a same-paper merge
    #: needs to recompute the state: ``species_evidence_ids`` (model-cited only),
    #: ``species_span_ok``, ``claimed_inferred``, ``target_species``,
    #: ``paper_evidence_type``.
    species_support: Dict[str, Any] = field(default_factory=dict)
    #: Participants the model asserted that no evidence supports (no valid citation
    #: and not verbatim in a cited span): ``{"side", "name", "reason"}``.
    unsupported_participants: List[Dict[str, str]] = field(default_factory=list)

    model: str = ""
    actual_model: str = ""
    prompt_version: str = ""
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    model_call_id: str = ""
    extraction_status: str = EXTRACTION_OK

    #: The paper presents this as a proposal, not an established result.
    hypothesis: bool = False

    conflicts_with: List[str] = field(default_factory=list)
    duplicate_of: str = ""
    equivalent_claim_ids: List[str] = field(default_factory=list)

    first_seen_round: int = 0
    last_updated_round: int = 0
    confidence: str = ""

    # -- derived ------------------------------------------------------------
    @property
    def substrate_names(self) -> List[str]:
        return [p.name for p in self.substrates]

    @property
    def product_names(self) -> List[str]:
        return [p.name for p in self.products]

    @property
    def admissible_extraction(self) -> bool:
        """A claim from an empty/truncated/repaired completion is never admitted."""
        return self.extraction_status not in NON_ADMISSIBLE_EXTRACTION_STATUSES

    def evidence_for(self, field_name: str) -> List[str]:
        return list(self.field_evidence.get(field_name) or [])

    def unsupported_fields(self) -> List[str]:
        """Material fields carrying a value with no evidence span attached.

        This is the "do not let the model make a stronger claim than its
        evidence supports" check, expressed as data. A claim with any unsupported
        material field cannot be accepted; the offending field is either dropped
        or the claim is held as ``incomplete``.
        """
        offenders: List[str] = []
        for name, value in (
            ("enzyme", self.enzyme),
            ("protein", self.protein),
            ("gene", self.gene),
            ("compartment", self.compartment),
            ("direction", self.direction if self.direction != DIRECTION_UNSPECIFIED else ""),
            ("reversible", "" if self.reversible is None else str(self.reversible)),
            ("species", self.species_text),
            ("tissue", self.tissue),
        ):
            if _clean(value) and not self.evidence_for(name):
                offenders.append(name)
        for side, participants in (("substrates", self.substrates), ("products", self.products)):
            if participants and not (self.evidence_for(side) or all(p.evidence_ids for p in participants)):
                offenders.append(side)
        return offenders

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["substrates"] = [p.to_dict() for p in self.substrates]
        payload["products"] = [p.to_dict() for p in self.products]
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExtractedClaim":
        payload = dict(data or {})
        payload["substrates"] = [
            ReactionParticipant(**{k: v for k, v in dict(p).items()
                                   if k in ReactionParticipant.__dataclass_fields__})  # type: ignore[attr-defined]
            for p in payload.get("substrates") or []
        ]
        payload["products"] = [
            ReactionParticipant(**{k: v for k, v in dict(p).items()
                                   if k in ReactionParticipant.__dataclass_fields__})  # type: ignore[attr-defined]
            for p in payload.get("products") or []
        ]
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


def reaction_identity(
    substrates: Iterable[Any], products: Iterable[Any], *, species: str = ""
) -> str:
    """Stable normalized identity of a reaction claim.

    Deliberately independent of enzyme, compartment, direction, paper and
    wording: two papers describing the same chemistry with different enzyme
    attributions must land on the SAME identity so their disagreement becomes a
    visible conflict rather than two unrelated reactions.

    Species is part of the identity because the same chemistry in two species is
    two different biological claims, and folding them would be exactly the
    cross-species transfer the guardrail forbids.
    """
    left = stable_join(comparison_key(s) for s in substrates)
    right = stable_join(comparison_key(p) for p in products)
    # Direction-insensitive core so a forward and a reverse report of the same
    # chemistry collide and can be compared; direction itself is an attribute
    # that can conflict.
    core = "|".join(sorted([left, right]))
    sp = comparison_key(species)
    return f"{sp}::{core}" if sp else f"::{core}"


def make_claim_id(identity: str, paper: str, round_index: int) -> str:
    return claim_id(identity, paper, round_index)


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------
@dataclass
class ClaimDecision:
    """One versioned admission decision. Decisions accumulate; none is erased.

    ``supersedes`` points at the previous decision for the same claim identity,
    so the full history ("proposed round 2, incomplete; species evidence arrived
    round 4; promoted round 6") reads as a chain rather than a final state.
    """

    decision_id: str
    claim_id: str
    claim_identity: str
    status: str
    reason: str = ""
    round_index: int = 0
    #: ``primary`` | ``candidate:<component_id>`` | ``none``
    target: str = "none"
    evidence_level: str = ""
    species_state: str = ""
    #: The claim's ``species_support["grade"]`` when it was decided.
    species_grade: str = ""
    supersedes: str = ""
    decided_at: str = ""
    supporting_paper_ids: List[str] = field(default_factory=list)
    supporting_evidence_ids: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def reconsiderable(self) -> bool:
        return self.status in RECONSIDERABLE_STATUSES

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def build(
        cls,
        *,
        claim: ExtractedClaim,
        status: str,
        reason: str,
        round_index: int,
        target: str = "none",
        evidence_level: str = "",
        supersedes: str = "",
        decided_at: str = "",
        detail: Optional[Mapping[str, Any]] = None,
        supporting_paper_ids: Optional[Iterable[str]] = None,
    ) -> "ClaimDecision":
        return cls(
            decision_id=decision_id(claim.claim_id, round_index, status),
            claim_id=claim.claim_id,
            claim_identity=claim.claim_identity,
            status=status,
            reason=reason,
            round_index=int(round_index),
            target=target,
            evidence_level=evidence_level,
            species_state=claim.species_state,
            species_grade=str((claim.species_support or {}).get("grade") or ""),
            supersedes=supersedes,
            decided_at=decided_at,
            supporting_paper_ids=_str_list(supporting_paper_ids or [claim.paper_id]),
            supporting_evidence_ids=list(claim.evidence_ids),
            detail=dict(detail or {}),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClaimDecision":
        payload = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------
@dataclass
class ConflictRecord:
    """Two papers disagreeing about one attribute of one reaction.

    A conflict is retained and displayed, never resolved by picking a winner.
    ``resolved`` flips only when later evidence genuinely settles it, and the
    original claims stay readable either way.
    """

    conflict_id: str
    claim_identity: str
    attribute: str
    positions: List[Dict[str, Any]] = field(default_factory=list)
    claim_ids: List[str] = field(default_factory=list)
    paper_ids: List[str] = field(default_factory=list)
    first_seen_round: int = 0
    last_updated_round: int = 0
    resolved: bool = False
    resolution_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConflictRecord":
        payload = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


# ---------------------------------------------------------------------------
# Frontier and queries
# ---------------------------------------------------------------------------
FRONTIER_DOWNSTREAM = "downstream_expansion"
FRONTIER_UPSTREAM = "upstream_expansion"
FRONTIER_BRANCH = "branch_reaction"
FRONTIER_ENZYME = "enzyme_reactions"
FRONTIER_COFACTOR = "cofactor_investigation"
FRONTIER_PATHWAY_REFERENCE = "pathway_reference"
FRONTIER_BRIDGE = "bridge_search"
FRONTIER_CONFLICT = "conflict_resolution"
FRONTIER_DIRECTION = "missing_direction"
FRONTIER_COMPARTMENT = "missing_compartment"
FRONTIER_SPECIES = "missing_species_evidence"
FRONTIER_SEED = "seed_expansion"

FRONTIER_KINDS = (
    FRONTIER_SEED,
    FRONTIER_DOWNSTREAM,
    FRONTIER_UPSTREAM,
    FRONTIER_BRANCH,
    FRONTIER_ENZYME,
    FRONTIER_COFACTOR,
    FRONTIER_PATHWAY_REFERENCE,
    FRONTIER_BRIDGE,
    FRONTIER_CONFLICT,
    FRONTIER_DIRECTION,
    FRONTIER_COMPARTMENT,
    FRONTIER_SPECIES,
)


@dataclass
class FrontierItem:
    """One thing the next round should go and look for."""

    frontier_item_id: str
    kind: str
    target: str
    reason: str = ""
    origin_round: int = 0
    priority: float = 0.0
    #: For bridge searches: the candidate component being bridged.
    component_id: str = ""
    entity_ids: List[str] = field(default_factory=list)
    claim_identity: str = ""
    exhausted: bool = False
    attempts: int = 0
    #: On-route relevance of a ``missing_species_evidence`` item (0-3), which orders
    #: the frontier slots reserved for that kind. 0 for every other kind. See
    #: :func:`t2pw.mining.frontier.cap_frontier`.
    relevance: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrontierItem":
        payload = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class LiteratureQuery:
    """One traceable literature query and what it returned."""

    query_id: str
    query_text: str
    frontier_item_id: str = ""
    frontier_kind: str = ""
    round_index: int = 0
    sources: List[str] = field(default_factory=list)
    source_terms: List[str] = field(default_factory=list)
    executed_at: str = ""
    result_count: int = 0
    new_paper_ids: List[str] = field(default_factory=list)
    cache_hit: bool = False
    error: str = ""
    #: Which attempt for the frontier item this is: 1 = its own terms; 2 = narrowed by the
    #: topic or, when that is no different, attempt 1's query again over a wider window.
    attempt: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LiteratureQuery":
        payload = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


# ---------------------------------------------------------------------------
# Graph delta
# ---------------------------------------------------------------------------
@dataclass
class GraphDelta:
    """Exactly what one round changed about the research graph."""

    round_index: int = 0
    graph_hash_before: str = ""
    graph_hash_after: str = ""
    added_entity_ids: List[str] = field(default_factory=list)
    added_reaction_ids: List[str] = field(default_factory=list)
    updated_reaction_ids: List[str] = field(default_factory=list)
    promoted_reaction_ids: List[str] = field(default_factory=list)
    added_candidate_reaction_ids: List[str] = field(default_factory=list)
    created_component_ids: List[str] = field(default_factory=list)
    merged_component_ids: List[str] = field(default_factory=list)
    new_conflict_ids: List[str] = field(default_factory=list)

    @property
    def useful(self) -> bool:
        """Did this round add anything worth continuing for?

        A round that only re-read papers it already had is unproductive even
        though it burned tokens; two of those in a row stop the run.
        """
        return bool(
            self.added_reaction_ids
            or self.promoted_reaction_ids
            or self.added_candidate_reaction_ids
            or self.updated_reaction_ids
            or self.new_conflict_ids
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["useful"] = self.useful
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraphDelta":
        payload = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------
@dataclass
class ModelCallRecord:
    """One crossing of the model boundary, priced and explained.

    ``requested_model`` and ``actual_model`` are separate fields on purpose: a
    silent provider substitution is the thing this record exists to make loud.
    ``estimated_cost_usd`` is priced from the configured rates and is labelled
    an estimate everywhere it appears -- it is not a billed amount.
    """

    call_id: str
    round_index: int = 0
    purpose: str = ""
    requested_model: str = ""
    actual_model: str = ""
    provider: str = ""
    prompt_version: str = ""
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    reasoning: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: int = 0
    retries: int = 0
    finish_reason: str = ""
    status: str = "ok"
    parse_error: str = ""
    schema_error: str = ""
    fallback_used: bool = False
    fallback_reason: str = ""
    request_hash: str = ""
    response_hash: str = ""
    response_chars: int = 0
    attempt_log: List[Dict[str, Any]] = field(default_factory=list)
    paper_id: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    started_at: str = ""
    #: ``ok``; ``retried_after_thin`` (a thin answer was re-asked once and the
    #: re-ask was used); ``thin_after_retry`` (the answer kept is thin, or the re-ask
    #: failed and the thin answer was kept); ``thin_not_retried`` (no budget for the
    #: re-ask); ``""`` when no answer was accepted. See
    #: :func:`t2pw.mining.modelclient.thin_answer`.
    answer_quality: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelCallRecord":
        payload = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
@dataclass
class RoundCheckpoint:
    """Enough state to resume the run after this round, and nothing more."""

    round_index: int
    completed: bool = False
    written_at: str = ""
    graph_hash: str = ""
    frontier: List[Dict[str, Any]] = field(default_factory=list)
    processed_paper_ids: List[str] = field(default_factory=list)
    discovered_paper_ids: List[str] = field(default_factory=list)
    seen_query_ids: List[str] = field(default_factory=list)
    unproductive_streak: int = 0
    usage: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoundCheckpoint":
        payload = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


# ---------------------------------------------------------------------------
# Provenance chain verification
# ---------------------------------------------------------------------------
def provenance_chain_complete(
    *,
    reaction: Any,
    claims: Mapping[str, ExtractedClaim],
    evidence: Mapping[str, EvidenceSpan],
    papers: Mapping[str, PaperRecord],
) -> List[str]:
    """Return the broken links in ``reaction -> claim -> span -> paper -> id``.

    Empty list means the chain is intact. This is the machine form of acceptance
    criterion 9 and is asserted on every accepted reaction by the controller
    before a run is allowed to report success.
    """
    problems: List[str] = []
    claim_ids = list(getattr(reaction, "claim_ids", None) or [])
    if not claim_ids:
        problems.append("reaction has no claim")
    for cid in claim_ids:
        claim = claims.get(cid)
        if claim is None:
            problems.append(f"claim {cid} missing")
            continue
        if not claim.evidence_ids:
            problems.append(f"claim {cid} has no evidence span")
        for eid in claim.evidence_ids:
            span = evidence.get(eid)
            if span is None:
                problems.append(f"evidence {eid} missing")
                continue
            if not _clean(span.text):
                problems.append(f"evidence {eid} has no supporting text")
            paper = papers.get(span.paper_id)
            if paper is None:
                problems.append(f"paper {span.paper_id} missing for evidence {eid}")
                continue
            if not paper.has_source_identifier:
                problems.append(
                    f"paper {paper.paper_id} has no resolvable source identifier "
                    "(PMID/PMCID/DOI/URL/file hash)"
                )
    return problems


__all__ = [
    "CLAIM_TYPES",
    "CLAIM_REACTION",
    "CLAIM_ENZYME_ASSOCIATION",
    "CLAIM_SPECIES_SUPPORT",
    "CLAIM_COMPARTMENT",
    "CLAIM_DIRECTION",
    "CLAIM_PATHWAY_MEMBERSHIP",
    "DECISION_STATUSES",
    "STATUS_ACCEPTED",
    "STATUS_CANDIDATE",
    "STATUS_CONFLICTING",
    "STATUS_INCOMPLETE",
    "STATUS_REJECTED",
    "STATUS_HYPOTHESIS_ONLY",
    "RECONSIDERABLE_STATUSES",
    "EXTRACTION_OK",
    "EXTRACTION_EMPTY",
    "EXTRACTION_TRUNCATED",
    "EXTRACTION_SCHEMA_ERROR",
    "EXTRACTION_PARSE_ERROR",
    "EXTRACTION_PARTIAL_REPAIR",
    "NON_ADMISSIBLE_EXTRACTION_STATUSES",
    "EVIDENCE_PRIMARY",
    "EVIDENCE_REVIEW",
    "EVIDENCE_PREPRINT",
    "EVIDENCE_UNKNOWN",
    "DIRECTION_FORWARD",
    "DIRECTION_REVERSE",
    "DIRECTION_UNSPECIFIED",
    "FRONTIER_KINDS",
    "FRONTIER_SEED",
    "FRONTIER_DOWNSTREAM",
    "FRONTIER_UPSTREAM",
    "FRONTIER_BRANCH",
    "FRONTIER_ENZYME",
    "FRONTIER_COFACTOR",
    "FRONTIER_PATHWAY_REFERENCE",
    "FRONTIER_BRIDGE",
    "FRONTIER_CONFLICT",
    "FRONTIER_DIRECTION",
    "FRONTIER_COMPARTMENT",
    "FRONTIER_SPECIES",
    "PaperRecord",
    "EvidenceSpan",
    "ReactionParticipant",
    "ExtractedClaim",
    "ClaimDecision",
    "ConflictRecord",
    "FrontierItem",
    "LiteratureQuery",
    "GraphDelta",
    "ModelCallRecord",
    "RoundCheckpoint",
    "reaction_identity",
    "make_claim_id",
    "provenance_chain_complete",
]
