"""Separated denominators, scientific error counts, and failure taxonomies.

The problem this replaces
========================

``t2pw.batch.report`` reports one rate per mode -- passes over attempted pairs --
and one triage class per paper. Both are honest about what they measure and both
are useless for deciding what to fix, for three reasons:

1. **One denominator for five questions.** A paper that was never a pathway
   paper, a paper that failed extraction, and a paper whose biology is perfect
   but lacks a UniProt accession all land in the same "fail" bucket over the same
   denominator. The resulting rate moves when *retrieval* changes and is read as
   if the *exporter* changed.

2. **No notion of scientific error.** Nothing counts a fabricated accession, a
   cofactor filed as a protein, or a reaction with no source. A run can improve
   its pass rate by hallucinating more confidently.

3. **A research failure is assumed to be a code defect.** The existing doctrine
   is "research mode is fail-open by construction, so a research failure can only
   mean the code itself is wrong" (``report.py``). That was true when research
   mode relaxed everything, and it is false now: the structural guards
   (``entities_required``, ``processes_required``, ...) abort in *both* modes by
   design, and a paper that yields no parseable payload fails research mode for a
   reason that has nothing to do with research mode. Filing those as
   implementation defects sends a reader to debug the export-mode code when the
   actual fault is upstream in extraction.

What this module provides instead
---------------------------------

:class:`Denominator` -- a rate that carries its own population, its exclusions,
and the reason for each exclusion, so no rate can be quoted without the reader
being able to see what it was computed over.

:func:`classify_research_failure` and :func:`classify_strict_boundary` -- the two
failure taxonomies, replacing "contract" with the boundary the run actually died
at.

:func:`rank_blockers` -- remaining blockers ordered by how many papers each one
would release if fixed, which is the only ranking that answers "what next?".
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from t2pw.bench.semantic import ERROR_ORDER


# ---------------------------------------------------------------------------
# Research-mode failure taxonomy (requirement 5).
# ---------------------------------------------------------------------------
#: The run reached the export boundary and was stopped by a rule research mode
#: is supposed to relax. This is a *relaxation coverage* bug: the rule fired,
#: ``export_mode.relax_report`` did not recognise it as relaxable, and the run
#: died at a gate that by policy should only have annotated.
RESEARCH_FAIL_RELAXED_GATE = "relaxed_export_gate_failure"

#: Extraction never produced a payload with the required top-level shape. The
#: structural guards abort in BOTH modes on purpose -- there is nothing to relax
#: when there are no entities or no processes -- so this is not a research-mode
#: fault and must not be counted as one.
RESEARCH_FAIL_STRUCTURAL = "structural_extraction_failure"

#: The run stopped itself because the requested scope was ambiguous or the paper
#: contradicted the request. This is the system working: a deliberate refusal is
#: a correct outcome, not a failure of research mode.
RESEARCH_FAIL_AMBIGUITY = "deliberate_ambiguity_stop"

#: The LLM or the network failed. Infrastructure, not biology and not code.
RESEARCH_FAIL_PROVIDER = "provider_failure"

#: The pair exceeded its wall-clock budget and was killed.
RESEARCH_FAIL_TIMEOUT = "timeout"

#: Nothing above explains it: research mode failed on its own terms. THIS is the
#: bucket the old report put everything in, and it should now be nearly empty.
RESEARCH_FAIL_DEFECT = "research_mode_implementation_defect"

RESEARCH_FAILURE_ORDER: Tuple[str, ...] = (
    RESEARCH_FAIL_RELAXED_GATE,
    RESEARCH_FAIL_STRUCTURAL,
    RESEARCH_FAIL_AMBIGUITY,
    RESEARCH_FAIL_PROVIDER,
    RESEARCH_FAIL_TIMEOUT,
    RESEARCH_FAIL_DEFECT,
)

RESEARCH_FAILURE_LABELS: Dict[str, str] = {
    RESEARCH_FAIL_RELAXED_GATE: "relaxed export-gate failure (relaxation did not cover the rule that fired)",
    RESEARCH_FAIL_STRUCTURAL: "structural extraction failure (no parseable payload; aborts in both modes)",
    RESEARCH_FAIL_AMBIGUITY: "deliberate ambiguity stop (the run refused on purpose)",
    RESEARCH_FAIL_PROVIDER: "provider failure (LLM or network)",
    RESEARCH_FAIL_TIMEOUT: "timeout",
    RESEARCH_FAIL_DEFECT: "TRUE research-mode implementation defect",
}

#: Codes that abort regardless of export mode. Imported lazily from
#: ``pipeline.export_mode`` so the two lists cannot drift; the literal fallback
#: exists only so this module stays importable in isolation.
_STRUCTURAL_GUARD_FALLBACK: Tuple[str, ...] = (
    "invalid_payload",
    "entities_required",
    "processes_required",
    "entity_not_object",
    "process_not_object",
)


def structural_guard_codes() -> frozenset:
    try:
        from t2pw.pipeline.export_mode import STRUCTURAL_GUARD_CODES

        return frozenset(STRUCTURAL_GUARD_CODES)
    except Exception:  # pragma: no cover - defensive
        return frozenset(_STRUCTURAL_GUARD_FALLBACK)


# ---------------------------------------------------------------------------
# Strict-mode boundary taxonomy.
# ---------------------------------------------------------------------------
BOUNDARY_NOT_ATTEMPTED = "not_attempted"
BOUNDARY_EXTRACTION = "stage1_extraction"
BOUNDARY_SCOPE = "scope_ambiguity"
BOUNDARY_STAGE3_GATE = "stage3_normalization_gate"
BOUNDARY_QUARANTINE = "pre_export_quarantine"
BOUNDARY_PWML_REQUIRED_FIELD = "pwml_required_field_gate"
BOUNDARY_PROVIDER = "provider"
BOUNDARY_TIMEOUT = "timeout"
BOUNDARY_CRASH = "crash"
BOUNDARY_NONE = "passed"
BOUNDARY_UNKNOWN = "unknown"

BOUNDARY_ORDER: Tuple[str, ...] = (
    BOUNDARY_EXTRACTION,
    BOUNDARY_SCOPE,
    BOUNDARY_STAGE3_GATE,
    BOUNDARY_QUARANTINE,
    BOUNDARY_PWML_REQUIRED_FIELD,
    BOUNDARY_PROVIDER,
    BOUNDARY_TIMEOUT,
    BOUNDARY_CRASH,
    BOUNDARY_NOT_ATTEMPTED,
    BOUNDARY_UNKNOWN,
    BOUNDARY_NONE,
)

BOUNDARY_LABELS: Dict[str, str] = {
    BOUNDARY_EXTRACTION: "Stage-1 extraction produced nothing usable",
    BOUNDARY_SCOPE: "scope ambiguity / requested-vs-observed conflict",
    BOUNDARY_STAGE3_GATE: "Stage-3 post-normalization hard gate",
    BOUNDARY_QUARANTINE: "pre-export strict quarantine",
    BOUNDARY_PWML_REQUIRED_FIELD: "PWML required-field gate",
    BOUNDARY_PROVIDER: "LLM / network provider",
    BOUNDARY_TIMEOUT: "wall-clock timeout",
    BOUNDARY_CRASH: "unhandled crash",
    BOUNDARY_NOT_ATTEMPTED: "never attempted",
    BOUNDARY_UNKNOWN: "unclassified",
    BOUNDARY_NONE: "no failure",
}


# ---------------------------------------------------------------------------
# Root causes -- what a gate code is really telling you to fix.
# ---------------------------------------------------------------------------
#: Ordered, first match wins. Keyed on a substring of the issue code, because the
#: codes are slugified free text (``gate.protein_coenzyme_a_coa_is_missing_a_...``)
#: and the entity name sits in the middle of them.
#:
#: The needles are deliberately shorter than the phrases they come from.
#: ``driver._slug_code`` truncates a code at 68 characters, and the entity name
#: sits *before* the diagnosis, so a long name eats the end of the sentence:
#: ``gate.protein_hydroxyacyl_acyl_carrier_protein_acp_is_missing_a_unipro``
#: is the same defect as ``..._is_missing_a_uniprot_or_drugbank_identifier`` but
#: shares only the prefix. Matching on the full phrase silently routes every
#: long-named protein into the "other" bucket, which is exactly the ranking
#: error this list exists to prevent.
_ROOT_CAUSE_RULES: Tuple[Tuple[str, str, str], ...] = (
    (
        "is_missing_a_unipro",
        "protein_identity_unresolved",
        "a protein reached the gate with no UniProt/DrugBank identity",
    ),
    (
        "protein_missing_external_identity",
        "protein_identity_unresolved",
        "a protein reached the gate with no UniProt/DrugBank identity",
    ),
    (
        "is_missing_species",
        "protein_organism_unset",
        "a protein reached the gate with no species/organism",
    ),
    (
        "protein_missing_species",
        "protein_organism_unset",
        "a protein reached the gate with no species/organism",
    ),
    (
        "degree_after_normalization",
        "protein_disconnected_after_normalization",
        "a protein survived normalization attached to no reaction",
    ),
    (
        "declared_as_both_a_protein",
        "entity_type_conflict",
        "one name is declared as two different entity types",
    ),
    (
        "entity_type",
        "entity_type_conflict",
        "one name is declared as two different entity types",
    ),
    (
        "unknown_protein_modifier_reference",
        "dangling_modifier_reference",
        "a reaction modifier points at a protein that no bucket declares",
    ),
    (
        "component_protein_unresolved",
        "complex_component_unresolved",
        "a protein-complex component references a protein that does not exist",
    ),
    (
        "reaction_missing_left",
        "reaction_side_empty",
        "a reaction reached the gate with an empty participant side",
    ),
    (
        "reaction_missing_right",
        "reaction_side_empty",
        "a reaction reached the gate with an empty participant side",
    ),
    (
        "ambiguous_review_scope",
        "scope_ambiguity",
        "the requested scope could not be pinned to one pathway",
    ),
    ("processes_required", "extraction_empty", "extraction returned no processes"),
    ("entities_required", "extraction_empty", "extraction returned no entities"),
    ("invalid_payload", "extraction_empty", "extraction returned an unparseable payload"),
    ("no_biological_states", "compartment_unresolved", "no biological state could be built"),
    ("biological_state_missing", "compartment_unresolved", "a biological state is missing species or location"),
    ("species_missing", "species_record_incomplete", "the species record lacks taxonomy or classification"),
)


def root_cause(issue_code: str) -> Tuple[str, str]:
    """Map an issue code to ``(root_cause_id, human explanation)``."""

    code = str(issue_code or "").casefold()
    for needle, cause, explanation in _ROOT_CAUSE_RULES:
        if needle in code:
            return cause, explanation
    return "other", f"unclassified issue code: {issue_code}"


# ---------------------------------------------------------------------------
# Classification.
# ---------------------------------------------------------------------------
def _codes(row: Mapping[str, Any]) -> List[str]:
    raw = row.get("issue_codes")
    if not isinstance(raw, list):
        return []
    return [str(code) for code in raw if code]


def _text(value: Any) -> str:
    return str(value or "").strip().casefold()


def classify_research_failure(row: Mapping[str, Any]) -> Tuple[str, str]:
    """Why a *research-mode* run failed, as ``(category, evidence)``.

    Order is most-specific-first and is load-bearing. Timeout and provider
    faults are checked before anything content-shaped, because a run killed
    mid-flight leaves partial gate reports behind that would otherwise be read as
    a gate failure.
    """

    status = _text(row.get("status"))
    kind = _text(row.get("failure_kind"))
    stage = _text(row.get("stage"))
    codes = _codes(row)

    if status == "timeout" or kind == "timeout":
        return RESEARCH_FAIL_TIMEOUT, f"status={status or 'fail'} kind={kind or 'timeout'}"
    if kind in ("llm", "network"):
        return RESEARCH_FAIL_PROVIDER, f"failure_kind={kind}"
    if kind == "ambiguous_review_scope" or status == "scope_conflict":
        return RESEARCH_FAIL_AMBIGUITY, f"status={status} kind={kind}"

    guards = structural_guard_codes()
    hit = [code for code in codes if code in guards]
    if hit:
        return RESEARCH_FAIL_STRUCTURAL, f"structural guard code(s): {', '.join(sorted(hit))}"
    if kind == "no_reactions":
        return RESEARCH_FAIL_STRUCTURAL, "extraction produced no reactions"
    if stage == "stage1" and not codes:
        return RESEARCH_FAIL_STRUCTURAL, "died at Stage 1 with no issue code"

    gate_codes = [code for code in codes if code.startswith("gate.")]
    if kind == "contract" or gate_codes:
        detail = ", ".join(sorted(gate_codes)[:4]) or "contract errors survived relaxation"
        return RESEARCH_FAIL_RELAXED_GATE, detail

    if kind == "crash":
        return RESEARCH_FAIL_DEFECT, "unhandled crash inside the research path"
    return RESEARCH_FAIL_DEFECT, f"unexplained research failure (kind={kind or 'none'}, stage={stage or 'none'})"


def classify_strict_boundary(row: Optional[Mapping[str, Any]]) -> Tuple[str, str]:
    """The boundary a *strict-mode* run actually died at, as ``(id, evidence)``.

    ``failure_kind`` is deliberately not trusted on its own: it reports
    ``contract`` for the Stage-3 hard gate, for the PWML required-field gate and
    for a structural guard alike, which is exactly the conflation that made the
    old fix-list unrankable.
    """

    if row is None:
        return BOUNDARY_NOT_ATTEMPTED, "no manifest row for this paper+mode"

    status = _text(row.get("status"))
    if status == "pass":
        return BOUNDARY_NONE, ""

    kind = _text(row.get("failure_kind"))
    stage = _text(row.get("stage"))
    codes = _codes(row)
    message = _text(row.get("message"))

    if status == "timeout" or kind == "timeout":
        return BOUNDARY_TIMEOUT, message or "killed at the wall-clock ceiling"
    if kind in ("llm", "network"):
        return BOUNDARY_PROVIDER, f"failure_kind={kind}"
    if kind == "ambiguous_review_scope" or status == "scope_conflict":
        return BOUNDARY_SCOPE, message or "scope ambiguity"

    guards = structural_guard_codes()
    if any(code in guards for code in codes) or kind == "no_reactions" or stage == "stage1":
        return BOUNDARY_EXTRACTION, ", ".join(sorted(set(codes))[:4]) or message or "died at Stage 1"

    if stage == "pwml_export":
        return BOUNDARY_PWML_REQUIRED_FIELD, ", ".join(sorted(set(codes))[:4]) or message
    if "quarantine" in message or any("quarantin" in code for code in codes):
        return BOUNDARY_QUARANTINE, message
    if stage == "post_pipeline" or any(code.startswith("gate.") for code in codes):
        return BOUNDARY_STAGE3_GATE, ", ".join(sorted(c for c in codes if c.startswith("gate."))[:4]) or message
    if kind == "crash":
        return BOUNDARY_CRASH, message
    return BOUNDARY_UNKNOWN, message or f"kind={kind or 'none'} stage={stage or 'none'}"


# ---------------------------------------------------------------------------
# Denominators.
# ---------------------------------------------------------------------------
@dataclass
class Denominator:
    """A rate that carries its own population and exclusions.

    Every rate in the acceptance report is one of these. The ``excluded`` list
    is not decoration: "0 of 0" and "0 of 9" are different findings, and a rate
    printed without its exclusions is the specific thing that made the old
    report unreadable.
    """

    key: str
    question: str
    population: str
    numerator_names: List[str] = field(default_factory=list)
    denominator_names: List[str] = field(default_factory=list)
    excluded: List[Dict[str, str]] = field(default_factory=list)

    @property
    def numerator(self) -> int:
        return len(self.numerator_names)

    @property
    def denominator(self) -> int:
        return len(self.denominator_names)

    @property
    def rate(self) -> Optional[float]:
        """``None`` when the denominator is empty -- never silently 0.0."""

        if not self.denominator_names:
            return None
        return self.numerator / self.denominator

    def render(self) -> str:
        if self.rate is None:
            return f"n/a (0 papers in the population: {self.population})"
        return f"{self.numerator}/{self.denominator} = {self.rate * 100:.0f}%"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "question": self.question,
            "population": self.population,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "rate": self.rate,
            "rendered": self.render(),
            "numerator_papers": sorted(self.numerator_names),
            "denominator_papers": sorted(self.denominator_names),
            "excluded": self.excluded,
        }


#: NOT "eligibility yield". Pinned acquisition bypasses the screener entirely --
#: every paper is admitted by ``OUTCOME_PINNED_OVERRIDE``, and its recorded score
#: is computed against whatever scope the spec carried. So this number measures
#: the composition of the gold set the author chose, not the screener's
#: performance. Calling it a yield would credit or blame ``rag.eligibility`` for a
#: value it had no part in producing.
#:
#: To measure the screener instead, run it in non-vetoing shadow mode over the
#: gold pathway/organism and report precision / recall / specificity against the
#: gold relevance labels. That needs a title+abstract, which a pinned candidate
#: does not carry (``CandidatePaper(id=..., source="pinned")`` has neither), so it
#: must run over a derived lead window -- which is exactly what
#: ``scripts/eligibility_dry_run.py`` already does, marking such decisions
#: ``provisional``. Point at that tool rather than duplicating it here.
DENOM_RELEVANCE = "gold_relevance_prevalence"
DENOM_EXTRACTION = "extraction_success"
DENOM_SEMANTIC = "semantic_pathway_success"
DENOM_STRICT = "strict_pwml_success"

#: Did research mode PRODUCE something? An output rate: the citation report exists
#: and is reviewable. It says nothing about whether the science in it is right,
#: and on its own it is the most over-readable number in the report -- the old
#: combined `research_partial_success` was routinely quoted as if it meant the
#: pathway was correct.
DENOM_RESEARCH_DELIVERABLE = "research_deliverable_produced"

#: Is the science in it right? Requires `semantic.confirmed` -- every check passed
#: AND every check was evaluable -- so a deliverable built on missing admission
#: data, identity/name conflicts, unsupported reactions or orphaned references
#: cannot enter it.
DENOM_RESEARCH_CONFIRMED = "research_semantically_confirmed"

DENOMINATOR_ORDER: Tuple[str, ...] = (
    DENOM_RELEVANCE,
    DENOM_EXTRACTION,
    DENOM_SEMANTIC,
    DENOM_STRICT,
    DENOM_RESEARCH_DELIVERABLE,
    DENOM_RESEARCH_CONFIRMED,
)


# ---------------------------------------------------------------------------
# Scientific error aggregation.
# ---------------------------------------------------------------------------
@dataclass
class ErrorLedger:
    """Scientific errors, kept in separate columns on purpose.

    Summing these would be meaningless: one fabricated accession is a worse
    outcome than forty Unknown-backed proteins, and the acceptance priorities
    order them explicitly rather than weighting them into a score.
    """

    totals: Dict[str, int] = field(default_factory=lambda: {key: 0 for key in ERROR_ORDER})
    by_paper: Dict[str, Dict[str, int]] = field(default_factory=dict)
    papers_affected: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add(self, paper_id: str, mode: str, errors: Mapping[str, int]) -> None:
        key = f"{paper_id}:{mode}"
        row = {name: int(errors.get(name, 0) or 0) for name in ERROR_ORDER}
        self.by_paper[key] = row
        for name, count in row.items():
            self.totals[name] = self.totals.get(name, 0) + count
            if count:
                self.papers_affected[name].add(paper_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "totals": {name: self.totals.get(name, 0) for name in ERROR_ORDER},
            "papers_affected": {
                name: sorted(self.papers_affected.get(name, set())) for name in ERROR_ORDER
            },
            "by_paper_mode": self.by_paper,
        }


# ---------------------------------------------------------------------------
# Blocker ranking.
# ---------------------------------------------------------------------------
@dataclass
class Blocker:
    """One remaining obstacle, with the papers it is holding back."""

    root_cause: str
    explanation: str
    papers: Set[str] = field(default_factory=set)
    codes: Set[str] = field(default_factory=set)
    boundaries: Set[str] = field(default_factory=set)
    #: Papers where this is the ONLY root cause standing between the run and a
    #: pass. Fixing it releases these papers outright; the rest need it *and*
    #: something else, so they are potential rather than expected benefit.
    sole_blocker_for: Set[str] = field(default_factory=set)

    @property
    def expected_benefit(self) -> int:
        return len(self.sole_blocker_for)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_cause": self.root_cause,
            "explanation": self.explanation,
            "papers_blocked": sorted(self.papers),
            "papers_blocked_count": len(self.papers),
            "sole_blocker_for": sorted(self.sole_blocker_for),
            "expected_benefit": self.expected_benefit,
            "issue_codes": sorted(self.codes)[:12],
            "boundaries": sorted(self.boundaries),
        }


#: The three blocker populations, kept apart because "papers released if fixed"
#: is only meaningful within one of them.
BLOCKERS_STRICT = "strict_export"
BLOCKERS_RESEARCH = "research_deliverable"
BLOCKERS_EXTRACTION = "extraction"

BLOCKER_SCOPES: Tuple[str, ...] = (BLOCKERS_STRICT, BLOCKERS_RESEARCH, BLOCKERS_EXTRACTION)

BLOCKER_SCOPE_LABELS: Dict[str, str] = {
    BLOCKERS_STRICT: (
        "strict-export blockers -- strict legs of strict_exportable cases ONLY. "
        "Fixing anything here can raise the strict PWML rate."
    ),
    BLOCKERS_RESEARCH: (
        "research-deliverable blockers -- research legs of relevant cases. "
        "Fixing these cannot change the strict rate."
    ),
    BLOCKERS_EXTRACTION: (
        "extraction blockers -- legs that never produced a payload, in either mode. "
        "Upstream of both rates."
    ),
}


def rank_blockers(
    failures: Sequence[Mapping[str, Any]],
) -> List[Blocker]:
    """Group failures by root cause and rank by papers released if fixed.

    ``failures`` is a sequence of ``{paper_id, boundary, issue_codes}``.

    Ranked by *expected* benefit -- papers for which this is the only root cause
    -- and only then by total papers touched. A cause that appears alongside four
    others on every paper it touches releases nothing on its own, and ranking it
    first (which counting occurrences does) sends the next day's work at a fix
    that cannot change the pass rate.

    **Feed this one population at a time.** Mixing strict and research failures,
    or including ``partial_only`` papers, produces a "papers released" number that
    no rate can actually collect. PMC12312563 is ``partial_only``: fixing the
    Stage-3 gate that stops its strict leg is real work, but it cannot move the
    strict-PWML benchmark rate by even one paper, because the paper is not in that
    denominator. Ranking it as the top strict blocker is precisely the misdirection
    this split exists to prevent. See :data:`BLOCKER_SCOPES`.
    """

    per_paper_causes: Dict[str, Set[str]] = defaultdict(set)
    blockers: Dict[str, Blocker] = {}

    for failure in failures:
        paper = str(failure.get("paper_id") or "")
        boundary = str(failure.get("boundary") or "")
        codes = [str(c) for c in (failure.get("issue_codes") or []) if c]
        if not codes:
            codes = [boundary or "unknown"]
        for code in codes:
            cause, explanation = root_cause(code)
            entry = blockers.get(cause)
            if entry is None:
                entry = Blocker(root_cause=cause, explanation=explanation)
                blockers[cause] = entry
            entry.papers.add(paper)
            entry.codes.add(code)
            if boundary:
                entry.boundaries.add(boundary)
            per_paper_causes[paper].add(cause)

    for paper, causes in per_paper_causes.items():
        if len(causes) == 1:
            only = next(iter(causes))
            blockers[only].sole_blocker_for.add(paper)

    return sorted(
        blockers.values(),
        key=lambda b: (-b.expected_benefit, -len(b.papers), b.root_cause),
    )


def tally(values: Iterable[str]) -> Dict[str, int]:
    return dict(Counter(v for v in values if v))


__all__ = [
    "BLOCKERS_EXTRACTION",
    "BLOCKERS_RESEARCH",
    "BLOCKERS_STRICT",
    "BLOCKER_SCOPES",
    "BLOCKER_SCOPE_LABELS",
    "BOUNDARY_CRASH",
    "BOUNDARY_EXTRACTION",
    "BOUNDARY_LABELS",
    "BOUNDARY_NONE",
    "BOUNDARY_NOT_ATTEMPTED",
    "BOUNDARY_ORDER",
    "BOUNDARY_PROVIDER",
    "BOUNDARY_PWML_REQUIRED_FIELD",
    "BOUNDARY_QUARANTINE",
    "BOUNDARY_SCOPE",
    "BOUNDARY_STAGE3_GATE",
    "BOUNDARY_TIMEOUT",
    "BOUNDARY_UNKNOWN",
    "DENOMINATOR_ORDER",
    "DENOM_RELEVANCE",
    "DENOM_EXTRACTION",
    "DENOM_RESEARCH_CONFIRMED",
    "DENOM_RESEARCH_DELIVERABLE",
    "DENOM_SEMANTIC",
    "DENOM_STRICT",
    "RESEARCH_FAILURE_LABELS",
    "RESEARCH_FAILURE_ORDER",
    "RESEARCH_FAIL_AMBIGUITY",
    "RESEARCH_FAIL_DEFECT",
    "RESEARCH_FAIL_PROVIDER",
    "RESEARCH_FAIL_RELAXED_GATE",
    "RESEARCH_FAIL_STRUCTURAL",
    "RESEARCH_FAIL_TIMEOUT",
    "Blocker",
    "Denominator",
    "ErrorLedger",
    "classify_research_failure",
    "classify_strict_boundary",
    "rank_blockers",
    "root_cause",
    "structural_guard_codes",
    "tally",
]
