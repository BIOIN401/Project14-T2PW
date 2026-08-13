"""Stage R5a — gap-bound admission of retrieved reactions.

A retrieved reaction may enter the pathway **only** when it fills a specific
detected gap and passes pathway, organism and evidence admission. This module is
the gate that decides that, and it is the only place the decision is made.

Why a gate exists at all
------------------------
Before it, :mod:`t2pw.rag.synthesize` transcribed *every* reaction stated by
*every* retrieved passage and merged them all. Because a passage is retrieved per
gap but states whatever the paper says, that is functionally "broad pathway
retrieval, then merge everything": run ``2026-07-28_0919`` leg PMC12444477/strict
shipped 27 reactions of which reactions 10-27 were phospholipid-biosynthesis
steps imported from PMC12898747 — correct biology, from a real paper, filling no
gap of the lipid A pathway that was actually requested. Nothing downstream can
tell those apart from the requested pathway's own chemistry once they are merged.

The four things this gate is careful about
------------------------------------------
1. **Connectivity is chemical.** Only substrates and products are graph nodes.
   Catalysts, reaction names, pathway names and evidence vocabulary never expand
   the frontier. Two unrelated reactions that share an enzyme are not connected —
   which is exactly how thirteen of those phospholipid imports first slipped
   through a draft of this gate, on the strength of a sprayed ``WaaA`` catalyst.
2. **Evidence is local and relational.** A candidate must point at ONE bounded
   span that states its whole claim — every non-currency participant, a relation
   or arrow giving the direction, and every catalyst. Names scattered across two
   sentences of one chunk are not a reaction (see :func:`validate_evidence_span`).
3. **Name matching is boundary aware.** ``LpxA`` does not match ``LpxAB``, ``NdmA``
   does not match ``NdmAB``, ``AMP`` does not match ``cAMP``; but ``lipid IV A``
   still matches ``lipid IV_A`` (see :func:`name_pattern`).
4. **A gap has a type.** A reaction candidate cannot "fill" a missing-compartment
   or unmapped-enzyme gap merely by naming the entity — those need a compartment
   assertion or identifier evidence respectively (see :func:`_gap_type_verdict`).

Deterministic and offline
-------------------------
No network, no LLM, no randomness: the same candidates in any order produce the
same decision — chain hop distances come from a breadth-first pass over the
metabolite graph, not from input order. Heavier siblings (``synthesize`` for the
cofactor / name-canonicalization vocabulary, ``eligibility`` for the taxonomy and
pathway lexicons) are imported lazily inside the functions that need them — the
established pattern in this package — so importing this module stays cheap and
free of any circular dependency (``synthesize`` imports *this* module at module
level).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# Module level, unlike the heavy siblings lazy-imported inside functions: lineage is a
# LEAF (it imports nothing from ``t2pw``), so it adds no cost and cannot cycle --
# ``rag.graph_delta`` already imports it beside ``rag.admission`` the same way.
from t2pw.pipeline.lineage import LineageEntry, LineageSource

__all__ = [
    "RagReactionCandidate",
    "AdmissionPolicy",
    "AdmissionReport",
    "TypedResolution",
    "extract_typed_resolution",
    "parse_identity_assertion",
    "parse_localization_assertion",
    "propose_typed_resolutions",
    "screen_typed_resolution",
    "missing_reaction_side",
    "admit_candidates",
    "admission_lineage_entry",
    "policy_from_config",
    "name_pattern",
    "states_name",
    "parse_span_relation",
    "parse_span_relations",
    "SpanRelation",
    "split_statements",
    "locate_span",
    "validate_evidence_span",
    "compare_organism",
    "compare_requested_pathway",
    "pathways_named_in",
    "PATHWAY_MIXED",
    "ADMISSION_RULES",
    "STATUS_ACCEPTED",
    "STATUS_REJECTED",
    "STATUS_PROPOSED",
    "STATUS_APPLIED",
    "RESOLUTION_IDENTIFIER",
    "RESOLUTION_COMPARTMENT",
    "RESOLUTION_PRECURSOR",
    "ROUTE_IDENTITY_RESOLVER",
    "SCOPE_ADMITTED",
    "SCOPE_REJECTED",
    "HUB_METABOLITES",
]


# ---------------------------------------------------------------------------
# Vocabulary.
# ---------------------------------------------------------------------------
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
#: A typed resolution that was parsed out of evidence but not yet applied to a
#: payload. Proposing is the gate's job; applying is ``t2pw.rag.synthesize``'s.
STATUS_PROPOSED = "proposed"
STATUS_APPLIED = "applied"
STATUS_REJECTED_PROPOSAL = "rejected"

#: ``scope_membership`` written onto an ADMITTED candidate. ``core`` is the
#: Stage-1 taxonomy value (``pwml_system.txt``: core | anaplerotic |
#: cataplerotic | auxiliary | out_of_scope) and is the honest label here: a
#: candidate only reaches ``accepted`` by filling a detected gap of the requested
#: pathway and connecting into it, which is what "core step of this pathway"
#: means. A rejected candidate is labelled ``out_of_scope`` in the report; it
#: never reaches a payload, so the label is documentation, not a filter input.
SCOPE_ADMITTED = "core"
SCOPE_REJECTED = "out_of_scope"

# Organism-compatibility verdicts (mirrors t2pw.rag.eligibility's vocabulary so
# the two agree on what "genus_level" means).
ORGANISM_MATCH = "match"
ORGANISM_GENUS_LEVEL = "genus_level"
ORGANISM_MISMATCH = "mismatch"
ORGANISM_UNKNOWN = "unknown"

PATHWAY_MATCH = "match"
PATHWAY_MISMATCH = "mismatch"
PATHWAY_UNKNOWN = "unknown"
#: The span names the requested pathway AND another one. Recorded explicitly
#: rather than rounded up to ``match``: mixed local evidence is a real state, and
#: silently calling it a match is how another pathway's chemistry gets in.
PATHWAY_MIXED = "mixed"

# --- rejection reason codes (stable; the report is diffed across runs) -------
REASON_NO_GAP_ID = "no_gap_id"
REASON_UNKNOWN_GAP_ID = "unknown_gap_id"
REASON_INVALID_ENTITY_TYPE = "invalid_entity_type"
REASON_ORGANISM_MISMATCH = "organism_mismatch"
REASON_ORGANISM_UNKNOWN_REFUSED = "organism_unknown_refused"
REASON_PATHWAY_MISMATCH = "requested_pathway_mismatch"
REASON_NO_EVIDENCE_SPAN = "no_local_evidence_span"
REASON_SPAN_NOT_LOCAL = "evidence_span_is_not_one_statement"
REASON_EVIDENCE_UNSUPPORTED = "evidence_does_not_support_participants"
REASON_NO_RELATION = "evidence_states_no_reaction_relation"
REASON_ROLES_UNASSIGNABLE = "evidence_relation_roles_unassignable"
REASON_RELATION_DISAGREES = "evidence_relation_disagrees_with_claim"
REASON_DIRECTION_DISAGREES = "evidence_direction_disagrees_with_claim"
REASON_UNSUPPORTED_CATALYST = "unsupported_catalyst_injection"
REASON_TYPE_CANNOT_FILL = "candidate_type_cannot_fill_gap"
REASON_NOT_CONNECTED = "not_connected_to_pathway_or_anchor"
REASON_ADJACENT_CONTEXT = "adjacent_pathway_context_only"
REASON_DOES_NOT_FILL_GAP = "does_not_fill_named_gap"
REASON_CHAIN_TOO_FAR = "chain_distance_exceeds_limit"
REASON_HUB_TARGET = "hub_target_needs_additional_evidence"
#: A typed resolution aimed at the wrong SHAPE of row — a UniProt accession for a
#: protein complex, a location for an entity element_locations cannot address.
#: Not a scope question and not a policy question: the schema has no
#: representation for it, so applying it would write a reference nothing resolves.
REASON_UNSUPPORTED_TARGET_TYPE = "unsupported_target_entity_type"
#: Evidence naming several participants for the missing side, of which only some
#: can be added. A repair is all-or-nothing: taking the first of "glycine and
#: succinyl-CoA" writes a reaction no paper states.
REASON_MULTI_PARTICIPANT_REPAIR = "unsupported_multi_participant_repair"
#: The reaction is no longer missing the side the proposal was made about — both
#: sides populated, both empty, or the other side is the empty one now.
REASON_SIDE_NO_LONGER_MISSING = "precursor_side_no_longer_missing"
#: Two admissible proposals for the SAME gap, kind and target state incompatible
#: values — two accessions for one protein, two participant sets for one missing
#: side. Applying either would be a deterministic arbitrary write: the winner is
#: whichever sorts first, which is not evidence. Neither is applied and both
#: values are reported with their sources.
REASON_CONFLICTING_RESOLUTION = "conflicting_typed_resolution"

# --- acceptance reason codes -------------------------------------------------
REASON_FILLS_GAP_DIRECT = "fills_named_gap_directly"
REASON_FILLS_GAP_CHAINED = "fills_named_gap_via_admitted_chain"

#: Where an unmapped-enzyme gap goes when no candidate carries identifier
#: evidence for it. A reaction that merely mentions the protein is not an
#: identifier resolution, so the gap stays open and is handed to the identity
#: resolver rather than being silently marked filled.
ROUTE_IDENTITY_RESOLVER = "identity_resolver"

#: The rules, in evaluation order. Documented as data so the gate's contract can
#: be printed in a report and asserted in a test rather than re-read out of the
#: control flow below.
ADMISSION_RULES: Tuple[Tuple[str, str], ...] = (
    ("named_gap", "the candidate names a gap_id that was actually detected"),
    (
        "valid_entity_types",
        "every participant and catalyst is a plausible species / protein token",
    ),
    (
        "local_evidence_span",
        "the candidate points at ONE bounded span of its chunk, which is a single "
        "statement (one sentence, or one parsed equation line)",
    ),
    (
        "evidence_supports_participants",
        "the span's own parsed substrates and products are the claim's, matched "
        "on whole-name boundaries",
    ),
    (
        "relation_agrees",
        "the span STATES this reaction — its parsed substrates, products and "
        "direction agree with the claim; co-occurrence of names is not a relation "
        "and a reversed span refutes a forward claim",
    ),
    (
        "no_unsupported_catalyst",
        "every catalyst claimed is the catalyst the span attaches to THIS "
        "relation, not merely a protein named nearby",
    ),
    (
        "organism_compatible",
        "the organism observed in the span (else the paper's observed metadata) "
        "matches the requested one, is a related taxon, or is explicitly unknown "
        "and the configured policy allows unknown",
    ),
    (
        "requested_pathway",
        "the span / observed metadata does not name a DIFFERENT pathway (strict "
        "policy additionally requires a positive pathway match)",
    ),
    (
        "gap_type_compatible",
        "the candidate is the KIND of evidence the gap asks for: a reaction can "
        "fill a connectivity gap, but a missing_compartment gap needs a location "
        "assertion and an unmapped_enzyme gap needs identifier evidence",
    ),
    (
        "connects_to_pathway",
        "a SUBSTRATE OR PRODUCT of the candidate is an existing pathway "
        "metabolite or the gap's expected anchor — catalysts never connect",
    ),
    (
        "fills_its_named_gap",
        "a substrate or product of the candidate is the gap's target, directly "
        "or within the configured hop limit through metabolites contributed by "
        "reactions already admitted for that same gap",
    ),
)


# ---------------------------------------------------------------------------
# Bounds. The admission report is a diagnostic, not a payload, and the RAG
# subsystem has already shipped one 4.70 MB payload of which 4.6 MB was repeated
# evidence text (run 2026-07-28_0919). Every text-bearing field here is capped.
# ---------------------------------------------------------------------------
#: Characters of the cited span quoted per candidate in the report.
EVIDENCE_SNIPPET_CHARS = 320
#: Entries kept per bucket (accepted / rejected). Overflow is counted, not kept.
DEFAULT_MAX_REPORT_ENTRIES = 200
#: Longest evidence span accepted. A span is meant to be ONE statement; anything
#: longer is a paragraph, and a paragraph lets two unrelated reactions' names
#: co-occur.
DEFAULT_MAX_SPAN_CHARS = 600
#: Maximum hop distance from the gap target. 0 = the candidate touches the target
#: itself. Conservative by design: each hop is a reaction admitted on the
#: strength of the previous one, so error compounds with distance.
DEFAULT_MAX_CHAIN_HOPS = 2
#: Connected/participant name lists quoted per candidate row.
_MAX_NAMES_PER_ROW = 12


# ---------------------------------------------------------------------------
# Currency / hub metabolites.
# ---------------------------------------------------------------------------
#: Metabolites that connect everything to everything and therefore must not
#: extend a chain. The ubiquitous cofactors (``synthesize.COFACTOR_NAMES``) are
#: unioned in at call time; these are the additional *activated carriers* and
#: central hubs whose appearance in two reactions says nothing about whether
#: those reactions belong to the same pathway. Without this set, one admitted
#: reaction that releases CoA would open the frontier onto every acyl-CoA
#: reaction in the corpus.
#:
#: Used for **chain expansion only**. A gap's own declared target is never
#: filtered by this set: if the pathway's open end genuinely is ``UMP`` (it is,
#: in the real PMC12444477 payload), that gap must still be fillable.
HUB_METABOLITES = frozenset(
    {
        "coa",
        "coa-sh",
        "coenzyme a",
        "acetyl-coa",
        "acetyl coa",
        "malonyl-coa",
        "acyl-coa",
        "succinyl-coa",
        "acp",
        "holo-acp",
        "acyl-acp",
        "s-adenosyl-l-methionine",
        "s-adenosylmethionine",
        "sam",
        "s-adenosyl-l-homocysteine",
        "sah",
        "glutathione",
        "thiamine pyrophosphate",
        "pyridoxal phosphate",
        "pyridoxal 5'-phosphate",
        "tetrahydrofolate",
        "phosphate",
        "orthophosphate",
        "pyrophosphate",
        "diphosphate",
        "glutamate",
        "l-glutamate",
        "glutamine",
        "l-glutamine",
        "2-oxoglutarate",
        "alpha-ketoglutarate",
        "α-ketoglutarate",
        "ammonia",
        "ammonium",
        "electron",
        "quinone",
        "ubiquinone",
        "ubiquinol",
    }
)


# ---------------------------------------------------------------------------
# Lazy bridges to the sibling vocabularies (kept lazy: see module docstring).
# ---------------------------------------------------------------------------
def _cofactors() -> frozenset:
    try:
        from t2pw.rag.synthesize import COFACTOR_NAMES

        return frozenset(COFACTOR_NAMES)
    except Exception:  # pragma: no cover - defensive; synthesize is a sibling
        return frozenset()


def _hubs() -> frozenset:
    """Currency set for CHAIN EXPANSION: cofactors plus the activated carriers."""
    return _cofactors() | HUB_METABOLITES


def _canonical(name: Any) -> str:
    try:
        from t2pw.rag.synthesize import canonical_name

        return canonical_name(name)
    except Exception:  # pragma: no cover - defensive
        return str(name or "").strip()


def _is_invalid_token(name: str) -> bool:
    try:
        from t2pw.rag.synthesize import _is_invalid_species_token

        return bool(_is_invalid_species_token(name))
    except Exception:  # pragma: no cover - defensive
        return not str(name or "").strip()


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _fold(value: Any) -> str:
    return _text(value).casefold()


def _dedupe(names: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for name in names:
        token = _text(name)
        key = token.casefold()
        if token and key not in seen:
            seen.add(key)
            out.append(token)
    return out


# ---------------------------------------------------------------------------
# Boundary-aware name matching.
# ---------------------------------------------------------------------------
_ALNUM_RUN_RE = re.compile(r"[a-z0-9]+")

#: How much punctuation/whitespace may sit between two alphanumeric runs of one
#: name. ``lipid IV_A`` / ``lipid IV A`` / ``lipid-IV-A`` are the same name;
#: three separator characters is enough for every real spelling and short enough
#: that two adjacent names cannot be glued into one match.
_MAX_SEPARATOR_CHARS = 3

_PATTERN_CACHE: Dict[str, Optional[re.Pattern]] = {}


def name_pattern(name: Any) -> Optional[re.Pattern]:
    """Compile a whole-name matcher for ``name``, or ``None`` if it has no content.

    Substring containment is the wrong test for biochemical names and produced
    real false positives: ``LpxA`` matched ``LpxAB``, ``NdmA`` matched ``NdmAB``,
    and ``AMP`` matched any longer identifier that happened to contain those
    three letters. The compiled pattern anchors both ends against alphanumerics
    (``(?<![a-z0-9]) ... (?![a-z0-9])``), so a name only matches when it is the
    whole token.

    Punctuation inside a name is treated as flexible, which is what keeps
    ``lipid IV A`` matching a passage that writes ``lipid IV_A``: the name is
    split into alphanumeric runs and rejoined with a bounded separator class.
    (Graph-level identity of such spellings is the offline synonym resolver's
    job — see :func:`_make_token`; this is only about finding the name in text.)
    """
    key = _fold(name)
    if key in _PATTERN_CACHE:
        return _PATTERN_CACHE[key]
    runs = _ALNUM_RUN_RE.findall(key)
    if not runs:
        _PATTERN_CACHE[key] = None
        return None
    body = (r"[^a-z0-9]{0," + str(_MAX_SEPARATOR_CHARS) + r"}").join(
        re.escape(run) for run in runs
    )
    compiled = re.compile(r"(?<![a-z0-9])" + body + r"(?![a-z0-9])")
    _PATTERN_CACHE[key] = compiled
    return compiled


def states_name(span: str, name: Any) -> bool:
    """True when ``span`` names ``name`` as a whole token.

    Both the written form and its canonical form are tried, because a transcriber
    canonicalizes synonyms through ``BIOCHEMICAL_ALIAS_MAP`` while the passage
    naturally uses whichever synonym its authors preferred.
    """
    haystack = _fold(span)
    if not haystack:
        return False
    for variant in (_text(name), _canonical(name)):
        pattern = name_pattern(variant)
        if pattern is not None and pattern.search(haystack):
            return True
    return False


# ---------------------------------------------------------------------------
# Statement segmentation + relation detection.
# ---------------------------------------------------------------------------
#: Tokens that state a direction between two sides of a reaction.
_ARROW_RE = re.compile(r"(<=>|<->|⇌|=>|-->|->|→|⇒)")

#: Mechanistic relation CUES, as whole words with their real morphology.
#:
#: The previous version of this was a prefix-matched alternation applied with a
#: single leading ``\b``, so ``form`` fired inside "formaldehyde" and ``transfer``
#: inside "transferase" — turning "A and B were measured with formaldehyde" into
#: a reaction assertion. Every entry now carries its own suffix alternation and a
#: trailing boundary, so only the verb forms match.
#:
#: A cue is NOT sufficient to admit anything. It only separates "this span states
#: no relation at all" from "this span states a relation whose roles could not be
#: assigned" — two different failures that need two different fixes. What admits
#: a claim is :func:`parse_span_relation` agreeing with it.
_RELATION_CUE_RE = re.compile(
    r"\b("
    r"catal(?:yse|yze|yses|yzes|ysed|yzed|ysing|yzing|ysis)|"
    r"convert(?:s|ed|ing)?|conversion|"
    r"produc(?:e|es|ed|ing|tion)|yield(?:s|ed|ing)?|"
    r"form(?:s|ed|ing|ation)|generat(?:e|es|ed|ing|ion)|"
    r"transfer(?:s|red|ring)?|"
    r"synthesi[sz](?:e|es|ed|ing)|synthesis|"
    r"(?:de)?methylat(?:e|es|ed|ing|ion)|"
    r"(?:de)?phosphorylat(?:e|es|ed|ing|ion)|"
    r"(?:de)?acylat(?:e|es|ed|ing|ion)|(?:de)?acetylat(?:e|es|ed|ing|ion)|"
    r"oxidi[sz](?:e|es|ed|ing)|oxidation|reduc(?:e|es|ed|ing)|reduction|"
    r"hydroly[sz](?:e|es|ed|ing)|hydrolysis|(?:de)?hydrat(?:e|es|ed|ing|ion)|"
    r"(?:de)?carboxylat(?:e|es|ed|ing|ion)|(?:de)?aminat(?:e|es|ed|ing|ion)|"
    r"isomeri[sz](?:e|es|ed|ing)|epimeri[sz](?:e|es|ed|ing)|"
    r"cleav(?:e|es|ed|ing|age)|condens(?:e|es|ed|ing|ation)|"
    r"ligat(?:e|es|ed|ing|ion)|adenylat(?:e|es|ed|ing|ion)|"
    r"glycosylat(?:e|es|ed|ing|ion)"
    r")\b",
    re.IGNORECASE,
)

# --- prose relation templates ---------------------------------------------
# Deterministic, role-assigning patterns. Each one has to say WHICH phrase is the
# substrate side, WHICH is the product side, and (optionally) which token is the
# catalyst — because that is the thing being validated. A sentence that merely
# contains a relation cue and the right names is refused: co-occurrence is not a
# relationship, and "Enz1 inhibits X, while A is converted to B" must not
# attribute the conversion to Enz1.
_ENTITY = r"[^.;]+?"
#: A catalyst phrase. Six tokens rather than four, because real extracted text
#: spaces its parentheses — ``acetyl-CoA synthetase ( acs )`` is five tokens as
#: written and three once :func:`_clean_actor` tightens it. The phrase is always
#: anchored by a following literal, and :func:`_clean_actor` re-imposes the
#: four-word cap after tightening, so widening the regex does not widen what is
#: accepted as a protein name.
_ACTOR = r"[A-Za-z][\w\-()/]*(?:\s+[\w\-()/]+){0,5}?"
_TAIL = r"(?=[.;]|$)"
_MAKE = r"(?:yield|form|give|produce|generate)\s+"
#: The head noun an appositive uses to introduce a catalyst: "acs, an enzyme
#: that converts acetate into acetyl-CoA".
_ACTOR_NOUN = (
    r"(?:enzyme|protein|catalyst|gene|subunit|synthase|synthetase|kinase|"
    r"transferase|reductase|oxidase|oxygenase|dehydrogenase|hydrolase|"
    r"isomerase|ligase|lyase|deacetylase|decarboxylase|carboxylase|"
    r"phosphatase|methyltransferase|acyltransferase)s?"
)

# The ORDER of these templates is load-bearing. A construction that explicitly
# binds the reaction to its actor must be tried before the bare shape nested
# inside it, or the wider match never happens and the sentence is read as
# stating no catalyst. "MenF catalyzes a reversible reaction CONVERTING
# chorismate to isochorismate" contains "converting X to Y"; matching that
# fragment first throws MenF away and then refuses the honest claim that names
# MenF. The generic actor-free forms therefore come last.
_PROSE_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    (
        # "NdmB catalyzes the N3-demethylation of theobromine, producing X and Y"
        "catalyzes_producing",
        re.compile(
            rf"(?P<enz>{_ACTOR})\s+catal(?:yz|ys)\w*\s+(?:the\s+)?"
            rf"(?:[\w\-]+\s+of\s+)?(?P<lhs>{_ENTITY})\s*,?\s+"
            rf"(?:producing|yielding|forming|generating|giving)\s+"
            rf"(?P<rhs>{_ENTITY}){_TAIL}",
            re.IGNORECASE,
        ),
    ),
    (
        # "The enzyme MenF catalyzes a reversible reaction converting chorismate
        # (A) to isochorismate (B)" — the actor-bearing wrapper around
        # ``converting_to``. Must precede both ``converting_to`` (which would
        # drop MenF) and ``catalyzes_to`` (whose lazy substrate group would
        # swallow "a reversible reaction converting chorismate (A)" whole).
        "catalyzes_converting",
        re.compile(
            rf"(?P<enz>{_ACTOR})\s+catal(?:yz|ys)\w*\s+[^.;]*?\bconverting\s+"
            rf"(?P<lhs>{_ENTITY})\s+(?:in)?to\s+(?P<rhs>{_ENTITY}){_TAIL}",
            re.IGNORECASE,
        ),
    ),
    (
        # "overexpressed acetyl-CoA synthetase ( acs ), an enzyme that converts
        # acetate into acetyl-CoA" — an appositive gloss binds the conversion to
        # the noun phrase it renames, so the actor IS deterministically bound.
        "appositive_converts",
        re.compile(
            rf"(?P<enz>{_ACTOR})\s*,\s*(?:an?|the)\s+(?:[\w\-]+\s+)?{_ACTOR_NOUN}\s+"
            rf"(?:that|which)\s+converts?\s+(?P<lhs>{_ENTITY})\s+(?:in)?to\s+"
            rf"(?P<rhs>{_ENTITY}){_TAIL}",
            re.IGNORECASE,
        ),
    ),
    (
        # "The acs gene, responsible for converting acetate to acetyl-CoA" /
        # "FadD, a protein responsible for converting oleate to oleoyl-CoA".
        "responsible_for_converting",
        re.compile(
            rf"(?P<enz>{_ACTOR})\s*,\s*(?:(?:an?|the)\s+(?:[\w\-]+\s+)?{_ACTOR_NOUN}\s+)?"
            rf"responsible\s+for\s+converting\s+(?P<lhs>{_ENTITY})\s+(?:in)?to\s+"
            rf"(?P<rhs>{_ENTITY}){_TAIL}",
            re.IGNORECASE,
        ),
    ),
    (
        # "The enzyme Idi , which converts IPP to farnesyl pyrophosphate (FPP)".
        # A non-restrictive relative clause has an explicit antecedent, so the
        # actor is bound. The comma is required: a span that OPENS on "which is
        # phosphorylated by LpxK ..." has no antecedent inside itself and must
        # keep being refused rather than reading "which" as a protein.
        "relative_converts",
        re.compile(
            rf"(?P<enz>{_ACTOR})\s*,\s*which\s+converts?\s+(?P<lhs>{_ENTITY})\s+"
            rf"(?:in)?to\s+(?P<rhs>{_ENTITY}){_TAIL}",
            re.IGNORECASE,
        ),
    ),
    (
        # "E catalyzes the conversion of A to B" / "Enz1 catalyzes A to B"
        "catalyzes_to",
        re.compile(
            rf"(?P<enz>{_ACTOR})\s+catal(?:yz|ys)\w*\s+(?:the\s+)?"
            rf"(?:[\w\-]+\s+of\s+)?(?P<lhs>{_ENTITY})\s+(?:in)?to\s+"
            rf"(?:{_MAKE})?(?P<rhs>{_ENTITY}){_TAIL}",
            re.IGNORECASE,
        ),
    ),
    (
        # "catalyzes the conversion of A to B" with no subject in this clause.
        # Tried AFTER the subject-bearing forms, or it would strip the subject off
        # "SeedEnz catalyzes the conversion of A to B" and report no catalyst for a
        # sentence that names one — which then refuses the honest claim.
        "catalyzes_to_subjectless",
        re.compile(
            rf"catal(?:yz|ys)\w*\s+the\s+(?:conversion|[\w\-]+ation)\s+of\s+"
            rf"(?P<lhs>{_ENTITY})\s+(?:in)?to\s+(?:{_MAKE})?(?P<rhs>{_ENTITY}){_TAIL}",
            re.IGNORECASE,
        ),
    ),
    (
        # "E converts A to B"
        "converts_to",
        re.compile(
            rf"(?P<enz>{_ACTOR})\s+converts?\s+(?P<lhs>{_ENTITY})\s+(?:in)?to\s+"
            rf"(?P<rhs>{_ENTITY}){_TAIL}",
            re.IGNORECASE,
        ),
    ),
    (
        # "A is converted to B (by E)"
        "is_converted_to",
        re.compile(
            rf"(?P<lhs>{_ENTITY})\s+(?:is|are|was|were)\s+converted\s+(?:in)?to\s+"
            rf"(?P<rhs>{_ENTITY})(?:\s+by\s+(?P<enz>{_ACTOR}))?{_TAIL}",
            re.IGNORECASE,
        ),
    ),
    (
        # "B is produced/formed from A (by E)"
        "is_produced_from",
        re.compile(
            rf"(?P<rhs>{_ENTITY})\s+(?:is|are|was|were)\s+"
            rf"(?:produced|formed|generated|synthesi[sz]ed|derived|made)\s+from\s+"
            rf"(?P<lhs>{_ENTITY})(?:\s+by\s+(?P<enz>{_ACTOR}))?{_TAIL}",
            re.IGNORECASE,
        ),
    ),
    (
        # "E phosphorylates/deacetylates A to (form) B". The verb must carry a
        # verbal suffix: an earlier ``[a-z]+as\w*`` branch matched the NOUN
        # "phospholipase", so "the phospholipase PldA that degrades GPLs ... to
        # produce lysophospholipids" read PldA itself as the substrate.
        "enzyme_verb_to",
        re.compile(
            rf"(?P<enz>{_ACTOR})\s+[a-z]{{3,}}"
            rf"(?:ates|ated|ylates|ylated|olyses|olyzes|olysed|olyzed|ifies|ified)\s+"
            rf"(?P<lhs>{_ENTITY})\s+(?:in)?to\s+(?:{_MAKE})?(?P<rhs>{_ENTITY}){_TAIL}",
            re.IGNORECASE,
        ),
    ),
    (
        # "... by rapidly converting isochorismate to 2,3-diDHB" — the actor-free
        # residue. LAST, so it only ever runs on sentences where no wider,
        # actor-binding construction applied.
        "converting_to",
        re.compile(
            rf"\bconverting\s+(?P<lhs>{_ENTITY})\s+(?:in)?to\s+"
            rf"(?P<rhs>{_ENTITY}){_TAIL}",
            re.IGNORECASE,
        ),
    ),
)

#: Bond-forming verbs that take TWO OR MORE substrates and state their product
#: explicitly. Deliberately a closed lexicon of condensation verbs rather than a
#: morphological rule: the ``-ates`` family is already covered by
#: ``enzyme_verb_to``, and its suffix restriction exists because an open rule read
#: the NOUN "phospholipase" as a verb. Naming the verbs keeps that impossible.
#:
#: ``transfer`` is excluded on purpose. "transfers a methyl group to DMK" names a
#: DESTINATION, not a product, and reading it as one would invent chemistry.
_SYNTHESIS_VERB = (
    r"(?:join|condense|combine|couple|link|attach|ligate|fuse|conjugate|add)"
    r"(?:s|es|ed)?"
)

#: Templates added by C-061, kept OUT of :data:`_PROSE_PATTERNS` so that constant
#: stays byte-identical, and tried LAST for the same reason the actor-free forms
#: are last: a template only ever runs on a construction no earlier one claimed.
#:
#: One entry. "Subsequently, MenA joins DHNA and prenyl diphosphate to produce
#: demethylmenaquinone (DMK)" is a paper-explicit, two-substrate reaction with its
#: catalyst bound by the sentence's own subject, and NO existing template reads it
#: at any position — measured, not assumed. The claim quoting it verbatim was
#: therefore refused as disagreeing with its own evidence.
#:
#: The explicit ``to <produce|yield|form|give|generate>`` is load-bearing: it is
#: what makes the phrase after it a PRODUCT rather than a destination, a cofactor
#: or a location, so the template asserts a role the sentence actually assigns.
_EXTRA_PROSE_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    (
        # "MenA joins DHNA and prenyl diphosphate to produce DMK"
        "actor_combines_to_make",
        re.compile(
            rf"(?P<enz>{_ACTOR})\s+{_SYNTHESIS_VERB}\s+(?P<lhs>{_ENTITY})\s+to\s+"
            rf"{_MAKE}(?P<rhs>{_ENTITY}){_TAIL}",
            re.IGNORECASE,
        ),
    ),
)

#: What the parser actually walks. Priority order is preserved exactly: every
#: historical template is tried, in its historical order, before any new one.
_ALL_PROSE_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    _PROSE_PATTERNS + _EXTRA_PROSE_PATTERNS
)

#: Clause connectors. Text before one of these on the substrate side, or after one
#: on the product side, belongs to a different clause and is dropped — this is what
#: keeps "Enz1 inhibits X, while A is converted to B" from reading ``Enz1 inhibits
#: X, while A`` as the substrate.
_CLAUSE_BREAKS = (
    " while ",
    " whereas ",
    " although ",
    " though ",
    " but ",
    " when ",
    " where ",
    " and then ",
    " because ",
    " since ",
    " if ",
)
#: Relative pronouns are NOT symmetric with the connectors above. "X, while A is
#: converted to B" puts the substrate AFTER the break; "the intermediate, which
#: is phosphorylated by LpxK" puts it BEFORE — the clause modifies what precedes
#: it. Treating the two alike read ``is phosphorylated by LpxK`` as a substrate.
_RELATIVE_BREAKS = (" which ", " that ", " who ")
#: A relative clause between the verb and its "to" means the product belongs to a
#: DIFFERENT predicate. "LpxB then catalyzes the formation of a tetra-acylated
#: disaccharide intermediate, which is phosphorylated by LpxK to produce lipid IV
#: A" says LpxB makes the intermediate and LpxK phosphorylates it; a template
#: that reaches across the boundary attributes LpxK's reaction to LpxB. Such a
#: match is abandoned rather than trimmed into something plausible.
_RELATIVE_CLAUSE_RE = re.compile(r"\b(?:which|who|whom|whose|that)\b", re.IGNORECASE)
#: Locative/instrumental tails: "B in the periplasm" is still just B.
_PHRASE_TAILS = (
    " in ",
    " at ",
    " within ",
    " during ",
    " under ",
    " via ",
    " using ",
    " through ",
    " upon ",
    " across ",
    " between ",
    " according ",
    " as ",
)
#: A figure/table callout is a CITATION, not part of the species name it follows.
#: "MenG demethylates DMK to generate MK ( Fig. 1A )" names the product MK; the
#: product-side phrase runs to the statement terminator, and the terminator here
#: is the period inside "Fig.", so the phrase arrived as ``MK ( Fig`` and was
#: short enough (four tokens) to pass :func:`_species_segment` as a name. The
#: candidate then disagreed with its own evidence over a compound the span never
#: mentioned. Trimmed here, in the one place both sides already normalize, so the
#: callout is dropped identically wherever it appears.
#:
#: Only the callout is removed. A parenthetical ALIAS — "demethylmenaquinone
#: (DMK)" — is the paper naming one compound twice and is left intact, which is
#: why this matches the citation keywords rather than "any parenthesis".
_FIGURE_CALLOUT_RE = re.compile(
    r"[\(\[]?\s*\b(?:extended\s+data\s+)?"
    r"(?:fig|figs|figure|figures|table|tables|scheme|schemes|panel|panels)\b\.?",
    re.IGNORECASE,
)
_LEADING_ARTICLES = ("the ", "a ", "an ", "its ", "their ")
#: A comma only separates two species when it is followed by whitespace. A comma
#: WITHIN a token is a locant — ``2,3-oxidosqualene`` and ``2,3-diDHB`` are one
#: compound each, and splitting them produced the substrate ``2``.
_SPECIES_SPLIT_RE = re.compile(r"\s\+\s|\s+and\s+|,(?=\s|$)")

#: Words that cannot begin a species name. A phrase starting with one of these is
#: a fragment of the surrounding sentence, not a metabolite — measured on the real
#: cached papers, where a loose splitter produced "by LpxK" and "depends on the
#: organism" as substrates.
_NON_SPECIES_LEAD = frozenset(
    {
        "by", "of", "with", "from", "in", "on", "at", "to", "for", "as", "into",
        "using", "via", "through", "and", "or", "but", "which", "that", "this",
        "these", "those", "it", "they", "there", "such", "subsequent", "then",
        "finally", "resulting", "depends", "allow", "release", "breakdown",
        "sequentially", "also", "both", "each", "its", "their", "his", "her",
    }
)
#: A prose species name is short. Twelve words (the payload-wide junk cap) is far
#: too generous for a name lifted out of a sentence.
_MAX_PROSE_SPECIES_WORDS = 6

#: A candidate species carrying a finite relation verb is a CLAUSE, not a name.
#: "HepPPS catalyzes the conversion of FPP to heptaprenyl pyrophosphate (HepPP),
#: MenA converts HepPP to DMK-7, and UbiE methylates DMK-7 to yield MK-7" lists
#: three reactions in one statement, and a comma splitter happily offered
#: ``MenA converts HepPP to DMK-7`` as a product of the first one.
#:
#: Deliberately restricted to third-person present forms and ``to <make>``:
#: participles occur inside real metabolite names (``reduced glutathione``,
#: ``oxidized NAD``), finite verbs do not.
_SPECIES_CLAUSE_RE = re.compile(
    r"\b(?:catal(?:yz|ys)\w+|converts|produces|yields|generates|forms|transfers|"
    r"[a-z]{3,}(?:at|ylat)es)\b|\bto\s+(?:yield|form|give|produce|generate)\b",
    re.IGNORECASE,
)

#: Nominalized reaction words that introduce a substrate rather than being one.
#: "the conversion of isochorismate" names isochorismate, not "conversion of
#: isochorismate" — measured on the real MenD sentences of PMC12421875.
_NOMINALIZATION_RE = re.compile(
    r"^\s*(?:the\s+|a\s+|an\s+)?(?:reversible\s+|irreversible\s+)?"
    r"(?:conversion|formation|synthesis|transfer|addition|removal|conjugation|"
    r"acylation|deacylation|acetylation|deacetylation|methylation|demethylation|"
    r"phosphorylation|dephosphorylation|oxidation|reduction|hydrolysis|"
    r"hydration|dehydration|carboxylation|decarboxylation|isomerisation|"
    r"isomerization|condensation|cleavage|ligation|prenylation|glycosylation)"
    r"\s+of\s+",
    re.IGNORECASE,
)

#: The span itself calling the reaction reversible. Parsed rather than assumed:
#: "LpxA catalyzes the REVERSIBLE acylation of UDP-GlcNAc" is a bidirectional
#: claim, and dropping that on the floor ships one-way chemistry the paper
#: explicitly called two-way.
_REVERSIBLE_WORD_RE = re.compile(
    r"\b(?:reversibl[ey]|bidirectional(?:ly)?|in both directions)\b", re.IGNORECASE
)

#: Actor phrases that name no protein: pronouns, relatives and bare descriptors.
#: The sentence "..., which converts IPP to FPP" states a conversion whose
#: catalyst is an ANAPHOR — reading "which" as the enzyme name is worse than
#: reading no enzyme at all, because a candidate could then be admitted with a
#: catalyst nobody named.
_ANAPHORIC_ACTORS = frozenset(
    {
        "which", "that", "it", "they", "them", "this", "these", "those", "such",
        "the enzyme", "the enzymes", "these enzymes", "those enzymes",
        "the protein", "the proteins", "both enzymes", "each enzyme",
    }
)
#: Function words that cannot END an actor phrase. A phrase ending in one is a
#: clause fragment, not a protein name ("LpxC is a", "MK biosynthesis starts
#: from", "catalyzes the").
_ACTOR_TAIL_STOPWORDS = frozenset(
    {
        "is", "are", "was", "were", "be", "been", "a", "an", "the", "of", "and",
        "or", "with", "from", "by", "to", "in", "on", "at", "for", "as", "that",
        "which", "starts", "depends", "resulting",
        "catalyzes", "catalyses", "produces", "converts", "forms",
    }
)
#: Extracted paper text spaces its parentheses, because the extractor put the
#: italic run ``acs`` on its own: ``acetyl-CoA synthetase ( acs )``. Tightening
#: is pure whitespace repair — it changes no token — and without it a three-token
#: name counts as five and is refused as a clause fragment.
_PAREN_OPEN_RE = re.compile(r"\(\s+")
_PAREN_CLOSE_RE = re.compile(r"\s+\)")
#: A parenthetical alias inside a catalyst phrase. "acetyl-CoA synthetase (acs)"
#: is the paper naming one enzyme twice; a claim may legitimately use either.
_PAREN_ALIAS_RE = re.compile(r"\(([^()]{1,40})\)")

#: Modifiers that describe an enzyme's provenance rather than its identity. They
#: lead the noun phrase in real methods prose ("overexpressed acetyl-CoA
#: synthetase") and are STRIPPED, because refusing the whole actor over them
#: throws away a correct attribution.
_ACTOR_LEAD_MODIFIERS = (
    "overexpressed",
    "expressed",
    "recombinant",
    "purified",
    "native",
    "endogenous",
    "heterologous",
    "encoded",
    "putative",
)

#: Adverbs that trail a subject without being part of its name. "NdmA also
#: converts X to Y" names NdmA; refusing the whole actor because of "also" throws
#: away a correct attribution, so these are STRIPPED rather than rejected.
_ACTOR_TAIL_ADVERBS = (
    "also",
    "then",
    "further",
    "subsequently",
    "additionally",
    "thus",
    "therefore",
    "however",
    "next",
    "finally",
    "sequentially",
    "in turn",
    "specifically",
)

#: A terminator followed by whitespace ends a statement, unless it closes a known
#: abbreviation or a single-letter initial (``E. coli`` must not split).
_ABBREVIATIONS = frozenset(
    {
        "fig",
        "figs",
        "eq",
        "eqs",
        "ref",
        "refs",
        "e.g",
        "eg",
        "i.e",
        "ie",
        "cf",
        "vs",
        "approx",
        "ca",
        "no",
        "al",
        "et",
        "sp",
        "spp",
        "min",
        "max",
        "etc",
        "dr",
        "st",
    }
)
_SENTENCE_BREAK_RE = re.compile(r"([.;!?])(\s+)")


def split_statements(text: str) -> List[str]:
    """Split ``text`` into single statements (approximately, sentences).

    This is the mechanism behind "names appearing in different sentences must not
    be combinable into a new reaction": a candidate's evidence span has to be ONE
    of these. Newlines split too, because the arrow-equation shape puts one
    reaction per line.

    Two guards keep biological prose intact: a terminator that closes a known
    abbreviation (``Fig.``, ``e.g.``, ``et al.``) or a single-letter initial
    (``E. coli``, ``S. aureus``) does not end a statement. Getting those wrong
    would split a species name in half and make every organism observation
    invisible.
    """
    body = str(text or "")
    if not body.strip():
        return []
    statements: List[str] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        start = 0
        for match in _SENTENCE_BREAK_RE.finditer(line):
            head = line[start : match.end(1)]
            tail_word = _ALNUM_RUN_RE.findall(_fold(head[:-1]))
            last = tail_word[-1] if tail_word else ""
            following = line[match.end() : match.end() + 1]
            # "Fig. 3", "et al." — a known abbreviation never ends a statement.
            if last in _ABBREVIATIONS:
                continue
            # "E. coli" — a single letter followed by a LOWERCASE word is a genus
            # initial, not a full stop. The lowercase test is what makes this
            # narrow enough to still split "... converts A to B. Enz2 ...", where
            # the single letter really is the end of a sentence; an initial-only
            # rule swallowed that boundary and let the two sentences be read as
            # one statement, which is exactly the combination this gate refuses.
            if len(last) <= 1 and following.islower():
                continue
            if head.strip():
                statements.append(head.strip())
            start = match.end()
        remainder = line[start:].strip()
        if remainder:
            statements.append(remainder)
    return statements


def locate_span(chunk_text: str, quote: Any, names: Sequence[str]) -> str:
    """Resolve a candidate's supporting span inside its own chunk, or ``""``.

    Two paths, both deterministic and both anchored in the chunk's real text —
    nothing here can invent a span:

    * a ``quote`` the extractor supplied is accepted only when it appears
      **verbatim** in the chunk (whitespace-normalized) and is a single statement.
      A paraphrase, a summary, or a quote stitched from two sentences fails and
      falls through;
    * otherwise the chunk's statements are scanned for the single one that names
      every entry of ``names``. This is the fallback for a model that omitted the
      quote — and it is also what makes the cross-sentence attack impossible:
      "Enz1 catalyzes A to B. Enz2 catalyzes X to Y." has no single statement
      naming ``A``, ``Y`` and ``Enz1``, so ``A -> Y / Enz1`` gets no span at all.

    Returns the located span, or ``""`` when the claim is not stated in one place.
    """
    body = str(chunk_text or "")
    wanted = [n for n in names if _text(n)]
    statements = split_statements(body)

    claimed = _text(quote)
    if claimed:
        normalized_body = " ".join(body.split()).casefold()
        normalized_quote = " ".join(claimed.split()).casefold()
        if normalized_quote and normalized_quote in normalized_body:
            if len(split_statements(claimed)) <= 1 and all(
                states_name(claimed, name) for name in wanted
            ):
                return claimed

    for statement in statements:
        if all(states_name(statement, name) for name in wanted):
            return statement
    return ""


def _trim_phrase(text: str, *, side: str) -> str:
    """Reduce a matched phrase to the species list it actually names."""
    body = " " + " ".join(str(text or "").split()) + " "
    lowered = body.casefold()
    for connector in _CLAUSE_BREAKS:
        idx = lowered.rfind(connector) if side == "lhs" else lowered.find(connector)
        if idx == -1:
            continue
        if side == "lhs":
            body = body[idx + len(connector) :]
        else:
            body = body[:idx]
        lowered = body.casefold()
    for connector in _RELATIVE_BREAKS:
        idx = lowered.find(connector)
        if idx > 0:
            body = body[:idx]
            lowered = body.casefold()
    for tail in _PHRASE_TAILS:
        idx = lowered.find(tail)
        if idx > 0:
            body = body[:idx]
            lowered = body.casefold()
    callout = _FIGURE_CALLOUT_RE.search(body)
    if callout is not None and callout.start() > 0:
        body = body[: callout.start()]
        lowered = body.casefold()
    body = _NOMINALIZATION_RE.sub("", body.strip())
    return body.strip()


def _split_species(phrase: str, *, stop_at_break: bool = False) -> List[str]:
    """Split a trimmed phrase into individual species names.

    Rejects fragments rather than accepting them. Measured on the real cached
    papers, a permissive splitter turned "which is phosphorylated by LpxK to
    produce lipid IVA" into the substrate "by LpxK" — a token that then had to be
    refused downstream by the agreement check, when it should never have been read
    as a species at all. Refusing here turns a wrong parse into no parse, which is
    the honest outcome and the one the recall measurement counts.

    ``stop_at_break`` ends the list at its first non-species segment instead of
    skipping over it, and is used on the PRODUCT side, where the phrase runs to
    the end of the statement. "converting acetate to acetyl-CoA, was overexpressed
    by replacing its native promoter with the strong, constitutive J23119
    promoter" is one product followed by a new clause — skipping the middle
    segment and keeping the last one shipped ``constitutive J23119 promoter`` as a
    metabolite. The substrate side does not use it: its junk is LEADING ("In this
    assay, isochorismate ..."), and stopping there would lose the substrate.
    """
    out: List[str] = []
    for raw in _SPECIES_SPLIT_RE.split(_trim_phrase(phrase, side="rhs") or ""):
        token = _species_segment(raw)
        if token is None:
            if stop_at_break and out:
                break
            continue
        out.append(token)
    return out


def _species_segment(raw: Any) -> Optional[str]:
    """One comma/``and``-delimited segment as a species name, or ``None``."""
    token = " ".join(str(raw or "").split()).strip()
    low = token.casefold()
    for article in _LEADING_ARTICLES:
        if low.startswith(article):
            token = token[len(article) :].strip()
            low = token.casefold()
    if not token or _is_invalid_token(token):
        return None
    words = low.split()
    if not words or words[0] in _NON_SPECIES_LEAD:
        return None
    if len(words) > _MAX_PROSE_SPECIES_WORDS:
        return None
    if _SPECIES_CLAUSE_RE.search(token):
        return None
    return token


def _clean_actor(text: Any) -> List[str]:
    """The catalyst a phrase names, or ``[]`` when it names none.

    ``[]`` is a real answer, not a failure: "..., which converts IPP to FPP"
    states a conversion and no catalyst. Returning ``which`` as the enzyme would
    let a candidate be admitted with a catalyst the sentence never named, so an
    anaphor, a bare descriptor, or a clause fragment all yield no catalyst and any
    claimed one is then refused as unsupported.
    """
    token = " ".join(str(text or "").split()).strip()
    token = _PAREN_CLOSE_RE.sub(")", _PAREN_OPEN_RE.sub("(", token)).strip()
    low = token.casefold()
    if low in _ANAPHORIC_ACTORS:
        return []
    for prefix in ("the ", "an ", "a ", "enzyme ", "protein "):
        if low.startswith(prefix):
            token = token[len(prefix) :].strip()
            low = token.casefold()
    for modifier in _ACTOR_LEAD_MODIFIERS:
        if low.startswith(modifier + " "):
            token = token[len(modifier) + 1 :].strip()
            low = token.casefold()
    changed = True
    while changed:
        changed = False
        for adverb in _ACTOR_TAIL_ADVERBS:
            if low.endswith(" " + adverb):
                token = token[: -(len(adverb) + 1)].strip()
                low = token.casefold()
                changed = True
    if not token or _is_invalid_token(token):
        return []
    words = low.split()
    if not words or words[-1] in _ACTOR_TAIL_STOPWORDS or words[0] in _NON_SPECIES_LEAD:
        return []
    if low in _ANAPHORIC_ACTORS or len(words) > 4:
        return []
    return [token]


def actor_spellings(phrase: Any) -> List[str]:
    """Every spelling of a catalyst phrase a claim may legitimately use.

    A span attaches the catalyst role to a PHRASE, and a transcriber names the
    protein inside it: the span says "EnzX (UniProt P12345) converts A to B" or
    "The enzyme NdmB catalyzes ...", the claim says ``EnzX`` / ``NdmB``. Only
    prefixes and suffixes of the phrase count — a name lifted from its middle is
    not what the sentence called the catalyst — and the phrase itself is at most
    six tokens, so the set stays small and checkable.

    One extra family: a parenthetical alias. "acetyl-CoA synthetase (acs)" is one
    enzyme named twice by the paper itself, so both the long form and the
    bracketed short form are legitimate spellings, and so is the long form with
    the bracket removed.
    """
    body = " ".join(str(phrase or "").split())
    body = _PAREN_CLOSE_RE.sub(")", _PAREN_OPEN_RE.sub("(", body)).strip()
    if not body:
        return []
    variants = [body]
    for alias in _PAREN_ALIAS_RE.findall(body):
        alias = " ".join(alias.split()).strip()
        if alias and alias not in variants:
            variants.append(alias)
    without = " ".join(_PAREN_ALIAS_RE.sub(" ", body).split()).strip()
    if without and without not in variants:
        variants.append(without)

    out: List[str] = []
    for variant in variants:
        tokens = [t for t in variant.split(" ") if t]
        for size in range(len(tokens), 0, -1):
            for window in ({" ".join(tokens[:size]), " ".join(tokens[-size:])}):
                if window and window not in out and not _is_invalid_token(window):
                    out.append(window)
    return out


@dataclass
class SpanRelation:
    """The relationship a span actually asserts, with roles assigned."""

    inputs: List[str]
    outputs: List[str]
    catalysts: List[str]
    reversible: bool = False
    pattern: str = ""


#: Every start offset a template may be tried from. A template is anchored either
#: on an actor (``[A-Za-z]``), on a verb keyword, or on ``_ENTITY`` — and an
#: ``_ENTITY`` match that begins on whitespace or punctuation trims to the same
#: phrase as the one beginning at the following word. Enumerating word starts
#: therefore yields the same RELATION set as enumerating every character, at a
#: fraction of the backtracking.
_WORD_START_RE = re.compile(r"\b\w")

#: The hyphens a chemical name is spelled with. ``-`` plus the Unicode dash block,
#: because extracted paper text carries both.
_COMPOUND_HYPHENS = "-‐‑‒–—―"

#: A word start that is BOUND INTO the token before it, and therefore is not the
#: start of a species name at all. ``\b\w`` fires after every hyphen and after the
#: locant comma of ``2,3-oxidosqualene``, so restarting there offers the templates
#: ``oxoglutarate`` (from ``2-oxoglutarate``) and ``3-oxidosqualene`` as substrates
#: — re-opening by the restart route the exact defect :data:`_SPECIES_SPLIT_RE`'s
#: comment records, where splitting a locant produced the substrate ``2``.
#:
#: The comma case is decided by :data:`_SPECIES_SPLIT_RE`'s own rule and nothing
#: else: that pattern separates species on ``,(?=\s|$)``, so a comma followed by a
#: word character is a locant, never a connector. A word start is therefore
#: excluded exactly when a hyphen or a locant comma sits between it and a word
#: character.
_TOKEN_INTERNAL_RESTART_RE = re.compile(rf"(?<=\w[{_COMPOUND_HYPHENS},])\w")

#: A span states at most this many distinct relations. Truncation can only make
#: the parser see FEWER readings, so it can only ever refuse more — it is not a
#: path to a permissive default (clause A8).
_MAX_SPAN_RELATIONS = 8


def _restart_positions(body: str) -> List[int]:
    """Every offset a template may be re-tried from, token-internal ones removed.

    A template match that BEGINS inside a hyphenated or locant compound can only
    ever report a truncated form of that compound, so such a start is not a
    reading of the span — it is a spelling error the parser would be inventing.
    Excluding it cannot cost a reading: ``pattern.search(body, pos)`` scans
    FORWARD, so every match reachable from an excluded offset ``p`` is still
    reachable from the nearest kept offset before ``p``, except a match that
    starts within the compound itself — which is precisely what is being refused.
    """
    skip = {m.start() for m in _TOKEN_INTERNAL_RESTART_RE.finditer(body)}
    return [0] + [m.start() for m in _WORD_START_RE.finditer(body) if m.start() not in skip]


def _pattern_matches(
    pattern: "re.Pattern", body: str, *, first_only: bool
) -> Iterator["re.Match"]:
    """Every distinct match of ``pattern`` in ``body``, leftmost start first.

    ``re.finditer`` is deliberately NOT used: it resumes at the END of each match,
    and these templates run to the statement terminator, so it reports exactly one
    match per pattern — which is the whole defect. Restarting the search one word
    later enumerates every start position instead, so a sentence stating two
    reactions yields both.

    Restarts that fall INSIDE a compound name are excluded
    (:func:`_restart_positions`). ``first_only=True`` restarts nowhere at all and
    reproduces ``pattern.search(body)`` exactly, which is what the single-relation
    shim needs — position ``0`` is always kept, so the shim is bit-for-bit
    unaffected by the exclusion.
    """
    seen: Set[Tuple[int, int]] = set()
    for pos in _restart_positions(body):
        match = pattern.search(body, pos)
        if match is None:
            # Starts ascend, so nothing later can match either.
            return
        if match.span() in seen:
            continue
        seen.add(match.span())
        yield match
        if first_only:
            return


def _relation_from_match(name: str, match: "re.Match", body: str) -> Optional[SpanRelation]:
    """One template match as a :class:`SpanRelation`, or ``None`` if unreadable.

    Byte-for-byte the acceptance logic the single-match loop used to inline: the
    same relative-clause abandonment, the same trimming, the same refusal to
    invent a species. Extracted so that every enumerated match is judged by
    exactly the same rules as the first one used to be.
    """
    groups = match.groupdict()
    if _RELATIVE_CLAUSE_RE.search(groups.get("lhs") or ""):
        # The "to" this template used sits on the far side of a relative
        # clause, so it belongs to a different predicate. Abandon the match.
        return None
    inputs = _split_species(_trim_phrase(groups.get("lhs") or "", side="lhs"))
    outputs = _split_species(groups.get("rhs") or "", stop_at_break=True)
    if not inputs or not outputs:
        return None
    return SpanRelation(
        inputs=inputs,
        outputs=outputs,
        catalysts=_clean_actor(groups.get("enz")),
        reversible=bool(_REVERSIBLE_WORD_RE.search(body)),
        pattern=name,
    )


def _participant_set(names: Sequence[str]) -> Set[str]:
    """A participant list as canonical spellings, currency INCLUDED.

    Deliberately not :func:`_non_currency`: this set answers "which participants
    does this reading say the span states", and a reading that states ATP states
    ATP. Projecting currency away here would make ``ATP -> G6P`` and
    ``glucose -> G6P`` look like the same reading of the same sentence.
    """
    return {_fold(_canonical(n)) for n in names if _text(n)}


def _drop_participant_subsets(relations: List[SpanRelation]) -> List[SpanRelation]:
    """Remove readings that are another reading with a stated participant deleted.

    ``_pattern_matches`` restarts every template at every word start, and two
    historical templates — ``is_converted_to`` and ``is_produced_from`` — open on
    an unanchored participant group. A restart past a coordinator therefore
    re-matches THE SAME reaction with one of its stated participants missing:
    "Isochorismate and 2-oxoglutarate are converted to SEPHCHC by MenD" also
    yields ``['2-oxoglutarate'] -> ['SEPHCHC']``. That is not a second reading of
    the sentence; it is the first reading with chemistry deleted, and admitting a
    claim against it contradicts the invariant
    :func:`validate_evidence_span` states two functions above — *"it is never
    exempt in the 'span has it, candidate dropped it' direction, because dropping
    a real substrate changes the chemistry"*.

    A reading is dropped only when another returned reading is strictly richer in
    exactly one participant role and no poorer anywhere else:

    * one side is a **strict** subset of the other reading's same side, and
    * the opposite side is **equal**, and
    * its catalysts are a subset of the other's — so a reading naming a catalyst
      the richer one does not name is never deleted, and
    * reversibility agrees.

    Direction is deliberately absent from the rule: a reversal is not a subset of
    anything, and :func:`_relation_verdict` already refuses it.

    **Only ever removes.** No branch constructs, mutates, reorders or copies a
    :class:`SpanRelation`; survivors are the input objects in the input order.

    **Cannot empty a non-empty list.** Both surviving branches require
    ``len(other.inputs') + len(other.outputs') > len(this.inputs') + len(this.outputs')``
    on the canonical sets, so a reading of maximal participant count has no
    possible eliminator and is always kept. Were that reasoning ever wrong the
    caller still fails CLOSED — :func:`validate_evidence_span` refuses an empty
    relation list; there is no permissive default anywhere on this path (A8).
    """
    if len(relations) < 2:
        return relations
    shapes = [
        (
            _participant_set(r.inputs),
            _participant_set(r.outputs),
            _participant_set(r.catalysts),
            bool(r.reversible),
        )
        for r in relations
    ]
    kept: List[SpanRelation] = []
    for index, relation in enumerate(relations):
        ins, outs, cats, rev = shapes[index]
        superseded = False
        for other, (o_ins, o_outs, o_cats, o_rev) in enumerate(shapes):
            if other == index or rev != o_rev or not cats <= o_cats:
                continue
            if (ins < o_ins and outs == o_outs) or (outs < o_outs and ins == o_ins):
                superseded = True
                break
        if not superseded:
            kept.append(relation)
    return kept


def parse_span_relations(span: str) -> List[SpanRelation]:
    """EVERY reaction a span states — substrates, products, direction, catalyst.

    Two paths, both deterministic:

    * **arrow** — the span is re-parsed with the very parser that produced the
      arrow-path candidates (``synthesize._parse_reaction_line``), so the span's
      own reading of itself is what the candidate is checked against. An arrow
      span short-circuits, exactly as it always has: its own parser already read
      the whole line, and letting the prose templates add readings on top of it
      would widen admission for a shape that never had the defect;
    * **prose** — the ordered templates in :data:`_ALL_PROSE_PATTERNS`, each of
      which assigns roles explicitly, tried at **every** start position. Nothing
      is inferred from co-occurrence: a sentence that names the right compounds
      and a relation cue but matches no template yields ``[]``, and the caller
      refuses the claim rather than guessing which name was the substrate.

    Why every position. A single sentence routinely states a whole segment of a
    pathway — "MenA joins DHNA and prenyl diphosphate to produce DMK, and MenG
    demethylates DMK to generate MK" states two reactions. Reading only the first
    match meant the second clause's reading was the ONLY thing the first clause's
    claim could be checked against, so a claim quoting the sentence verbatim was
    refused as disagreeing with its own evidence.

    Returned in template-priority order, then by position, duplicates collapsed,
    and then filtered by :func:`_drop_participant_subsets` — a restart past a
    coordinator re-matches the SAME reaction with a stated participant deleted,
    and a reading with chemistry removed is not a reading of the span. The FIRST
    surviving element is whatever the single-match parser used to return (the
    leftmost match of the highest-priority template is maximal on both sides, so
    it is never the one filtered), which is what keeps
    :func:`parse_span_relation` unchanged.

    This widens what a span can be read as SAYING. It does not touch what counts
    as agreement: each returned relation is judged by the unchanged predicate in
    :func:`validate_evidence_span`, and a claim is admitted only if some single
    relation agrees with it on substrates, products, direction and catalyst at
    once.
    """
    body = _text(span)
    if not body:
        return []

    if _ARROW_RE.search(body):
        try:
            from t2pw.rag.synthesize import _parse_reaction_line

            parsed = _parse_reaction_line(body)
        except Exception:  # pragma: no cover - defensive; synthesize is a sibling
            parsed = None
        if parsed:
            return [
                SpanRelation(
                    inputs=[p.name for p in parsed["inputs"]],
                    outputs=[p.name for p in parsed["outputs"]],
                    catalysts=list(parsed["enzymes"]),
                    reversible=bool(parsed["reversible"]),
                    pattern="arrow",
                )
            ]

    out: List[SpanRelation] = []
    seen: Set[Tuple[Any, ...]] = set()
    for name, pattern in _ALL_PROSE_PATTERNS:
        for match in _pattern_matches(pattern, body, first_only=False):
            relation = _relation_from_match(name, match, body)
            if relation is None:
                continue
            key = (
                tuple(relation.inputs),
                tuple(relation.outputs),
                tuple(relation.catalysts),
                relation.reversible,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(relation)
            if len(out) >= _MAX_SPAN_RELATIONS:
                break
        if len(out) >= _MAX_SPAN_RELATIONS:
            break
    return _drop_participant_subsets(out)


def parse_span_relation(span: str) -> Optional[SpanRelation]:
    """The single relation the span-relation parser used to return, unchanged.

    Kept for every existing caller (``_precursor_resolution`` here, and the
    relation-recall corpus) whose contract is "one relation or ``None``". It walks
    the templates in priority order and takes the first template whose LEFTMOST
    match is readable — the historical algorithm, reproduced by
    ``first_only=True`` rather than re-implemented.

    Prefer :func:`parse_span_relations` in new code: a span that states two
    reactions cannot be represented by this signature at all, and picking one of
    them silently is what made a verbatim claim disagree with its own evidence.
    """
    body = _text(span)
    if not body:
        return None

    if _ARROW_RE.search(body):
        try:
            from t2pw.rag.synthesize import _parse_reaction_line

            parsed = _parse_reaction_line(body)
        except Exception:  # pragma: no cover - defensive; synthesize is a sibling
            parsed = None
        if parsed:
            return SpanRelation(
                inputs=[p.name for p in parsed["inputs"]],
                outputs=[p.name for p in parsed["outputs"]],
                catalysts=list(parsed["enzymes"]),
                reversible=bool(parsed["reversible"]),
                pattern="arrow",
            )

    for name, pattern in _ALL_PROSE_PATTERNS:
        for match in _pattern_matches(pattern, body, first_only=True):
            relation = _relation_from_match(name, match, body)
            if relation is None:
                # The leftmost match of this template is unreadable. The historical
                # loop moved to the NEXT TEMPLATE here rather than to the next
                # position, and that is preserved: this shim exists to be
                # identical, not to be better.
                continue
            return relation
    return None


@dataclass
class SpanVerdict:
    """Why an evidence span did or did not support a claim."""

    ok: bool
    reasons: List[str] = field(default_factory=list)
    relation: Optional[SpanRelation] = None
    #: When the span states reversibility and the claim did not, the claim is
    #: NORMALIZED to the span rather than silently kept as one-way. ``None`` means
    #: no normalization was needed. The caller writes it onto the reaction, so the
    #: payload says what the evidence said.
    normalized_reversible: Optional[bool] = None


def _non_currency(names: Sequence[str]) -> Set[str]:
    cofactors = _cofactors()
    return {
        _fold(_canonical(n))
        for n in names
        if _text(n) and _fold(_canonical(n)) not in cofactors
    }


def validate_evidence_span(
    span: str,
    *,
    inputs: Sequence[str],
    outputs: Sequence[str],
    enzymes: Sequence[str],
    reversible: bool = False,
    max_span_chars: int = DEFAULT_MAX_SPAN_CHARS,
) -> SpanVerdict:
    """Check that ONE span *states this reaction*, roles and direction included.

    A span containing all the right names is not enough, and was the hole this
    replaced: "B -> A" contains exactly the names of "A -> B", and
    "Enz1 inhibits X, while A is converted to B" contains exactly the names of
    "A -> B catalysed by Enz1". Both are now refused, because the span is parsed
    into a :class:`SpanRelation` and the candidate has to AGREE with it on:

    * **substrates and products** — set-equal on non-currency names. Currency is
      exempt in the "candidate has extra" direction only, for the same reason it
      is exempt from the provenance requirement in :mod:`t2pw.rag.provenance`: a
      transcriber legitimately writes the implicit H2O / ATP of a step described
      in words. It is never exempt in the "span has it, candidate dropped it"
      direction, because dropping a real substrate changes the chemistry.
    * **direction** — a candidate whose sides are the span's sides *swapped* is
      refused unless the span itself states reversibility.
    * **catalysts** — every claimed catalyst must be the catalyst the span
      attaches to THIS relation, not merely a protein named nearby.

    A span whose roles cannot be assigned at all is refused and recorded
    (:data:`REASON_ROLES_UNASSIGNABLE`) rather than being guessed at.

    A span may state MORE THAN ONE reaction — "MenA joins DHNA and prenyl
    diphosphate to produce DMK, and MenG demethylates DMK to generate MK" states
    two, and both are paper-explicit. Each stated relation is judged separately by
    the criteria above and the claim is admitted if ANY ONE of them agrees with it
    outright. What is widened is what the span can be read as SAYING; what counts
    as agreement is untouched, so a claim still cannot be assembled from parts of
    two different relations: taking MenA's chemistry with MenG's name is refused
    by the relation that has the chemistry, and taking either reaction backwards
    is still refused for direction.
    """
    body = _text(span)
    reasons: List[str] = []
    if not body:
        return SpanVerdict(False, [f"{REASON_NO_EVIDENCE_SPAN}: no span was located"])
    if max_span_chars and len(body) > max_span_chars:
        reasons.append(
            f"{REASON_SPAN_NOT_LOCAL}: span is {len(body)} chars "
            f"(limit {max_span_chars}) — that is a paragraph, not a statement"
        )
    if len(split_statements(body)) > 1:
        reasons.append(
            f"{REASON_SPAN_NOT_LOCAL}: span covers more than one statement, so its "
            "names come from different sentences"
        )
    if reasons:
        return SpanVerdict(False, reasons)

    relations = parse_span_relations(body)
    if not relations:
        if _RELATION_CUE_RE.search(body):
            return SpanVerdict(
                False,
                [
                    f"{REASON_ROLES_UNASSIGNABLE}: the span states a relation but no "
                    "implemented pattern could assign substrate/product/catalyst "
                    "roles — recorded for review rather than inferred"
                ],
            )
        return SpanVerdict(
            False,
            [
                f"{REASON_NO_RELATION}: the span names the participants but asserts "
                "no reaction between them"
            ],
        )

    claim_in = _non_currency(inputs)
    claim_out = _non_currency(outputs)

    # Every relation the span states is a candidate support. The claim is admitted
    # when ONE of them agrees with it outright; it is refused when none does. The
    # agreement test itself is unchanged and is applied per relation, so a span
    # stating two reactions can support either of them and still supports neither
    # a third reaction nor a crossover between its own two.
    best: Optional[Tuple[Tuple[int, int], List[str], SpanRelation, Optional[bool]]] = None
    for relation in relations:
        rel_reasons, normalized = _relation_verdict(
            relation,
            claim_in=claim_in,
            claim_out=claim_out,
            enzymes=enzymes,
            reversible=reversible,
        )
        if not rel_reasons:
            return SpanVerdict(True, [], relation, normalized)
        rank = _refusal_rank(rel_reasons)
        if best is None or rank < best[0]:
            best = (rank, rel_reasons, relation, normalized)

    # Every stated relation refused the claim. Report the nearest miss — for a span
    # stating one relation that is the same relation, the same reasons and the same
    # wording as before, which is what keeps existing refusals byte-identical.
    assert best is not None  # relations is non-empty and each refusal has a reason
    return SpanVerdict(False, best[1], best[2], best[3])


def _refusal_rank(reasons: Sequence[str]) -> Tuple[int, int]:
    """How near a miss a refusal is — lower is nearer, ties broken by span order.

    Fewest reasons first, then chemistry before catalysis. A span stating two
    reactions produces one refusal per relation, and the reported one should be
    the reading the claim came closest to: for a catalyst crossover ("the span
    does say A + B -> C, but MenG is not the enzyme it attaches to that one") the
    useful report is the unsupported-catalyst refusal against the relation whose
    chemistry matched, not a substrate mismatch against the other clause.
    """
    chemistry = any(
        reason.startswith((REASON_RELATION_DISAGREES, REASON_DIRECTION_DISAGREES))
        for reason in reasons
    )
    return (len(reasons), 1 if chemistry else 0)


def _relation_verdict(
    relation: SpanRelation,
    *,
    claim_in: Set[str],
    claim_out: Set[str],
    enzymes: Sequence[str],
    reversible: bool,
) -> Tuple[List[str], Optional[bool]]:
    """Judge the claim against ONE stated relation. The predicate, unchanged.

    Extracted verbatim from the single-relation body so that widening the parser
    could not quietly widen the test: substrates and products still have to be
    set-equal on non-currency names, direction still has to agree unless the span
    states reversibility, and every claimed catalyst still has to be the catalyst
    THIS relation attaches. Nothing here consults the other relations.
    """
    reasons: List[str] = []
    span_in = _non_currency(relation.inputs)
    span_out = _non_currency(relation.outputs)

    swapped = claim_in == span_out and claim_out == span_in and claim_in != claim_out
    if swapped:
        if not relation.reversible:
            return (
                [
                    f"{REASON_DIRECTION_DISAGREES}: the span states "
                    f"{sorted(span_in)} -> {sorted(span_out)}, the claim states the "
                    "reverse, and the span does not state reversibility"
                ],
                None,
            )
    else:
        missing_in = span_in - claim_in
        missing_out = span_out - claim_out
        extra_in = claim_in - span_in
        extra_out = claim_out - span_out
        if missing_in or missing_out or extra_in or extra_out:
            reasons.append(
                f"{REASON_RELATION_DISAGREES}: the span states "
                f"{sorted(span_in)} -> {sorted(span_out)} but the claim states "
                f"{sorted(claim_in)} -> {sorted(claim_out)}"
            )

    # Reversibility agreement is SYMMETRIC. Both halves used to be silently
    # tolerated in one direction: a one-way claim backed by reversible evidence
    # shipped as one-way (losing a direction the paper asserted), and the reverse
    # was refused. A claim may only be MORE constrained than its evidence if the
    # evidence says so.
    normalized: Optional[bool] = None
    if bool(reversible) and not relation.reversible:
        reasons.append(
            f"{REASON_DIRECTION_DISAGREES}: the claim states a reversible reaction "
            "and the span states a one-way one"
        )
    elif relation.reversible and not bool(reversible):
        # Normalized, explicitly and visibly, rather than refused: the evidence
        # supports strictly more than the claim, and a swapped-sides claim is only
        # admissible AS reversible.
        normalized = True

    span_catalysts: Set[str] = set()
    for phrase in relation.catalysts:
        for spelling in actor_spellings(phrase):
            span_catalysts.add(_fold(_canonical(spelling)))
    injected = [
        name
        for name in enzymes
        if _text(name) and _fold(_canonical(name)) not in span_catalysts
    ]
    if injected:
        reasons.append(
            f"{REASON_UNSUPPORTED_CATALYST}: {', '.join(sorted(set(injected)))} "
            "is not the catalyst the span attaches to this reaction"
        )

    return reasons, normalized


# ---------------------------------------------------------------------------
# Policy.
# ---------------------------------------------------------------------------
#: ``organism_policy`` values.
ORGANISM_POLICY_STRICT = "strict"  # require match or genus_level; unknown refused
ORGANISM_POLICY_ALLOW_UNKNOWN = "allow_unknown"  # unknown admitted; mismatch refused
ORGANISM_POLICY_OFF = "off"  # organism never refuses a candidate

_ORGANISM_POLICIES = frozenset(
    {ORGANISM_POLICY_STRICT, ORGANISM_POLICY_ALLOW_UNKNOWN, ORGANISM_POLICY_OFF}
)


@dataclass(frozen=True)
class AdmissionPolicy:
    """The configured half of the admission decision.

    ``organism_policy`` implements "organism is compatible or explicitly unknown
    **under configured policy**": ``allow_unknown`` (the default) admits a
    candidate whose evidence names no organism and refuses an explicit mismatch;
    ``strict`` additionally refuses the unknown case; ``off`` disables the
    organism rule entirely. ``require_pathway_match`` turns the requested-pathway
    comparison from "must not contradict" into "must positively match".
    ``max_chain_hops`` bounds how far a chained admission may travel from the gap
    target, and ``max_span_chars`` bounds what counts as one evidence statement.

    Frozen: a policy is read many times during one admission pass and must not
    drift between candidates.
    """

    organism_policy: str = ORGANISM_POLICY_ALLOW_UNKNOWN
    require_pathway_match: bool = False
    max_report_entries: int = DEFAULT_MAX_REPORT_ENTRIES
    evidence_snippet_chars: int = EVIDENCE_SNIPPET_CHARS
    max_chain_hops: int = DEFAULT_MAX_CHAIN_HOPS
    max_span_chars: int = DEFAULT_MAX_SPAN_CHARS

    def normalized(self) -> "AdmissionPolicy":
        policy = _fold(self.organism_policy) or ORGANISM_POLICY_ALLOW_UNKNOWN
        if policy not in _ORGANISM_POLICIES:
            policy = ORGANISM_POLICY_ALLOW_UNKNOWN
        return AdmissionPolicy(
            organism_policy=policy,
            require_pathway_match=bool(self.require_pathway_match),
            max_report_entries=max(0, int(self.max_report_entries or 0)),
            evidence_snippet_chars=max(0, int(self.evidence_snippet_chars or 0)),
            max_chain_hops=max(0, int(self.max_chain_hops or 0)),
            max_span_chars=max(0, int(self.max_span_chars or 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "organism_policy": self.organism_policy,
            "require_pathway_match": self.require_pathway_match,
            "max_report_entries": self.max_report_entries,
            "evidence_snippet_chars": self.evidence_snippet_chars,
            "max_chain_hops": self.max_chain_hops,
            "max_span_chars": self.max_span_chars,
        }


def policy_from_config(config: Optional[Dict[str, Any]] = None) -> AdmissionPolicy:
    """Build an :class:`AdmissionPolicy` from ``rag_config()`` (read at call time)."""
    cfg = config
    if cfg is None:
        try:
            from t2pw.config import rag_config

            cfg = rag_config()
        except Exception:  # pragma: no cover - defensive; config is stdlib-only
            cfg = {}
    cfg = cfg if isinstance(cfg, dict) else {}
    return AdmissionPolicy(
        organism_policy=str(
            cfg.get("admission_organism_policy") or ORGANISM_POLICY_ALLOW_UNKNOWN
        ),
        require_pathway_match=bool(cfg.get("admission_require_pathway_match", False)),
        max_report_entries=int(
            cfg.get("admission_max_report_entries") or DEFAULT_MAX_REPORT_ENTRIES
        ),
        max_chain_hops=int(
            cfg.get("admission_max_chain_hops", DEFAULT_MAX_CHAIN_HOPS)
        ),
        max_span_chars=int(
            cfg.get("admission_max_span_chars") or DEFAULT_MAX_SPAN_CHARS
        ),
    ).normalized()


# ---------------------------------------------------------------------------
# The candidate.
# ---------------------------------------------------------------------------
@dataclass
class RagReactionCandidate:
    """One reaction a retrieved passage stated, before it is allowed into a payload.

    Every field below the chemistry is what makes the admission decision
    *reviewable*: which gap(s) this claims to fill, which paper and which passage
    said it, **which span of that passage**, what organism/pathway that passage is
    about, what it connects to, and — after :func:`admit_candidates` — the verdict,
    the reasons behind it, and (for a chained admission) how far from the gap it
    sits and through which metabolite.

    ``scope_membership`` is written by the gate, not by the transcriber: an
    accepted candidate is :data:`SCOPE_ADMITTED`, a rejected one
    :data:`SCOPE_REJECTED`. It rides with the reaction all the way into the
    payload (see :mod:`t2pw.rag.synthesize`), which is what lets the core
    out-of-scope filter and the lock manifest agree about a RAG-imported row.
    """

    gap_id: str
    name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    enzymes: List[str] = field(default_factory=list)
    reversible: bool = False
    #: Every gap this same canonical claim was retrieved for. ``gap_id`` is the
    #: primary one (the gap the admission decision was made against); this is the
    #: complete set, so a claim that filled two gaps loses neither attribution
    #: when the two candidates merge into one payload row.
    gap_ids: List[str] = field(default_factory=list)
    #: Provenance — the paper and the exact passage.
    source_paper: Dict[str, Any] = field(default_factory=dict)
    #: Pointer to the PARENT CHUNK: ``chunk_id`` / ``source_id`` / ``source_uri``
    #: / ``section`` / ``score``. Its ``text`` is the whole chunk.
    evidence: Dict[str, Any] = field(default_factory=dict)
    #: The exact span WITHIN that chunk that states this reaction — the parsed
    #: source line for an arrow equation, the validated quote/sentence for prose.
    #: The chunk pointer above is retained alongside it, never replaced.
    evidence_span: str = ""
    #: Scope — what the evidence reports vs what the run asked for.
    organism: str = ""
    observed_organisms: List[str] = field(default_factory=list)
    observed_pathways: List[str] = field(default_factory=list)
    requested_pathway: str = ""
    requested_organism: str = ""
    requested_pathway_match: str = PATHWAY_UNKNOWN
    organism_match: str = ORGANISM_UNKNOWN
    scope_membership: str = ""
    #: Graph — the existing metabolites this candidate attaches to.
    connected_entities: List[str] = field(default_factory=list)
    confidence: float = 0.0
    #: Explicit record of any field the gate normalized to its evidence (today:
    #: reversibility). Silent normalization is indistinguishable from the gate
    #: not having noticed, so it is written down.
    normalized: List[str] = field(default_factory=list)
    #: Verdict.
    status: str = ""
    reasons: List[str] = field(default_factory=list)
    #: Audit trail for a chained acceptance: hop distance, the parent claim it
    #: chained from, the metabolite they share, and the gap. Empty for a direct
    #: acceptance (hop 0) and for a rejection.
    chain: Dict[str, Any] = field(default_factory=dict)

    # --- derived ---------------------------------------------------------
    def participant_names(self) -> List[str]:
        """The CHEMICAL nodes: substrates and products only.

        Catalysts are deliberately absent. This is the list every graph rule
        keys on, which is what makes "two reactions sharing an enzyme are not
        connected" true by construction rather than by a check that can be
        forgotten.
        """
        return list(self.inputs) + list(self.outputs)

    def all_names(self) -> List[str]:
        """Every name the claim mentions — for evidence checks, never for the graph."""
        return self.participant_names() + list(self.enzymes)

    def claim_identity(self) -> Tuple[Any, ...]:
        """The canonical CLAIM this candidate makes — never its display name.

        Two candidates are the same claim when they assert the same canonical
        substrates → products with the same catalysts and the same directionality.
        The reaction *name* is deliberately excluded: papers name the same step
        differently ("R4 theobromine demethylation" vs "theobromine
        N3-demethylation"), and merging on a name would both miss real duplicates
        and fuse genuinely different steps that happen to share a label.
        """
        return (
            tuple(sorted(_fold(_canonical(n)) for n in self.inputs)),
            tuple(sorted(_fold(_canonical(n)) for n in self.outputs)),
            tuple(sorted(_fold(_canonical(n)) for n in self.enzymes)),
            bool(self.reversible),
        )

    def claim_key(self) -> str:
        """A stable, order-independent string form of :meth:`claim_identity`."""
        ins, outs, enz, rev = self.claim_identity()
        return f"{'+'.join(ins)}=>{'+'.join(outs)}|{'+'.join(enz)}|{int(rev)}"

    def provenance_identity(self) -> Tuple[str, str, str]:
        """``(source_id, chunk_id, span)`` — which paper, which passage, which span.

        Paired with :meth:`claim_identity` this is the full merge key: the same
        claim from a *different* span is corroboration and unions in; the same
        claim from the *same* span is one retrieval counted twice (a passage is
        top-k for several gaps) and must not amplify anything. The span is part of
        the identity because one chunk can legitimately state two different
        reactions, and each is evidenced by its own sentence.
        """
        return (
            _text(self.evidence.get("source_id"))
            or _text(self.source_paper.get("source_id")),
            _text(self.evidence.get("chunk_id")),
            _fold(self.evidence_span),
        )

    def merge_key(self) -> Tuple[Any, ...]:
        return (self.gap_id, self.claim_identity())

    def evidence_text(self) -> str:
        """The chunk's whole text (the parent pointer), NOT the supporting span."""
        return _text(self.evidence.get("text"))

    def source_id(self) -> str:
        return _text(self.source_paper.get("source_id")) or _text(
            self.evidence.get("source_id")
        )

    def to_dict(self, *, snippet_chars: int = EVIDENCE_SNIPPET_CHARS) -> Dict[str, Any]:
        """A bounded report row. The span is quoted, the chunk never carried."""
        span = self.evidence_span
        if snippet_chars >= 0 and len(span) > snippet_chars:
            span = span[:snippet_chars].rstrip() + "…"
        return {
            "gap_id": self.gap_id,
            "gap_ids": list(self.gap_ids or ([self.gap_id] if self.gap_id else [])),
            "name": self.name,
            "inputs": self.inputs[:_MAX_NAMES_PER_ROW],
            "outputs": self.outputs[:_MAX_NAMES_PER_ROW],
            "enzymes": self.enzymes[:_MAX_NAMES_PER_ROW],
            "reversible": bool(self.reversible),
            "source_paper": {
                "source_id": _text(self.source_paper.get("source_id")),
                "title": _text(self.source_paper.get("title")),
                "uri": _text(self.source_paper.get("uri")),
                "source_type": _text(self.source_paper.get("source_type")),
            },
            "evidence": {
                "chunk_id": _text(self.evidence.get("chunk_id")),
                "source_id": _text(self.evidence.get("source_id")),
                "source_uri": _text(self.evidence.get("source_uri")),
                "section": _text(self.evidence.get("section")),
                "score": float(self.evidence.get("score") or 0.0),
                "span": span,
            },
            "organism": self.organism,
            "observed_organisms": list(self.observed_organisms)[:_MAX_NAMES_PER_ROW],
            "observed_pathways": list(self.observed_pathways)[:_MAX_NAMES_PER_ROW],
            "requested_pathway": self.requested_pathway,
            "requested_organism": self.requested_organism,
            "requested_pathway_match": self.requested_pathway_match,
            "organism_match": self.organism_match,
            "scope_membership": self.scope_membership,
            "connected_entities": self.connected_entities[:_MAX_NAMES_PER_ROW],
            "confidence": round(float(self.confidence or 0.0), 6),
            "status": self.status,
            "reasons": list(self.reasons),
            "normalized": list(self.normalized),
            "chain": dict(self.chain),
        }


# ---------------------------------------------------------------------------
# The report.
# ---------------------------------------------------------------------------
@dataclass
class AdmissionReport:
    """Accepted *and* rejected candidates, bounded.

    Both halves are preserved: an accepted-only report cannot answer "why is the
    reaction I expected missing?", which is the question a gap-filling run
    actually raises. ``truncated`` records what the caps dropped, so a bounded
    report never reads as a complete one. ``acceptance_breakdown`` splits the
    accepted set into direct and chained admissions, because they carry different
    risk and a single "accepted: N" hides that. It is deliberately NOT called
    precision: nothing here computes a true-positive/false-positive denominator.
    """

    accepted: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    reason_counts: Dict[str, int] = field(default_factory=dict)
    by_gap: Dict[str, Dict[str, int]] = field(default_factory=dict)
    truncated: Dict[str, int] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)
    rules: List[Dict[str, str]] = field(default_factory=list)
    #: ``{"direct": n, "chained": n, "chained_by_hop": {"1": n, ...}}``.
    #: Named a BREAKDOWN, not "precision": nothing here computes a
    #: true-positive/false-positive denominator, so calling it precision would
    #: claim a measurement that was never made.
    acceptance_breakdown: Dict[str, Any] = field(default_factory=dict)
    #: Typed-gap resolutions proposed from evidence (identity / compartment /
    #: precursor). Proposed here, APPLIED — or not — in ``t2pw.rag.synthesize``.
    resolutions: List[Dict[str, Any]] = field(default_factory=list)
    #: Where a gap this gate cannot close SHOULD go. A recommendation, not an
    #: action: nothing here hands the gap to another component, and naming it
    #: ``routed`` claimed otherwise.
    recommended_route: List[Dict[str, str]] = field(default_factory=list)
    #: The live :class:`TypedResolution` objects behind ``resolutions``. Kept off
    #: :meth:`to_dict` on purpose — the serialized report is a diagnostic, while
    #: these are what ``t2pw.rag.synthesize`` tries to APPLY to the payload.
    proposals: List["TypedResolution"] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "counts": dict(self.counts),
            "reason_counts": dict(self.reason_counts),
            "by_gap": {k: dict(v) for k, v in self.by_gap.items()},
            "truncated": dict(self.truncated),
            "policy": dict(self.policy),
            "rules": list(self.rules),
            "acceptance_breakdown": dict(self.acceptance_breakdown),
            "resolutions": list(self.resolutions),
            "recommended_route": list(self.recommended_route),
        }


# ---------------------------------------------------------------------------
# Comparisons — observed vs requested.
# ---------------------------------------------------------------------------
def _compare_one_organism(requested: str, observed: str) -> str:
    try:
        from t2pw.rag.eligibility import _compare_one

        return str(_compare_one(requested, observed))
    except Exception:  # pragma: no cover - defensive
        return (
            ORGANISM_MATCH
            if _fold(requested) == _fold(observed)
            else ORGANISM_MISMATCH
        )


def compare_organism(requested: Any, observed: Any) -> str:
    """``match`` / ``genus_level`` / ``mismatch`` / ``unknown`` for the observations.

    ``observed`` may be a single name or a list of them. Delegates the taxonomy to
    ``t2pw.rag.eligibility._compare_one`` so this gate and the paper-level
    eligibility screen never disagree about whether *Escherichia coli* and
    *Escherichia fergusonii* are the same organism.

    A bare genus, or a different species of the requested genus, comes back as
    ``genus_level`` — never ``match``. That distinction is the point: such
    evidence is related-taxon support, and calling it an exact match is how a
    *Pseudomonas aeruginosa* reaction would silently become a *Pseudomonas
    putida* pathway step.
    """
    req = _text(requested)
    names = (
        [observed] if isinstance(observed, str) else list(observed or [])
    )
    names = [n for n in (_text(n) for n in names) if n]
    if not req or not names:
        return ORGANISM_UNKNOWN
    verdicts = [_compare_one_organism(req, name) for name in names]
    for level in (ORGANISM_MATCH, ORGANISM_GENUS_LEVEL):
        if level in verdicts:
            return level
    return ORGANISM_MISMATCH if verdicts else ORGANISM_UNKNOWN


def organisms_in_span(span: str) -> List[str]:
    """Organisms the span itself names, read with the eligibility lexicon.

    This is the LOCAL observation — what the sentence backing this reaction is
    about — and it takes precedence over the paper-level metadata. A review that
    covers *E. coli* and human cells is "about" both at paper level; the sentence
    describing one reaction is about one of them.
    """
    body = _text(span)
    if not body:
        return []
    try:
        from t2pw.rag.eligibility import (
            _ORGANISM_ENTRIES,
            _TermSet,
            _binomials,
            _normalize,
            _prune_genus_observations,
        )
    except Exception:  # pragma: no cover - defensive
        return []
    haystack = _normalize(body)
    found: List[str] = []
    for entry in _ORGANISM_ENTRIES:
        if _TermSet(entry.aliases).hits(haystack):
            found.append(entry.canonical)
    found.extend(_binomials(haystack))
    return _prune_genus_observations(_dedupe(found))


def _requested_alias_set(requested: str) -> Tuple[Set[str], Dict[str, Any]]:
    try:
        from t2pw.rag.eligibility import _PATHWAY_LEXICON, _pathway_aliases

        aliases = list(_pathway_aliases(requested)) or [requested]
        lexicon = _PATHWAY_LEXICON
    except Exception:  # pragma: no cover - defensive
        aliases, lexicon = [requested], {}
    alias_set = {_fold(a) for a in aliases if _text(a)}
    alias_set.add(_fold(requested))
    return alias_set, lexicon


def pathways_named_in(text: Any, alias_set: Set[str], lexicon: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """``(names_the_requested_pathway, other_lexicon_pathways_named)``."""
    body = _fold(text)
    if not body:
        return (False, [])
    names_requested = any(
        name_pattern(alias) is not None and name_pattern(alias).search(body)
        for alias in alias_set
        if alias
    )
    others: List[str] = []
    for key, entry in lexicon.items():
        spellings = {_fold(key)} | {_fold(a) for a in entry.aliases}
        if spellings & alias_set:
            continue
        for spelling in sorted(spellings):
            pattern = name_pattern(spelling)
            if pattern is not None and pattern.search(body):
                others.append(key)
                break
    return (names_requested, sorted(set(others)))


def compare_requested_pathway(
    requested: Any,
    span: Any,
    observed_pathways: Optional[Sequence[str]] = None,
    paper_haystack: Any = "",
) -> str:
    """Does the cited evidence read as being about the requested pathway?

    **The span wins.** A paper that genuinely covers the requested pathway will
    also contain sentences about other pathways, and the sentence backing THIS
    reaction is the one being admitted. So the local evidence is read first:

    * span names another known pathway and not the requested one -> ``mismatch``,
      even when the paper as a whole was observed to be about the requested
      pathway. This is the case that previously slipped through as ``match`` on
      the paper-level reading and imported another pathway's chemistry.
    * span names the requested pathway and another one -> :data:`PATHWAY_MIXED`,
      recorded explicitly rather than being silently rounded up to ``match``.
    * span names only the requested pathway -> ``match``.
    * span names no pathway at all -> fall back to the paper-level
      ``observed_pathways`` / title, since silence locally says nothing.

    ``unknown`` is the honest answer when neither level says anything: a passage
    stating one reaction routinely names neither the pathway nor a synonym of it,
    and treating that silence as contradiction would reject exactly the
    mechanistic evidence this subsystem exists to find.
    """
    req = _text(requested)
    if not req:
        return PATHWAY_UNKNOWN
    alias_set, lexicon = _requested_alias_set(req)

    local_requested, local_others = pathways_named_in(span, alias_set, lexicon)
    if local_requested and local_others:
        return PATHWAY_MIXED
    if local_others:
        return PATHWAY_MISMATCH
    if local_requested:
        return PATHWAY_MATCH

    observed_text = " ; ".join(_text(p) for p in (observed_pathways or []) if _text(p))
    observed_requested, observed_others = pathways_named_in(
        observed_text, alias_set, lexicon
    )
    if observed_requested and observed_others:
        return PATHWAY_MIXED
    if observed_requested:
        return PATHWAY_MATCH
    if observed_others:
        return PATHWAY_MISMATCH

    paper_requested, paper_others = pathways_named_in(paper_haystack, alias_set, lexicon)
    if paper_requested:
        return PATHWAY_MATCH
    if paper_others:
        return PATHWAY_MISMATCH
    return PATHWAY_UNKNOWN


# ---------------------------------------------------------------------------
# Graph tokens — chemical only.
# ---------------------------------------------------------------------------
def _make_token(resolver: Optional[Any]) -> Any:
    """Return the ``name -> graph token`` function the connection rules key on.

    Without ``resolver`` this is the canonical-name casefold the rest of the
    module uses. With one — ``t2pw.rag.synonyms.build_offline_synonym_resolver``,
    the same GROUPING-only resolver :func:`t2pw.rag.synthesize._resolve_reactions`
    already uses — two spellings of one compound collapse to a single token, so a
    retrieved reaction whose paper writes ``lipid IV A`` still connects to a seed
    that writes ``lipid IV_A``. It affects MATCHING only: no name is ever
    rewritten, and a resolver that raises degrades to the plain token rather than
    breaking admission.
    """
    if resolver is None:
        return lambda value: _fold(_canonical(value))

    def _token(value: Any) -> str:
        try:
            resolved = resolver(_text(value))
        except Exception:  # noqa: BLE001 - a broken resolver must not sink the gate
            resolved = ""
        return _fold(resolved) or _fold(_canonical(value))

    return _token


def _pathway_metabolites(payload: Any, token: Any) -> Set[str]:
    """Every non-currency METABOLITE the existing pathway already knows, as tokens.

    Reaction substrates/products and the compound-ish entity buckets only.
    Proteins, protein complexes and reaction names are excluded on purpose: they
    are not chemical nodes, and admitting them here would re-open the
    shared-catalyst path that let thirteen phospholipid reactions read as
    connected to the lipid A pathway in the ``2026-07-28_0919`` replay.
    """
    hubs = _cofactors()
    names: Set[str] = set()

    def _add(value: Any) -> None:
        if _fold(_canonical(value)) in hubs:
            return
        resolved = token(value)
        if resolved:
            names.add(resolved)

    data = payload if isinstance(payload, dict) else {}
    entities = data.get("entities")
    if isinstance(entities, dict):
        for bucket in ("compounds", "element_collections", "nucleic_acids"):
            rows = entities.get(bucket)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict):
                    _add(row.get("name"))
                elif isinstance(row, str):
                    _add(row)
    processes = data.get("processes")
    if isinstance(processes, dict):
        for rows in processes.values():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for side in ("inputs", "outputs"):
                    for participant in row.get(side) or []:
                        if isinstance(participant, str):
                            _add(participant)
                        elif isinstance(participant, dict):
                            _add(
                                participant.get("name") or participant.get("entity")
                            )
    return names


def _gap_target_tokens(gap: Any, token: Any) -> Set[str]:
    """The gap's own target — what a candidate must touch to be FILLING it.

    Delegates to :meth:`t2pw.rag.retrieve.Gap.target_names`, which is narrower
    than ``symbols`` on purpose: ``symbols`` carries retrieval aids (the
    neighbouring reaction's name and catalysts) that must not count as the thing
    being connected. Currency is NOT filtered out here — if the pathway's open end
    genuinely is ``UMP``, that gap must still be fillable.
    """
    target = getattr(gap, "target_names", None)
    names = list(target()) if callable(target) else [getattr(gap, "label", "")]
    return {t for t in (token(n) for n in names) if t}


def _gap_anchor_tokens(gap: Any, token: Any) -> Set[str]:
    """The gap's expected anchor: its target plus the adjacent existing metabolites.

    Wider than :func:`_gap_target_tokens` because the two rules ask different
    questions — "does this connect to the pathway or the expected anchor?" versus
    "does this FILL the named gap?" — but still strictly chemical.
    ``adjacent_metabolites`` is used rather than ``adjacent_entities``: the latter
    includes the neighbouring reaction's catalysts and name.
    """
    tokens = _gap_target_tokens(gap, token)
    adjacent = getattr(gap, "adjacent_metabolites", None)
    if not adjacent:
        # A duck-typed / hand-built gap without the chemical list contributes only
        # its target. Falling back to ``adjacent_entities`` would smuggle catalysts
        # back into the graph, which is the defect this method exists to close.
        adjacent = []
    hubs = _cofactors()
    for value in adjacent:
        if _fold(_canonical(value)) in hubs:
            continue
        resolved = token(value)
        if resolved:
            tokens.add(resolved)
    return tokens


def _metabolite_tokens(names: Iterable[str], token: Any, *, drop_hubs: bool) -> Set[str]:
    """Tokenize metabolite names, optionally dropping currency/hub metabolites."""
    blocked = _hubs() if drop_hubs else _cofactors()
    out: Set[str] = set()
    for name in names:
        if _fold(_canonical(name)) in blocked:
            continue
        resolved = token(name)
        if resolved:
            out.add(resolved)
    return out


# ---------------------------------------------------------------------------
# Gap-type compatibility.
# ---------------------------------------------------------------------------
GAP_DANGLING_REACTION = "dangling_reaction"
GAP_ORPHAN_METABOLITE = "orphan_metabolite"
GAP_UNMAPPED_ENZYME = "unmapped_enzyme"
GAP_MISSING_PRECURSOR = "missing_precursor"
GAP_MISSING_COMPARTMENT = "missing_compartment"

#: Gap kinds a REACTION candidate can fill on connectivity evidence alone. These
#: are the connectivity gaps: what is missing is a reaction, and a reaction is
#: what a candidate is.
_CONNECTIVITY_GAP_KINDS = frozenset({GAP_DANGLING_REACTION, GAP_ORPHAN_METABOLITE})

#: UniProt accession grammar (the official one). A candidate identifier that does
#: not parse is not an identifier, whatever the sentence around it says.
_UNIPROT_RE = re.compile(
    r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b"
)
_EC_RE = re.compile(r"\bEC\s*([0-9]+\.[0-9]+\.[0-9]+\.[0-9\-]+)\b", re.IGNORECASE)

#: Canonical compartment names this gate will write into a payload, keyed by the
#: spellings a paper uses. Only these are applied: a location the schema has no
#: representation for cannot "resolve" a missing-compartment gap.
_COMPARTMENT_CANON = {
    "cytosol": "cytosol",
    "cytoplasm": "cytosol",
    "cytoplasmic": "cytosol",
    "periplasm": "periplasm",
    "periplasmic": "periplasm",
    "inner membrane": "inner membrane",
    "cytoplasmic membrane": "inner membrane",
    "outer membrane": "outer membrane",
    "plasma membrane": "plasma membrane",
    "cell membrane": "plasma membrane",
    "extracellular": "extracellular",
    "nucleus": "nucleus",
    "nuclear": "nucleus",
    "mitochondrion": "mitochondrion",
    "mitochondria": "mitochondrion",
    "mitochondrial": "mitochondrion",
    "endoplasmic reticulum": "endoplasmic reticulum",
    "golgi": "golgi apparatus",
    "peroxisome": "peroxisome",
    "lysosome": "lysosome",
    "vacuole": "vacuole",
    "chloroplast": "chloroplast",
    "thylakoid": "thylakoid",
    "cell wall": "cell wall",
    "cell envelope": "cell envelope",
}

#: What kind of thing a typed resolution proposes.
RESOLUTION_IDENTIFIER = "identifier"
RESOLUTION_COMPARTMENT = "compartment"
RESOLUTION_PRECURSOR = "precursor"


@dataclass
class TypedResolution:
    """A proposed resolution for a gap that a REACTION cannot fill.

    Kept deliberately separate from :class:`RagReactionCandidate`. An
    unmapped-enzyme gap does not want a reaction, it wants an identity; a
    missing-compartment gap wants a location; a missing-precursor gap wants a
    named participant added to an existing reaction. Admitting a *reaction*
    because its evidence happened to also mention a UniProt id is what let
    "identifier mention" masquerade as "identifier resolved".

    A resolution is only ever *proposed* here. Whether it is APPLIED — and
    therefore whether the gap is actually closed — is decided in
    :mod:`t2pw.rag.synthesize` against the emitted payload, after the identity
    policy has had its say. ``applied`` records what really happened.
    """

    gap_id: str
    gap_kind: str
    target: str
    kind: str
    value: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    evidence_span: str = ""
    source_paper: Dict[str, Any] = field(default_factory=dict)
    #: The bucket/type of the ROW this proposal would write to, carried from the
    #: gap contract. An ``unmapped_enzyme`` gap expects "a protein" whether its
    #: target is in ``entities.proteins`` or ``entities.protein_complexes``; only
    #: the former takes a UniProt accession.
    target_entity_bucket: str = ""
    target_entity_type: str = ""
    #: Scope — the same three-way picture a reaction candidate carries. A typed
    #: proposal mutates the payload directly, so it needs at least the scrutiny a
    #: reaction gets: an accession lifted from a Homo sapiens cholesterol paper
    #: must not become an E. coli caffeine protein's identity.
    requested_pathway: str = ""
    requested_organism: str = ""
    span_observed_organisms: List[str] = field(default_factory=list)
    span_observed_pathways: List[str] = field(default_factory=list)
    paper_observed_organisms: List[str] = field(default_factory=list)
    paper_observed_pathways: List[str] = field(default_factory=list)
    organism_match: str = ORGANISM_UNKNOWN
    requested_pathway_match: str = PATHWAY_UNKNOWN
    #: Refusals decided at EXTRACTION, about the shape of the proposal itself
    #: rather than its scope — e.g. evidence naming a participant set the repair
    #: cannot add in full. Seeded into ``reasons`` by
    #: :func:`screen_typed_resolution` so a structurally impossible proposal is
    #: reported rather than silently dropped or silently narrowed.
    structural_reasons: List[str] = field(default_factory=list)
    status: str = ""
    reasons: List[str] = field(default_factory=list)
    applied: bool = False

    def to_dict(self, *, snippet_chars: int = EVIDENCE_SNIPPET_CHARS) -> Dict[str, Any]:
        span = self.evidence_span
        if snippet_chars >= 0 and len(span) > snippet_chars:
            span = span[:snippet_chars].rstrip() + "…"
        return {
            "gap_id": self.gap_id,
            "gap_kind": self.gap_kind,
            "target": self.target,
            "kind": self.kind,
            "value": dict(self.value),
            "evidence": {
                "chunk_id": _text(self.evidence.get("chunk_id")),
                "source_id": _text(self.evidence.get("source_id")),
                "source_uri": _text(self.evidence.get("source_uri")),
                "span": span,
            },
            "source_paper": {"source_id": _text(self.source_paper.get("source_id"))},
            "target_entity_bucket": self.target_entity_bucket,
            "target_entity_type": self.target_entity_type,
            "requested_pathway": self.requested_pathway,
            "requested_organism": self.requested_organism,
            "span_observed_organisms": list(self.span_observed_organisms),
            "span_observed_pathways": list(self.span_observed_pathways),
            "paper_observed_organisms": list(self.paper_observed_organisms),
            "paper_observed_pathways": list(self.paper_observed_pathways),
            "organism_match": self.organism_match,
            "requested_pathway_match": self.requested_pathway_match,
            "status": self.status,
            "applied": bool(self.applied),
            "reasons": list(self.reasons),
        }


# --- typed-evidence relationship parsing ------------------------------------
#: Phrases that ATTACH an accession to a name, rather than merely putting the two
#: in one sentence. "EnzX interacts with EnzY (UniProt P12345)" states EnzY's
#: identity, not EnzX's, and a bare proximity rule gets that backwards.
_IDENTITY_LEAD = r"(?:uniprot(?:kb)?|swiss-?prot|accession|entry)"
_EC_LEAD = r"(?:ec(?:\s+number)?)"


def parse_identity_assertion(span: str, target: str) -> Dict[str, Any]:
    """Identifiers the span asserts are ``target``'s OWN identity, else ``{}``.

    The accession has to be bound to the target by an explicit construction — a
    parenthetical straight after the name, an apposition, an "X has UniProt Y", or
    the reverse "UniProt Y corresponds to X". Mere co-occurrence is refused, which
    is what stops "EnzX interacts with EnzY (UniProt P12345)" from handing EnzX
    somebody else's accession.
    """
    body = _text(span)
    pattern = name_pattern(target)
    if not body or pattern is None:
        return {}
    name = pattern.pattern  # already boundary-anchored
    acc = r"(?P<acc>[A-Z0-9]{6,10})"
    ec = r"(?P<ec>[0-9]+\.[0-9]+\.[0-9]+\.[0-9\-]+)"
    forms = [
        # EnzX (UniProt P12345) / EnzX [UniProtKB: P12345] / EnzX (EC 1.2.3.4)
        rf"{name}\s*[\(\[]\s*(?:{_IDENTITY_LEAD}|{_EC_LEAD})\s*[:#]?\s*(?:{acc}|{ec})\s*[\)\]]",
        # EnzX, UniProt P12345, ... / EnzX UniProtKB: P12345
        rf"{name}\s*,?\s+(?:{_IDENTITY_LEAD}|{_EC_LEAD})\s*[:#]?\s*(?:{acc}|{ec})\b",
        # EnzX has/carries/is assigned (the) UniProt accession P12345
        rf"{name}\s+(?:has|carries|is assigned|bears|is annotated as)\s+(?:the\s+)?"
        rf"(?:{_IDENTITY_LEAD}|{_EC_LEAD})\s*(?:accession|id|identifier|number)?\s*[:#]?\s*(?:{acc}|{ec})\b",
        # UniProt P12345 corresponds to / is / = EnzX
        rf"(?:{_IDENTITY_LEAD}|{_EC_LEAD})\s*[:#]?\s*(?:{acc}|{ec})\s*(?:corresponds to|refers to|is|=)\s*{name}",
    ]
    out: Dict[str, Any] = {}
    for form in forms:
        match = re.search(form, body, re.IGNORECASE)
        if not match:
            continue
        groups = match.groupdict()
        raw_acc = _text(groups.get("acc"))
        raw_ec = _text(groups.get("ec"))
        if raw_acc and _UNIPROT_RE.fullmatch(raw_acc.upper()):
            out["uniprot"] = raw_acc.upper()
        if raw_ec:
            out["ec"] = raw_ec
        if out:
            return out
    return out


#: Constructions that LOCATE something. "B and nuclear extracts were analyzed"
#: contains a compartment word and the target and states no localization at all.
_LOCALIZED_VERBS = (
    r"localis(?:e|es|ed|ing)|localiz(?:e|es|ed|ing)|"
    r"occurs?|occurred|resides?|is found|are found|was found|were found|"
    r"is located|are located|was located|were located|"
    r"is present|are present|was present|were present|"
    r"accumulates?|accumulated|is targeted|is anchored|is embedded|is retained"
)


def parse_localization_assertion(span: str, target: str) -> str:
    """The canonical compartment the span asserts for ``target``, else ``""``.

    Only explicit localization constructions count:

    * ``B is localized in/to the nucleus``
    * ``B occurs / resides / is found / is present in the periplasm``
    * ``periplasmic B was detected`` (the adjectival form)
    * ``the periplasmic localization of B``

    A compartment word merely sharing a sentence with the target is not a
    localization — "B and nuclear extracts were analyzed" says nothing about where
    B is, and treating it as evidence is how a payload acquires a fabricated
    compartment that every downstream stage then trusts. Nor is
    "A is converted to B in the periplasm" a localization OF B: it locates the
    reaction, and a reaction compartment is a different claim on a different row.
    """
    body = _text(span)
    name = name_pattern(target)
    if not body or name is None:
        return ""
    hits: List[str] = []
    for spelling, canon in _COMPARTMENT_CANON.items():
        loc = name_pattern(spelling)
        if loc is None:
            continue
        loc_src = loc.pattern
        forms = (
            # B is localized in/to the periplasm
            rf"{name.pattern}\b[^.;]{{0,40}}?\b(?:{_LOCALIZED_VERBS})\b"
            rf"[^.;]{{0,20}}?\b(?:in|to|at|within|on)\s+(?:the\s+)?{loc_src}",
            # periplasmic B / periplasm-associated B
            rf"{loc_src}[a-z]{{0,3}}[-\s]+{name.pattern}",
            # the periplasmic localization of B
            rf"{loc_src}[a-z]{{0,3}}\s+(?:localis|localiz)\w*\s+of\s+{name.pattern}",
        )
        for form in forms:
            if re.search(form, body, re.IGNORECASE):
                hits.append(canon)
                break
    if not hits:
        return ""
    # Deterministic: the longest canonical wins ("inner membrane" over
    # "membrane"), ties broken alphabetically.
    return sorted(set(hits), key=lambda c: (-len(c), c))[0]


def _payload_reaction(payload: Any, name: str) -> Optional[Dict[str, Any]]:
    """The reaction row ``name`` names in ``payload``, or ``None`` (read-only)."""
    target = _fold(name)
    data = payload if isinstance(payload, dict) else {}
    processes = data.get("processes")
    rows = processes.get("reactions") if isinstance(processes, dict) else None
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and _fold(row.get("name")) == target:
            return row
    return None


def _reaction_side_names(row: Dict[str, Any], side: str) -> List[str]:
    out: List[str] = []
    for token in row.get(side) or []:
        name = token if isinstance(token, str) else (token or {}).get("name")
        if _text(name):
            out.append(_text(name))
    return out


def missing_reaction_side(row: Any) -> str:
    """Which side of a reaction is EMPTY — ``"inputs"``, ``"outputs"`` or ``""``.

    A reaction with both sides populated is not missing a precursor, whatever a
    report said about it, and appending a participant to it would be inventing
    chemistry rather than repairing an omission.
    """
    if not isinstance(row, dict):
        return ""
    has_in = bool(_reaction_side_names(row, "inputs"))
    has_out = bool(_reaction_side_names(row, "outputs"))
    if has_in and not has_out:
        return "outputs"
    if has_out and not has_in:
        return "inputs"
    return ""


def _precursor_resolution(
    gap: Any, span: str, target: str, payload: Any, base: Dict[str, Any]
) -> Optional[TypedResolution]:
    """Propose a repair for a genuinely half-specified reaction, or ``None``.

    Every condition here exists because its absence produced a wrong repair:

    * the reaction must be IN the payload and must really have an empty side —
      otherwise "X is converted to Q by EnzY" appends Q to an unrelated, complete
      R0 and calls it a fix;
    * the evidence must share the reaction's KNOWN side, so the sentence is about
      this reaction rather than about some other chemistry that happens to be in
      the same passage;
    * it must supply a NEW participant on the MISSING side, in the right
      direction — a sentence that names the known participants again identifies
      nothing;
    * and it must supply the evidence's participants for that side **in full**.
      "condensation of glycine and succinyl-CoA to produce ALA" names two
      substrates; taking the first and dropping the second writes a reaction the
      paper does not state. When the full set cannot be added — because one of
      them already sits on the reaction's other side — the proposal is emitted
      REJECTED (:data:`REASON_MULTI_PARTICIPANT_REPAIR`) rather than narrowed.
    """
    row = _payload_reaction(payload, target)
    if row is None:
        return None
    side = _text(getattr(gap, "missing_side", "")) or missing_reaction_side(row)
    if side not in ("inputs", "outputs"):
        return None
    if _reaction_side_names(row, side):
        return None  # that side is populated; there is nothing missing to supply
    known_side = "outputs" if side == "inputs" else "inputs"
    known = {_fold(_canonical(n)) for n in _reaction_side_names(row, known_side)}
    if not known:
        return None

    relation = parse_span_relation(span)
    if relation is None:
        return None
    # Direction agreement: what the reaction already has must sit on the SAME side
    # of the evidence's relation, and the new participants on the other.
    evidence_known = relation.outputs if known_side == "outputs" else relation.inputs
    evidence_missing = relation.inputs if side == "inputs" else relation.outputs
    if not known & {_fold(_canonical(n)) for n in evidence_known}:
        return None

    fresh = [n for n in evidence_missing if _fold(_canonical(n)) not in known]
    if not fresh:
        return None
    value = {
        "participants": list(fresh),
        # Kept for readers of a single-participant repair; ``participants`` is the
        # field the applier uses, and it is always the complete set.
        "participant": fresh[0],
        "side": side,
        "reaction": target,
    }
    structural: List[str] = []
    if len(fresh) != len(evidence_missing):
        dropped = [n for n in evidence_missing if _fold(_canonical(n)) in known]
        structural.append(
            f"{REASON_MULTI_PARTICIPANT_REPAIR}: the evidence states "
            f"{sorted(evidence_missing)} on {side!r}, but {sorted(dropped)} already "
            f"sit on {known_side!r} — the stated set cannot be added in full, and a "
            "partial repair writes chemistry the sentence does not state"
        )
    return TypedResolution(
        kind=RESOLUTION_PRECURSOR,
        value=value,
        structural_reasons=structural,
        **base,
    )


def extract_typed_resolution(
    gap: Any,
    span: str,
    *,
    evidence: Optional[Dict[str, Any]] = None,
    source_paper: Optional[Dict[str, Any]] = None,
    payload: Any = None,
    observed_organisms: Optional[Sequence[str]] = None,
    observed_pathways: Optional[Sequence[str]] = None,
) -> Optional[TypedResolution]:
    """Parse a typed-gap resolution out of ONE evidence span, or ``None``.

    Deliberately independent of any reaction candidate. The sentence that answers
    an unmapped-enzyme gap is usually "EnzX is a UniProt P12345 acyltransferase",
    which states no reaction at all — tying extraction to a transcribed reaction
    would miss exactly the evidence the gap asked for, and tying the gap's closure
    to that reaction is the conflation this separation exists to end.

    Parsing is strict and structured: an identifier has to be a well-formed
    accession attached to the gap's own protein, a compartment has to be one the
    payload schema can express, a precursor has to be a participant the incomplete
    reaction genuinely lacks. Anything looser puts the guessing back in.
    """
    kind = _text(getattr(gap, "kind", ""))
    span = _text(span)
    target = _text(getattr(gap, "label", ""))
    if not span or not target:
        return None
    requested_pathway = _text(getattr(gap, "requested_pathway", ""))
    requested_organism = _text(getattr(gap, "requested_organism", ""))
    span_organisms = organisms_in_span(span)
    paper_organisms = [_text(o) for o in (observed_organisms or []) if _text(o)]
    paper_pathways = [_text(pw) for pw in (observed_pathways or []) if _text(pw)]
    alias_set, lexicon = _requested_alias_set(requested_pathway)
    span_named, span_others = pathways_named_in(span, alias_set, lexicon)
    span_pathways = ([requested_pathway] if span_named else []) + list(span_others)
    base = dict(
        gap_id=_text(getattr(gap, "gap_id", "")),
        gap_kind=kind,
        target=target,
        evidence=dict(evidence or {}),
        evidence_span=span,
        source_paper=dict(source_paper or {}),
        target_entity_bucket=_text(getattr(gap, "target_entity_bucket", "")),
        target_entity_type=_text(getattr(gap, "target_entity_type", "")),
        requested_pathway=requested_pathway,
        requested_organism=requested_organism,
        span_observed_organisms=list(span_organisms),
        span_observed_pathways=[pw for pw in span_pathways if pw],
        paper_observed_organisms=list(paper_organisms),
        paper_observed_pathways=list(paper_pathways),
        organism_match=compare_organism(
            requested_organism, span_organisms or paper_organisms
        ),
        requested_pathway_match=compare_requested_pathway(
            requested_pathway, span, paper_pathways
        ),
    )

    if kind == GAP_UNMAPPED_ENZYME:
        value = parse_identity_assertion(span, target)
        if not value:
            return None
        return TypedResolution(kind=RESOLUTION_IDENTIFIER, value=value, **base)

    if kind == GAP_MISSING_COMPARTMENT:
        canon = parse_localization_assertion(span, target)
        if not canon:
            return None
        return TypedResolution(
            kind=RESOLUTION_COMPARTMENT, value={"location": canon}, **base
        )

    if kind == GAP_MISSING_PRECURSOR:
        return _precursor_resolution(gap, span, target, payload, base)

    return None


def propose_typed_resolutions(
    gap: Any,
    chunk_text: str,
    *,
    evidence: Optional[Dict[str, Any]] = None,
    source_paper: Optional[Dict[str, Any]] = None,
    payload: Any = None,
    observed_organisms: Optional[Sequence[str]] = None,
    observed_pathways: Optional[Sequence[str]] = None,
    policy: Optional[AdmissionPolicy] = None,
    max_span_chars: int = DEFAULT_MAX_SPAN_CHARS,
) -> List[TypedResolution]:
    """Scan a retrieved passage for statements that resolve a TYPED gap.

    One proposal per passage at most, and it is the FIRST statement (in reading
    order) that yields one — deterministic, and bounded so a long passage cannot
    turn into a list of near-duplicate proposals. Connectivity gaps get nothing
    here: what they need is a reaction, and that is the reaction path's job.
    """
    kind = _text(getattr(gap, "kind", ""))
    if kind in _CONNECTIVITY_GAP_KINDS or not kind:
        return []
    for statement in split_statements(chunk_text):
        if max_span_chars and len(statement) > max_span_chars:
            continue
        proposal = extract_typed_resolution(
            gap,
            statement,
            evidence=evidence,
            source_paper=source_paper,
            payload=payload,
            observed_organisms=observed_organisms,
            observed_pathways=observed_pathways,
        )
        if proposal is None:
            continue
        screen_typed_resolution(proposal, policy)
        return [proposal]
    return []


#: Entity types each typed resolution may legitimately be written onto, and what
#: the schema uses instead for the ones it may not.
#:
#: * a **UniProt accession is a protein's identity**. ``entities.protein_complexes``
#:   rows are identified by ``pathbank_complex_id`` / ``pathwhiz_id`` (see
#:   ``t2pw.pwml.ir``), which no sentence in a paper supplies. Writing
#:   ``mapped_ids.uniprot`` onto a complex would make ``identity_status`` report
#:   ``verified`` for a row whose real identity is still unknown, and would remove
#:   it from the Unknown-protein-component fallback that carries it honestly.
#: * a **location** is expressed per element bucket, and ``element_locations`` has
#:   no bucket for a complex: ``protein_locations`` resolves its ``protein`` field
#:   against ``entities.proteins`` only, so a complex placed there is a dangling
#:   reference the IR reports and no exporter can follow.
_RESOLUTION_TARGET_TYPES = {
    RESOLUTION_IDENTIFIER: {"protein", ""},
    RESOLUTION_COMPARTMENT: {
        "compound",
        "protein",
        "nucleic_acid",
        "element_collection",
        "",
    },
    RESOLUTION_PRECURSOR: {"compound", "element_collection", "nucleic_acid", ""},
}


def _target_type_refusals(proposal: TypedResolution) -> List[str]:
    """Refusals arising purely from the SHAPE of the row being written to."""
    allowed = _RESOLUTION_TARGET_TYPES.get(proposal.kind)
    if allowed is None:
        return []
    etype = _text(proposal.target_entity_type)
    if etype in allowed:
        return []
    detail = {
        RESOLUTION_IDENTIFIER: (
            "a UniProt/EC accession is a protein identity; a "
            f"{etype!r} row takes a complex-level identifier "
            "(pathbank_complex_id / pathwhiz_id), which no sentence supplies"
        ),
        RESOLUTION_COMPARTMENT: (
            f"element_locations has no bucket that resolves a {etype!r}; "
            "protein_locations.protein resolves against entities.proteins only"
        ),
        RESOLUTION_PRECURSOR: (
            f"a reaction participant is a metabolite, not a {etype!r}"
        ),
    }.get(proposal.kind, f"{proposal.kind!r} cannot be written onto a {etype!r}")
    return [
        f"{REASON_UNSUPPORTED_TARGET_TYPE}: {proposal.target!r} is a {etype!r} "
        f"({proposal.target_entity_bucket or 'unknown bucket'}) — {detail}"
    ]


def screen_typed_resolution(
    proposal: TypedResolution, policy: Optional[AdmissionPolicy] = None
) -> TypedResolution:
    """Apply the SAME scope admission a reaction candidate gets, in place.

    A typed proposal writes straight into the payload — an identity, a location,
    a participant — so it cannot be held to a weaker standard than a reaction that
    merely adds a row. The organism policy and the local-first pathway policy are
    the ones :func:`_screen_candidate` uses; an explicit different species or a
    different named pathway rejects the proposal outright, ``mixed`` obeys
    ``require_pathway_match``, and ``unknown`` keeps the configured behaviour.

    Type discipline is checked here too, before scope: a proposal aimed at the
    wrong SHAPE of row is not a scope question and no policy setting makes it
    admissible.
    """
    active = (policy or AdmissionPolicy()).normalized()
    reasons: List[str] = list(proposal.structural_reasons) + list(
        _target_type_refusals(proposal)
    )

    if active.organism_policy != ORGANISM_POLICY_OFF:
        if proposal.organism_match == ORGANISM_MISMATCH:
            reasons.append(
                f"{REASON_ORGANISM_MISMATCH}: the evidence reports "
                f"{(proposal.span_observed_organisms or proposal.paper_observed_organisms)!r}, "
                f"the run requested {proposal.requested_organism!r}"
            )
        elif (
            proposal.organism_match == ORGANISM_UNKNOWN
            and active.organism_policy == ORGANISM_POLICY_STRICT
            and _text(proposal.requested_organism)
        ):
            reasons.append(
                f"{REASON_ORGANISM_UNKNOWN_REFUSED}: the evidence names no organism "
                "and the organism policy is 'strict'"
            )
        elif (
            proposal.organism_match == ORGANISM_GENUS_LEVEL
            and active.organism_policy == ORGANISM_POLICY_STRICT
        ):
            reasons.append(
                f"{REASON_ORGANISM_MISMATCH}: same-genus/different-species evidence "
                "is related-taxon support, not an exact match"
            )

    if proposal.requested_pathway_match == PATHWAY_MISMATCH:
        reasons.append(
            f"{REASON_PATHWAY_MISMATCH}: the evidence names a different pathway than "
            f"{proposal.requested_pathway!r}"
        )
    elif proposal.requested_pathway_match == PATHWAY_MIXED and (
        active.require_pathway_match
    ):
        reasons.append(
            f"{REASON_PATHWAY_MISMATCH}: the span names "
            f"{proposal.requested_pathway!r} AND another pathway, which is not the "
            "unambiguous match the policy requires"
        )
    elif (
        active.require_pathway_match
        and _text(proposal.requested_pathway)
        and proposal.requested_pathway_match != PATHWAY_MATCH
    ):
        reasons.append(
            f"{REASON_PATHWAY_MISMATCH}: no positive match to "
            f"{proposal.requested_pathway!r} and the policy requires one"
        )

    if reasons:
        proposal.status = STATUS_REJECTED
        proposal.reasons = reasons
        proposal.applied = False
    else:
        proposal.status = STATUS_PROPOSED
    return proposal


def _gap_type_verdict(candidate: "RagReactionCandidate", gap: Any) -> Optional[str]:
    """``None`` when this candidate is the right KIND of evidence for ``gap``.

    A reaction candidate must not fill every gap merely by mentioning the gap's
    label. What each gap asks for differs, and the difference is not cosmetic:

    * **connectivity** (``dangling_reaction`` / ``orphan_metabolite``) — a
      reaction is exactly what is missing; connectivity evidence is enough.
    * **missing_compartment** — the gap is "where does this happen?". A reaction
      says nothing about location unless the span asserts one, so a bare
      ``A -> B`` cannot fill it.
    * **unmapped_enzyme** — the gap is "which protein IS this?". A reaction that
      mentions the protein is not an identifier resolution; the span has to carry
      identity evidence (UniProt / EC number / accession / gene locus). Otherwise
      the gap stays open and is routed to the identity resolver
      (:data:`ROUTE_IDENTITY_RESOLVER`), which is the component that can actually
      answer it.
    * **missing_precursor** — the gap is "which participant is this reaction
      missing?". Filling it means supplying a metabolite the incomplete reaction
      does NOT already have, while sharing one that it does; a candidate that only
      restates the known participants identifies nothing.
    """
    kind = _text(getattr(gap, "kind", ""))
    gap_id = _text(getattr(gap, "gap_id", ""))
    label = _text(getattr(gap, "label", ""))

    if kind in _CONNECTIVITY_GAP_KINDS or not kind:
        return None

    # A REACTION never fills a typed gap, however good its evidence. What such a
    # gap wants is an identity, a location, or a participant added to an existing
    # reaction — none of which a new reaction row expresses. The evidence that
    # DOES answer it is lifted out separately as a :class:`TypedResolution` and
    # only closes the gap once the payload actually carries the resolution.
    what = {
        GAP_UNMAPPED_ENZYME: (
            f"the identity of {label!r}; a reaction is not an identifier "
            f"resolution — recommended route: {ROUTE_IDENTITY_RESOLVER}"
        ),
        GAP_MISSING_COMPARTMENT: (
            f"a compartment/location for {label!r}; a reaction row expresses no "
            "location"
        ),
        GAP_MISSING_PRECURSOR: (
            f"the participant reaction {label!r} is missing; adding a SEPARATE "
            "reaction does not repair the incomplete one"
        ),
    }.get(kind, f"evidence of kind {kind!r}")
    return f"{REASON_TYPE_CANNOT_FILL}: gap {gap_id} asks for {what}"


# ---------------------------------------------------------------------------
# The gate.
# ---------------------------------------------------------------------------
def _screen_candidate(
    candidate: RagReactionCandidate,
    gap: Any,
    policy: AdmissionPolicy,
) -> List[str]:
    """Per-candidate rules (everything except the graph-connection BFS).

    Returns the rejection reasons; empty means the candidate survives to the
    connection phase. Rules are evaluated in :data:`ADMISSION_RULES` order and
    ALL failures are collected — a report that names only the first failure sends
    a reviewer round the loop once per rule.
    """
    reasons: List[str] = []
    cofactors = _cofactors()

    # Rule: valid entity types.
    for name in candidate.participant_names():
        if not _text(name) or _is_invalid_token(name):
            reasons.append(f"{REASON_INVALID_ENTITY_TYPE}: participant {name!r}")
    for enzyme in candidate.enzymes:
        if not _text(enzyme):
            # A blank actor row claims no catalyst, so there is nothing to refuse.
            # Real payloads carry actor rows that are pure ``{"evidence": ...}``
            # with no entity name at all; discarding an evidenced reaction over a
            # malformed sibling field would be the wrong trade.
            continue
        if _is_invalid_token(enzyme):
            reasons.append(f"{REASON_INVALID_ENTITY_TYPE}: catalyst {enzyme!r}")
        elif _fold(_canonical(enzyme)) in cofactors:
            reasons.append(
                f"{REASON_INVALID_ENTITY_TYPE}: catalyst {enzyme!r} is a cofactor, "
                "not a protein"
            )
    if not candidate.inputs or not candidate.outputs:
        reasons.append(f"{REASON_INVALID_ENTITY_TYPE}: reaction is missing a side")

    # Rules: local relational evidence — the span must STATE this reaction, with
    # the same direction and the same catalyst, not merely contain its names.
    span_verdict = validate_evidence_span(
        candidate.evidence_span,
        inputs=candidate.inputs,
        outputs=candidate.outputs,
        enzymes=candidate.enzymes,
        reversible=candidate.reversible,
        max_span_chars=policy.max_span_chars,
    )
    reasons.extend(span_verdict.reasons)
    if span_verdict.normalized_reversible is not None:
        candidate.reversible = bool(span_verdict.normalized_reversible)
        candidate.normalized.append(
            "reversible: normalized to the span, which states a reversible reaction"
        )

    # Rule: organism compatible / explicitly unknown, under the configured policy.
    if policy.organism_policy != ORGANISM_POLICY_OFF:
        if candidate.organism_match == ORGANISM_MISMATCH:
            reasons.append(
                f"{REASON_ORGANISM_MISMATCH}: evidence reports "
                f"{(candidate.observed_organisms or [candidate.organism])!r}, run "
                f"requested {candidate.requested_organism!r}"
            )
        elif (
            candidate.organism_match == ORGANISM_UNKNOWN
            and policy.organism_policy == ORGANISM_POLICY_STRICT
            and _text(candidate.requested_organism)
        ):
            reasons.append(
                f"{REASON_ORGANISM_UNKNOWN_REFUSED}: evidence names no organism and "
                "the organism policy is 'strict'"
            )
        elif (
            candidate.organism_match == ORGANISM_GENUS_LEVEL
            and policy.organism_policy == ORGANISM_POLICY_STRICT
        ):
            reasons.append(
                f"{REASON_ORGANISM_MISMATCH}: evidence is same-genus/different-species "
                f"({candidate.observed_organisms!r}), which is related-taxon support "
                "and not an exact match; the organism policy is 'strict'"
            )

    # Rule: requested pathway not contradicted (strict: positively matched).
    if candidate.requested_pathway_match == PATHWAY_MISMATCH:
        reasons.append(
            f"{REASON_PATHWAY_MISMATCH}: the evidence names a different pathway than "
            f"{candidate.requested_pathway!r}"
        )
    elif candidate.requested_pathway_match == PATHWAY_MIXED and (
        policy.require_pathway_match
    ):
        reasons.append(
            f"{REASON_PATHWAY_MISMATCH}: the span names "
            f"{candidate.requested_pathway!r} AND another pathway, which is not the "
            "unambiguous match the policy requires"
        )
    elif (
        policy.require_pathway_match
        and _text(candidate.requested_pathway)
        and candidate.requested_pathway_match != PATHWAY_MATCH
    ):
        reasons.append(
            f"{REASON_PATHWAY_MISMATCH}: no positive match to "
            f"{candidate.requested_pathway!r} and the policy requires one"
        )

    # Rule: the candidate is the KIND of evidence this gap asks for.
    type_reason = _gap_type_verdict(candidate, gap)
    if type_reason:
        reasons.append(type_reason)

    return reasons


def admit_candidates(
    candidates: Sequence[RagReactionCandidate],
    *,
    gaps: Sequence[Any] = (),
    seed_payload: Any = None,
    policy: Optional[AdmissionPolicy] = None,
    name_resolver: Optional[Any] = None,
) -> Tuple[List[RagReactionCandidate], AdmissionReport]:
    """Apply :data:`ADMISSION_RULES` and return ``(accepted, report)``.

    ``gaps`` are the detected :class:`~t2pw.rag.retrieve.Gap` objects; a candidate
    naming a ``gap_id`` that is not among them is rejected, which is what makes
    "every retrieval query references a specific gap" enforceable rather than
    aspirational. ``seed_payload`` is the pathway the candidates would join —
    read-only, used for the set of metabolites already in it.

    The graph rules run as a **breadth-first pass per gap**, not per candidate in
    isolation: a retrieved passage frequently states a short chain (``A -> B``,
    ``B -> C``, ``C -> D``) whose first link touches the gap target and whose later
    links only touch each other. Hop 0 is a candidate touching the target itself;
    hop *k* is one reachable through *k* reactions already admitted for that same
    gap, and ``policy.max_chain_hops`` bounds *k*. Because hops come from BFS
    levels rather than iteration order, the verdict and every recorded hop
    distance are independent of the order the candidates arrive in.

    Chain expansion travels **only** through non-currency substrates and products
    (:data:`HUB_METABOLITES`): an admitted reaction that releases CoA must not open
    the frontier onto every acyl-CoA reaction in the corpus, and an enzyme shared
    between two unrelated reactions must not connect them at all.

    A candidate that never reaches the frontier is rejected — as
    :data:`REASON_ADJACENT_CONTEXT` when it does touch the surrounding pathway
    (the "correct biology, wrong gap" case), :data:`REASON_CHAIN_TOO_FAR` when it
    was reachable but beyond the hop limit, and :data:`REASON_NOT_CONNECTED` when
    it touches nothing known at all.

    ``name_resolver`` (optional) is the GROUPING-only synonym resolver from
    :mod:`t2pw.rag.synonyms`; supplying it makes the connection rules
    synonym-aware without rewriting any name (see :func:`_make_token`).

    Nothing is mutated except the candidates' own verdict fields — that is the
    point of the pass. ``seed_payload`` and ``gaps`` are read-only.
    """
    active = (policy or AdmissionPolicy()).normalized()
    token = _make_token(name_resolver)
    gaps_by_id: Dict[str, Any] = {}
    for gap in gaps or ():
        gap_id = _text(getattr(gap, "gap_id", ""))
        if gap_id:
            gaps_by_id.setdefault(gap_id, gap)

    pathway_tokens = _pathway_metabolites(seed_payload, token)

    pending_by_gap: Dict[str, List[RagReactionCandidate]] = {}
    rejected: List[RagReactionCandidate] = []
    routed: List[Dict[str, str]] = []
    resolutions: List[TypedResolution] = []

    # --- Phase 1: per-candidate rules -------------------------------------
    for candidate in candidates:
        gap_id = _text(candidate.gap_id)
        if not candidate.gap_ids and gap_id:
            candidate.gap_ids = [gap_id]
        if not gap_id:
            _reject(
                candidate,
                [
                    f"{REASON_NO_GAP_ID}: the candidate names no gap, so it cannot be "
                    "admitted as filling one"
                ],
            )
            rejected.append(candidate)
            continue
        gap = gaps_by_id.get(gap_id)
        if gap is None:
            _reject(
                candidate,
                [f"{REASON_UNKNOWN_GAP_ID}: {gap_id!r} is not a detected gap"],
            )
            rejected.append(candidate)
            continue
        reasons = _screen_candidate(candidate, gap, active)
        if reasons:
            _reject(candidate, reasons)
            rejected.append(candidate)
            if any(ROUTE_IDENTITY_RESOLVER in reason for reason in reasons):
                routed.append(
                    {
                        "gap_id": gap_id,
                        "label": _text(getattr(gap, "label", "")),
                        "recommended_route": ROUTE_IDENTITY_RESOLVER,
                    }
                )
            # A reaction refused for being the wrong KIND of evidence may still
            # carry, in the same span, the thing the gap actually asked for. That
            # is lifted out as a separate proposal rather than being allowed to
            # admit the reaction.
            if any(reason.startswith(REASON_TYPE_CANNOT_FILL) for reason in reasons):
                proposal = extract_typed_resolution(
                    gap,
                    candidate.evidence_span,
                    evidence=candidate.evidence,
                    source_paper=candidate.source_paper,
                    payload=seed_payload,
                    observed_organisms=candidate.observed_organisms,
                    observed_pathways=candidate.observed_pathways,
                )
                if proposal is not None:
                    screen_typed_resolution(proposal, active)
                    resolutions.append(proposal)
        else:
            pending_by_gap.setdefault(gap_id, []).append(candidate)

    # --- Phase 2: per-gap breadth-first connection -------------------------
    accepted: List[RagReactionCandidate] = []
    for gap_id in sorted(pending_by_gap):
        gap = gaps_by_id[gap_id]
        accepted.extend(
            _admit_for_gap(
                gap_id,
                gap,
                pending_by_gap[gap_id],
                pathway_tokens=pathway_tokens,
                token=token,
                policy=active,
                rejected=rejected,
            )
        )

    # Keep the caller's input order for the accepted list so the payload's
    # reaction order does not depend on the gap id sort above.
    order = {id(c): i for i, c in enumerate(candidates)}
    accepted.sort(key=lambda c: order.get(id(c), 0))

    report = _build_report(accepted, rejected, active, routed, resolutions)
    return accepted, report


def _admit_for_gap(
    gap_id: str,
    gap: Any,
    pending: List[RagReactionCandidate],
    *,
    pathway_tokens: Set[str],
    token: Any,
    policy: AdmissionPolicy,
    rejected: List[RagReactionCandidate],
) -> List[RagReactionCandidate]:
    """BFS over the metabolite graph for one gap. See :func:`admit_candidates`."""
    target = _gap_target_tokens(gap, token)
    anchors = _gap_anchor_tokens(gap, token) | pathway_tokens
    kind = _text(getattr(gap, "kind", "")) or GAP_ORPHAN_METABOLITE


    # Chemical tokens per candidate, computed once. Two views: the full
    # participant set (what may MATCH a frontier) and the hub-free set (what may
    # EXTEND it).
    match_tokens = {
        id(c): _metabolite_tokens(c.participant_names(), token, drop_hubs=False)
        for c in pending
    }
    extend_tokens = {
        id(c): _metabolite_tokens(c.participant_names(), token, drop_hubs=True)
        for c in pending
    }

    # A gap whose target is itself a currency/hub metabolite cannot be filled on
    # "touches the target" alone: every reaction in metabolism touches acetyl-CoA,
    # CoA, ACP or glutamate, so that test admits the corpus. Such a gap needs a
    # SECOND, non-hub connection into its own neighbourhood and positive pathway
    # evidence — and if neither is there it stays open for review rather than
    # being closed by whatever was retrieved first. Non-hub targets (theobromine,
    # UMP, ...) are unaffected: ``hub_target`` is False for them.
    hubs = _hubs()
    hub_target = bool(target) and all(t in hubs for t in target)
    non_hub_anchors = {a for a in anchors if a not in hubs}

    accepted: List[RagReactionCandidate] = []
    remaining = list(pending)
    frontier: Dict[str, RagReactionCandidate] = {}  # token -> contributing parent
    reachable = set(target)
    hop = 0

    if hub_target:
        survivors: List[RagReactionCandidate] = []
        for candidate in pending:
            tokens = match_tokens[id(candidate)]
            extra = (tokens & non_hub_anchors) - target
            if extra and candidate.requested_pathway_match == PATHWAY_MATCH:
                survivors.append(candidate)
                continue
            _reject(
                candidate,
                [
                    f"{REASON_HUB_TARGET}: gap {gap_id}'s target is a currency/hub "
                    "metabolite, so touching it proves nothing; this candidate adds "
                    f"{'no' if not extra else 'a'} non-hub connection to the gap's "
                    "neighbourhood and its pathway match is "
                    f"{candidate.requested_pathway_match!r}"
                ],
            )
            rejected.append(candidate)
        remaining = list(survivors)

    while remaining and hop <= policy.max_chain_hops:
        level: List[Tuple[RagReactionCandidate, str, Optional[RagReactionCandidate]]] = []
        for candidate in remaining:
            hits = match_tokens[id(candidate)] & reachable
            if not hits:
                continue
            shared = sorted(hits)[0]
            parent = frontier.get(shared) if hop > 0 else None
            level.append((candidate, shared, parent))
        if not level:
            break
        for candidate, shared, parent in level:
            anchor_hits = match_tokens[id(candidate)] & (anchors | set(frontier))
            if not anchor_hits:
                # Unreachable in practice (the frontier is a subset of the anchor
                # set at hop 0 and grows from accepted candidates after), but the
                # rule is explicit in ADMISSION_RULES so it is checked explicitly.
                continue
            candidate.connected_entities = _dedupe(
                name
                for name in candidate.participant_names()
                if token(name) in anchor_hits
            )
            _accept(candidate, hop, shared, parent, gap_id)
            accepted.append(candidate)
        accepted_ids = {id(c) for c, _s, _p in level}
        remaining = [c for c in remaining if id(c) not in accepted_ids]
        # Extend the frontier ONLY through non-currency metabolites of the
        # reactions just admitted.
        # Deterministic lineage: when several reactions admitted at the SAME hop
        # introduce the same metabolite, the recorded parent is the one with the
        # smallest canonical claim key — a property of the claims themselves, not
        # of the order they arrived in, so the audit trail is stable under any
        # input permutation.
        for tok in sorted(
            {t for c, _s, _p in level for t in extend_tokens[id(c)]} - reachable
        ):
            contributors = [
                c for c, _s, _p in level if tok in extend_tokens[id(c)]
            ]
            if contributors:
                frontier.setdefault(tok, min(contributors, key=lambda c: c.claim_key()))
        for candidate, _shared, _parent in level:
            reachable |= extend_tokens[id(candidate)]
        hop += 1

    for candidate in remaining:
        tokens = match_tokens[id(candidate)]
        if tokens & reachable:
            reason = (
                f"{REASON_CHAIN_TOO_FAR}: reachable from gap {gap_id} only beyond the "
                f"configured limit of {policy.max_chain_hops} hop(s)"
            )
        elif tokens & (anchors | pathway_tokens):
            reason = (
                f"{REASON_ADJACENT_CONTEXT}: shares a metabolite with the pathway but "
                f"nothing in gap {gap_id}'s target — related context, not a gap fill"
            )
        elif tokens:
            reason = (
                f"{REASON_DOES_NOT_FILL_GAP}: no substrate or product of this "
                f"candidate is in gap {gap_id}'s target or its admitted chain"
            )
        else:
            reason = (
                f"{REASON_NOT_CONNECTED}: the candidate contributes no usable "
                f"metabolite, so it cannot connect to gap {gap_id}"
            )
        _reject(candidate, [reason])
        rejected.append(candidate)

    return accepted


def _accept(
    candidate: RagReactionCandidate,
    hop: int,
    shared: str,
    parent: Optional[RagReactionCandidate],
    gap_id: str,
) -> None:
    candidate.status = STATUS_ACCEPTED
    candidate.scope_membership = SCOPE_ADMITTED
    if hop == 0:
        candidate.chain = {"hop": 0, "gap_id": gap_id, "shared_metabolite": shared}
        candidate.reasons = [f"{REASON_FILLS_GAP_DIRECT}: via {shared}"]
        return
    candidate.chain = {
        "hop": hop,
        "gap_id": gap_id,
        "shared_metabolite": shared,
        "parent_claim": parent.claim_key() if parent is not None else "",
        "parent_name": parent.name if parent is not None else "",
    }
    candidate.reasons = [
        f"{REASON_FILLS_GAP_CHAINED}: hop {hop} via {shared} from "
        f"{(parent.name if parent is not None else '?')!r}"
    ]


def _reject(candidate: RagReactionCandidate, reasons: Sequence[str]) -> None:
    candidate.status = STATUS_REJECTED
    candidate.scope_membership = SCOPE_REJECTED
    candidate.reasons = list(reasons)
    candidate.chain = {}


# ---------------------------------------------------------------------------
# Lineage (C-035) -- ATTRIBUTION ONLY.
#
# Nothing below participates in the verdict. It is a pure read of a decision
# ``_accept`` / ``_reject`` already made, so recording it cannot change what is
# admitted, rejected or exported. The gate keeps its decision surface
# (``claim_identity`` / ``merge_key`` / ``AdmissionPolicy`` / ``_gap_type_verdict``)
# exactly as it was.
# ---------------------------------------------------------------------------
def admission_lineage_entry(
    candidate: RagReactionCandidate,
) -> Optional[LineageEntry]:
    """The gate's attribution for a candidate it ADMITTED -- or ``None``.

    ``None`` for anything the gate did not accept, and for an accepted claim whose
    paper cannot be named. A :data:`~t2pw.pipeline.lineage.SOURCED_ORIGINS` origin
    must identify its supporting record, and a row asserting provenance it does not
    have is worse than a row with none: an unciteable claim gets no entry rather
    than an anonymous one.

    Every field is read off the verdict, never inferred:

    ``origin`` is ``rag_literature`` because the claim came from a retrieved
    paper -- the gate admitted content, it did not author any. ``support`` is
    ``direct``: a candidate only reaches ``accepted`` by passing
    :func:`_screen_candidate` with zero reasons, and that runs
    :func:`validate_evidence_span`, which refuses an empty span outright and
    otherwise requires ONE bounded span to state the whole claim -- substrates,
    products, direction and catalysts. So a named passage does state this fact,
    which is what ``direct`` means. The span is checked anyway rather than assumed,
    because "the caller only ever passes an accepted candidate" is a caller
    contract, not a property of this function.

    ``paper_explicit`` is ``not_evaluated``, and that is the point of the record.
    This gate compares a claim to the REQUESTED pathway and organism; it never asks
    whether the SUPPLIED paper stated it, so it may not answer that question.
    ``not_evaluated`` is never ``not_explicit``. R-003 found a RAG-imported row
    without a carrier to be indistinguishable between "paper-explicit" and
    "predates instrumentation"; an entry that says which stage imported it, and
    says in the same breath that paper-explicitness was not evaluated here,
    separates the two.

    ``reason`` names **which gap the claim was admitted against**, plus the
    complete set it was retrieved for. R-004 established the verdict is keyed on
    ``(gap_id, claim_identity())`` while at least one consumer compares on a key
    that omits ``gap_id``, producing false "reintroduction" findings; a stored
    per-row record of the gap is what makes such a dispute decidable rather than
    arguable.

    ``review_required`` is ``False``. The gate's verdict is *admit*, and a lineage
    that contradicted it would be this record changing what ships -- exactly what
    an attribution writer may not do. Real nuance the gate observed (a non-exact
    organism match, a chained rather than direct admission) goes to ``uncertainty``,
    which is documentation at every support level and gates nothing.
    """
    if candidate.status != STATUS_ACCEPTED:
        return None
    source = _admission_source(candidate)
    if source is None:
        return None

    primary = _text(candidate.gap_id)
    others = [g for g in candidate.gap_ids if _text(g) and _text(g) != primary]
    verdict = _text(candidate.reasons[0]) if candidate.reasons else ""
    reason = f"admission gate accepted this claim against gap {primary!r}"
    if others:
        reason += f" (also retrieved for {', '.join(sorted(others))})"
    if verdict:
        reason += f"; {verdict}"

    notes: List[str] = []
    if candidate.organism_match != ORGANISM_MATCH:
        notes.append(
            f"organism match is {candidate.organism_match!r} against requested "
            f"{candidate.requested_organism!r}"
        )
    hop = int(candidate.chain.get("hop", 0) or 0)
    if hop:
        notes.append(
            f"admitted at chain hop {hop}, not directly on the gap's own target"
        )
    if candidate.requested_pathway_match != PATHWAY_MATCH:
        notes.append(
            f"requested-pathway match is {candidate.requested_pathway_match!r}"
        )

    return LineageEntry(
        stage="rag_admission",
        origin="rag_literature",
        support="direct" if _text(candidate.evidence_span) else "indirect",
        paper_explicit="not_evaluated",
        reason=reason,
        review_required=False,
        uncertainty="; ".join(notes),
        sources=(source,),
    )


def _admission_source(candidate: RagReactionCandidate) -> Optional[LineageSource]:
    """A POINTER to the paper and passage backing ``candidate``, or ``None``.

    A pointer, never a copy: the passage itself already rides in the candidate's
    ``evidence`` / ``evidence_span`` and on the emitted row, and this module's size
    history is why that is not duplicated here. ``source_id`` is checked only for
    non-emptiness -- accepting an identifier because its FORMAT looks right is a
    standing invariant violation, so no shape test is applied to it.
    """
    source_id = candidate.source_id()
    uri = _text(candidate.evidence.get("source_uri")) or _text(
        candidate.source_paper.get("uri")
    )
    if not (source_id or uri):
        return None
    return LineageSource(
        source_id=source_id,
        source_type=_text(candidate.source_paper.get("source_type")) or "paper",
        uri=uri,
        locator=_text(candidate.evidence.get("chunk_id"))
        or _text(candidate.evidence.get("section")),
    )


def _reason_code(reason: str) -> str:
    return str(reason or "").split(":", 1)[0].strip() or "unknown"


def _build_report(
    accepted: Sequence[RagReactionCandidate],
    rejected: Sequence[RagReactionCandidate],
    policy: AdmissionPolicy,
    routed: Sequence[Dict[str, str]],
    resolutions: Sequence[TypedResolution] = (),
) -> AdmissionReport:
    cap = policy.max_report_entries
    snippet = policy.evidence_snippet_chars
    reason_counts: Dict[str, int] = {}
    by_gap: Dict[str, Dict[str, int]] = {}
    for bucket, candidates in (("accepted", accepted), ("rejected", rejected)):
        for candidate in candidates:
            gap_row = by_gap.setdefault(
                candidate.gap_id or "<none>", {"accepted": 0, "rejected": 0}
            )
            gap_row[bucket] = gap_row.get(bucket, 0) + 1
            for reason in candidate.reasons:
                code = _reason_code(reason)
                reason_counts[code] = reason_counts.get(code, 0) + 1

    by_hop: Dict[str, int] = {}
    direct = 0
    for candidate in accepted:
        hop = int(candidate.chain.get("hop", 0) or 0)
        if hop == 0:
            direct += 1
        else:
            by_hop[str(hop)] = by_hop.get(str(hop), 0) + 1

    seen_routes: Set[Tuple[str, str]] = set()
    unique_routes: List[Dict[str, str]] = []
    for entry in routed:
        key = (entry.get("gap_id", ""), entry.get("route", ""))
        if key in seen_routes:
            continue
        seen_routes.add(key)
        unique_routes.append(dict(entry))

    return AdmissionReport(
        accepted=[c.to_dict(snippet_chars=snippet) for c in list(accepted)[:cap]],
        rejected=[c.to_dict(snippet_chars=snippet) for c in list(rejected)[:cap]],
        counts={
            "accepted": len(accepted),
            "rejected": len(rejected),
            "considered": len(accepted) + len(rejected),
        },
        reason_counts=dict(sorted(reason_counts.items())),
        by_gap={k: by_gap[k] for k in sorted(by_gap)},
        truncated={
            "accepted": max(0, len(accepted) - cap),
            "rejected": max(0, len(rejected) - cap),
        },
        policy=policy.to_dict(),
        rules=[{"rule": name, "requires": text} for name, text in ADMISSION_RULES],
        acceptance_breakdown={
            "direct": direct,
            "chained": len(accepted) - direct,
            "chained_by_hop": {k: by_hop[k] for k in sorted(by_hop)},
        },
        resolutions=[r.to_dict(snippet_chars=snippet) for r in list(resolutions)[:cap]],
        recommended_route=unique_routes,
        proposals=list(resolutions),
    )
