"""The pinned gold set: schema, loader, and the name-matching rules.

A *gold case* is one paper plus everything a validator needs in order to decide
whether a pathway extracted from that paper is scientifically acceptable. Every
biological expectation carries a ``quote`` -- a verbatim span from the paper --
so a disagreement between the pipeline and the benchmark can always be settled by
reading the paper rather than by arguing about the benchmark.

Three design rules, each of which exists because the obvious alternative produces
a benchmark that rewards the wrong thing:

**1. Expectations are what the paper supports, not what the pathway is.**
``minimum_connected_reactions`` for a regulation review is 3, not the 9 steps of
the textbook Raetz pathway, because the review does not lay out 9 steps. Scoring
against the textbook would mark a faithful extraction as a failure and reward a
pipeline that hallucinated the missing steps from its own weights. The gold set
grades *fidelity to the source*, and ``expected_export`` records separately
whether a complete importable pathway is even obtainable from this paper.

**2. Enzymes come in two tiers.** ``expected_enzymes`` are the ones the paper
describes as catalysing a step; missing one is a coverage failure.
``acceptable_enzymes`` are regulators and accessory proteins the paper mentions;
their presence is neither required nor an error. Without the second tier every
correct extraction looks over-broad.

**3. Forbidden identifiers are the load-bearing half.** ``forbidden_identifiers``
lists strings that are *not real distinct chemical or protein identities* --
placeholder product names like ``LpxA product``, cofactors that keep getting
filed as proteins (``coenzyme A``), strain and plasmid labels, section headings.
Emitting one of these carrying a real external accession is the single worst
outcome the pipeline can produce, because every structural gate passes and the
result is silently wrong. Acceptance priority 1 is defined on this list.

Name matching
-------------

Payload entity names are written the way the paper writes them, so they arrive
with Greek letters, en dashes, charge notation and parenthetical abbreviations.
:func:`normalize_name` folds those into a comparable form and
:func:`match_term` reports *which* rule matched, so a coverage number can always
be audited down to the pair of strings that produced it. Matching is deliberately
conservative: an expected term shorter than :data:`MIN_SUBSTRING_TERM` never
matches by containment, because a 2-character term like ``AA`` would otherwise
match half the payload.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


#: Bumped whenever the *schema* changes in a way that invalidates stored results.
#: Adding a case does not bump it; adding a required field does.
GOLD_SET_VERSION = "2026-08-01.1"

#: Mechanistic relevance, most relevant first. ``core`` means the paper's subject
#: IS the requested pathway's chemistry; ``partial`` means it covers a genuine
#: fragment (one enzyme, one transport step, one regulatory loop); ``context_only``
#: means the pathway is named but no step of it is described mechanistically.
RELEVANCE_CORE = "core"
RELEVANCE_PARTIAL = "partial"
RELEVANCE_CONTEXT_ONLY = "context_only"
RELEVANCE_VALUES: Tuple[str, ...] = (RELEVANCE_CORE, RELEVANCE_PARTIAL, RELEVANCE_CONTEXT_ONLY)

#: Whether a *complete, ID-resolvable, connected* pathway can be built from this
#: paper at all. ``partial_only`` is not a lowered bar for the pipeline -- it is a
#: statement about the source, and it moves the paper out of the strict-PWML
#: denominator so a 0% strict rate is not blamed on the exporter.
EXPORT_STRICT = "strict_exportable"
EXPORT_PARTIAL = "partial_only"
EXPORT_VALUES: Tuple[str, ...] = (EXPORT_STRICT, EXPORT_PARTIAL)

#: Below this length an expected term may only match by exact normalized equality.
MIN_SUBSTRING_TERM = 4

#: How a term matched, most exact first. Carried into the report so every
#: coverage number can be audited back to the two strings that produced it.
MATCH_EXACT = "exact"
MATCH_ALIAS = "alias"
MATCH_CONTAINS = "contains"
MATCH_NONE = ""


# ---------------------------------------------------------------------------
# Normalization.
# ---------------------------------------------------------------------------
#: Greek letters are spelled out rather than stripped: "beta-hydroxyacyl-ACP" and
#: "β-hydroxyacyl-ACP" must compare equal, but "α-ketoglutarate" and
#: "β-ketoglutarate" must NOT -- stripping the letter would collapse two
#: different molecules onto one name.
_GREEK: Dict[str, str] = {
    "α": "alpha", "Α": "alpha",
    "β": "beta", "Β": "beta",
    "γ": "gamma", "Γ": "gamma",
    "δ": "delta", "Δ": "delta",
    "ε": "epsilon", "Ε": "epsilon",
    "κ": "kappa", "Κ": "kappa",
    "λ": "lambda", "Λ": "lambda",
    "μ": "mu", "Μ": "mu",
    "σ": "sigma", "Σ": "sigma",
    "ω": "omega", "Ω": "omega",
}

_DASHES = {"‐", "‑", "‒", "–", "—", "―", "−"}

#: Stripped before comparison. These are formatting, not identity: an entity
#: called "LpxC (deacetylase)" is the same protein as "LpxC".
_TRAILING_NOISE = re.compile(r"\s*\((?:protein|enzyme|gene|complex|abbrev\.?)\)\s*$", re.IGNORECASE)


def canonical_text(value: Any) -> str:
    """Collapse unicode punctuation and whitespace without changing identity."""

    text = str(value if value is not None else "")
    text = unicodedata.normalize("NFKC", text)
    out = []
    for char in text:
        if char in _DASHES:
            out.append("-")
        elif char == " ":
            out.append(" ")
        else:
            out.append(char)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def normalize_name(value: Any) -> str:
    """Fold an entity or term name into its comparable form.

    Greek letters become their Latin names, punctuation becomes a single space,
    and case is dropped. Digits are kept: ``MK-7`` and ``MK-4`` must not collide.
    """

    text = canonical_text(value)
    text = _TRAILING_NOISE.sub("", text)
    for greek, latin in _GREEK.items():
        text = text.replace(greek, latin)
    text = text.casefold()
    # Charge notation is identity-bearing for an ion but noise for a name match;
    # "NAD+" and "NAD" are the same entity for coverage purposes.
    text = re.sub(r"(?<=[a-z0-9])\s*[⁺⁻+]\s*$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: str) -> List[str]:
    return [tok for tok in normalize_name(value).split(" ") if tok]


def contains_term(haystack: str, needle: str) -> bool:
    """Whether ``needle`` occurs in ``haystack`` on whole-token boundaries.

    Token boundaries matter: without them ``ala`` (5-aminolevulinic acid's
    abbreviation) matches ``alanine``, and the anchor check silently passes on a
    pathway that never mentions ALA.
    """

    hay = normalize_name(haystack)
    ndl = normalize_name(needle)
    if not hay or not ndl or len(ndl.replace(" ", "")) < MIN_SUBSTRING_TERM:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(ndl)}(?![a-z0-9])", hay) is not None


# ---------------------------------------------------------------------------
# Gold-set records.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GoldTerm:
    """One expectation, with the paper span that justifies it.

    ``quote`` is not decoration. A benchmark whose expectations cannot be traced
    to the source is indistinguishable from a benchmark that encodes the author's
    memory of the biology, and the whole point here is to grade fidelity to the
    paper.
    """

    name: str
    aliases: Tuple[str, ...] = ()
    quote: str = ""
    #: Free-text: the gene symbol, EC number or UniProt accession the paper
    #: states. Empty when the paper does not state one -- which is itself a fact
    #: the validator uses, since an enzyme with no stated identity cannot be
    #: held against the mapper.
    identifier: str = ""
    role: str = ""

    def all_names(self) -> Tuple[str, ...]:
        return (self.name, *self.aliases)

    def match(self, candidate: Any) -> str:
        """How ``candidate`` matches this term, or :data:`MATCH_NONE`."""

        norm = normalize_name(candidate)
        if not norm:
            return MATCH_NONE
        if norm == normalize_name(self.name):
            return MATCH_EXACT
        for alias in self.aliases:
            if norm == normalize_name(alias):
                return MATCH_ALIAS
        for name in self.all_names():
            if contains_term(candidate, name) or contains_term(name, candidate):
                return MATCH_CONTAINS
        return MATCH_NONE

    def match_any(self, candidates: Iterable[Any]) -> Tuple[str, str]:
        """Best match over ``candidates`` as ``(rule, matched_candidate)``."""

        best_rule, best_value = MATCH_NONE, ""
        order = {MATCH_EXACT: 3, MATCH_ALIAS: 2, MATCH_CONTAINS: 1, MATCH_NONE: 0}
        for candidate in candidates:
            rule = self.match(candidate)
            if order[rule] > order[best_rule]:
                best_rule, best_value = rule, canonical_text(candidate)
                if rule == MATCH_EXACT:
                    break
        return best_rule, best_value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "quote": self.quote,
            "identifier": self.identifier,
            "role": self.role,
        }


@dataclass(frozen=True)
class SupportedReaction:
    """One reaction the paper actually states, as a matchable signature.

    This exists because carrying an ``evidence`` field is not the same as being
    supported. The old check asked "does this reaction cite *something*?", which
    a bare string like ``"the paper says"`` satisfies, as does an arbitrary
    ``source_ref`` pointing at a chunk that discusses something else entirely. It
    could therefore report a hallucinated reaction as fully sourced. That check
    still exists -- renamed to say what it measures, a *source carrier* -- and
    this is the one that measures support: a retained reaction is supported only
    when its chemistry matches a signature read out of the paper by hand, and
    the signature's ``quote`` is verifiable in the stored paper text.

    Participants are :class:`GoldTerm` so a signature can accept the spellings a
    paper actually uses (``lipid IV A`` for ``lipid IVA``) without accepting a
    different molecule.
    """

    inputs: Tuple[GoldTerm, ...]
    outputs: Tuple[GoldTerm, ...]
    enzyme: Optional[GoldTerm] = None
    reversible: bool = False
    quote: str = ""
    label: str = ""

    def quote_hash(self) -> str:
        """Stable hash of the normalized quote, for cross-run comparison."""

        return hashlib.sha256(fold_for_quote(self.quote).encode("utf-8")).hexdigest()[:16] if self.quote else ""

    def describe(self) -> str:
        left = " + ".join(t.name for t in self.inputs) or "?"
        right = " + ".join(t.name for t in self.outputs) or "?"
        arrow = "<->" if self.reversible else "->"
        enzyme = f" [{self.enzyme.name}]" if self.enzyme else ""
        return f"{left} {arrow} {right}{enzyme}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label or self.describe(),
            "signature": self.describe(),
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "enzyme": self.enzyme.to_dict() if self.enzyme else None,
            "reversible": self.reversible,
            "quote": self.quote,
            "quote_hash": self.quote_hash(),
        }


def fold_for_quote(value: Any) -> str:
    """Fold a span for quote verification against stored paper text.

    All non-alphanumeric characters are dropped, not just collapsed. The cached
    full text carries italic-tag stripping artifacts that insert spaces *inside*
    tokens -- ``lipid IV A``, ``R -3-hydroxymyristoyl-ACP``, ``UDP- N -acetyl-
    glucosamine``, ``sigma 32`` -- so any whitespace-preserving comparison fails
    on spans that are genuinely present. At the length of a real quote the
    collision risk from dropping punctuation is negligible.
    """

    return re.sub(r"[^a-z0-9]+", "", canonical_text(value).casefold())


@dataclass(frozen=True)
class ForbiddenIdentifier:
    """A string that must never be emitted as a real, distinct identity.

    ``kind`` explains *why* it is forbidden, which is what makes a violation
    actionable rather than just a red number:

    ``placeholder_product``  a reaction output named after its enzyme rather than
                             a molecule ("LpxA product") -- the extractor did not
                             read a product, so the reaction is unsupported
    ``cofactor_as_protein``  a small molecule filed under ``entities.proteins``
                             ("coenzyme A") -- the downstream mapper will then go
                             looking for a UniProt accession for a metabolite
    ``strain_or_construct``  a strain, plasmid or construct label treated as a
                             biological entity
    ``heading_or_prose``     a section heading or sentence fragment
    ``regulator_as_metabolite`` a protein filed under ``entities.compounds``
                             (sigma factors are the recurring case)
    """

    name: str
    kind: str
    reason: str = ""
    aliases: Tuple[str, ...] = ()

    def as_term(self) -> GoldTerm:
        return GoldTerm(name=self.name, aliases=self.aliases)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "reason": self.reason,
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True)
class GoldCase:
    """Everything the validator needs about one pinned paper."""

    paper_id: str
    title: str
    requested_pathway: str
    requested_organism: str
    source_uri: str = ""
    #: What the paper is *actually* about, which is not always what was
    #: requested. A menaquinone query returned a Listeria structural paper; the
    #: organism check has to grade against reality, not against the query.
    actual_organism: str = ""
    mechanistic_relevance: str = RELEVANCE_PARTIAL
    relevance_note: str = ""

    expected_pathway_anchors: Tuple[GoldTerm, ...] = ()
    expected_enzymes: Tuple[GoldTerm, ...] = ()
    acceptable_enzymes: Tuple[GoldTerm, ...] = ()
    expected_substrates: Tuple[GoldTerm, ...] = ()
    expected_products: Tuple[GoldTerm, ...] = ()
    #: Reactions the paper actually states, with a verifiable quote each.
    supported_reactions: Tuple[SupportedReaction, ...] = ()
    #: Whether :attr:`supported_reactions` is EXHAUSTIVE for this paper.
    #:
    #: This flag is what makes "unsupported" a defensible word. When it is
    #: ``False`` (the default), the signature list is a hand-read subset -- only
    #: the steps where the paper names both sides -- so a retained reaction that
    #: matches nothing may be (a) genuinely invented, (b) a step the paper states
    #: in a form no signature was written for, or (c) a legitimate cross-paper
    #: addition from RAG synthesis, which cannot match a seed-paper signature by
    #: construction. Those are indistinguishable from the signature list alone,
    #: so unmatched rows are reported as *unattributed* and do NOT count as
    #: scientific errors. Recall is unaffected: a declared signature the payload
    #: fails to contain is a real miss either way.
    #:
    #: Only set this ``True`` after reading the whole paper and writing a
    #: signature for every reaction in it -- and note that it is incompatible
    #: with multi-paper RAG synthesis unless the run is seed-only.
    supported_reactions_complete: bool = False

    forbidden_organisms: Tuple[str, ...] = ()
    forbidden_identifiers: Tuple[ForbiddenIdentifier, ...] = ()

    min_connected_reactions: int = 1
    #: Ceiling on retained reactions, for **negative-control** cases only.
    #: ``0`` means the paper contains no reaction of the requested pathway and the
    #: only correct output is an empty one. This is the sharpest hallucination
    #: detector in the set: a floor rewards recall and can always be satisfied by
    #: inventing chemistry, whereas a ceiling of zero can only be satisfied by
    #: declining to invent it. ``None`` means no ceiling.
    max_retained_reactions: Optional[int] = None
    unknown_backed_proteins_acceptable: bool = False
    unknown_backed_rationale: str = ""
    expected_export: str = EXPORT_PARTIAL
    export_rationale: str = ""
    notes: str = ""

    # -- convenience ------------------------------------------------------
    @property
    def is_strict_expected(self) -> bool:
        return self.expected_export == EXPORT_STRICT

    @property
    def is_negative_control(self) -> bool:
        """A paper that must yield nothing -- a retrieval false positive.

        Kept out of the "mechanistically relevant" headcount and out of every
        coverage denominator, because grading recall on a paper with no pathway
        in it is meaningless. It is graded on one thing only: did the pipeline
        decline to invent a pathway?
        """

        return self.max_retained_reactions == 0

    @property
    def expected_metabolites(self) -> Tuple[GoldTerm, ...]:
        return (*self.expected_substrates, *self.expected_products)

    @property
    def all_enzymes(self) -> Tuple[GoldTerm, ...]:
        return (*self.expected_enzymes, *self.acceptable_enzymes)

    def forbidden_match(self, candidate: Any) -> Optional[ForbiddenIdentifier]:
        """The forbidden identifier ``candidate`` names, if any.

        Exact and alias matches only. Containment is deliberately refused here:
        ``coenzyme A`` is forbidden as a *protein*, but ``coenzyme A ligase`` is a
        perfectly real enzyme, and a containment rule would condemn it.
        """

        norm = normalize_name(candidate)
        if not norm:
            return None
        for entry in self.forbidden_identifiers:
            if norm == normalize_name(entry.name):
                return entry
            if any(norm == normalize_name(alias) for alias in entry.aliases):
                return entry
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "source_uri": self.source_uri,
            "requested_pathway": self.requested_pathway,
            "requested_organism": self.requested_organism,
            "actual_organism": self.actual_organism,
            "mechanistic_relevance": self.mechanistic_relevance,
            "relevance_note": self.relevance_note,
            "expected_pathway_anchors": [t.to_dict() for t in self.expected_pathway_anchors],
            "expected_enzymes": [t.to_dict() for t in self.expected_enzymes],
            "acceptable_enzymes": [t.to_dict() for t in self.acceptable_enzymes],
            "expected_substrates": [t.to_dict() for t in self.expected_substrates],
            "expected_products": [t.to_dict() for t in self.expected_products],
            "supported_reactions": [r.to_dict() for r in self.supported_reactions],
            "supported_reactions_complete": self.supported_reactions_complete,
            "forbidden_organisms": list(self.forbidden_organisms),
            "forbidden_identifiers": [f.to_dict() for f in self.forbidden_identifiers],
            "min_connected_reactions": self.min_connected_reactions,
            "max_retained_reactions": self.max_retained_reactions,
            "is_negative_control": self.is_negative_control,
            "unknown_backed_proteins_acceptable": self.unknown_backed_proteins_acceptable,
            "unknown_backed_rationale": self.unknown_backed_rationale,
            "expected_export": self.expected_export,
            "export_rationale": self.export_rationale,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class GoldSet:
    """An ordered, id-addressable collection of gold cases."""

    version: str
    description: str
    cases: Tuple[GoldCase, ...]
    source_path: str = ""

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def by_id(self, paper_id: str) -> Optional[GoldCase]:
        wanted = canonical_text(paper_id).casefold()
        for case in self.cases:
            if case.paper_id.casefold() == wanted:
                return case
        return None

    @property
    def paper_ids(self) -> Tuple[str, ...]:
        return tuple(case.paper_id for case in self.cases)

    @property
    def strict_expected_ids(self) -> Tuple[str, ...]:
        return tuple(c.paper_id for c in self.cases if c.is_strict_expected)

    def topics_lines(self) -> List[str]:
        """The ``topics.txt`` body that pins exactly these papers **with scope**.

        Each line is ``PAPERID | requested pathway | requested organism``, which
        ``fetch.parse_topics`` reads as a pinned spec carrying a scope: fetched by
        id with zero search calls, eligibility score bypassed
        (``OUTCOME_PINNED_OVERRIDE``), and the request preserved exactly.

        The scope fields are not decoration. A *bare* pinned id yields
        ``topic=""`` and ``organism=""``, which ``_as_batch_paper`` copies into
        ``requested_pathway`` / ``requested_organism`` -- so extraction is handed
        no scope, Stage-0 reconciliation has no request to contradict, and the
        screening record is computed against an empty pathway. That is what made
        ``runs/2026-08-01_1724`` unusable as an acceptance run despite fetching
        all ten papers correctly.
        """

        lines = [
            "# GENERATED from the pinned gold set -- do not hand-edit.",
            f"# gold set: {self.version}",
            "#",
            "# Format: PAPERID | requested pathway | requested organism",
            "#",
            "# fetch.parse_topics reads a first field that looks like a paper id as a",
            "# PINNED spec and takes the remaining fields as the requested scope:",
            "# fetched by id, zero search calls, score bypassed, request preserved.",
            "# A bare id would fetch the same paper with an EMPTY scope, which is not",
            "# a valid acceptance run -- see GoldSet.topics_lines.",
            "#",
            "# Regenerate with: python scripts/bench_acceptance.py --write-topics <path>",
            "",
        ]
        width = max((len(case.paper_id) for case in self.cases), default=12)
        pathway_width = max((len(case.requested_pathway) for case in self.cases), default=20)
        for case in self.cases:
            lines.append(
                f"{case.paper_id:<{width}} | {case.requested_pathway:<{pathway_width}} | "
                f"{case.requested_organism}"
            )
        return lines


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------
class GoldSetError(ValueError):
    """The gold-set file is malformed. Always names the offending case."""


def _terms(raw: Any, *, where: str) -> Tuple[GoldTerm, ...]:
    if raw in (None, ""):
        return ()
    if not isinstance(raw, list):
        raise GoldSetError(f"{where}: expected a list, found {type(raw).__name__}")
    out: List[GoldTerm] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            out.append(GoldTerm(name=item))
            continue
        if not isinstance(item, dict):
            raise GoldSetError(f"{where}[{index}]: expected an object or string")
        name = canonical_text(item.get("name"))
        if not name:
            raise GoldSetError(f"{where}[{index}]: missing 'name'")
        out.append(
            GoldTerm(
                name=name,
                aliases=tuple(canonical_text(a) for a in item.get("aliases") or () if canonical_text(a)),
                quote=str(item.get("quote") or ""),
                identifier=canonical_text(item.get("identifier")),
                role=canonical_text(item.get("role")),
            )
        )
    return tuple(out)


def _supported_reactions(raw: Any, *, where: str) -> Tuple[SupportedReaction, ...]:
    if raw in (None, ""):
        return ()
    if not isinstance(raw, list):
        raise GoldSetError(f"{where}: expected a list")
    out: List[SupportedReaction] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise GoldSetError(f"{where}[{index}]: expected an object")
        spot = f"{where}[{index}]"
        inputs = _terms(item.get("inputs"), where=f"{spot}.inputs")
        outputs = _terms(item.get("outputs"), where=f"{spot}.outputs")
        if not inputs or not outputs:
            raise GoldSetError(f"{spot}: a supported reaction needs both inputs and outputs")
        enzyme_raw = item.get("enzyme")
        enzyme: Optional[GoldTerm] = None
        if enzyme_raw not in (None, ""):
            enzyme = _terms([enzyme_raw], where=f"{spot}.enzyme")[0]
        quote = str(item.get("quote") or "")
        if not quote.strip():
            raise GoldSetError(
                f"{spot}: a supported reaction must carry the quote that supports it, "
                "or it is an assertion rather than evidence"
            )
        out.append(
            SupportedReaction(
                inputs=inputs,
                outputs=outputs,
                enzyme=enzyme,
                reversible=bool(item.get("reversible", False)),
                quote=quote,
                label=canonical_text(item.get("label")),
            )
        )
    return tuple(out)


def _forbidden(raw: Any, *, where: str) -> Tuple[ForbiddenIdentifier, ...]:
    if raw in (None, ""):
        return ()
    if not isinstance(raw, list):
        raise GoldSetError(f"{where}: expected a list")
    out: List[ForbiddenIdentifier] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise GoldSetError(f"{where}[{index}]: expected an object")
        name = canonical_text(item.get("name"))
        kind = canonical_text(item.get("kind"))
        if not name:
            raise GoldSetError(f"{where}[{index}]: missing 'name'")
        if not kind:
            raise GoldSetError(f"{where}[{index}] ({name}): missing 'kind'")
        out.append(
            ForbiddenIdentifier(
                name=name,
                kind=kind,
                reason=str(item.get("reason") or ""),
                aliases=tuple(canonical_text(a) for a in item.get("aliases") or () if canonical_text(a)),
            )
        )
    return tuple(out)


def _case(raw: Mapping[str, Any], *, index: int) -> GoldCase:
    paper_id = canonical_text(raw.get("paper_id"))
    if not paper_id:
        raise GoldSetError(f"cases[{index}]: missing 'paper_id'")
    where = f"cases[{index}] ({paper_id})"

    relevance = canonical_text(raw.get("mechanistic_relevance")) or RELEVANCE_PARTIAL
    if relevance not in RELEVANCE_VALUES:
        raise GoldSetError(f"{where}: mechanistic_relevance must be one of {RELEVANCE_VALUES}")
    export = canonical_text(raw.get("expected_export")) or EXPORT_PARTIAL
    if export not in EXPORT_VALUES:
        raise GoldSetError(f"{where}: expected_export must be one of {EXPORT_VALUES}")

    try:
        min_connected = int(raw.get("min_connected_reactions", 1))
    except (TypeError, ValueError):
        raise GoldSetError(f"{where}: min_connected_reactions must be an integer") from None
    if min_connected < 0:
        raise GoldSetError(f"{where}: min_connected_reactions must not be negative")

    max_retained_raw = raw.get("max_retained_reactions", None)
    max_retained: Optional[int]
    if max_retained_raw is None:
        max_retained = None
    else:
        try:
            max_retained = int(max_retained_raw)
        except (TypeError, ValueError):
            raise GoldSetError(f"{where}: max_retained_reactions must be an integer or null") from None
        if max_retained < 0:
            raise GoldSetError(f"{where}: max_retained_reactions must not be negative")
        if max_retained < min_connected:
            raise GoldSetError(
                f"{where}: max_retained_reactions ({max_retained}) is below "
                f"min_connected_reactions ({min_connected}); no result could satisfy both"
            )

    pathway = canonical_text(raw.get("requested_pathway"))
    if not pathway:
        raise GoldSetError(f"{where}: missing 'requested_pathway'")

    case = GoldCase(
        paper_id=paper_id,
        title=canonical_text(raw.get("title")),
        source_uri=canonical_text(raw.get("source_uri")),
        requested_pathway=pathway,
        requested_organism=canonical_text(raw.get("requested_organism")),
        actual_organism=canonical_text(raw.get("actual_organism")),
        mechanistic_relevance=relevance,
        relevance_note=str(raw.get("relevance_note") or ""),
        expected_pathway_anchors=_terms(raw.get("expected_pathway_anchors"), where=f"{where}.expected_pathway_anchors"),
        expected_enzymes=_terms(raw.get("expected_enzymes"), where=f"{where}.expected_enzymes"),
        acceptable_enzymes=_terms(raw.get("acceptable_enzymes"), where=f"{where}.acceptable_enzymes"),
        expected_substrates=_terms(raw.get("expected_substrates"), where=f"{where}.expected_substrates"),
        expected_products=_terms(raw.get("expected_products"), where=f"{where}.expected_products"),
        supported_reactions=_supported_reactions(
            raw.get("supported_reactions"), where=f"{where}.supported_reactions"
        ),
        supported_reactions_complete=bool(raw.get("supported_reactions_complete", False)),
        forbidden_organisms=tuple(
            canonical_text(o) for o in raw.get("forbidden_organisms") or () if canonical_text(o)
        ),
        forbidden_identifiers=_forbidden(raw.get("forbidden_identifiers"), where=f"{where}.forbidden_identifiers"),
        min_connected_reactions=min_connected,
        max_retained_reactions=max_retained,
        unknown_backed_proteins_acceptable=bool(raw.get("unknown_backed_proteins_acceptable", False)),
        unknown_backed_rationale=str(raw.get("unknown_backed_rationale") or ""),
        expected_export=export,
        export_rationale=str(raw.get("export_rationale") or ""),
        notes=str(raw.get("notes") or ""),
    )

    # A case with no anchors cannot fail the anchor check, which would make it
    # look like a free pass in the coverage column. Refuse it at load time rather
    # than let it quietly inflate the score.
    #
    # The one legitimate exception is a negative control: a paper with no pathway
    # content has nothing to anchor on, and demanding an anchor would force the
    # author to invent one. Such a case must instead carry an explicit ceiling,
    # so it still has teeth -- it is graded on refusing to hallucinate.
    if not case.expected_pathway_anchors and not case.is_negative_control:
        raise GoldSetError(
            f"{where}: at least one expected_pathway_anchor is required, "
            "unless the case is a negative control (max_retained_reactions: 0)"
        )
    if case.is_negative_control and case.mechanistic_relevance != RELEVANCE_CONTEXT_ONLY:
        raise GoldSetError(
            f"{where}: a negative control (max_retained_reactions: 0) must be "
            f"declared '{RELEVANCE_CONTEXT_ONLY}', not '{case.mechanistic_relevance}'"
        )
    return case


def pinned_gold_set_path() -> Path:
    """Location of the checked-in pinned gold set."""

    return Path(__file__).resolve().parent / "gold" / "pinned_v1.json"


def load_gold_set(path: Optional[Any] = None) -> GoldSet:
    """Read and validate a gold-set file.

    Raises :class:`GoldSetError` on any malformed case rather than silently
    dropping it: a gold set that quietly loses a case reports a better score than
    it earned.
    """

    target = Path(path) if path else pinned_gold_set_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise GoldSetError(f"gold set not found: {target}") from None
    except json.JSONDecodeError as exc:
        raise GoldSetError(f"gold set {target} is not valid JSON: {exc}") from None

    if not isinstance(raw, dict):
        raise GoldSetError(f"gold set {target}: top level must be an object")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise GoldSetError(f"gold set {target}: 'cases' must be a non-empty list")

    cases = tuple(_case(item, index=i) for i, item in enumerate(cases_raw))

    seen: Dict[str, int] = {}
    for i, case in enumerate(cases):
        key = case.paper_id.casefold()
        if key in seen:
            raise GoldSetError(f"gold set {target}: duplicate paper_id {case.paper_id} (cases {seen[key]} and {i})")
        seen[key] = i

    return GoldSet(
        version=canonical_text(raw.get("version")) or GOLD_SET_VERSION,
        description=str(raw.get("description") or ""),
        cases=cases,
        source_path=str(target),
    )


__all__ = [
    "EXPORT_PARTIAL",
    "EXPORT_STRICT",
    "EXPORT_VALUES",
    "GOLD_SET_VERSION",
    "ForbiddenIdentifier",
    "GoldCase",
    "GoldSet",
    "GoldSetError",
    "GoldTerm",
    "MATCH_ALIAS",
    "MATCH_CONTAINS",
    "MATCH_EXACT",
    "MATCH_NONE",
    "RELEVANCE_CONTEXT_ONLY",
    "RELEVANCE_CORE",
    "RELEVANCE_PARTIAL",
    "RELEVANCE_VALUES",
    "SupportedReaction",
    "canonical_text",
    "contains_term",
    "fold_for_quote",
    "load_gold_set",
    "normalize_name",
    "pinned_gold_set_path",
]
