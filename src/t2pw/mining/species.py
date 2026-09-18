"""The species guardrail.

A mining run declares one target species. Every claim the run extracts is
classified into exactly one species state, and only
:data:`SPECIES_TARGET_SUPPORTED` claims may enter the accepted target-species
graph.

The rule this module exists to enforce
--------------------------------------
**A shared enzyme name, a shared gene symbol or an ortholog relationship never
transfers a reaction between species.** Evidence read in mouse stays mouse
evidence, however confidently a model asserts the human orthologue does the same
thing. Such a claim is retained -- labelled, auditable, and available as a lead
-- but it is not an established target-species reaction.

Normalization here is *local*: lowercasing, whitespace collapse, the common
abbreviated binomial (``A. thaliana`` -> ``arabidopsis thaliana``) and a short
common-name table. It deliberately does **not** call NCBI Taxonomy or any other
external service -- version 1 admits on paper evidence alone, and a normalized
string must never be mistaken for external verification.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# The species states. Exhaustive and explicit -- there is no "probably fine".
# ---------------------------------------------------------------------------
SPECIES_TARGET_SUPPORTED = "target_species_supported"
SPECIES_NON_TARGET = "non_target_species"
SPECIES_MULTIPLE = "multiple_species"
SPECIES_AMBIGUOUS = "species_ambiguous"
SPECIES_NOT_STATED = "species_not_stated"
SPECIES_IN_VITRO = "in_vitro_or_cell_free"
SPECIES_INFERRED_ONLY = "inferred_only"

SPECIES_STATES = (
    SPECIES_TARGET_SUPPORTED,
    SPECIES_NON_TARGET,
    SPECIES_MULTIPLE,
    SPECIES_AMBIGUOUS,
    SPECIES_NOT_STATED,
    SPECIES_IN_VITRO,
    SPECIES_INFERRED_ONLY,
)

#: The ONLY state admitted to the accepted target-species graph.
ADMISSIBLE_SPECIES_STATES = frozenset({SPECIES_TARGET_SUPPORTED})

#: States that are retained as clearly-labelled leads rather than discarded.
LEAD_SPECIES_STATES = frozenset(
    {
        SPECIES_NON_TARGET,
        SPECIES_MULTIPLE,
        SPECIES_AMBIGUOUS,
        SPECIES_NOT_STATED,
        SPECIES_IN_VITRO,
        SPECIES_INFERRED_ONLY,
    }
)

# ---------------------------------------------------------------------------
# Local normalization
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9. ]+")
_ABBREV = re.compile(r"^([a-z])\.\s*([a-z][a-z-]+)$")

#: Common names and shorthands seen in paper text. Local convenience only; this
#: is not a taxonomy service and is never presented as external verification.
_COMMON_NAMES: Dict[str, str] = {
    "human": "homo sapiens",
    "humans": "homo sapiens",
    "man": "homo sapiens",
    "mouse": "mus musculus",
    "mice": "mus musculus",
    "murine": "mus musculus",
    "rat": "rattus norvegicus",
    "rats": "rattus norvegicus",
    "yeast": "saccharomyces cerevisiae",
    "budding yeast": "saccharomyces cerevisiae",
    "fission yeast": "schizosaccharomyces pombe",
    "e. coli": "escherichia coli",
    "ecoli": "escherichia coli",
    "arabidopsis": "arabidopsis thaliana",
    "thale cress": "arabidopsis thaliana",
    "fruit fly": "drosophila melanogaster",
    "drosophila": "drosophila melanogaster",
    "zebrafish": "danio rerio",
    "c. elegans": "caenorhabditis elegans",
    "worm": "caenorhabditis elegans",
    "tomato": "solanum lycopersicum",
    "rice": "oryza sativa",
    "maize": "zea mays",
    "corn": "zea mays",
    "tobacco": "nicotiana tabacum",
    "cow": "bos taurus",
    "bovine": "bos taurus",
    "pig": "sus scrofa",
    "porcine": "sus scrofa",
    "chicken": "gallus gallus",
}

#: Phrases that mean "no organism was involved", not "the organism is unknown".
_IN_VITRO_MARKERS = (
    "in vitro",
    "cell-free",
    "cell free",
    "recombinant enzyme assay",
    "purified enzyme",
    "reconstituted system",
)

#: Phrases that mark an inference rather than an observation in this species.
_INFERENCE_MARKERS = (
    "by analogy",
    "presumably",
    "is expected to",
    "likely catalyzes",
    "ortholog",
    "ortholog of",
    "orthologue",
    "homolog of",
    "homologue of",
    "extrapolat",
    "inferred from",
    "by similarity",
)


def normalize_species(text: Any) -> str:
    """Local canonical form of a species string. Never an external lookup.

    Lowercases, strips punctuation and strain suffixes, expands an abbreviated
    binomial and maps a short table of common names. Returns ``""`` for empty
    input.
    """
    raw = _WS.sub(" ", str(text or "")).strip().lower()
    if not raw:
        return ""
    # strip parenthetical strain/cultivar notes: "Arabidopsis thaliana (Col-0)"
    raw = re.sub(r"\(.*?\)", " ", raw)
    raw = _PUNCT.sub(" ", raw)
    raw = _WS.sub(" ", raw).strip()
    if not raw:
        return ""
    if raw in _COMMON_NAMES:
        return _COMMON_NAMES[raw]
    abbrev = _ABBREV.match(raw)
    if abbrev:
        initial, epithet = abbrev.group(1), abbrev.group(2)
        for canonical in set(_COMMON_NAMES.values()):
            genus, _, species = canonical.partition(" ")
            if genus.startswith(initial) and species == epithet:
                return canonical
        # Unknown genus: keep the abbreviation rather than inventing a genus.
        return f"{initial}. {epithet}"
    tokens = raw.split()
    if len(tokens) >= 2:
        binomial = " ".join(tokens[:2])
        if binomial in _COMMON_NAMES:
            return _COMMON_NAMES[binomial]
        # Drop strain/subspecies tail: "escherichia coli k 12 mg1655".
        if len(tokens[0]) > 1 and tokens[1].isalpha():
            return binomial
    if raw in _COMMON_NAMES:
        return _COMMON_NAMES[raw]
    return raw


def species_matches_target(candidate: Any, target_normalized: str) -> bool:
    """Exact normalized equality. Genus-only text never matches a binomial.

    "Arabidopsis" does not satisfy a target of "Arabidopsis thaliana" unless the
    common-name table maps it, because a genus is not a species and the guardrail
    does not round in the permissive direction. (The table does map the bare
    genus ``arabidopsis``; that is a documented, auditable convenience, not an
    inference rule.)
    """
    norm = normalize_species(candidate)
    return bool(norm) and norm == str(target_normalized or "").strip().lower()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
@dataclass
class SpeciesVerdict:
    """Why a claim got the species state it did, in a form an auditor can read."""

    state: str
    target_species: str = ""
    observed_species: List[str] = field(default_factory=list)
    observed_normalized: List[str] = field(default_factory=list)
    reason: str = ""
    #: Evidence ids that carry the species support specifically. Empty for every
    #: state except ``target_species_supported``, where it must be non-empty.
    species_evidence_ids: List[str] = field(default_factory=list)
    inference_markers: List[str] = field(default_factory=list)

    @property
    def admissible(self) -> bool:
        return self.state in ADMISSIBLE_SPECIES_STATES

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_inference_markers(text: Any) -> List[str]:
    """Phrases in ``text`` that mark an inference rather than an observation."""
    haystack = str(text or "").lower()
    return [marker for marker in _INFERENCE_MARKERS if marker in haystack]


def detect_in_vitro(text: Any) -> bool:
    haystack = str(text or "").lower()
    return any(marker in haystack for marker in _IN_VITRO_MARKERS)


def classify_species(
    *,
    target_species_normalized: str,
    observed: Optional[Sequence[str]] = None,
    species_evidence_ids: Optional[Sequence[str]] = None,
    evidence_text: str = "",
    claimed_inferred: bool = False,
) -> SpeciesVerdict:
    """Assign exactly one species state to a claim.

    Ordering matters and is deliberate, because each earlier test describes a
    *stronger* reason the claim is not an established target-species observation:

    1. **inference** -- an ortholog/analogy claim is ``inferred_only`` even when
       it names the target species. This is the ortholog-transfer guard and it
       outranks a naive name match.
    2. **in vitro / cell-free** -- no organism performed the reaction.
    3. **nothing observed** -- ``species_not_stated``.
    4. **several distinct species** -- ``multiple_species``, never "one of them
       is the target, good enough".
    5. **one species, is it the target?** -- ``target_species_supported`` only
       when it matches AND at least one evidence span is cited for the species
       specifically; otherwise ``species_ambiguous``.
    """
    target = str(target_species_normalized or "").strip().lower()
    raw_observed = [str(s or "").strip() for s in (observed or []) if str(s or "").strip()]
    normalized = []
    for item in raw_observed:
        norm = normalize_species(item)
        if norm and norm not in normalized:
            normalized.append(norm)
    ev_ids = [str(e) for e in (species_evidence_ids or []) if str(e or "").strip()]
    markers = detect_inference_markers(evidence_text)

    def verdict(state: str, reason: str, ids: Optional[List[str]] = None) -> SpeciesVerdict:
        return SpeciesVerdict(
            state=state,
            target_species=target,
            observed_species=raw_observed,
            observed_normalized=normalized,
            reason=reason,
            species_evidence_ids=list(ids or []),
            inference_markers=list(markers),
        )

    if claimed_inferred or markers:
        detail = ", ".join(markers) if markers else "extraction reported an inferred species"
        return verdict(
            SPECIES_INFERRED_ONLY,
            f"species support is an inference, not an observation ({detail}); "
            "an ortholog or analogy never transfers a reaction between species",
        )

    if detect_in_vitro(evidence_text):
        return verdict(
            SPECIES_IN_VITRO,
            "the supporting passage describes an in vitro / cell-free system, "
            "so no organism performed the reaction",
        )

    if not normalized:
        return verdict(
            SPECIES_NOT_STATED,
            "no species is stated anywhere in the supporting evidence",
        )

    if len(normalized) > 1:
        if target in normalized:
            return verdict(
                SPECIES_MULTIPLE,
                "the evidence names more than one species including the target; "
                "which one the reaction belongs to is not established",
            )
        return verdict(
            SPECIES_MULTIPLE,
            "the evidence names more than one species, none of them the target",
        )

    only = normalized[0]
    if only != target:
        return verdict(
            SPECIES_NON_TARGET,
            f"evidence is for {only!r}, not the target species {target!r}",
        )

    if not ev_ids:
        return verdict(
            SPECIES_AMBIGUOUS,
            "the target species is named but no evidence span was attached to the "
            "species claim specifically, so the support cannot be audited",
        )

    return verdict(
        SPECIES_TARGET_SUPPORTED,
        f"target species {target!r} explicitly supported by {len(ev_ids)} evidence span(s)",
        ev_ids,
    )


__all__ = [
    "SPECIES_STATES",
    "SPECIES_TARGET_SUPPORTED",
    "SPECIES_NON_TARGET",
    "SPECIES_MULTIPLE",
    "SPECIES_AMBIGUOUS",
    "SPECIES_NOT_STATED",
    "SPECIES_IN_VITRO",
    "SPECIES_INFERRED_ONLY",
    "ADMISSIBLE_SPECIES_STATES",
    "LEAD_SPECIES_STATES",
    "SpeciesVerdict",
    "classify_species",
    "normalize_species",
    "species_matches_target",
    "detect_in_vitro",
    "detect_inference_markers",
]
