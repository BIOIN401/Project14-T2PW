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

Reading the claim's own sentences (ORCH-745 F-1)
------------------------------------------------
:func:`classify_species` reads a whole passage: one "in vitro" anywhere in a
5,000-character span decided every claim citing it, and a span that named the
target anywhere made the target "supported". :func:`grade_sentences` and
:func:`aggregate_species` read instead the sentences of the claim's cited spans
that name one of ITS participants, its enzyme or its gene, and grade each one.
A claim is ``target_species_supported`` only on POSITIVE evidence that the
organism itself performed the reaction (a primary paper's in-planta sentence) or
on a review's plain statement about the target (graded ``review``, never
primary). The enzyme alone earns nothing; a marker of inference, a host system
or a cell-free system is never positive. The rules are the pwml-bio-auditor's
adjudication of ORCH-745 F-1 (its section 1c) and every word list below is a
documented ceiling, like the screen rescue's host list.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from t2pw.mining.normalize import _DASHES, _GREEK, is_cofactor

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
    #: :func:`aggregate_species` only: what kind of evidence decided the state
    #: (``GRADE_*``), and the graded supporting sentences it read.
    grade: str = ""
    sentences: List[Dict[str, Any]] = field(default_factory=list)

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


# ---------------------------------------------------------------------------
# Reading the claim's own sentences (ORCH-745 F-1)
# ---------------------------------------------------------------------------
#: What kind of evidence a supporting sentence is, and (on a claim) what decided
#: its species state.
GRADE_IN_PLANTA = "in_planta"
GRADE_REVIEW = "review"
GRADE_IN_VITRO = "in_vitro"
GRADE_INFERENCE = "inference"
GRADE_HOST = "host"
GRADE_NONE = "none"

#: acquisition's sentence rule, and no break after "et al.", "Fig.", "e.g." either. A
#: single capital ("A.") breaks only before a capital ("clade A. In opr3 mutant
#: leaves"), never before an epithet ("A. thaliana"). Its own copy: acquisition
#: imports models, which imports this.
_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])(?<!\bal\.)(?<!\bal \.)(?<!\bFig\.)(?<!\bfig\.)(?<!\bFigs\.)"
    r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<!\bca\.)(?<!\bcf\.)"
    r"(?:(?<!\b[A-Z]\.)\s+|\s+(?=[A-Z]))"
)
#: Clauses: ";", a colon not inside a name ("OPC-8:0", "16:1n9" keep theirs), and
#: the words and / but / whereas / while / or.
_CLAUSE = re.compile(r";|:(?![0-9])|\b(?:and|but|whereas|while|or)\b")
_NBSP = re.compile(r"&nbsp;|\u00a0")
#: A spaced em / en dash is punctuation, not a hyphen: "JA-Ile \u2014 the" must not
#: fold to "ja-ile-the".
_SPACED_DASH = re.compile("\\s+[\u2013\u2014\u2015]\\s+")
_HYPHEN_GAP = re.compile(r"\s*-\s*")
_DIGIT_COMMA = re.compile(r"(?<=\d),\s+(?=\d)")
#: "X (Y)": the paper's "long name (abbreviation)", or a name with its note.
_PAREN_TAIL = re.compile(r"^(.+?)\s+\(([^()]+)\)$")


def _any(*patterns: str) -> "re.Pattern[str]":
    return re.compile("|".join(f"(?:{p})" for p in patterns))


# Negative markers: any one makes a supporting sentence non-positive. Matched on
# folded text (lowercase, hyphens and Greek normalised).
_INFERENCE = _any(r"by analogy", r"presumably", r"is expected to", r"likely cataly[sz]",
                  r"ortholog", r"homolog(?:ue)? of", r"extrapolat", r"\binferred\b",
                  r"by similarity")
#: Homology transfer without the word "ortholog" (repair B4): graded ``inference``
#: like it. Not "compared with / to", which is ordinary loss-of-function prose.
_HOMOLOGY = _any(r"\bcounterpart", r"closest .{0,24}relative", r"\brelative of\b",
                 r"corresponding .{0,24}(?:enzyme|protein|gene|reaction)", r"\bequivalent\b",
                 r"by (?:sequence )?similarity", r"\bhomologous\b", r"\bresembl", r"\banalogous\b",
                 r"the same (?:conversion|reaction)")
_HOST = _any(r"transgenic", r"heterologous", r"ectopic", r"over-?express", r"\b35s\b",
             r"\bexpressing\b", r"\bexpression of\b", r"\btransient", r"agro-?infiltrat",
             r"\binfiltrated\b", r"protoplast", r"\btransformed\b",
             r"expressed in (?:tobacco|nicotiana|n\. ?benthamiana|arabidopsis|a\. ?thaliana|plants?)",
             # repair B5: the construct / transfer vocabulary
             r"\bcarry(?:ing)?\b", r"\bcarries\b", r"\bharbou?r", r"\bintroduc(?:ed|tion|ing)\b",
             r"\bconstructs?\b", r"\bpromoter\b", r"\bcdna\b", r"\btransgene")
#: Complementation is host evidence only with a foreign-gene cue (repair B5): native-gene
#: complementation ("jar1 complemented with JAR1") stays readable.
_COMPLEMENT = re.compile(r"\bcomplement")
_FOREIGN_CUE = re.compile(r"\bbryophyte|\bmoss\b|\bliverwort|\bforeign\b|\bheterolog")
_IN_VITRO = _any(r"in vitro", r"cell-?free", r"cell free", r"recombinant", r"purified",
                 r"reconstitut", r"expressed in (?:e\. ?coli|escherichia|yeast|pichia|insect cells|sf9)",
                 r"enzyme assays?", r"\bincubations?\b", r"incubated with", r"\bk ?m\b", r"\bv ?max\b",
                 r"\bk ?cat\b", r"kinetic", r"\blysates?\b", r"ex vivo",
                 r"(?:crude|tissue|protein) extracts?", r"homogenates?", r"microsom",
                 r"isolated chloroplasts?",
                 # repair B1: preparations of an organ, organelle or cell, affinity tags,
                 # heterologous production, cell culture (BIO-745B-5)
                 r"(?:leaf|leaves|flowers?|roots?|rosettes?|seedlings?|siliques?|thall(?:us|i)|tissues?|"
                 r"plants?|cells?|chloroplasts?|plastids?)[-\s](?:extracts?|homogenates?|lysates?|"
                 r"fractions?|assays?|discs?|preparations?|supernatants?|powder)",
                 r"(?:soluble|membrane|stromal|microsomal|crude|desalted|clarified|total|cell-free)\s+"
                 r"(?:protein\s+)?(?:extracts?|fractions?)",
                 r"(?:isolated|intact)\s+(?:chloroplasts?|plastids?|peroxisomes?|mitochondri)",
                 r"\bhis-?tag", r"strep-?tag", r"\bgst-?tag", r"\bflag-?tag", r"\b6x\s?his", r"\bmbp-",
                 r"produced in (?:e\. ?coli|escherichia|bacteria|yeast|pichia|insect cells?|sf9|baculovirus)",
                 r"insect[-\s]cell", r"baculovirus", r"\bassays? with",
                 r"cell (?:suspension|culture)s?", r"suspension cultures?", r"\bcall(?:us|i)\b")

# Positive credit (rules 3 and 5 of the adjudication).
_IN_PLANTA_CONTEXT = _any(r"in planta", r"in vivo", r"\bmutants?\b", r"knock-?outs?",
                          r"knock-?downs?", r"loss-of-function", r"null alleles?", r"t-dna",
                          r"salk_", r"wild-?type", r"wild type", r"\bcol-0\b", r"\bseedlings?\b",
                          r"\bleaf\b", r"\bleaves\b", r"\bflowers?\b", r"\broots?\b", r"\brosettes?\b",
                          r"\bsiliques?\b", r"\btissues?\b", r"\bwounded\b", r"\bwounding\b", r"\blines\b")
_LOSS_OF_FUNCTION = _any(r"\bmutants?\b", r"knock-?outs?", r"knock-?downs?", r"loss-of-function",
                         r"null alleles?", r"t-dna", r"salk_", r"\brnai\b", r"amirna", r"crispr")
_NO_EFFECT = _any(r"unchanged", r"unaffected", r"not altered", r"no significant",
                  r"similar to (?:the )?wild-?type", r"comparable to")
#: A negation governing the conversion (repair B2): "was not converted", "no
#: conversion", "lack the ability to convert", "failed to convert", or the conversion
#: then reported absent. Rule 5(a) only: a loss-of-function sentence keeps its "no"
#: ("no JA-Ile was detected in gh3.10 jar1 lines"). Words between the negation and
#: the verb are joined by spaces only, so "cyclase, not synthase, converts" keeps its
#: credit.
_NEGATED_CONVERSION = _any(
    r"\b(?:not|no|never|nor|without|cannot|fail(?:s|ed)? to|unable to|lacks?|lacking)\b"
    r"(?:\s+[\w-]+){0,3}?\s+(?:be\s+)?(?:convert|conversion|formed|formation|produced|"
    r"metaboli[sz]|synthesi[sz])",
    r"\b(?:convert\w*|conversion)\b(?:\s+[\w()-]+){0,8}?\s+(?:was|were|is|are)\s+"
    r"(?:not|never|undetect\w*|absent)\b")
_CONVERSION = _any(r"\bconvert", r"\bconversion\b", r"formed from", r"metaboli[sz]ed to",
                   r"produced from", r"synthesi[sz]ed from")
_APPLICATION = _any(r"\btreat", r"\bappl(?:y|ied|ication)", r"\bspray", r"exogenous", r"\bfed\b",
                    r"\bfeeding\b",
                    # repair B6: a fed precursor is held back pending the product-owner hinge
                    r"\bsuppl(?:y|ied|ies|ement)", r"\binfiltrat", r"\binject", r"\bdosed?\b",
                    r"\badministered\b", r"\bwatered\b", r"\bsoaked\b", r"\bdipped\b",
                    r"\bperfused\b", r"\bdeuterated\b", r"\blabell?ed\b")
_HEDGE = _any(r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bwould\b", r"\bpotential", r"\bpossibl",
              r"\bputative", r"\blikely\b", r"\bsuggest", r"\banticipated\b", r"supports? the view",
              r"\bappears?\b", r"\bpresumabl", r"\bpropos", r"\bhypothe", r"\bpredicted\b",
              r"\bperhaps\b",
              # repair B3
              r"thought to", r"\bpresumed\b", r"\bseems? to\b", r"consistent with",
              r"\bpoints? (?:to|towards)\b", r"\bspeculat", r"\bimpl(?:y|ies|ied)\b", r"\bunclear\b",
              r"remains? (?:unknown|unclear|to be|an open)", r"open question", r"\bassume[ds]?\b",
              r"\bconceivabl")
_PROCEDURAL = _any(r"\b(?:was|were)\s+(?:\w+\s+)?(?:grown|sown|harvested|sprayed|treated|extracted|"
                   r"analy[sz]ed|quantified|measured|incubated|used)\b",
                   r"\bwe\s+(?:employed|used)\b",
                   # repair N3
                   r"plant material", r"growth conditions",
                   r"\b(?:was|were)\s+(?:\w+\s+)?(?:obtained|purchased|provided|surface-sterili[sz]ed|"
                   r"stratified|transferred)\b")
_PURPOSE = _any(r"\bto (?:study|test|investigate|examine|evaluate|assess|analy[sz]e|determine|explore)\b",
                r"\bas an? (?:new|novel) tool\b", r"\bin order to\b")
#: Matched on the ORIGINAL text: a bracketed reference number, "et al", an author-year
#: close such as "(Suza & Staswick 2008 )", a numeric parenthetical "(12, 13)", a
#: superscript flattened after a full stop ("...12-OPDA.12,13"), or a narrative
#: citation (repair B9 / F3).
_CITATION = re.compile(
    r"\[\s*\d|\bet al\b|\b(?:19|20)\d{2}[a-z]?\s*[);]"
    r"|\(\s*\d{1,3}(?:\s*[,\u2013-]\s*\d{1,3})*\s*\)"
    r"|[A-Za-z)\]]\.\d{1,3}(?:[,\u2013-]\d{1,3})*(?=\s|$)"
    r"|previously (?:reported|shown|described|demonstrated)|as (?:shown|reported|described) (?:by|in)"
    r"|(?:prior|earlier) (?:work|studies|reports)|(?:has|had|have) been (?:shown|reported|described|demonstrated)"
    r"|co-?workers",
    re.IGNORECASE)
_COMPARISON = _any(r"\bunlike\b", r"in contrast", r"by contrast", r"\bwhereas\b",
                   r"compared (?:with|to)", r"relative to", r"equivalent to", r"\bversus\b",
                   r"counterpart",
                   # repair N2: the target named only by comparison
                   r"\bresembl", r"similar to", r"\banalogous\b", r"\bas in\b",
                   r"also (?:found|present) in", r"\bboth\b")
_NEGATION = _any(r"\blacks?\b", r"\blacking\b", r"\babsen(?:t|ce)\b", r"\bnot\b", r"\bno\b",
                 r"\bnever\b", r"\bwithout\b",
                 # repair B7
                 r"\bfail(?:s|ed)? to\b", r"\bunclear\b", r"open question",
                 r"remains? (?:unknown|unclear|to be)", r"\bwhether\b")
#: A review's statement names the conversion itself (repair F4): co-occurrence ("OPDA
#: and OPC-8:0 accumulate") is not one.
_REVIEW_CONVERSION = _any(r"\bconvert", r"\bconversion\b", r"\breduc(?:e|es|ed|ing|tion)\b",
                          r"\bproduc(?:e|es|ed|ing|tion)\b", r"\bform(?:ed|s|ation)\b",
                          r"\bsynthesi[sz]", r"\bcataly[sz]", r"\boxidi[sz]", r"\boxygenat",
                          r"\bconjugat", r"\bmetaboli[sz]", r"\byield", r"\bgenerat", r"\bcycli[sz]",
                          r"\b(?:give|gives|gave|giving) rise")
#: Spans from these sections never earn positive credit: methods describe what was
#: done, an introduction or background what others found. Discussion and conclusion
#: restate and interpret (an orchestrator decision the product owner may revisit).
#: Searched, not prefix-matched: "2. Materials and Methods" is a methods section.
_NO_CREDIT_SECTION = re.compile(r"method|material|introduction|background|discussion|conclusion")

#: Organisms the sentence-level rules recognise. A documented ceiling, like the
#: screen rescue's ``_EXTRA_HOSTS``: an organism named only by a word missing here
#: (and not as an abbreviated binomial such as "P. patens") is not seen.
_ORGANISM_WORDS = re.compile(
    r"\b(?:arabidopsis|thale cress|marchantia|physcomitrium|physcomitrella|selaginella|"
    r"klebsormidium|chlamydomonas|anthoceros|pisum|pea|nicotiana|benthamiana|tobacco|capsicum|"
    r"pepper|solanum|tomato|potato|oryza|rice|zea mays|maize|corn|triticum|wheat|hordeum|barley|"
    r"soybean|vitis|grapes?|grapevine|brassica|medicago|populus|escherichia coli|e\. ?coli|yeast|"
    r"saccharomyces|pichia|human|humans|homo sapiens|mouse|mice|murine|rats?|drosophila|"
    r"zebrafish|liverworts?|moss|mosses|hornworts?|bryophytes?|algae|alga)\b"
)
#: A gene symbol carrying another species' prefix ("MpFAD5", "Mp gh3a", "PpAOC1",
#: "CaLOX2") names that organism (repair B8). Matched on the ORIGINAL text, and only
#: where the sentence does not name the target. A documented ceiling like the organism
#: words: an unlisted prefix is not seen.
_FOREIGN_GENE = re.compile(
    r"\b(?:Mp|Pp|Os|Sl|Zm|Ta|Hv|Gm|Nt|Nb|Bn|Ca|Vv|Md|St|Le|Cs|Sb|Br|Mt|Pt)"
    r"(?:[A-Z]{2,}[0-9A-Za-z.]*|\s[a-z]{2,}[0-9a-z]*)\b")
_ABBREVIATED_BINOMIAL = re.compile(r"\b[A-Z]\. ?[a-z]{4,}\b")
#: Arabidopsis-only shorthands: an accession and an AGI locus name the plant.
_ARABIDOPSIS_MARKS = re.compile(r"\bcol-0\b|\bat[1-5cm]g\d{5}\b")

#: Never a reaction's evidence on their own (adjudication rule 6, W11): a sentence
#: naming ATP or Ile is a buffer recipe as often as a result.
_EXTRA_COFACTORS = frozenset({"nad(p)h", "nad(p)+", "nad(p)", "coa"})
_AMINO_ACIDS = frozenset(
    "ala arg asn asp cys gln glu gly his ile leu lys met phe pro ser thr trp tyr val "
    "alanine arginine asparagine aspartate cysteine glutamine glutamate glycine histidine "
    "isoleucine leucine lysine methionine phenylalanine proline serine threonine tryptophan "
    "tyrosine valine".split()
)


def fold_text(text: Any) -> str:
    """Matching form of passage text and names: lowercase; Unicode hyphens and
    Greek letters as in :func:`t2pw.mining.normalize.normalize_name`; the italic
    gaps around hyphens closed ("dn- cis -OPDA" -> "dn-cis-opda"); "9, 11, 15" ->
    "9,11,15"; whitespace collapsed."""
    folded = _SPACED_DASH.sub(", ", _NBSP.sub(" ", str(text or ""))).lower().translate(_DASHES)
    for greek, latin in _GREEK.items():
        folded = folded.replace(greek, latin)
    folded = _DIGIT_COMMA.sub(",", _HYPHEN_GAP.sub("-", folded))
    return _WS.sub(" ", folded).strip()


def split_sentences(text: Any) -> List[str]:
    return [s for s in (p.strip() for p in _SENTENCE_SPLIT.split(str(text or ""))) if s]


def name_variants(name: Any, *, bare: bool = False) -> List[str]:
    """The folded name and, for "X (Y)", the parenthetical Y. X alone only when
    ``bare`` (the loose, negative-marker set): the note may be what disambiguates it
    ("ALA (18:3n3)"; a bare ALA is also 5-aminolevulinic acid or alanine), the rule
    ``queries.rescue_terms`` keeps for anything that earns credit. A sentence naming
    bare ALA with an in-vitro marker still vetoes (BIO-745B-1)."""
    raw = " ".join(str(name or "").split())
    out = [fold_text(raw)] if raw else []
    tail = _PAREN_TAIL.match(raw)
    if tail:
        for part in (tail.group(2), tail.group(1)) if bare else (tail.group(2),):
            if fold_text(part) not in out:
                out.append(fold_text(part))
    return [v for v in out if v]


def _matchers(names: Iterable[Any], *, strict: bool) -> List[List["re.Pattern[str]"]]:
    """One list of variant patterns per name. Loose: any non-word edge, so "OPDA"
    occurs in "dn-OPDA", and bare X counts. Strict: a hyphen is part of the token
    (rule 6) and only the full name or Y counts."""
    edge = r"[\w-]" if strict else r"\w"
    return [
        [re.compile(rf"(?<!{edge}){re.escape(v)}(?!{edge})") for v in variants]
        for variants in (name_variants(n, bare=not strict) for n in names) if variants
    ]


def _names(text: str, matchers: Sequence[Sequence["re.Pattern[str]"]]) -> bool:
    return any(p.search(text) for variants in matchers for p in variants)


def _creditable(name: Any) -> bool:
    """A cofactor or a free amino acid never counts for positive credit."""
    key = fold_text(name)
    key = key[2:] if key.startswith(("l-", "d-")) else key
    return bool(key) and not is_cofactor(key) and key not in _EXTRA_COFACTORS and key not in _AMINO_ACIDS


def _catalyst_names(*values: Any) -> List[str]:
    """Enzyme and gene values split into names: "LOX2, LOX3", "GH3.11/JAR1"."""
    out: List[str] = []
    for value in values:
        for part in re.split(r"\s*[,;/]\s*", str(value or "")):
            if part.strip() and part.strip() not in out:
                out.append(part.strip())
    return out


def organisms_named(text: Any) -> List[str]:
    """The organisms a piece of text names, as :func:`normalize_species` keys."""
    folded = fold_text(text)
    hits = [m.group(0) for m in _ORGANISM_WORDS.finditer(folded)]
    hits += [m.group(0) for m in _ABBREVIATED_BINOMIAL.finditer(str(text or ""))]
    out = [normalize_species(h) for h in hits]
    if _ARABIDOPSIS_MARKS.search(folded):
        out.append("arabidopsis thaliana")
    return list(dict.fromkeys(o for o in out if o))


def _found(pattern: "re.Pattern[str]", text: str) -> str:
    match = pattern.search(text)
    return match.group(0) if match else ""


def _others_named(sentence: str, target: str) -> List[str]:
    """Organisms other than the target that ``sentence`` names: organism words,
    abbreviated binomials, and -- when the sentence does not name the target -- a gene
    symbol carrying another species' prefix ("MpFAD5", "Mp gh3a"; repair B8)."""
    named = organisms_named(sentence)
    others = [o for o in named if o != target]
    if target not in named:
        others += [f"gene {m.group(0)!r}" for m in _FOREIGN_GENE.finditer(sentence)]
    return others


def _host_marker(sentence: str, folded: str, target: str) -> str:
    """The host-system marker of a sentence, or "". Complementation counts only with
    a foreign-gene cue (repair B5): native-gene complementation stays readable."""
    found = _found(_HOST, folded)
    if found:
        return found
    if _COMPLEMENT.search(folded) and (_FOREIGN_CUE.search(folded) or _FOREIGN_GENE.search(sentence)
                                       or any(o != target for o in organisms_named(sentence))):
        return "complementation with a foreign gene"
    return ""


def species_span_names_target(spans: Iterable[Any], target_species_normalized: str) -> bool:
    """W13: a species span counts only if one of its sentences names the target
    plainly -- not in an ortholog, host-system or comparison sentence ("Arabidopsis
    Ortholog" as a table header, "MpFAD5 is orthologous to AtFAD5, an Arabidopsis
    enzyme", "unlike Arabidopsis")."""
    target = str(target_species_normalized or "").strip().lower()
    for span in spans:
        for sentence in split_sentences(getattr(span, "text", "")):
            named = organisms_named(sentence)
            if target not in named:
                continue
            folded = fold_text(sentence)
            if (_INFERENCE.search(folded) or _HOMOLOGY.search(folded) or _host_marker(sentence, folded, target)
                    or _COMPARISON.search(folded)):
                continue
            # Repair N2: the target alongside another organism is not a plain statement
            # either ("Arabidopsis thaliana and Marchantia polymorpha both make ...").
            if any(o != target for o in named) or _FOREIGN_GENE.search(sentence):
                continue
            return True
    return False


def grade_sentences(
    spans: Iterable[Any],
    *,
    target_species_normalized: str,
    substrates: Sequence[Any] = (),
    products: Sequence[Any] = (),
    enzyme: Any = "",
    gene: Any = "",
    asserted: Sequence[Any] = (),
    review: bool = False,
    paper_observed_organisms: Sequence[Any] = (),
) -> List[Dict[str, Any]]:
    """Grade every supporting sentence of ``spans`` (the claim's cited passages).

    A sentence supports the claim when it names one of its participants, its
    enzyme or its gene (loose naming; ``asserted`` adds names the model gave but
    extraction could not support, so their negative markers still count). Each
    supporting sentence gets one grade, tested in this order: ``inference``,
    ``host``, ``in_vitro``, then -- only in a primary paper -- ``in_planta`` under
    rules 3-6 of the adjudication, or -- only in a review -- ``review`` under rule
    7. Positive credit names ``substrates`` / ``products`` strictly and never by a
    cofactor or a free amino acid. Returns one dict per supporting sentence:
    ``evidence_id``, ``sentence`` (index in its span), ``text``, ``grade``,
    ``reason``.
    """
    target = str(target_species_normalized or "").strip().lower()
    catalysts = _catalyst_names(enzyme, gene)
    loose = _matchers([*substrates, *products, *asserted, *catalysts], strict=False)
    subs = _matchers([n for n in substrates if _creditable(n)], strict=True)
    prods = _matchers([n for n in products if _creditable(n)], strict=True)
    cats = _matchers(catalysts, strict=True)
    paper_other = any(normalize_species(o) not in ("", target) for o in paper_observed_organisms)

    def in_planta(sentence: str, folded: str, span: Any, span_other: bool) -> Tuple[str, str]:
        for pattern, why in ((_HEDGE, "hedged"), (_PROCEDURAL, "a procedure, not a result"),
                             (_PURPOSE, "a purpose clause")):
            if pattern.search(folded):
                return GRADE_NONE, f"no positive credit: {why} ({_found(pattern, folded)!r})"
        if _CITATION.search(sentence):
            return GRADE_NONE, f"no positive credit: cites other work ({_found(_CITATION, sentence)!r})"
        section = str(getattr(span, "section", "") or "").lower()
        if _NO_CREDIT_SECTION.search(section):
            return GRADE_NONE, f"no positive credit: {section} passage"
        other = _others_named(sentence, target)
        if other:
            return GRADE_NONE, f"no positive credit: names a non-target organism ({other[0]})"
        if target not in organisms_named(sentence) and (paper_other or span_other):
            return GRADE_NONE, ("no positive credit: the passage or the paper names another organism "
                                "and this sentence does not name the target")
        applied = _APPLICATION.search(folded)
        no_effect = _NO_EFFECT.search(folded)
        for clause in _CLAUSE.split(folded):
            if not _IN_PLANTA_CONTEXT.search(clause):
                continue
            sub, prod = _names(clause, subs), _names(clause, prods)
            if (sub and prod and _CONVERSION.search(clause) and not applied
                    and not _NEGATED_CONVERSION.search(clause)):
                return GRADE_IN_PLANTA, ("in planta: substrate, product and a conversion in one clause "
                                         f"({_found(_IN_PLANTA_CONTEXT, clause)!r})")
            if (sub or prod) and _names(clause, cats) and _LOSS_OF_FUNCTION.search(clause) and not no_effect:
                return GRADE_IN_PLANTA, ("in planta: loss of function of the enzyme or gene changes a "
                                         f"participant ({_found(_LOSS_OF_FUNCTION, clause)!r})")
        return GRADE_NONE, "no in-planta statement of this reaction in one clause"

    def review_route(sentence: str, folded: str) -> Tuple[str, str]:
        for pattern, why in ((_HEDGE, "hedged"), (_COMPARISON, "a comparison"), (_NEGATION, "a negation"),
                             (_HOMOLOGY, "a homology transfer")):
            if pattern.search(folded):
                return GRADE_NONE, f"no review credit: {why} ({_found(pattern, folded)!r})"
        named = organisms_named(sentence)
        if target not in named:
            return GRADE_NONE, "no review credit: the sentence does not name the target"
        if any(o != target for o in named) or _FOREIGN_GENE.search(sentence):
            return GRADE_NONE, "no review credit: the sentence names another organism too"
        if not _REVIEW_CONVERSION.search(folded):
            return GRADE_NONE, "no review credit: no conversion is stated (co-occurrence is not one)"
        sub, prod = _names(folded, subs), _names(folded, prods)
        if (sub and prod) or ((sub or prod) and _names(folded, cats)):
            return GRADE_REVIEW, "a review states this reaction for the target species"
        return GRADE_NONE, "no review credit: substrate and product are not both named"

    graded: List[Dict[str, Any]] = []
    for span in spans:
        text = str(getattr(span, "text", "") or "")
        span_other = any(o != target for o in organisms_named(text))
        for index, sentence in enumerate(split_sentences(text)):
            folded = fold_text(sentence)
            if not _names(folded, loose):
                continue
            host = _host_marker(sentence, folded, target)
            if _INFERENCE.search(folded):
                grade, reason = GRADE_INFERENCE, f"inference marker {_found(_INFERENCE, folded)!r}"
            elif _HOMOLOGY.search(folded):
                grade, reason = GRADE_INFERENCE, f"homology-transfer marker {_found(_HOMOLOGY, folded)!r}"
            elif host:
                grade, reason = GRADE_HOST, f"host-system marker {host!r}"
            elif _IN_VITRO.search(folded):
                grade, reason = GRADE_IN_VITRO, f"in vitro / cell-free marker {_found(_IN_VITRO, folded)!r}"
            elif review:
                grade, reason = review_route(sentence, folded)
            else:
                grade, reason = in_planta(sentence, folded, span, span_other)
            graded.append({"evidence_id": str(getattr(span, "evidence_id", "")), "sentence": index,
                           "text": sentence, "grade": grade, "reason": reason})
    return graded


def aggregate_species(
    *,
    target_species_normalized: str,
    observed: Optional[Sequence[str]] = None,
    species_evidence_ids: Optional[Sequence[str]] = None,
    sentences: Sequence[Mapping[str, Any]] = (),
    species_span_ok: bool = False,
    claimed_inferred: bool = False,
) -> SpeciesVerdict:
    """One species state from graded sentences (rule 8 of the adjudication).

    The order of :func:`classify_species` is kept: inference (a marker in a
    supporting sentence, or the model's ``species_inferred``) > in vitro (an in
    vitro sentence and no in-planta one) > not stated > multiple > non-target >
    target. The target is ``target_species_supported`` only with a model-cited
    species span that names it plainly (:func:`species_span_names_target`) AND an
    ``in_planta`` sentence, or a ``review`` sentence from a review. Anything less
    is ``species_ambiguous``: a lead for a ``missing_species_evidence`` query.
    """
    target = str(target_species_normalized or "").strip().lower()
    raw_observed = [str(s or "").strip() for s in (observed or []) if str(s or "").strip()]
    normalized = list(dict.fromkeys(n for n in (normalize_species(s) for s in raw_observed) if n))
    ids = [str(e) for e in (species_evidence_ids or []) if str(e or "").strip()]
    rows = [dict(s) for s in sentences]
    by_grade: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_grade.setdefault(str(row.get("grade") or ""), []).append(row)

    def cite(grade: str) -> str:
        row = (by_grade.get(grade) or [{}])[0]
        return f"{row.get('evidence_id', '')}[{row.get('sentence', '')}]: {str(row.get('text', ''))[:160]!r}"

    def verdict(state: str, reason: str, grade: str = GRADE_NONE) -> SpeciesVerdict:
        return SpeciesVerdict(
            state=state, target_species=target, observed_species=raw_observed,
            observed_normalized=normalized, reason=reason,
            species_evidence_ids=list(ids) if state == SPECIES_TARGET_SUPPORTED else [],
            inference_markers=[str(r.get("reason", "")) for r in by_grade.get(GRADE_INFERENCE, [])],
            grade=grade, sentences=rows,
        )

    if claimed_inferred or GRADE_INFERENCE in by_grade:
        detail = (cite(GRADE_INFERENCE) if GRADE_INFERENCE in by_grade
                  else "extraction reported an inferred species")
        return verdict(SPECIES_INFERRED_ONLY,
                       f"species support is an inference, not an observation ({detail}); an ortholog "
                       "or analogy never transfers a reaction between species", GRADE_INFERENCE)
    if GRADE_IN_VITRO in by_grade and GRADE_IN_PLANTA not in by_grade:
        return verdict(SPECIES_IN_VITRO,
                       f"a supporting sentence describes an in vitro / cell-free system and none shows "
                       f"the reaction in the organism ({cite(GRADE_IN_VITRO)})", GRADE_IN_VITRO)
    if not normalized:
        return verdict(SPECIES_NOT_STATED, "the claim states no species")
    if len(normalized) > 1:
        return verdict(SPECIES_MULTIPLE, "the claim names more than one species")
    if normalized[0] != target:
        return verdict(SPECIES_NON_TARGET, f"evidence is for {normalized[0]!r}, not the target species {target!r}")
    if not ids:
        return verdict(SPECIES_AMBIGUOUS, "the target species is named but no evidence span was cited for "
                       "the species claim specifically, so the support cannot be audited")
    if not species_span_ok:
        return verdict(SPECIES_AMBIGUOUS, "no cited species passage names the target in a plain statement "
                       "(only in an ortholog, host-system or comparison sentence)")
    if GRADE_IN_PLANTA in by_grade:
        return verdict(SPECIES_TARGET_SUPPORTED,
                       f"target species {target!r}: the organism performs the reaction ({cite(GRADE_IN_PLANTA)})",
                       GRADE_IN_PLANTA)
    if GRADE_REVIEW in by_grade:
        return verdict(SPECIES_TARGET_SUPPORTED,
                       f"target species {target!r}, stated by a review article ({cite(GRADE_REVIEW)}); "
                       "review evidence, never primary", GRADE_REVIEW)
    if GRADE_HOST in by_grade:
        return verdict(SPECIES_AMBIGUOUS, f"the target appears only as a host system ({cite(GRADE_HOST)}); "
                       "a host never performs its own reaction there", GRADE_HOST)
    return verdict(SPECIES_AMBIGUOUS, "the target species is named but no supporting sentence shows the "
                   "organism performing the reaction")


__all__ = [
    "GRADE_IN_PLANTA",
    "GRADE_REVIEW",
    "GRADE_IN_VITRO",
    "GRADE_INFERENCE",
    "GRADE_HOST",
    "GRADE_NONE",
    "fold_text",
    "split_sentences",
    "name_variants",
    "organisms_named",
    "species_span_names_target",
    "grade_sentences",
    "aggregate_species",
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
