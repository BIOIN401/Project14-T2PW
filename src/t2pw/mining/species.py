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
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
#: Homology transfer without the word "ortholog" (repair B4, R2-B6): graded
#: ``inference`` like it. Not "compared with / to", which is ordinary loss-of-function
#: prose (a comparison still vetoes positive credit, see ``_COMPARISON_ANY``).
_HOMOLOGY = _any(r"\bcounterpart", r"closest .{0,24}relative", r"\brelative of\b",
                 r"corresponding .{0,24}(?:enzyme|protein|gene|reaction)", r"\bequivalent\b",
                 r"by (?:sequence )?similarity", r"\bhomologous\b", r"\bresembl", r"\banalogous\b",
                 r"the same (?:conversion|reaction)", r"\bhomolog(?:ue)?s?\b")
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
_IN_VITRO = _any(r"in[- ]vitro", r"cell-?free", r"cell free", r"recombinant", r"purified",
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
                 r"cell (?:suspension|culture)s?", r"suspension cultures?", r"\bcall(?:us|i)\b",
                 # CS-D D3 (the R1-B3 family): any extract or enzyme preparation is cell-free
                 r"\bextracts?\b", r"\benzyme preparations?\b", r"\bcrude enzyme\b",
                 r"\b(?:enzyme|crude|protein|cell|leaf|tissue)\s+(?:[\w-]+\s+)?(?:extracts?|preparations?|"
                 r"fractions?|homogenates?)")

# Positive credit (ORCH-745 CS-B repair 2, an orchestrator decision the product owner may
# revisit): in a primary paper ONLY loss-of-function genetics earns in_planta. The old
# rule 5(a) -- a conversion stated in an organ / genotype context -- is removed: under
# the adjudicated policy such a sentence is background, a fed precursor or an in-vitro
# description, and every round of word lists guarding it let paraphrases through.
#: A plant loss-of-function genotype.
_LOSS_OF_FUNCTION = _any(r"\bmutants?\b", r"knock-?outs?", r"knock-?downs?", r"loss-of-function",
                         r"null alleles?", r"t-dna", r"salk_", r"\brnai\b", r"amirna", r"crispr")
#: ...but not a protein made mutant on the bench (on folded text; the substitution
#: notation "R117A" is matched on the original text by ``_SUBSTITUTION``).
_PROTEIN_MUTANT = _any(r"mutant (?:proteins?|enzymes?|variants?|forms?)", r"\bvariants?\b",
                       r"site-directed", r"wild-?type (?:enzymes?|proteins?)", r"\b(?:in)?active\b")
_SUBSTITUTION = re.compile(r"\b[A-Z]\d{2,4}[A-Z]\b")
#: The effect a loss of function must show, on the RIGHT participant (CS-D D1): a
#: product that is absent or decreased, or a substrate that accumulates. Each form is
#: (pattern, side it must apply to, forward). A forward form names its participant right
#: after it ("lacked JA-Ile", "deficient in JA-Ile", "accumulated no JA-Ile"), never the
#: genotype ("mutants lacking GH3.10"); any other form applies to the first participant
#: before it in the clause (its subject), or failing that the first one after it.
_EFFECT_FORMS = (
    (_any(r"\bnot detect(?:ed|able)\b", r"\bno detectable\b", r"\bundetectable\b",
          r"\bbelow (?:the )?(?:detection|quantification) limit", r"\babsent\b", r"\babolish\w*",
          r"\babrogat\w*", r"\beliminat\w*", r"\bdepleted\b",
          r"\b(?:did|does|do) not (?:accumulate|form|produce)\b",
          r"\bfail(?:ed|s)? to (?:accumulate|form|produce)\b",
          r"\bno longer (?:\w+ )?(?:detect\w*|accumulat\w*|form\w*)",
          r"\bno\b(?:\s+[\w:()+-]+){1,4}?\s+(?:was|were)\s+(?:detected|found|formed)\b",
          r"\b(?:reduced|decreased|lower|diminished|impaired|compromised)\b",
          # CS-D repair 1 (R2): "aos mutants accumulated less 12-OPDA" is a decrease; a
          # participant inside the gap earns nothing (:func:`_effect_under`, ORCH-746 CS-E
          # repair 1, RT-E2-3), nor does "nowhere near less" (CSE-R0-N5)
          r"\baccumulated (?:(?!(?:not|no|hardly|scarcely|barely|never|nowhere)\b)\w+ ){0,2}?"
          r"(?:less|lower|reduced|decreased)\b"), "product", False),
    # forward; repair 1 (R1) dropped the unbriefed "produced / formed / contained no"
    (_any(r"\black(?:ed|ing)\b", r"\bdeficient in\b", r"\baccumulated no\b"), "product", True),
    (re.compile(r"\b(?:accumulated|increased|elevated|higher)\b"), "substrate", False),
)
#: A substrate-side increase form counts only in a clause that says nothing went down
#: (CS-D repair 1, R2): "accumulated less JA", "accumulated to lower levels" is a decrease.
_DECREASE_WORD = re.compile(r"\b(?:less|fewer|lesser|lower|reduced|decreased|diminished|no|weakly|slightly|"
                            r"marginally)\b")
#: A participant name followed by an enzyme noun is the enzyme, not the participant
#: ("OPDA reductase levels", "the other allene oxide cyclases"; CS-D review N1 / N4).
#: ponytail: "-ase" plus a short list of English "-ase" words; an unlisted one ("x
#: phrases") only costs credit.
_ENZYME_HEAD = re.compile(r"\s+(?![a-z-]*(?:creases?|phases?|cases?|bases?|leases?|chases?|rases?)\b)"
                          r"[a-z-]+ases?\b")
#: A whole-plant phenotype is the effect's subject, not the metabolite (CS-D repair 1, R2):
#: "Root growth of the jar1 mutant was reduced on medium containing JA-Ile."
_PHENOTYPE = re.compile(r"\b(?:growth|fertility|fertile|resistance|susceptibility|tolerance|senescence|yield|"
                        r"biomass|germination|flowering|development|phenotypes?)\b")
#: The opener of a comma-led relative or adverbial clause (CS-D repair 1, review N3; how a
#: participant is cut off by one is :data:`_READINGS`, CS-D repair 2 and ORCH-746 CS-E).
_NEW_SUBJECT = re.compile(r";|,\s*(?:which|that|who|whose|with|whereas|while|although|though|but|and)\b")
#: The noun after the gene says the marker is not OF the gene: "mutants in the JAR1
#: pathway", "the myc2 mutant in the JAR1 background" (CS-D repair 1, RT-D-4).
_CONTAINER = re.compile(r"\s*(?:pathway|background|cluster|family|signall?ing|network)s?\b")
#: A negation before an effect form in the clause refuses it ("was not abolished",
#: "never abolished", "were not reduced"); the forms that START at their negation ("not
#: detected", "no detectable", "no JA-Ile was detected", "did not accumulate") are the
#: absence statement itself.
_CHANGE_NEGATION = re.compile(r"\b(?:not|no|never|neither|nor|without|none|hardly|scarcely|barely)\b|n't")
_NO_EFFECT = _any(r"\bunchanged\b", r"\bunaffected\b",
                  r"\bnot\s+(?:significantly\s+)?(?:affected|changed|altered|different)\b",
                  r"\bnot significantly\b", r"\bno significant", r"\bdid not (?:differ|change)\b",
                  r"(?:similar|comparable) to (?:the )?wild-?type", r"\bwild-?type levels\b",
                  r"\bnormal levels\b", r"\bno (?:change|difference|effect)s?\b",
                  # CS-D D1: the adjudicated bare "comparable to" (repair 2 had narrowed it)
                  r"\bcomparable to\b", r"\bindistinguishable\b", r"\bnormally\b", r"\bresidual\b",
                  r"\bsimilar\b",
                  r"\bstill (?:present|accumulat\w*|detect\w*|formed|produced|increased|elevated)\b",
                  r"\bnot (?:fully |completely |entirely )?(?:abolished|absent|eliminated|abrogated|depleted|"
                  r"deficient|lacking)\b",
                  # CS-D repair 1 (R1): a negated comparison or delay is not an absence
                  # ("did not accumulate significantly less JA-Ile than", "did not accumulate
                  # until 6 h", "accumulated no more JA-Ile than"). The window runs to a
                  # semicolon or the sentence end (ORCH-746 CS-E, BIO-D-R2-2): the text is one
                  # split sentence already, so "(Fig. 3A)", "ca." or "i.e." inside it must not
                  # hide the "until" / "than"; a wider veto window only refuses more.
                  r"\bno (?:more|less|greater|fewer|higher|lower|lesser)\b",
                  r"\b(?:(?:did|does|do) not|fail(?:ed|s)? to)\b[^;]{0,60}?"
                  r"\b(?:more|less|lesser|lower|higher|"
                  r"fewer|greater|further|again|than|until|significantly)\b",
                  # a negation after the effect (CS-D review N4 / BIO-D-4)
                  r"\bneither\b[^;:]*\bnor\b", r"\bnone of\b",
                  # ORCH-746 CS-E repair 1, RT-E2-1: a delay written with ANY absence form ("was
                  # not detected until 30 min", "absent until stage 13", "did not accumulate
                  # before 1 h") is not an absence
                  r"\buntil\b", r"\b(?:before|prior to)\s+\d",
                  # RT-E2-2: a non-significant change ("not statistically significant", "n.s.",
                  # "P > 0.05", "P = 0.08", "tended to") ...
                  r"\bnot (?:statistically )?significant\b", r"\bnot statistically\b",
                  r"\b(?:non-?|in)significant", r"\bn\.\s?s\.", r"\bp\s*(?:>|\u2265)\s*0?\.\d",
                  r"\bp\s*=\s*0?\.(?:0[5-9]|[1-9])", r"\btend(?:ed|s)? to\b",
                  # ... and the genotype joined to the wild type ("in jar1 mutant and wild-type
                  # plants", "in opr3 mutants or in the wild type", "and Col-0 alike")
                  r"\b(?:mutants?|lines?|knock-?outs?|plants?)\s+(?:and|or)\s+(?:also\s+)?(?:in\s+)?(?:the\s+)?"
                  r"(?:wild[- ]?type|col-0|wt)\b",
                  r"\balike\b",
                  # RT-E2-4: a concessive clause ("Despite high OPC-8:0 levels in opr3 mutants, JA was
                  # reduced", "Even though jar1 mutants accumulated JA-Ile, ...")
                  r"\b(?:although|though|despite|in spite of)\b")
#: The effect's subject is a protein or transcript measure, not a metabolite (CS-D D2).
_MEASURE = _any(r"\bproteins?\b", r"\babundance\b", r"\btranscripts?\b", r"\bmrnas?\b",
                r"\bexpression\b", r"\bexpressed\b", r"\bactivit(?:y|ies)\b")
#: A loss-of-function marker followed by the gene it is of: "knockout of GH3.10",
#: "T-DNA insertion lines of AOS", "mutants in JAR1", "The SALK_034543 insertion line of
#: GH3.10" (at most three tokens, then "of" / "in", then an optional "the"). No gap token is
#: itself "of" / "in" (CS-D repair 2) or names another gene's role (ORCH-746 CS-E repair 1,
#: BIO-D-R2-3 / RT-D-R2-3: "Mutants lacking a regulator of AOS", "with defects upstream of JAR1").
_MARKER_OF = re.compile(r"\s*(?:(?!(?:of|in|regulat\w*|upstream|downstream|genes?|factors?|activators?|repressors?|"
                        r"defects?|targets?)\b)[\w-]+\s+){0,3}?(?:of|in)\s+(?:the\s+)?")
#: The participant + enzyme branch of the review route (rule 7) keeps the strict set.
_CONVERSION = _any(r"\bconvert", r"\bconversion\b", r"formed from", r"metaboli[sz]ed to",
                   r"produced from", r"synthesi[sz]ed from")
_APPLICATION = _any(r"\btreat", r"\bappl(?:y|ied|ication)", r"\bspray", r"exogenous", r"\bfed\b",
                    r"\bfeeding\b",
                    # repair B6: a fed precursor is held back pending the product-owner hinge
                    r"\bsuppl(?:y|ied|ies|ement)", r"\binfiltrat", r"\binject", r"\bdosed?\b",
                    r"\badministered\b", r"\bwatered\b", r"\bsoaked\b", r"\bdipped\b",
                    r"\bperfused\b", r"\bdeuterated\b", r"\blabell?ed\b")
#: Hedge and modality (repair 2): a broad STEM set, matched at a word start, plus the
#: earlier phrases. Any one refuses positive credit and review credit.
_HEDGE = _any(r"\b(?:may|might|could|would|should|probabl|possibl|likely|unlikely|believ|thought|"
              r"suggest|propos|postulat|hypothe|presum|assum|apparent|seem|appear|putative|potential|"
              r"speculat|conceivabl|yet to|remain|unclear|unknown|whether|debat|controvers|elusive|"
              r"unresolved|expected|perhaps|anticipated|predicted|indicat|reportedly)",
              r"\bconsider(?:ed)? to\b", r"supports? the view", r"consistent with",
              r"\bpoints? (?:to|towards)\b", r"\bimpl(?:y|ies|ied)\b", r"open question",
              r"\bcannot (?:be )?(?:exclud|rul)", r"\bnot (?:be )?(?:exclud|ruled? out)")
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
    r"|co-?workers|\(\s*refs?\.?\s*\d|well[- ]known|it is (?:well )?established that"
    r"|as (?:we )?reported|is known to",
    re.IGNORECASE)
_COMPARISON = _any(r"\bunlike\b", r"in contrast", r"by contrast", r"\bwhereas\b",
                   r"compared (?:with|to)", r"relative to", r"equivalent to", r"\bversus\b",
                   r"counterpart",
                   # repair N2: the target named only by comparison
                   r"\bresembl", r"similar to", r"\banalogous\b", r"\bas in\b",
                   r"also (?:found|present) in", r"\bboth\b")
#: Any comparison at all refuses positive credit and review credit (repair 2): the
#: list above plus bare "as" and "like" and a homologue.
_COMPARISON_ANY = _any(_COMPARISON.pattern, r"\bas\b", r"\blike\b", r"in contrast to",
                       r"\bhomolog(?:ue)?s?\b")
#: The review route's negation veto (repair 2).
_NEGATION = _any(r"\b(?:not|no|never|neither|nor|none|without|cannot|unable|incapable|absent|absence|"
                 r"negligible)\b", r"\b(?:fail|lack|undetect)\w*", r"n't", r"rather than",
                 r"to be (?:determined|established|shown|demonstrated)")
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
    r"soybean|vitis|grapes?|grapevine|brassica|medicago|populus|"
    # CS-D D4: plant genera a jasmonate / oxylipin corpus hits ("glycine" only as the
    # binomial: the bare word is the amino acid)
    r"brachypodium|camelina|setaria|lotus|gossypium|petunia|catharanthus|salvia|mentha|artemisia|"
    r"panicum|sorghum|glycine max|"
    r"escherichia coli|e\. ?coli|yeast|"
    r"saccharomyces|pichia|human|humans|homo sapiens|mouse|mice|murine|rats?|drosophila|"
    r"zebrafish|liverworts?|moss|mosses|hornworts?|bryophytes?|algae|alga)\b"
)
#: A gene symbol carrying another species' prefix ("MpFAD5", "Mp FAD5", "Mpgh3a",
#: "Mp gh3a", "Mp-aos1", "PpAOC1", "CaLOX2") names that organism (repair B8, R2-B7).
#: Matched on the ORIGINAL text, always: a sentence naming the target AND a foreign gene
#: is a host or comparison construct, never native evidence. A documented ceiling like
#: the organism words: an unlisted prefix is not seen, and a word such as "Table1" can be
#: read as one (fail closed).
_FOREIGN_GENE = re.compile(
    r"\b(?:Mp|Pp|Os|Sl|Zm|Ta|Hv|Gm|Nt|Nb|Bn|Ca|Vv|Md|St|Le|Cs|Sb|Br|Mt|Pt|Lj|Ph|Ps|Sm|Gh|Bd|Sv|Ptr|Mtr)"
    r"(?:[A-Z]{2,}[0-9A-Za-z.]*|[\s-]?[a-z]{2,}\d[0-9a-z]*|[\s-][A-Z]{2,}\d?[0-9A-Za-z.]*)\b")
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
    folded = _SPACED_DASH.sub(", ", _NBSP.sub(" ", _nfkc(text))).lower().translate(_DASHES)
    for greek, latin in _GREEK.items():
        folded = folded.replace(greek, latin)
    folded = _DIGIT_COMMA.sub(",", _HYPHEN_GAP.sub("-", folded))
    return _WS.sub(" ", folded).strip()


def _nfkc(text: Any) -> str:
    """Compatibility-normalised text: a PDF ligature ("\ufb01") must not defeat a marker."""
    return unicodedata.normalize("NFKC", str(text or ""))


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
    text = _nfkc(text)
    folded = fold_text(text)
    hits = [m.group(0) for m in _ORGANISM_WORDS.finditer(folded)]
    hits += [m.group(0) for m in _ABBREVIATED_BINOMIAL.finditer(text)]
    out = [normalize_species(h) for h in hits]
    if _ARABIDOPSIS_MARKS.search(folded):
        out.append("arabidopsis thaliana")
    return list(dict.fromkeys(o for o in out if o))


def _found(pattern: "re.Pattern[str]", text: str) -> str:
    match = pattern.search(text)
    return match.group(0) if match else ""


def _without_target_comparisons(sentence: str, target: str) -> str:
    """``sentence`` with every comparison mention of the target removed ("as in
    Arabidopsis", "like its Arabidopsis homologue", "unlike (in) Arabidopsis", "similar
    to Arabidopsis", "compared with Arabidopsis"; repair 2): such a mention does not
    name the target as the organism of THIS statement."""
    words = target.split()
    if not words:
        return sentence
    names = [re.escape(words[0])]
    if len(words) > 1:
        names = [rf"{re.escape(words[0])}(?:\s+{re.escape(words[1])})?", rf"{words[0][0]}\.\s?{re.escape(words[1])}"]
    pattern = (r"\b(?:as in|unlike(?: in)?|like(?: its| the| in)?|similar to|in contrast to|compared (?:with|to)|"
               r"as (?:is the case|seen|observed) (?:in|for))\s+(?:the\s+)?(?:" + "|".join(names) + r")\b(?:\s+\w+)?")
    return re.sub(pattern, " ", _nfkc(sentence), flags=re.IGNORECASE)


def _names_target(sentence: str, target: str) -> bool:
    """Does ``sentence`` name the target as its own organism (comparisons removed)?"""
    return target in organisms_named(_without_target_comparisons(sentence, target))


def _others_named(sentence: str, target: str) -> List[str]:
    """Organisms other than the target that ``sentence`` names: organism words,
    abbreviated binomials, and ALWAYS a gene symbol carrying another species' prefix
    ("MpFAD5", "Mp gh3a"; repair B8, R2-B5): a sentence naming the target and a
    foreign gene is a host or comparison construct, never native evidence."""
    others = [o for o in organisms_named(sentence) if o != target]
    return others + [f"gene {m.group(0)!r}" for m in _FOREIGN_GENE.finditer(_nfkc(sentence))]


def _host_marker(sentence: str, folded: str, target: str) -> str:
    """The host-system marker of a sentence, or "". A foreign gene in a sentence that
    names the target is a host construct (repair 2); complementation counts only with
    a foreign-gene cue (repair B5), so native-gene complementation is not a host marker."""
    found = _found(_HOST, folded)
    if found:
        return found
    foreign = _FOREIGN_GENE.search(_nfkc(sentence))
    if foreign and _names_target(sentence, target):
        return f"a foreign gene ({foreign.group(0)!r}) in the target"
    if _COMPLEMENT.search(folded) and (_FOREIGN_CUE.search(folded) or _others_named(sentence, target)):
        return "complementation with a foreign gene"
    return ""


def species_span_names_target(spans: Iterable[Any], target_species_normalized: str) -> bool:
    """W13: a species span counts only if one of its sentences names the target
    plainly -- not in an ortholog, homology, host-system or comparison sentence
    ("Arabidopsis Ortholog" as a table header, "MpFAD5 is orthologous to AtFAD5, an
    Arabidopsis enzyme", "unlike Arabidopsis"), and alone (no other organism, no
    foreign gene symbol)."""
    target = str(target_species_normalized or "").strip().lower()
    for span in spans:
        for sentence in split_sentences(_nfkc(getattr(span, "text", ""))):
            if not _names_target(sentence, target):
                continue
            folded = fold_text(sentence)
            if (_INFERENCE.search(folded) or _HOMOLOGY.search(folded) or _host_marker(sentence, folded, target)
                    or _COMPARISON.search(folded)):
                continue
            if _others_named(sentence, target):
                continue
            return True
    return False


def _hits(text: str, matchers: Sequence[Sequence["re.Pattern[str]"]]) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for variants in matchers for p in variants for m in p.finditer(text)]


def _blank(text: str, matchers: Sequence[Sequence["re.Pattern[str]"]]) -> str:
    """``text`` with every match blanked to spaces (positions kept)."""
    for start, end in _hits(text, matchers):
        text = text[:start] + " " * (end - start) + text[end:]
    return text


def _bound(clause: str, cats: Sequence[Sequence["re.Pattern[str]"]]) -> bool:
    """Is a loss-of-function marker in ``clause`` OF the claim's own enzyme or gene (CS-D
    D2)? The symbol sits at most three tokens before the marker ("gh3.10 jar1 knockout
    lines", "the jar1 gh3.10 double mutant", "lox2 (At3g45140) knockout lines"), or right
    after it with "of" / "in" ("knockout of GH3.10", "T-DNA insertion lines of AOS").
    "arc mutants ... allene oxide cyclase" and "the myc2 mutant ... by JAR1" are not."""
    cat_hits = [(a, b) for a, b in _hits(clause, cats) if not _CONTAINER.match(clause, b)]
    for marker in _LOSS_OF_FUNCTION.finditer(clause):
        for start, end in cat_hits:
            if end <= marker.start():
                between = clause[end:marker.start()]
                if len(between.split()) <= 3 and not re.search(r"[,;]", between):
                    return True
            elif start >= marker.end() and _MARKER_OF.fullmatch(clause[marker.end():start]):
                return True
    return False


def _excluded(clause: str, hit: Tuple[int, int, str], form_start: int) -> bool:
    """Is a participant before an effect form cut off from it by a comma-led relative or
    adverbial clause (CS-D repair 2)? Either the clause that follows the participant is
    still OPEN at the form ("JA-Gln accumulated ..., which had reduced JA-Ile"), or the
    participant sits INSIDE such a clause that closes before the form ("OPC-8:0, which is
    formed from OPDA, was undetectable": OPDA is excluded, the head OPC-8:0 stays).

    The 686ae134 reading, kept verbatim as the first of :data:`_READINGS`: any comma after
    the first opener reads as the clause closing, so it is fooled by an inner comma."""
    gap = clause[hit[1]:form_start]
    opened = _NEW_SUBJECT.search(gap)
    if opened and "," not in gap[opened.end():]:
        return True
    opener = None
    for opener in _NEW_SUBJECT.finditer(clause[:hit[0]]):
        pass
    return bool(opener and "," not in clause[opener.end():hit[0]] and "," in gap)


#: A parenthetical ("(Fig. 3A, B)", "(3r,7s)", "(p < 0.01, t-test)"): the bio-auditor's
#: reading blanks it, so its commas close no clause.
_PARENTHETICAL = re.compile(r"\([^()]*\)")
#: In the bio-auditor's reading only a comma followed by whitespace closes a clause.
_CLAUSE_CLOSE = re.compile(r",\s")


def _excluded_blanked(clause: str, hit: Tuple[int, int, str], form_start: int) -> bool:
    """The bio-auditor's reading of :func:`_excluded` (ORCH-746 CS-E, BIO-D-R2-1;
    rev745d_r2_proto.py). Parentheticals are blanked and only ", " closes a clause; the
    LAST opener in the gap decides whether a clause is still open at the form ("JA-Ile,
    which ..., accumulated after wounding, with reduced 12-OH-JA-Ile": the ", with" clause
    is open, JA-Ile is cut off); a head is kept across a closed clause only in the shape
    "X, <opener> ..., <form>" with one opener right after X. Any other gap holding an
    opener cuts the participant off (fail closed)."""
    text = _PARENTHETICAL.sub(lambda m: " " * len(m.group(0)), clause)
    gap = text[hit[1]:form_start]
    openers = list(_NEW_SUBJECT.finditer(gap))
    if openers:
        if not _CLAUSE_CLOSE.search(gap[openers[-1].end():]):
            return True
        if len(openers) > 1 or gap[:openers[0].start()].strip():
            return True
    opener = None
    for opener in _NEW_SUBJECT.finditer(text[:hit[0]]):
        pass
    return bool(opener and not _CLAUSE_CLOSE.search(text[opener.end():hit[0]]) and _CLAUSE_CLOSE.search(gap))


def _excluded_adjacent(clause: str, hit: Tuple[int, int, str], form_start: int) -> bool:
    """The red team's reading of :func:`_excluded` (ORCH-746 CS-E, RT-D-R2-1; rt745dr2_fix.py):
    a head is kept across a closed clause only when the clause opens right after it, so that
    it modifies the head itself ("JA-Ile, which ..., accumulated"). A clause that opens after
    other words ("JA-Ile accumulated after wounding, which, in turn, ...", "OPC-8:0 levels,
    which ...") cuts the head off whatever commas it holds."""
    gap = clause[hit[1]:form_start]
    opened = _NEW_SUBJECT.search(gap)
    if opened and ("," not in gap[opened.end():] or gap[:opened.start()].strip()):
        return True
    opener = None
    for opener in _NEW_SUBJECT.finditer(clause[:hit[0]]):
        pass
    return bool(opener and "," not in clause[opener.end():hit[0]] and "," in gap)


def _excluded_unambiguous_close(clause: str, hit: Tuple[int, int, str], form_start: int) -> bool:
    """The code reviewer's reading (ORCH-746 CS-E repair 1, CSE-R0-1; rev746e2_r4.py): the
    686ae134 reading, plus, with parentheticals blanked, a participant is cut off when two or
    more ", " follow the opener before the form. One ", " closes a relative clause that modifies
    the head ("JA-Ile, which is the substrate of CYP94B3, accumulated"); two or more make the
    close ambiguous, as an appositive pair (", with 12-OH-JA-Ile, the major catabolite, reduced")
    or an inner pair (", which, in turn, resulted in reduced ...") inside a clause that is still
    open at the form, where the three readings above all kept the head."""
    if _excluded(clause, hit, form_start):
        return True
    text = _PARENTHETICAL.sub(lambda m: " " * len(m.group(0)), clause)
    gap = text[hit[1]:form_start]
    opened = _NEW_SUBJECT.search(gap)
    return bool(opened and len(_CLAUSE_CLOSE.findall(gap[opened.end():])) > 1)


#: "with" written without its comma (CSE-R0-N1), and the 686ae134 clause openers plus it.
_BARE_WITH = re.compile(r"(?<!,)\swith\b")
_NEW_SUBJECT_BARE_WITH = re.compile(_NEW_SUBJECT.pattern + "|" + _BARE_WITH.pattern)


def _excluded_bare_with(clause: str, hit: Tuple[int, int, str], form_start: int) -> bool:
    """The 686ae134 reading (:func:`_excluded`, the same lines) in which a bare " with " also
    opens a clause (ORCH-746 CS-E repair 1, CSE-R0-N1): "JA-Ile accumulated after wounding with
    reduced 12-OH-JA-Ile" cuts JA-Ile off from "reduced", as ", with reduced" already did. The
    686ae134 lines decide on the FIRST opener only, so a bare " with " after a closed relative
    clause ("JA-Ile, which is the substrate of CYP94B3 (Fig. 3A, B), accumulated after wounding
    with reduced 12-OH-JA-Ile") also cuts the participant off when no comma follows it."""
    gap = clause[hit[1]:form_start]
    if any("," not in gap[bare.end():] for bare in _BARE_WITH.finditer(gap)):
        return True
    opened = _NEW_SUBJECT_BARE_WITH.search(gap)
    if opened and "," not in gap[opened.end():]:
        return True
    opener = None
    for opener in _NEW_SUBJECT_BARE_WITH.finditer(clause[:hit[0]]):
        pass
    return bool(opener and "," not in clause[opener.end():hit[0]] and "," in gap)


#: The 686ae134 clause openers plus a comma before a participle or consequence word
#: (ORCH-746 CS-E repair 1, item 9). ponytail: any "-ing" word counts, so ", including",
#: ", during" open a clause too; that only cuts participants off (a reading can only refuse).
_NEW_SUBJECT_CONSEQUENCE = re.compile(
    _NEW_SUBJECT.pattern
    + r"|,\s*(?:\w+ing|accompanied|followed|coupled|paralleled|associated|yet|thus|thereby|hence|so)\b")


def _excluded_consequence(clause: str, hit: Tuple[int, int, str], form_start: int) -> bool:
    """The 686ae134 reading (:func:`_excluded`, the same lines) in which a participle or
    consequence clause also opens a clause (ORCH-746 CS-E repair 1, item 9): "JA-Ile
    accumulated after wounding, resulting in reduced 12-OH-JA-Ile levels" cuts JA-Ile off from
    "reduced", as ", which resulted in reduced" already did; so do ", leading to", ",
    accompanied by" and ", yet"."""
    gap = clause[hit[1]:form_start]
    opened = _NEW_SUBJECT_CONSEQUENCE.search(gap)
    if opened and "," not in gap[opened.end():]:
        return True
    opener = None
    for opener in _NEW_SUBJECT_CONSEQUENCE.finditer(clause[:hit[0]]):
        pass
    return bool(opener and "," not in clause[opener.end():hit[0]] and "," in gap)


#: The readings of which participants a comma-led clause cuts off (ORCH-746 CS-E); an effect
#: counts only when ALL of them credit it. The first three (CS-E): each reviewer prototype
#: misreads a shape the other reads right: the red team's keeps the head across a closed
#: relative clause while a SECOND comma-led clause is open at the form ("JA-Ile, which is the
#: substrate of CYP94B3, accumulated after wounding, with reduced 12-OH-JA-Ile" credits the
#: reversed reaction), and takes a callout's comma for a close; the bio-auditor's reads a
#: relative clause whose closing comma is missing as open to the form, and credits
#: correct-direction rows that the 686ae134 reading refuses, a recall gain that round forbade.
#: Repair 1 appends three: an ambiguous close (two or more ", " after the opener), a bare
#: " with ", and a participle or consequence clause. Agreement is the fail-closed choice, and
#: keeping the 686ae134 reading among them means a reading can only remove credit, never add
#: it; appending one removes credit only.
_READINGS = (_excluded, _excluded_blanked, _excluded_adjacent, _excluded_unambiguous_close, _excluded_bare_with,
             _excluded_consequence)


def _effect(clause: str, subs: Sequence[Sequence["re.Pattern[str]"]],
            prods: Sequence[Sequence["re.Pattern[str]"]]) -> str:
    """The effect of a loss-of-function clause on the right participant, or "": the
    686ae134 reading's reason when every reading of :data:`_READINGS` finds an effect
    (:func:`_effect_under`), else "" (ORCH-746 CS-E). Credit here is credit under each
    reading, so nothing is credited that 686ae134 did not credit."""
    reasons = [_effect_under(clause, subs, prods, excluded) for excluded in _READINGS]
    return reasons[0] if all(reasons) else ""


def _effect_under(clause: str, subs: Sequence[Sequence["re.Pattern[str]"]],
                  prods: Sequence[Sequence["re.Pattern[str]"]],
                  excluded: Callable[[str, Tuple[int, int, str], int], bool]) -> str:
    """The effect of a loss-of-function clause on the right participant under one reading
    of the cut-off rule (``excluded``), or "" (CS-D D1):
    an absence or decrease form whose participant is a PRODUCT, or an accumulation or
    increase form whose participant is a SUBSTRATE. ``clause`` has the claim's catalyst
    names blanked (D2), so an enzyme named after its substrate supplies no participant.
    A negation before the form in the clause refuses it.

    The form's subject (CS-D repair 1): the participants inside the form ("no JA-Ile was
    detected"); else every participant before it in the clause that a comma-led relative
    or adverbial clause does not cut off (``excluded``; none for a forward form);
    else the participants right after it, at most three tokens away (two for a forward
    form) with no comma between. If those candidates name BOTH a substrate and a product
    the subject is ambiguous and the form earns nothing ("the OPDA metabolite OPC-8:0
    accumulated", "13-HPOT formed from alpha-linolenic acid was undetectable"). An
    increase form never counts in a clause that also says something went down."""
    hits = sorted(h for h in [(a, b, "product") for a, b in _hits(clause, prods)]
                  + [(a, b, "substrate") for a, b in _hits(clause, subs)]
                  if not _ENZYME_HEAD.match(clause, h[1]))
    for pattern, side, forward in _EFFECT_FORMS:
        if side == "substrate" and (_DECREASE_WORD.search(clause) or _EFFECT_FORMS[0][0].search(clause)
                                    or _EFFECT_FORMS[1][0].search(clause)):
            continue
        for form in pattern.finditer(clause):
            if _CHANGE_NEGATION.search(clause[:form.start()]):
                continue
            inside = [h for h in hits if form.start() <= h[0] < form.end()]
            if inside and form.group(0).startswith("accumulated "):
                # ORCH-746 CS-E repair 1 (RT-E2-3): a participant inside the "accumulated ...
                # less" gap is not the decreased one ("accumulated OPDA with lower OPC-8:0")
                continue
            before = [] if forward else [h for h in hits if h[1] <= form.start()
                                         and not excluded(clause, h, form.start())]
            after = [h for h in hits if h[0] >= form.end()
                     and len(clause[form.end():h[0]].split()) <= (2 if forward else 3)
                     and not re.search(r"[,;]", clause[form.end():h[0]])]
            sides = {h[2] for h in (inside or before or after)}
            if sides == {side}:
                return f"{form.group(0)!r} on a {side}"
    return ""


def _linked(folded: str, subs: Sequence[Sequence["re.Pattern[str]"]],
            prods: Sequence[Sequence["re.Pattern[str]"]]) -> bool:
    """Are a substrate and a product LINKED by a conversion, not merely co-occurring
    (repair 2, rule 7)? "X ... <verb> ... to / into Y", "Y ... <verb> ... from X", or a
    "respectively" sentence with the verb between them."""
    sub_hits = [m for variants in subs for p in variants for m in p.finditer(folded)]
    prod_hits = [m for variants in prods for p in variants for m in p.finditer(folded)]
    respectively = "respectively" in folded
    for sub in sub_hits:
        for prod in prod_hits:
            if sub.end() <= prod.start():
                between = folded[sub.end():prod.start()]
                if _REVIEW_CONVERSION.search(between) and (re.search(r"\b(?:to|into)\b", between) or respectively):
                    return True
            elif prod.end() <= sub.start():
                between = folded[prod.end():sub.start()]
                if _REVIEW_CONVERSION.search(between) and re.search(r"\bfrom\b", between):
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
    ``host``, ``in_vitro``, then -- only in a primary paper -- ``in_planta`` from
    loss-of-function genetics alone (repair 2 and CS-D: a loss-of-function genotype OF the
    claim's own enzyme or gene, and in the same clause an absent or decreased product or
    an accumulated substrate, named apart from the catalyst's own name, with no negation
    before the effect; and no hedge, modality, no-effect phrase, application, comparison,
    procedure, protein mutant, protein or transcript measure, citation, other organism or
    no-credit section), or -- only in a
    review -- ``review`` under a strict rule 7 (a conversion LINKING substrate and
    product, or the claim's enzyme with a strict conversion verb, and no negation,
    hedge, comparison, homology, application or other organism). Positive credit
    names ``substrates`` / ``products`` strictly and never by a cofactor or a free
    amino acid. Returns one dict per supporting sentence:
    ``evidence_id``, ``sentence`` (index in its span), ``text``, ``grade``,
    ``reason``.
    """
    target = str(target_species_normalized or "").strip().lower()
    catalysts = _catalyst_names(enzyme, gene)
    loose = _matchers([*substrates, *products, *asserted, *catalysts], strict=False)
    subs = _matchers([n for n in substrates if _creditable(n)], strict=True)
    prods = _matchers([n for n in products if _creditable(n)], strict=True)
    # The claim's OWN enzyme or gene: a foreign-prefixed symbol never earns credit.
    cats = _matchers([c for c in catalysts if not _FOREIGN_GENE.search(c)], strict=True)
    every_catalyst = _matchers(catalysts, strict=True)
    paper_other = any(normalize_species(o) not in ("", target) for o in paper_observed_organisms)

    def in_planta(sentence: str, folded: str, span: Any, span_other: bool) -> Tuple[str, str]:
        """Positive credit in a primary paper: loss-of-function genetics only (repair 2)."""
        for pattern, why in ((_HEDGE, "hedged or modal"), (_NO_EFFECT, "no effect reported"),
                             (_APPLICATION, "an applied compound"), (_COMPARISON_ANY, "a comparison"),
                             (_PROCEDURAL, "a procedure, not a result"), (_PURPOSE, "a purpose clause"),
                             (_PROTEIN_MUTANT, "a protein mutant, not a plant genotype"),
                             (_MEASURE, "a protein or transcript measure, not a metabolite")):
            if pattern.search(folded):
                return GRADE_NONE, f"no positive credit: {why} ({_found(pattern, folded)!r})"
        if _SUBSTITUTION.search(sentence):
            return GRADE_NONE, (f"no positive credit: an amino-acid substitution "
                                f"({_found(_SUBSTITUTION, sentence)!r}) is a protein mutant")
        if _CITATION.search(sentence):
            return GRADE_NONE, f"no positive credit: cites other work ({_found(_CITATION, sentence)!r})"
        section = str(getattr(span, "section", "") or "").lower()
        if _NO_CREDIT_SECTION.search(section):
            return GRADE_NONE, f"no positive credit: {section} passage"
        other = _others_named(sentence, target)
        if other:
            return GRADE_NONE, f"no positive credit: names a non-target organism ({other[0]})"
        if not _names_target(sentence, target) and (paper_other or span_other):
            return GRADE_NONE, ("no positive credit: the passage or the paper names another organism "
                                "and this sentence does not name the target")
        for clause in _CLAUSE.split(folded):
            if not _bound(clause, cats) or _PHENOTYPE.search(clause):
                continue
            effect = _effect(_blank(clause, every_catalyst), subs, prods)
            if effect:
                return GRADE_IN_PLANTA, (f"in planta: loss of function of the claim's enzyme or gene "
                                         f"({_found(_LOSS_OF_FUNCTION, clause)!r}): {effect}")
        return GRADE_NONE, ("no positive credit: no loss-of-function genotype of the claim's own enzyme or "
                            "gene with an absent or decreased product, or an accumulated substrate, in one "
                            "clause")

    def review_route(sentence: str, folded: str) -> Tuple[str, str]:
        for pattern, why in ((_HEDGE, "hedged or modal"), (_NEGATION, "a negation"),
                             (_COMPARISON_ANY, "a comparison"), (_HOMOLOGY, "a homology transfer"),
                             (_APPLICATION, "an applied compound"), (_PROCEDURAL, "a procedure")):
            if pattern.search(folded):
                return GRADE_NONE, f"no review credit: {why} ({_found(pattern, folded)!r})"
        if not _names_target(sentence, target):
            return GRADE_NONE, "no review credit: the sentence does not name the target"
        if _others_named(sentence, target):
            return GRADE_NONE, "no review credit: the sentence names another organism too"
        sub, prod = _names(folded, subs), _names(folded, prods)
        if sub and prod and _linked(folded, subs, prods):
            return GRADE_REVIEW, "a review states this conversion for the target species"
        if (sub or prod) and _names(folded, cats) and _CONVERSION.search(folded):
            return GRADE_REVIEW, "a review states this enzyme's conversion for the target species"
        return GRADE_NONE, "no review credit: no conversion linking this reaction's participants"

    graded: List[Dict[str, Any]] = []
    for span in spans:
        text = _nfkc(getattr(span, "text", "") or "")
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
