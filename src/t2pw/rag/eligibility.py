"""Deterministic title/abstract eligibility screening, *before* full-text fetch.

Why this exists
---------------
The batch runner's acquisition step (``t2pw.batch.fetch``) turned every literature
hit with retrievable full text into a pipeline run. A topic query is a blunt
instrument, so the 2026-07-28_2122 plan spent two expensive runs each (LLM
extraction + PathBank mapping + RAG) on a Fournier's-gangrene case report, two
poultry ESBL virulence surveys, a COVID-19 lncRNA association study and a
gene-set-evolution tool -- none of which contain mechanistic information about the
requested pathway. Worse, those runs then showed up in the morning triage as
*extraction failures*, sending debugging into the extractor for papers that
should never have entered the pipeline.

This module is the missing gate: a **deterministic, offline** scorer over the
cheap metadata (title + abstract) that decides whether a paper is plausibly about
the *requested* pathway in the *requested* organism. It runs before
``fetch_full_text``, so an ineligible paper costs one keyword scan rather than a
full-text download and two app runs.

Requested vs observed -- the central distinction
------------------------------------------------
The old code stamped the requested organism straight onto every
``CandidatePaper`` (``organism=organism`` in each acquire fetcher), which made a
paper *claim* to report the organism we asked for. Downstream that is
indistinguishable from a paper that really did, and it silently defeated
``select._organism_score`` (always 1.0). So:

* :class:`RequestedScope` -- what the *batch asked for*. Frozen: nothing
  downstream, Stage 0 included, may rewrite it.
* :class:`ObservedContext` -- what the *paper itself* reports, read from its own
  title/abstract (plus, later, Stage 0's read of its full text).
* ``organism_match`` -- the comparison of the two: ``match`` / ``genus_level`` /
  ``mismatch`` / ``unknown``. A paper that names no organism is ``unknown``; the
  requested organism is never copied in to fill the hole.

Species precision
-----------------
``match`` means the same **species**: an exact binomial, or a strain-qualified
form of it (``Escherichia coli K-12`` matches ``Escherichia coli``). Same genus,
different species -- ``E. coli`` vs ``E. fergusonii``, ``B. subtilis`` vs
``B. cereus`` -- is ``genus_level``: permitted (it is not a mismatch, because the
mechanism often transfers) but carrying **no full organism bonus** and an explicit
warning. A bare genus never infers a species: a paper that says only
``Escherichia`` is ``genus_level`` against a request for ``Escherichia coli``, not
a match.

Contextual mechanistic evidence
-------------------------------
A pathway alias occurring *somewhere* and a generic word like "mechanism" or
"inhibition" occurring *elsewhere* is not evidence of mechanistic content -- that
combination is what let five cholesterol-adjacent immunology/cancer papers
through. Eligibility therefore requires either

* **(a)** a pathway-specific enzyme/metabolite term (``lpxc``, ``mend``,
  ``ppox``, ``entb``, ``squalene``, …), or
* **(b)** a pathway alias **together with** reaction/enzyme evidence in the same
  sentence, or within a bounded token window.

A paper with the alias but no local mechanistic evidence is classified
``context_only``; one with neither, but with omics/screening vocabulary, is
``omics_only``.

Determinism / offline
---------------------
Pure keyword and regex matching over fixed lexicons plus arithmetic. No network,
no LLM, no clock, no randomness -- the same (title, abstract, scope, thresholds)
always yields the same decision, which is what makes the screening auditable and
the dry-run reproducible. There is deliberately **no** LLM-based paper selector:
a model would make the gate non-reproducible and would cost exactly what the gate
exists to save. The exact screening input is persisted with the decision (bounded
abstract + SHA-256 of the full one) so a rejection can be reproduced offline.

Name normalization is a local 3-line copy of the ``map_ids._normalize_name`` idea
(lowercase, non-alphanumerics to spaces, collapse) rather than an import, so this
module -- and the offline dry-run tool that uses it -- never drags the
network-capable ``t2pw.mapping`` stack into an offline path. Same precedent as
``t2pw.rag.synonyms``. It also takes no ``CandidatePaper`` import: callers pass
either a paper-shaped object or plain strings, so ``t2pw.rag.acquire`` stays
free to import nothing from here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from t2pw.config import rag_config

# ---------------------------------------------------------------------------
# Outcomes. One explicit label per way a candidate can leave the gate, so a
# skipped paper never has to be inferred from a free-text reason string.
# ---------------------------------------------------------------------------
OUTCOME_ELIGIBLE = "eligible"
OUTCOME_INELIGIBLE_PATHWAY = "ineligible_pathway"
OUTCOME_INELIGIBLE_ORGANISM = "ineligible_organism"
OUTCOME_INSUFFICIENT_METADATA = "insufficient_metadata"
OUTCOME_DUPLICATE = "duplicate"
OUTCOME_NO_FULL_TEXT = "no_full_text"
OUTCOME_PINNED_OVERRIDE = "pinned_override"
#: Stage 0 read a pathway/organism that strongly contradicts the batch request.
#: Kept distinct from ``ineligible_pathway`` so the batch request is never
#: silently re-pointed at whatever the paper turned out to be about.
OUTCOME_SCOPE_CONFLICT = "scope_conflict"

ELIGIBILITY_OUTCOMES = (
    OUTCOME_ELIGIBLE,
    OUTCOME_INELIGIBLE_PATHWAY,
    OUTCOME_INELIGIBLE_ORGANISM,
    OUTCOME_INSUFFICIENT_METADATA,
    OUTCOME_DUPLICATE,
    OUTCOME_NO_FULL_TEXT,
    OUTCOME_PINNED_OVERRIDE,
    OUTCOME_SCOPE_CONFLICT,
)

#: Outcomes that mean "this paper was screened out". These are **not** pipeline
#: failures: nothing was attempted, so nothing failed. ``batch.report`` relies on
#: this set so a screened paper can never be counted as an extraction or
#: research-mode defect.
INELIGIBLE_OUTCOMES = frozenset(
    {
        OUTCOME_INELIGIBLE_PATHWAY,
        OUTCOME_INELIGIBLE_ORGANISM,
        OUTCOME_INSUFFICIENT_METADATA,
        OUTCOME_SCOPE_CONFLICT,
    }
)

#: Outcomes that let a paper through to the expensive pipeline.
ACCEPTING_OUTCOMES = frozenset({OUTCOME_ELIGIBLE, OUTCOME_PINNED_OVERRIDE})

# organism_match values.
ORGANISM_MATCH = "match"
#: Same genus, different (or unstated) species. Permitted, but weak: no full
#: organism bonus, an explicit warning, and never a "match".
ORGANISM_GENUS_LEVEL = "genus_level"
ORGANISM_MISMATCH = "mismatch"
ORGANISM_UNKNOWN = "unknown"

ORGANISM_MATCH_VALUES = (
    ORGANISM_MATCH,
    ORGANISM_GENUS_LEVEL,
    ORGANISM_MISMATCH,
    ORGANISM_UNKNOWN,
)

# Mechanistic classifications.
CLASS_MECHANISTIC = "mechanistic"
#: The pathway is named but nothing local to that mention is mechanistic.
CLASS_CONTEXT_ONLY = "context_only"
#: No pathway anchor at all, and the paper reads as an omics / screening study.
CLASS_OMICS_ONLY = "omics_only"
CLASS_OFF_TOPIC = "off_topic"


def outcome_is_ineligible(outcome: Any) -> bool:
    """True when ``outcome`` means the paper was screened out (never run)."""
    return str(outcome or "").strip().lower() in INELIGIBLE_OUTCOMES


def outcome_accepts(outcome: Any) -> bool:
    """True when ``outcome`` admits the paper to the pipeline."""
    return str(outcome or "").strip().lower() in ACCEPTING_OUTCOMES


# ---------------------------------------------------------------------------
# Normalization. Everything below matches against this one shape: lowercase,
# every non-alphanumeric run collapsed to a single space. That deliberately
# folds journal-title noise -- "&lt;i&gt;Escherichia coli&lt;/i&gt;",
# "UDP-GlcNAc", "2,3-dihydroxybenzoic" -- onto the same space-separated tokens
# the lexicons are written in, so a lexicon entry needs one spelling, not six.
# ---------------------------------------------------------------------------
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
#: Sentence boundary on the RAW text, before normalization erases punctuation.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")


def _normalize(value: Any) -> str:
    return _NON_ALNUM.sub(" ", str(value or "").casefold()).strip()


def _pattern(term: str) -> Optional[re.Pattern]:
    """Word-boundary matcher for one normalized term (``None`` when empty).

    Boundaries matter: ``kdo`` must not match inside ``kdometer``, ``mend`` must
    not match inside ``mending``, and -- the case that made this load-bearing --
    ``escherichia coli`` must not match inside ``escherichia colicinogenes``.
    Because the haystack is already reduced to ``[a-z0-9 ]``, ``\\b`` is exact.
    """
    norm = _normalize(term)
    if not norm:
        return None
    return re.compile(r"\b" + re.escape(norm) + r"\b")


class _TermSet:
    """A compiled, ordered set of terms matched against a normalized haystack.

    Order is the declaration order of the lexicon, so ``matched`` is stable and
    the resulting eligibility report diffs cleanly between runs.
    """

    __slots__ = ("_terms",)

    def __init__(self, terms: Iterable[str]) -> None:
        compiled: List[Tuple[str, re.Pattern]] = []
        seen: set = set()
        for term in terms:
            norm = _normalize(term)
            if not norm or norm in seen:
                continue
            pattern = _pattern(norm)
            if pattern is None:
                continue
            seen.add(norm)
            compiled.append((norm, pattern))
        self._terms = tuple(compiled)

    def matched(self, haystack: str) -> List[str]:
        return [term for term, pattern in self._terms if pattern.search(haystack)]

    def hits(self, haystack: str) -> bool:
        return any(pattern.search(haystack) for _term, pattern in self._terms)

    def spans(self, haystack: str) -> List[Tuple[int, int]]:
        """Character spans of every occurrence -- used for local-window checks."""
        out: List[Tuple[int, int]] = []
        for _term, pattern in self._terms:
            out.extend((m.start(), m.end()) for m in pattern.finditer(haystack))
        return sorted(out)


# ---------------------------------------------------------------------------
# Pathway lexicons.
#
# One entry per pathway the batch actually asks for (topics.txt). An unknown
# pathway is NOT rejected: it falls back to aliases generated from the requested
# name, so the gate still works -- just with less discriminating power. Adding an
# entry here is the way to sharpen screening for a new topic; nothing else has to
# change.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PathwayLexicon:
    """Pathway-specific evidence: extra name aliases plus enzyme/metabolite terms."""

    aliases: Tuple[str, ...] = ()
    terms: Tuple[str, ...] = ()


_PATHWAY_LEXICON: Dict[str, PathwayLexicon] = {
    "lipid a": PathwayLexicon(
        aliases=(
            "lipid a biosynthesis",
            "lipid a synthesis",
            "lipid a biosynthetic",
            "lipid a modification",
            "raetz pathway",
            "lipopolysaccharide biosynthesis",
            "lps biosynthesis",
        ),
        terms=(
            "lpxa",
            "lpxb",
            "lpxc",
            "lpxd",
            "lpxh",
            "lpxk",
            "lpxl",
            "lpxm",
            "lpxp",
            "lpxo",
            "kdta",
            "waaa",
            "msba",
            "kdo",
            "udp glcnac",
            "udp n acetylglucosamine",
            "hydroxymyristoyl",
            "myristoyl",
            "lauroyl",
            "3 deoxy d manno oct 2 ulosonic",
            "phosphoethanolamine transferase",
            "eptb",
            "arnt",
            "pagp",
        ),
    ),
    "menaquinone": PathwayLexicon(
        aliases=(
            "menaquinone biosynthesis",
            "menaquinone synthesis",
            "menaquinone biosynthetic",
            "vitamin k2 biosynthesis",
            "vitamin k biosynthesis",
            "futalosine pathway",
        ),
        terms=(
            "mena",
            "menb",
            "menc",
            "mend",
            "mene",
            "menf",
            "meng",
            "menh",
            "meni",
            "menj",
            "ubie",
            "mqna",
            "1 4 dihydroxy 2 naphthoate",
            "dhna",
            "o succinylbenzoate",
            "sephchc",
            "isochorismate",
            "demethylmenaquinone",
            "menaquinone 7",
            "mk 7",
            "vitamin k2",
        ),
    ),
    "heme": PathwayLexicon(
        aliases=(
            "heme biosynthesis",
            "heme synthesis",
            "haem biosynthesis",
            "haem synthesis",
            "heme biosynthetic",
            "porphyrin biosynthesis",
            "tetrapyrrole biosynthesis",
        ),
        terms=(
            "alas1",
            "alas2",
            "alad",
            "hmbs",
            "uros",
            "urod",
            "cpox",
            "ppox",
            "fech",
            "ferrochelatase",
            "aminolevulinic",
            "aminolaevulinic",
            "protoporphyrin",
            "protoporphyrinogen",
            "coproporphyrinogen",
            "uroporphyrinogen",
            "porphobilinogen",
            "hydroxymethylbilane",
            "porphyria",
            "slc25a38",
        ),
    ),
    "enterobactin": PathwayLexicon(
        aliases=(
            "enterobactin biosynthesis",
            "enterobactin synthesis",
            "enterobactin biosynthetic",
            "enterochelin biosynthesis",
            "siderophore biosynthesis",
        ),
        terms=(
            "enta",
            "entb",
            "entc",
            "entd",
            "ente",
            "entf",
            "fepa",
            "2 3 dihydroxybenzoate",
            "2 3 dihydroxybenzoic",
            "isochorismatase",
            "isochorismate synthase",
            "enterochelin",
            "aryl carrier protein",
        ),
    ),
    "cholesterol": PathwayLexicon(
        aliases=(
            "cholesterol biosynthesis",
            "cholesterol synthesis",
            "cholesterol biosynthetic",
            "sterol biosynthesis",
            "mevalonate pathway",
            "bloch pathway",
            "kandutsch russell pathway",
        ),
        terms=(
            "hmgcr",
            "hmg coa reductase",
            "hmgcs1",
            "squalene",
            "lanosterol",
            "mevalonate",
            "mevalonic",
            "dhcr7",
            "dhcr24",
            "sqle",
            "fdft1",
            "cyp51a1",
            "msmo1",
            "nsdhl",
            "sc5d",
            "idi1",
            "farnesyl diphosphate",
            "geranylgeranyl",
            "7 dehydrocholesterol",
            "desmosterol",
            "zymosterol",
        ),
    ),
}

# Words that make a phrase a *pathway-name* alias rather than a bare compound
# name. Used both to generate aliases for an unknown pathway and to strip the
# head compound out of a requested name.
_PATHWAY_HEADS = (
    "biosynthesis",
    "synthesis",
    "biosynthetic",
    "metabolism",
    "degradation",
    "pathway",
)
_ALIAS_SUFFIXES = (
    "biosynthesis",
    "synthesis",
    "biosynthetic pathway",
    "biosynthetic",
    "biogenesis",
    "pathway",
)


# ---------------------------------------------------------------------------
# Organism lexicon.
#
# ``aliases`` are SPECIES-level spellings only -- the full binomial and its
# abbreviated form. Bare genus names are deliberately NOT aliases: "Escherichia"
# alone must never be read as "Escherichia coli". Genus names are collected
# separately in ``_KNOWN_GENERA`` and drive two things: a binomial scan that can
# observe a species this table has never heard of (``Escherichia fergusonii``),
# and a genus-only observation when no species epithet follows.
#
# ``genus`` is empty for canonical names that are not binomials (viruses); those
# compare by exact equality only.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrganismEntry:
    canonical: str
    aliases: Tuple[str, ...]
    genus: str = ""


def _entry(canonical: str, *aliases: str, genus: Optional[str] = None) -> OrganismEntry:
    inferred = genus if genus is not None else canonical.split()[0]
    return OrganismEntry(canonical=canonical, aliases=tuple(aliases), genus=inferred)


_ORGANISM_ENTRIES: Tuple[OrganismEntry, ...] = (
    _entry("Escherichia coli", "escherichia coli", "e coli"),
    _entry("Escherichia fergusonii", "escherichia fergusonii", "e fergusonii"),
    _entry("Bacillus subtilis", "bacillus subtilis", "b subtilis"),
    _entry("Bacillus cereus", "bacillus cereus", "b cereus"),
    _entry("Listeria monocytogenes", "listeria monocytogenes", "l monocytogenes"),
    _entry("Klebsiella pneumoniae", "klebsiella pneumoniae", "k pneumoniae"),
    _entry("Pseudomonas aeruginosa", "pseudomonas aeruginosa", "p aeruginosa"),
    _entry("Pseudomonas putida", "pseudomonas putida", "p putida"),
    _entry("Acinetobacter baumannii", "acinetobacter baumannii", "a baumannii"),
    _entry("Salmonella enterica", "salmonella enterica", "s enterica", "s typhimurium"),
    _entry("Staphylococcus aureus", "staphylococcus aureus", "s aureus", "mrsa"),
    _entry("Streptococcus pneumoniae", "streptococcus pneumoniae", "s pneumoniae"),
    _entry("Enterococcus faecalis", "enterococcus faecalis", "e faecalis"),
    _entry("Mycobacterium tuberculosis", "mycobacterium tuberculosis", "m tuberculosis"),
    _entry("Ochrobactrum anthropi", "ochrobactrum anthropi", "o anthropi"),
    _entry("Burkholderia cenocepacia", "burkholderia cenocepacia"),
    _entry("Vibrio cholerae", "vibrio cholerae", "v cholerae"),
    _entry("Helicobacter pylori", "helicobacter pylori", "h pylori"),
    _entry("Campylobacter jejuni", "campylobacter jejuni", "c jejuni"),
    _entry(
        "Clostridioides difficile",
        "clostridioides difficile",
        "clostridium difficile",
        "c difficile",
    ),
    _entry("Lactococcus lactis", "lactococcus lactis", "l lactis"),
    _entry(
        "Lactobacillus plantarum",
        "lactobacillus plantarum",
        "lactiplantibacillus plantarum",
    ),
    _entry("Corynebacterium glutamicum", "corynebacterium glutamicum", "c glutamicum"),
    _entry("Yersinia pestis", "yersinia pestis", "y pestis"),
    _entry("Shigella flexneri", "shigella flexneri", "s flexneri"),
    _entry("Proteus mirabilis", "proteus mirabilis", "p mirabilis"),
    _entry("Serratia marcescens", "serratia marcescens", "s marcescens"),
    _entry("Enterobacter cloacae", "enterobacter cloacae", "e cloacae"),
    _entry("Neisseria gonorrhoeae", "neisseria gonorrhoeae", "n gonorrhoeae"),
    _entry("Brucella abortus", "brucella abortus", "b abortus"),
    _entry(
        "Saccharomyces cerevisiae",
        "saccharomyces cerevisiae",
        "s cerevisiae",
        "budding yeast",
    ),
    _entry("Candida albicans", "candida albicans", "c albicans"),
    _entry("Arabidopsis thaliana", "arabidopsis thaliana", "a thaliana"),
    _entry("Quercus robur", "quercus robur", "q robur"),
    _entry(
        "Homo sapiens",
        "homo sapiens",
        "human",
        "humans",
        "patient",
        "patients",
        "hela",
        "hek293",
    ),
    _entry("Mus musculus", "mus musculus", "mouse", "mice", "murine"),
    _entry("Rattus norvegicus", "rattus norvegicus", "rat", "rats"),
    _entry(
        "Gallus gallus",
        "gallus gallus",
        "chicken",
        "chickens",
        "broiler",
        "broilers",
        "poultry",
    ),
    _entry("Meleagris gallopavo", "meleagris gallopavo", "turkey", "turkeys"),
    _entry(
        "Sus scrofa",
        "sus scrofa",
        "pig",
        "pigs",
        "piglet",
        "piglets",
        "swine",
        "porcine",
    ),
    _entry("Bos taurus", "bos taurus", "cattle", "bovine", "calves"),
    _entry("Danio rerio", "danio rerio", "zebrafish"),
    _entry("Drosophila melanogaster", "drosophila melanogaster", "d melanogaster"),
    # Non-binomial canonicals: genus="" -> compared by exact equality only.
    _entry("SARS-CoV-2", "sars cov 2", "covid 19", "covid19", "covid", genus=""),
    _entry("Influenza A virus", "influenza a virus", "influenza", genus=""),
)

_ORGANISM_BY_CANONICAL: Dict[str, OrganismEntry] = {
    _normalize(e.canonical): e for e in _ORGANISM_ENTRIES
}
_ORGANISM_BY_ALIAS: Dict[str, OrganismEntry] = {}
for _e in _ORGANISM_ENTRIES:
    _ORGANISM_BY_ALIAS.setdefault(_normalize(_e.canonical), _e)
    for _a in _e.aliases:
        _ORGANISM_BY_ALIAS.setdefault(_normalize(_a), _e)

#: normalized genus token -> display genus. Built from the entries, so adding an
#: entry widens the binomial scan automatically.
_KNOWN_GENERA: Dict[str, str] = {
    _normalize(e.genus): e.genus for e in _ORGANISM_ENTRIES if e.genus
}

#: Tokens that are never a species epithet, so ``Escherichia and Klebsiella``
#: does not become the species ``Escherichia and``.
_NOT_SPECIES = frozenset(
    {
        "and",
        "or",
        "the",
        "in",
        "of",
        "from",
        "was",
        "were",
        "is",
        "are",
        "has",
        "have",
        "spp",
        "species",
        "strain",
        "strains",
        "isolate",
        "isolates",
        "sp",
        "genus",
        "cells",
        "cell",
        "bacteria",
        "as",
        "with",
        "that",
        "this",
        "for",
        "but",
        "not",
        "can",
        "may",
        "which",
        "when",
        "also",
        "such",
    }
)

#: Animal/agricultural hosts. A bacterial-pathway request plus one of these plus
#: virulence/typing language is the poultry-survey shape.
_ANIMAL_HOST_ORGANISMS = frozenset(
    {
        "Gallus gallus",
        "Meleagris gallopavo",
        "Sus scrofa",
        "Bos taurus",
        "Mus musculus",
        "Rattus norvegicus",
        "Danio rerio",
    }
)


# ---------------------------------------------------------------------------
# Generic (pathway-independent) evidence vocabularies.
# ---------------------------------------------------------------------------
#: Biochemical reaction / mechanism language. This is what distinguishes a paper
#: that *works out* a step from one that merely names the pathway.
_MECHANISM_TERMS = _TermSet(
    (
        "catalyzes",
        "catalyses",
        "catalyzed",
        "catalysed",
        "catalytic",
        "catalysis",
        "enzymatic",
        "enzymatically",
        "substrate",
        "substrates",
        "cofactor",
        "kinetic",
        "kinetics",
        "km",
        "kcat",
        "active site",
        "intermediate",
        "intermediates",
        "condensation",
        "decarboxylation",
        "decarboxylase",
        "transamination",
        "acylation",
        "acyltransferase",
        "methylation",
        "methyltransferase",
        "phosphorylation",
        "prenylation",
        "reduction",
        "oxidation",
        "hydrolysis",
        "hydrolyzes",
        "isomerization",
        "rate limiting",
        "reaction mechanism",
        "mechanism",
        "mechanistic",
        "inhibitor",
        "inhibition",
        "feedback",
        "flux",
        "stoichiometry",
        "committed step",
    )
)

#: Pathway-reconstruction / enzyme-characterization language: the paper is doing
#: the kind of work whose output the pipeline can actually extract.
_RECONSTRUCTION_TERMS = _TermSet(
    (
        "pathway reconstruction",
        "reconstitution",
        "reconstituted",
        "biosynthetic pathway",
        "biosynthetic gene cluster",
        "gene cluster",
        "heterologous expression",
        "enzyme characterization",
        "enzymatic characterization",
        "biochemical characterization",
        "kinetic characterization",
        "functional characterization",
        "purified enzyme",
        "recombinant enzyme",
        "in vitro assay",
        "enzyme assay",
        # NOT "metabolic pathway": every review that mentions a pathway in passing
        # says it, so it admitted immunology and cancer-biology papers whose only
        # link to the request was the phrase itself.
        "structural basis",
        "crystal structure",
        "metabolic engineering",
        "pathway engineering",
        "knockout",
        "deletion mutant",
    )
)

#: Omics / screening vocabulary. Not negative on its own. Two jobs: it classifies
#: a paper with no pathway anchor as ``omics_only``, and -- the load-bearing one --
#: when it is *local to a pathway-term mention* it means that term is a hit in a
#: screen's gene list, not an enzyme the paper works on. Three of the four
#: cholesterol false positives were exactly that: SQLE / CYP51A1 / DHCR24 named as
#: differentially expressed proteins or transcripts.
_OMICS_TERMS = _TermSet(
    (
        "bioinformatics",
        "transcriptomic",
        "transcriptome",
        "transcriptomics",
        "proteomic",
        "proteome",
        "proteomics",
        "metabolomic",
        "metabolomics",
        "gene set",
        "gene sets",
        "gene family",
        "expression profiling",
        "differentially expressed",
        "upregulated",
        "downregulated",
        "up regulated",
        "down regulated",
        "enrichment analysis",
        "enriched",
        "integrative analysis",
        "multi omics",
        "rna seq",
        "rna sequencing",
        "sequencing",
        "single cell",
        "genome wide",
        "machine learning",
        "prognostic",
        "signature",
    )
)

#: Reaction/enzyme evidence STRONG enough to make a pathway mention mechanistic.
#:
#: Deliberately narrower than ``_MECHANISM_TERMS``. The words left out --
#: "mechanism", "mechanistic", "inhibition", "inhibitor", "feedback", "flux",
#: "reduction", "oxidation" -- appear in the framing sentence of almost any
#: molecular-biology abstract, so accepting them here is what "a pathway alias
#: somewhere and a generic word elsewhere" looks like once the window is closed.
#: A bare ``-ase`` word is likewise not strong: a gene symbol in a hit list ends
#: in "-ase" too.
_STRONG_MECHANISM_TERMS = _TermSet(
    (
        "catalyzes",
        "catalyses",
        "catalyzed",
        "catalysed",
        "catalytic",
        "catalysis",
        "enzymatic",
        "enzymatically",
        "substrate",
        "substrates",
        "cofactor",
        "kinetic",
        "kinetics",
        "km",
        "kcat",
        "active site",
        "intermediate",
        "intermediates",
        "condensation",
        "decarboxylation",
        "transamination",
        "acylation",
        "prenylation",
        "hydrolysis",
        "hydrolyzes",
        "hydrolyses",
        "isomerization",
        "rate limiting",
        "committed step",
        "reaction mechanism",
        "precursor",
        "precursors",
        "converts",
        "conversion of",
        "biosynthetic step",
        # Characterization / reconstruction: the paper is doing the work.
        "pathway reconstruction",
        "reconstitution",
        "reconstituted",
        "biosynthetic gene cluster",
        "heterologous expression",
        "enzyme characterization",
        "enzymatic characterization",
        "biochemical characterization",
        "kinetic characterization",
        "purified enzyme",
        "purified",
        "recombinant enzyme",
        "in vitro assay",
        "enzyme assay",
        "structural basis",
        "crystal structure",
        "metabolic engineering",
        "pathway engineering",
        "knockout",
        "knockdown",
        "deletion mutant",
        "overexpression",
        # NOT "biosynthesis" / "biosynthetic": those words are *inside* the pathway
        # aliases themselves ("cholesterol biosynthesis"), so including them let
        # every alias certify itself as mechanistic.
    )
)

#: Words ending in "-ase" that are not enzymes.
_NOT_ENZYMES = frozenset(
    {
        "increase",
        "increases",
        "decrease",
        "decreases",
        "disease",
        "diseases",
        "release",
        "releases",
        "database",
        "databases",
        "purchase",
        "please",
        "case",
        "cases",
        "phase",
        "phases",
        "base",
        "bases",
        "use",
        "cease",
    }
)
_ENZYME_WORD = re.compile(r"\b([a-z][a-z0-9]{4,}ases?)\b")
_EC_NUMBER = re.compile(r"\bec \d+ \d+ \d+ \d+\b")

# --- negative-evidence vocabularies ---------------------------------------
_CASE_REPORT_TERMS = _TermSet(
    (
        "case report",
        "case reports",
        "a case of",
        "rare case",
        "report of a rare case",
        "case series",
        "clinical case",
        "autopsy",
        "we report a patient",
        "first reported case",
    )
)

_EPIDEMIOLOGY_TERMS = _TermSet(
    (
        "prevalence",
        "epidemiology",
        "epidemiological",
        "seroprevalence",
        "cross sectional",
        "cohort study",
        "retrospective study",
        "questionnaire",
        "comorbidity",
        "comorbidities",
        "risk factors",
        "surveillance",
        "antimicrobial susceptibility",
        "antibiogram",
        "incidence of",
    )
)

_VIRULENCE_TERMS = _TermSet(
    (
        "virulence",
        "virulence gene",
        "virulence factors",
        "serotype",
        "serotypes",
        "serogroup",
        "pathotype",
        "phylogroup",
        "antimicrobial resistance",
        "esbl",
        "extended spectrum",
        "resistome",
        "whole genome sequencing",
        "isolates",
    )
)

_SOFTWARE_TERMS = _TermSet(
    (
        "web server",
        "webserver",
        "web tool",
        "web application",
        "software",
        "r package",
        "python package",
        "command line tool",
        "toolkit",
        "graphical user interface",
        "user friendly interface",
        "database resource",
        "workflow manager",
        "we present a tool",
        "in silico platform",
    )
)

#: Wet-lab signals that disqualify the "software-only" verdict.
_WET_LAB_TERMS = _TermSet(
    (
        "in vitro",
        "in vivo",
        "assay",
        "purified",
        "mutant",
        "knockout",
        "enzymatic",
        "expressed",
        "measured",
        "cells were",
        "experimental validation",
    )
)

# Negative categories and their weights. Categories in ``_VETO_NEGATIVES`` are
# decisive on their own (subject to the ``negative_veto`` threshold flag): a case
# report that happens to name the pathway in its title is still a case report.
_NEG_INCOMPATIBLE_ORGANISM = "incompatible_organism"
_NEG_CASE_REPORT = "clinical_case_report"
_NEG_EPIDEMIOLOGY = "epidemiology_survey"
_NEG_ANIMAL_VIRULENCE = "animal_virulence_survey"
_NEG_SOFTWARE_ONLY = "software_only"
_NEG_BACKGROUND_ONLY = "pathway_only_in_background"
_NEG_CONTEXT_ONLY = "pathway_context_only"
_NEG_NO_MECHANISM = "no_mechanistic_pathway_terms"

_NEGATIVE_WEIGHTS: Dict[str, float] = {
    _NEG_INCOMPATIBLE_ORGANISM: 3.0,
    _NEG_CASE_REPORT: 3.0,
    _NEG_EPIDEMIOLOGY: 3.0,
    _NEG_ANIMAL_VIRULENCE: 3.0,
    _NEG_SOFTWARE_ONLY: 3.0,
    _NEG_BACKGROUND_ONLY: 1.5,
    _NEG_CONTEXT_ONLY: 1.5,
    _NEG_NO_MECHANISM: 1.5,
}

_VETO_NEGATIVES = frozenset(
    {
        _NEG_CASE_REPORT,
        _NEG_EPIDEMIOLOGY,
        _NEG_ANIMAL_VIRULENCE,
        _NEG_SOFTWARE_ONLY,
    }
)

# Positive categories: (weight per distinct hit, category cap).
_POS_PATHWAY_ALIAS = "pathway_alias"
_POS_PATHWAY_TERM = "pathway_term"
_POS_PATHWAY_HEAD = "pathway_head"
_POS_MECHANISM = "mechanism_term"
_POS_ENZYME = "enzyme_term"
_POS_ORGANISM = "organism_match"
_POS_ORGANISM_GENUS = "organism_genus_level"
_POS_RECONSTRUCTION = "reconstruction_term"

_POSITIVE_WEIGHTS: Dict[str, Tuple[float, float]] = {
    _POS_PATHWAY_ALIAS: (2.0, 2.0),
    _POS_PATHWAY_TERM: (1.5, 3.0),
    _POS_PATHWAY_HEAD: (0.5, 0.5),
    _POS_MECHANISM: (0.75, 1.5),
    _POS_ENZYME: (0.5, 1.0),
    _POS_ORGANISM: (1.0, 1.0),
    # A quarter of the species bonus: same genus is weak, related-taxon evidence.
    # Deliberately NOT the full organism bonus (see module docstring).
    _POS_ORGANISM_GENUS: (0.25, 0.25),
    _POS_RECONSTRUCTION: (1.0, 2.0),
}

#: Categories that can anchor a paper to the REQUESTED pathway. The bare head
#: compound (``cholesterol``) is deliberately excluded: naming the molecule is
#: not evidence the paper is about its biosynthesis, and treating it as an anchor
#: let cholesterol-signalling and cholesterol-in-cancer papers through. A
#: ``pathway_alias`` anchors only with LOCAL mechanistic evidence -- see
#: :func:`_local_mechanistic_evidence`.
_ANCHOR_CATEGORIES = frozenset({_POS_PATHWAY_ALIAS, _POS_PATHWAY_TERM})


# ---------------------------------------------------------------------------
# Requested vs observed scope.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RequestedScope:
    """What the batch asked for. **Immutable by construction.**

    ``frozen=True`` is the enforcement mechanism for the rule that Stage 0 may
    not rewrite the request: an assignment raises ``FrozenInstanceError`` rather
    than silently re-pointing the batch at whatever the paper turned out to be
    about. Stage 0's reading of the paper goes into :class:`ObservedContext`
    instead (see :func:`apply_stage0_observation`).
    """

    requested_pathway: str = ""
    requested_organism: str = ""
    pinned: bool = False

    @property
    def pathway_aliases(self) -> Tuple[str, ...]:
        """Name aliases for the requested pathway (lexicon + generated forms)."""
        return _pathway_aliases(self.requested_pathway)

    @property
    def pathway_terms(self) -> Tuple[str, ...]:
        """Expected enzyme / metabolite terms for the requested pathway."""
        return _pathway_terms(self.requested_pathway)

    @property
    def organism_aliases(self) -> Tuple[str, ...]:
        """Species-level taxonomy aliases for the requested organism."""
        return _organism_aliases(self.requested_organism)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_pathway": self.requested_pathway,
            "requested_organism": self.requested_organism,
            "pinned": self.pinned,
        }


@dataclass
class ObservedContext:
    """What the paper itself reports. Mutable: this is the half Stage 0 may fill.

    ``aliases`` and ``ambiguities`` exist so Stage 0 can contribute the extra
    naming/scope information it reads out of the full text (alternative pathway
    names, "this review covers three examples") without touching the request.
    """

    observed_pathways: List[str] = field(default_factory=list)
    observed_organisms: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observed_pathways": list(self.observed_pathways),
            "observed_organisms": list(self.observed_organisms),
            "aliases": list(self.aliases),
            "ambiguities": list(self.ambiguities),
        }

    def merge(self, other: "ObservedContext") -> "ObservedContext":
        """Union with ``other``, preserving first-seen order (no request fields)."""
        return ObservedContext(
            observed_pathways=_dedup(self.observed_pathways + other.observed_pathways),
            observed_organisms=_dedup(
                self.observed_organisms + other.observed_organisms
            ),
            aliases=_dedup(self.aliases + other.aliases),
            ambiguities=_dedup(self.ambiguities + other.ambiguities),
        )


def _dedup(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for item in items:
        text = str(item or "").strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


# ---------------------------------------------------------------------------
# Thresholds (configurable + deterministic).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EligibilityThresholds:
    """Every tunable number the gate uses, in one frozen object.

    Resolved from the environment by :func:`thresholds_from_config` so a run's
    behavior is reproducible from its config, and frozen so nothing can drift
    mid-run. ``title_only_min_score`` is the lower bar used when no abstract is
    available: a title carries fewer positive terms than a title+abstract, so
    scoring it against the full bar would reject almost everything. Decisions
    taken on a title alone are marked ``provisional``.
    """

    enabled: bool = True
    min_score: float = 2.0
    title_only_min_score: float = 1.5
    min_title_chars: int = 20
    require_pathway_anchor: bool = True
    organism_veto: bool = True
    negative_veto: bool = True
    review_margin: float = 0.5
    #: Tokens either side of a pathway-alias hit that count as "local" when
    #: looking for reaction/enzyme evidence in the same breath.
    local_window_tokens: int = 12
    #: Hard ceiling on candidates examined per topic before giving up on filling
    #: the requested count (see ``batch.fetch.fetch_papers``).
    candidate_ceiling: int = 60
    #: A Stage-0 reading that contradicts the request stops the run instead of
    #: only annotating it.
    stage0_conflict_aborts: bool = True

    def threshold_for(self, *, title_only: bool) -> float:
        return self.title_only_min_score if title_only else self.min_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_score": self.min_score,
            "title_only_min_score": self.title_only_min_score,
            "min_title_chars": self.min_title_chars,
            "require_pathway_anchor": self.require_pathway_anchor,
            "organism_veto": self.organism_veto,
            "negative_veto": self.negative_veto,
            "review_margin": self.review_margin,
            "local_window_tokens": self.local_window_tokens,
            "candidate_ceiling": self.candidate_ceiling,
            "stage0_conflict_aborts": self.stage0_conflict_aborts,
        }


def thresholds_from_config(
    overrides: Optional[Mapping[str, Any]] = None
) -> EligibilityThresholds:
    """Build :class:`EligibilityThresholds` from ``rag_config`` (env / ``.env``).

    ``overrides`` is passed straight through to :func:`t2pw.config.rag_config`,
    so a caller (or a test) can pin any threshold without touching the process
    environment.
    """
    cfg = rag_config(dict(overrides) if overrides else None)
    return EligibilityThresholds(
        enabled=bool(cfg["eligibility_enabled"]),
        min_score=float(cfg["eligibility_min_score"]),
        title_only_min_score=float(cfg["eligibility_title_only_min_score"]),
        min_title_chars=int(cfg["eligibility_min_title_chars"]),
        require_pathway_anchor=bool(cfg["eligibility_require_pathway_anchor"]),
        organism_veto=bool(cfg["eligibility_organism_veto"]),
        negative_veto=bool(cfg["eligibility_negative_veto"]),
        review_margin=float(cfg["eligibility_review_margin"]),
        local_window_tokens=int(cfg["eligibility_local_window_tokens"]),
        candidate_ceiling=int(cfg["eligibility_candidate_ceiling"]),
        stage0_conflict_aborts=bool(cfg["eligibility_stage0_conflict_aborts"]),
    )


# ---------------------------------------------------------------------------
# Persisted screening input.
# ---------------------------------------------------------------------------
#: How much abstract is stored with a decision. Generous enough that a real
#: abstract is never truncated (they run 1-3k chars); truncation is flagged.
MAX_PERSISTED_ABSTRACT = 4000


@dataclass
class ScreeningInput:
    """The exact bounded text a decision was taken on.

    ``abstract_sha256`` is of the **full supplied** abstract, not the stored
    slice. When the source is longer than :data:`MAX_PERSISTED_ABSTRACT`, the
    decision is deliberately taken on the stored slice so an offline replay sees
    exactly the same evidence. The full length and digest preserve an audit link
    to the original input without persisting unbounded text.
    """

    title: str = ""
    abstract: str = ""
    abstract_chars: int = 0
    abstract_sha256: str = ""
    abstract_truncated: bool = False
    abstract_source: str = ""
    abstract_authoritative: bool = True

    @classmethod
    def build(
        cls,
        title: str,
        abstract: str,
        *,
        source: str = "",
        authoritative: bool = True,
    ) -> "ScreeningInput":
        text = str(abstract or "")
        digest = (
            hashlib.sha256(text.encode("utf-8", "replace")).hexdigest() if text else ""
        )
        return cls(
            title=str(title or ""),
            abstract=text[:MAX_PERSISTED_ABSTRACT],
            abstract_chars=len(text),
            abstract_sha256=digest,
            abstract_truncated=len(text) > MAX_PERSISTED_ABSTRACT,
            abstract_source=source or ("supplied" if text else "none"),
            abstract_authoritative=bool(authoritative),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "abstract_chars": self.abstract_chars,
            "abstract_sha256": self.abstract_sha256,
            "abstract_truncated": self.abstract_truncated,
            "abstract_source": self.abstract_source,
            "abstract_authoritative": self.abstract_authoritative,
        }


# ---------------------------------------------------------------------------
# The decision record (this is what gets persisted).
# ---------------------------------------------------------------------------
@dataclass
class EligibilityDecision:
    """One paper's screening verdict, compact enough to persist per paper.

    ``matched_positive`` / ``matched_negative`` entries are
    ``"<category>:<term>"`` strings, so the report says *why* without needing a
    second lookup table. ``screening_input`` carries the exact text the verdict
    was taken on, so a rejection can be reproduced offline.
    """

    paper_id: str = ""
    title: str = ""
    outcome: str = OUTCOME_ELIGIBLE
    score: float = 0.0
    threshold: float = 0.0
    classification: str = ""
    matched_positive: List[str] = field(default_factory=list)
    matched_negative: List[str] = field(default_factory=list)
    requested_pathway: str = ""
    requested_organism: str = ""
    observed_pathways: List[str] = field(default_factory=list)
    observed_organisms: List[str] = field(default_factory=list)
    organism_match: str = ORGANISM_UNKNOWN
    reason: str = ""
    provisional: bool = False
    needs_manual_review: bool = False
    warnings: List[str] = field(default_factory=list)
    screening_input: ScreeningInput = field(default_factory=ScreeningInput)

    @property
    def eligible(self) -> bool:
        return outcome_accepts(self.outcome)

    @property
    def ineligible(self) -> bool:
        return outcome_is_ineligible(self.outcome)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "outcome": self.outcome,
            "score": round(float(self.score), 3),
            "threshold": round(float(self.threshold), 3),
            "classification": self.classification,
            "matched_positive": list(self.matched_positive),
            "matched_negative": list(self.matched_negative),
            "requested_pathway": self.requested_pathway,
            "requested_organism": self.requested_organism,
            "observed_pathways": list(self.observed_pathways),
            "observed_organisms": list(self.observed_organisms),
            "organism_match": self.organism_match,
            "reason": self.reason,
            "provisional": self.provisional,
            "needs_manual_review": self.needs_manual_review,
            "warnings": list(self.warnings),
            "screening_input": self.screening_input.to_dict(),
        }


# ---------------------------------------------------------------------------
# Alias / term derivation.
# ---------------------------------------------------------------------------
def _pathway_key(pathway: str) -> str:
    """The lexicon key for a requested pathway name, or ``""`` when unknown.

    Matching is on the normalized requested name containing the key, so
    ``"lipid A biosynthesis"`` and ``"regulation of lipid A biosynthesis"`` both
    resolve to ``"lipid a"``. Longest key first, so a future ``"heme b"`` entry
    would beat ``"heme"``.
    """
    norm = _normalize(pathway)
    if not norm:
        return ""
    for key in sorted(_PATHWAY_LEXICON, key=len, reverse=True):
        pattern = _pattern(key)
        if pattern is not None and pattern.search(norm):
            return key
    return ""


def _pathway_head(pathway: str) -> str:
    """The head compound: ``"cholesterol biosynthesis"`` -> ``"cholesterol"``."""
    norm = _normalize(pathway)
    if not norm:
        return ""
    tokens = norm.split()
    while tokens and tokens[-1] in _PATHWAY_HEADS:
        tokens.pop()
    # A leading "regulation of ..." / "the ..." is not part of the compound name.
    while tokens and tokens[0] in ("the", "regulation", "of", "a", "an"):
        tokens.pop(0)
    return " ".join(tokens)


def _pathway_aliases(pathway: str) -> Tuple[str, ...]:
    """Name aliases for a requested pathway: generated forms plus any lexicon ones.

    Generation runs for known and unknown pathways alike, so an unlisted topic
    still gets ``"<head> biosynthesis"`` / ``"<head> synthesis"`` / ``"<head>
    pathway"`` coverage and the gate degrades in power rather than in
    correctness.
    """
    norm = _normalize(pathway)
    out: List[str] = []
    if norm:
        out.append(norm)
    head = _pathway_head(pathway)
    if head:
        out.extend(f"{head} {suffix}" for suffix in _ALIAS_SUFFIXES)
    key = _pathway_key(pathway)
    if key:
        out.extend(_PATHWAY_LEXICON[key].aliases)
    # Single-word requests ("cholesterol") would otherwise alias to the bare
    # compound, which is explicitly NOT an anchor -- drop it here so it can only
    # score through _POS_PATHWAY_HEAD.
    return tuple(a for a in _dedup(out) if " " in a)


def _pathway_terms(pathway: str) -> Tuple[str, ...]:
    key = _pathway_key(pathway)
    return _PATHWAY_LEXICON[key].terms if key else ()


def _organism_entry(name: str) -> Optional[OrganismEntry]:
    return _ORGANISM_BY_ALIAS.get(_normalize(name))


def _organism_aliases(organism: str) -> Tuple[str, ...]:
    """Species-level aliases for one organism name (lexicon, else generated).

    Generated forms cover the two spellings papers use: the full binomial and the
    abbreviated genus (``Escherichia coli`` -> ``e coli``). The bare genus is
    **not** included -- ``Escherichia`` alone must never stand in for
    ``Escherichia coli``.
    """
    norm = _normalize(organism)
    if not norm:
        return ()
    entry = _organism_entry(organism)
    if entry is not None:
        return tuple(_dedup([_normalize(entry.canonical), *entry.aliases]))
    out = [norm]
    tokens = norm.split()
    if len(tokens) >= 2:
        out.append(f"{tokens[0][0]} {tokens[1]}")
    return tuple(_dedup(out))


def _canonical_organism(organism: str) -> str:
    """Map an organism spelling onto its lexicon canonical name, else ``""``."""
    entry = _organism_entry(organism)
    return entry.canonical if entry is not None else ""


def _taxon(name: str) -> Tuple[str, str, bool]:
    """``(genus, species, is_binomial)`` for one organism name.

    Strain suffixes are dropped: ``Escherichia coli K-12 MG1655`` yields
    ``("escherichia", "coli", True)``, so a strain-qualified name matches its
    species exactly. Non-binomial canonicals (``SARS-CoV-2``) come back with
    ``is_binomial=False`` and compare by equality only.
    """
    norm = _normalize(name)
    if not norm:
        return ("", "", False)
    entry = _organism_entry(name)
    if entry is not None:
        if not entry.genus:
            return (norm, "", False)
        canon = _normalize(entry.canonical).split()
        return (_normalize(entry.genus), canon[1] if len(canon) > 1 else "", True)
    tokens = norm.split()
    if len(tokens) == 1:
        return (tokens[0], "", tokens[0] in _KNOWN_GENERA)
    if tokens[0] in _KNOWN_GENERA or len(tokens[0]) > 2:
        species = tokens[1] if tokens[1] not in _NOT_SPECIES else ""
        return (tokens[0], species, True)
    return (norm, "", False)


def _compare_one(requested: str, observed: str) -> str:
    """``match`` / ``genus_level`` / ``mismatch`` for a single pair."""
    r_genus, r_species, r_binomial = _taxon(requested)
    o_genus, o_species, o_binomial = _taxon(observed)
    if not r_genus or not o_genus:
        return ORGANISM_UNKNOWN
    if not r_binomial or not o_binomial:
        # At least one side is not a species name (a virus, an unparseable label):
        # only exact equality can be a match.
        return (
            ORGANISM_MATCH
            if r_genus == o_genus and r_species == o_species
            else ORGANISM_MISMATCH
        )
    if r_genus != o_genus:
        return ORGANISM_MISMATCH
    if r_species and o_species:
        return ORGANISM_MATCH if r_species == o_species else ORGANISM_GENUS_LEVEL
    if not r_species and not o_species:
        # Both sides are the bare genus -- that IS what was asked for.
        return ORGANISM_MATCH
    # One side names a species, the other only the genus. A bare genus never
    # infers a species, so this is related-taxon evidence, not a match.
    return ORGANISM_GENUS_LEVEL


# ---------------------------------------------------------------------------
# Observation: what does this title/abstract actually report?
# ---------------------------------------------------------------------------
def _binomials(haystack: str) -> List[str]:
    """Genus-anchored species names found in ``haystack``.

    Scans for a known genus followed by a plausible species epithet, so a species
    the lexicon has never seen (``Escherichia fergusonii``) is still observed as
    itself rather than collapsing onto a lexicon neighbour. A genus with no
    epithet after it yields the bare genus.
    """
    out: List[str] = []
    for gnorm, display in _KNOWN_GENERA.items():
        for match in re.finditer(r"\b" + re.escape(gnorm) + r"\b(\s+([a-z]{3,}))?", haystack):
            epithet = (match.group(2) or "").strip()
            if epithet and epithet not in _NOT_SPECIES:
                canonical = _canonical_organism(f"{gnorm} {epithet}")
                out.append(canonical or f"{display} {epithet}")
            else:
                out.append(display)
    return out


def observe_metadata(
    title: str, abstract: str, scope: RequestedScope
) -> ObservedContext:
    """Read the observed pathways/organisms out of a paper's own metadata.

    Never falls back to the requested values: a paper that names no organism
    yields an empty ``observed_organisms``, which downstream reads as
    ``organism_match == "unknown"``. That is the whole point -- an unknown
    organism and a confirmed match must not look the same.
    """
    haystack = _haystack(title, abstract)
    observed_organisms: List[str] = []
    for entry in _ORGANISM_ENTRIES:
        if _TermSet(entry.aliases).hits(haystack):
            observed_organisms.append(entry.canonical)
    observed_organisms.extend(_binomials(haystack))
    # An organism named in the request but absent from the lexicon is still
    # detectable via its generated aliases -- observed because the PAPER says so,
    # not because the request did.
    if scope.requested_organism and _organism_entry(scope.requested_organism) is None:
        if _TermSet(_organism_aliases(scope.requested_organism)).hits(haystack):
            observed_organisms.append(scope.requested_organism.strip())

    observed_pathways: List[str] = []
    for alias in _pathway_aliases(scope.requested_pathway):
        pattern = _pattern(alias)
        if pattern is not None and pattern.search(haystack):
            observed_pathways.append(alias)

    return ObservedContext(
        observed_pathways=_dedup(observed_pathways),
        observed_organisms=_prune_genus_observations(_dedup(observed_organisms)),
    )


def _prune_genus_observations(names: Sequence[str]) -> List[str]:
    """Drop a bare genus when a species of that genus was also observed.

    "Escherichia coli" triggers both the species alias and the genus scan; the
    genus row adds nothing and would only weaken the comparison.
    """
    species_genera = set()
    for name in names:
        genus, species, binomial = _taxon(name)
        if binomial and species:
            species_genera.add(genus)
    out: List[str] = []
    for name in names:
        genus, species, binomial = _taxon(name)
        if binomial and not species and genus in species_genera:
            continue
        out.append(name)
    return out


def compare_organism(scope: RequestedScope, observed: ObservedContext) -> str:
    """Requested vs observed organism: the strongest verdict across observations.

    ``unknown`` whenever either side is empty -- an unnamed organism is not a
    mismatch. A ``match`` on **any** observed organism wins, because a paper that
    studies a pathway in E. coli and validates it in mice reports both; failing
    that, a ``genus_level`` relation beats a ``mismatch``.
    """
    requested = scope.requested_organism.strip()
    if not requested or not observed.observed_organisms:
        return ORGANISM_UNKNOWN
    verdicts = [_compare_one(requested, name) for name in observed.observed_organisms]
    for level in (ORGANISM_MATCH, ORGANISM_GENUS_LEVEL):
        if level in verdicts:
            return level
    return ORGANISM_MISMATCH if verdicts else ORGANISM_UNKNOWN


def _haystack(title: str, abstract: str) -> str:
    return _normalize(f"{title or ''} {abstract or ''}")


def _sentences(title: str, abstract: str) -> List[str]:
    """Normalized sentence units for locality checks.

    The title is its own unit even without terminal punctuation, so a pathway name
    in the title cannot borrow "catalysis" from the abstract's first sentence and
    call it local.
    """
    units: List[str] = []
    for block in (title, abstract):
        text = str(block or "").strip()
        if not text:
            continue
        for part in _SENTENCE_SPLIT.split(text):
            norm = _normalize(part)
            if norm:
                units.append(norm)
    return units


# ---------------------------------------------------------------------------
# Contextual mechanistic evidence.
# ---------------------------------------------------------------------------
def _local_units(
    terms: Sequence[str],
    haystack: str,
    sentences: Sequence[str],
    *,
    window_tokens: int,
) -> List[str]:
    """Every bounded text unit that mentions one of ``terms``.

    Two kinds of unit, per the requirement that a pathway mention plus a generic
    word from a different part of the paper is NOT evidence:

    1. **same sentence** -- the sentence containing the mention; and
    2. **local window** -- ``window_tokens`` tokens either side of it in the
       running text, which covers abstracts the sentence splitter joins into one
       long run.
    """
    term_set = _TermSet(terms)
    if not terms:
        return []
    units = [s for s in sentences if term_set.hits(s)]

    tokens = haystack.split()
    if tokens:
        starts: List[int] = []
        offset = 0
        for token in tokens:
            starts.append(offset)
            offset += len(token) + 1
        for begin, _end in term_set.spans(haystack):
            anchor = 0
            for index, start in enumerate(starts):
                if start > begin:
                    break
                anchor = index
            low = max(0, anchor - window_tokens)
            high = min(len(tokens), anchor + window_tokens + 1)
            units.append(" ".join(tokens[low:high]))
    return units


@dataclass
class _LocalEvidence:
    """What the text immediately around each pathway mention actually says."""

    strong: List[str] = field(default_factory=list)
    omics: List[str] = field(default_factory=list)
    #: A unit named a pathway-specific enzyme/metabolite, clear of screening
    #: language -- requirement (a).
    term_anchor: bool = False
    #: A unit paired a pathway mention with strong reaction/enzyme evidence, clear
    #: of screening language -- requirement (b).
    local_anchor: bool = False

    @property
    def anchored(self) -> bool:
        return self.term_anchor or self.local_anchor


def _local_evidence(
    term_hits: Sequence[str],
    alias_hits: Sequence[str],
    haystack: str,
    sentences: Sequence[str],
    *,
    window_tokens: int,
    title_names_pathway: bool = True,
) -> _LocalEvidence:
    """Judge each pathway mention **in its own local context**.

    Evaluating unit by unit rather than over the whole document is the point: an
    abstract that lists SQLE among differentially expressed proteins and, three
    sentences later, says "mechanism", must not combine the two into an anchor.
    A unit carrying screening vocabulary cannot anchor at all -- the pathway gene
    named there is a hit, not a subject -- while a *different* unit in the same
    abstract that genuinely characterizes an enzyme still anchors the paper.
    """
    out = _LocalEvidence()
    mentions = list(term_hits) + list(alias_hits)
    if not mentions:
        return out
    # Screening vocabulary is a DOCUMENT-level property, not only a local one. The
    # two hardest false positives (PMC12113831, PMC12782028) put "proteomic
    # analysis" / "transcriptome sequencing" in the methods framing sentence and
    # the pathway genes (SQLE, CYP51A1, DHCR24) many words later, so no bounded
    # window around the gene name contained the screening word.
    #
    # So in a paper that announces itself as a screen, path (a) survives only when
    # the pathway is named in the TITLE. That is the line between the two shapes:
    # PMC13264790 is titled "...analysis of PPOX...", so PPOX is its subject and a
    # heme-pathway term in the title is a real claim about what the paper is about;
    # PMC12113831 is titled about oak-leaf extract and wound healing, and SQLE only
    # turns up among its proteomic hits. Same word class, opposite meaning.
    out.omics = _OMICS_TERMS.matched(haystack)
    screen_without_title_claim = bool(out.omics) and not title_names_pathway
    term_set = _TermSet(term_hits) if term_hits else None
    for unit in _local_units(mentions, haystack, sentences, window_tokens=window_tokens):
        strong = _STRONG_MECHANISM_TERMS.matched(unit)
        out.strong.extend(strong)
        if _OMICS_TERMS.hits(unit):
            continue
        if (
            term_set is not None
            and not screen_without_title_claim
            and term_set.hits(unit)
        ):
            out.term_anchor = True
        if strong:
            out.local_anchor = True
    out.strong = _dedup(out.strong)
    return out


# ---------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------
def _score_positive(
    haystack: str, scope: RequestedScope, organism_match: str
) -> Tuple[float, List[str], Dict[str, List[str]]]:
    """Accumulate positive evidence -> (score, ``category:term`` labels, by-category)."""
    by_category: Dict[str, List[str]] = {}

    def _record(category: str, terms: Sequence[str]) -> None:
        if terms:
            by_category[category] = list(terms)

    _record(_POS_PATHWAY_ALIAS, _TermSet(scope.pathway_aliases).matched(haystack))
    _record(_POS_PATHWAY_TERM, _TermSet(scope.pathway_terms).matched(haystack))

    head = _pathway_head(scope.requested_pathway)
    if head and not by_category.get(_POS_PATHWAY_ALIAS):
        pattern = _pattern(head)
        if pattern is not None and pattern.search(haystack):
            _record(_POS_PATHWAY_HEAD, [head])

    _record(_POS_MECHANISM, _MECHANISM_TERMS.matched(haystack))
    _record(_POS_RECONSTRUCTION, _RECONSTRUCTION_TERMS.matched(haystack))

    enzymes = [
        word
        for word in _dedup(_ENZYME_WORD.findall(haystack))
        if word not in _NOT_ENZYMES
    ]
    enzymes.extend(_dedup(_EC_NUMBER.findall(haystack)))
    _record(_POS_ENZYME, enzymes)

    if organism_match == ORGANISM_MATCH:
        _record(_POS_ORGANISM, [scope.requested_organism.strip()])
    elif organism_match == ORGANISM_GENUS_LEVEL:
        _record(_POS_ORGANISM_GENUS, [scope.requested_organism.strip()])

    score = 0.0
    labels: List[str] = []
    for category, (weight, cap) in _POSITIVE_WEIGHTS.items():
        terms = by_category.get(category) or []
        if not terms:
            continue
        score += min(weight * len(terms), cap)
        labels.extend(f"{category}:{term}" for term in terms)
    return score, labels, by_category


def _score_negative(
    haystack: str,
    title_hay: str,
    scope: RequestedScope,
    observed: ObservedContext,
    organism_match: str,
    by_category: Mapping[str, List[str]],
    *,
    anchored: bool,
    has_abstract: bool,
    mechanism_hit: bool,
    classification: str,
) -> Tuple[float, List[str], List[str]]:
    """Accumulate negative evidence -> (penalty, labels, fired categories)."""
    fired: List[str] = []
    labels: List[str] = []

    if organism_match == ORGANISM_MISMATCH:
        fired.append(_NEG_INCOMPATIBLE_ORGANISM)
        labels.append(
            f"{_NEG_INCOMPATIBLE_ORGANISM}:"
            f"{'/'.join(observed.observed_organisms) or 'unknown'}"
        )

    case_terms = _CASE_REPORT_TERMS.matched(haystack)
    if case_terms:
        fired.append(_NEG_CASE_REPORT)
        labels.extend(f"{_NEG_CASE_REPORT}:{t}" for t in case_terms)

    epi_terms = _EPIDEMIOLOGY_TERMS.matched(haystack)
    if epi_terms:
        fired.append(_NEG_EPIDEMIOLOGY)
        labels.extend(f"{_NEG_EPIDEMIOLOGY}:{t}" for t in epi_terms)

    # Poultry / animal virulence survey: an animal host plus virulence-typing
    # language, for a pathway this paper shows no anchor for. Gated on
    # ``not anchored`` because a genuine lipid A paper done in chickens is not
    # this shape -- the requirement is specifically the UNRELATED-pathway survey.
    animal_hosts = [
        o for o in observed.observed_organisms if o in _ANIMAL_HOST_ORGANISMS
    ]
    virulence_terms = _VIRULENCE_TERMS.matched(haystack)
    if animal_hosts and virulence_terms and not anchored:
        fired.append(_NEG_ANIMAL_VIRULENCE)
        labels.append(f"{_NEG_ANIMAL_VIRULENCE}:{animal_hosts[0]}")
        labels.extend(f"{_NEG_ANIMAL_VIRULENCE}:{t}" for t in virulence_terms)

    software_terms = _SOFTWARE_TERMS.matched(haystack)
    if software_terms and not _WET_LAB_TERMS.hits(haystack):
        fired.append(_NEG_SOFTWARE_ONLY)
        labels.extend(f"{_NEG_SOFTWARE_ONLY}:{t}" for t in software_terms)

    # Pathway named only in the background: the pathway appears in the abstract
    # but not the title, and there is no mechanism language anywhere. Needs an
    # abstract to be knowable, so it never fires on a title-only screen.
    pathway_named = bool(
        by_category.get(_POS_PATHWAY_ALIAS) or by_category.get(_POS_PATHWAY_TERM)
    )
    if pathway_named and has_abstract and not mechanism_hit:
        in_title = _TermSet(
            tuple(scope.pathway_aliases) + tuple(scope.pathway_terms)
        ).hits(title_hay)
        if not in_title:
            fired.append(_NEG_BACKGROUND_ONLY)
            labels.append(f"{_NEG_BACKGROUND_ONLY}:pathway absent from the title")

    if classification in (CLASS_CONTEXT_ONLY, CLASS_OMICS_ONLY) and pathway_named:
        fired.append(_NEG_CONTEXT_ONLY)
        labels.append(
            f"{_NEG_CONTEXT_ONLY}:{classification} -- pathway named with no strong "
            "reaction/enzyme evidence in the same sentence or window"
        )
    elif not anchored:
        fired.append(_NEG_NO_MECHANISM)
        labels.append(f"{_NEG_NO_MECHANISM}:no requested-pathway term matched")

    penalty = sum(_NEGATIVE_WEIGHTS[c] for c in fired)
    return penalty, labels, fired


# ---------------------------------------------------------------------------
# Public API: screen one paper.
# ---------------------------------------------------------------------------
def screen_paper(
    *,
    paper_id: str,
    title: str,
    abstract: str = "",
    scope: RequestedScope,
    thresholds: Optional[EligibilityThresholds] = None,
    observed: Optional[ObservedContext] = None,
    abstract_source: str = "",
    abstract_is_authoritative: bool = True,
) -> EligibilityDecision:
    """Decide whether one paper may enter the expensive pipeline.

    Uses only ``title`` and ``abstract`` -- deliberately, since the whole point is
    to decide *before* paying for the full text. When ``abstract`` is empty the
    decision is taken against ``title_only_min_score`` and marked
    ``provisional``: a title carries a fraction of the evidence, so a title-only
    verdict is a screening hint, not a conclusion.

    ``scope.pinned`` short-circuits to :data:`OUTCOME_PINNED_OVERRIDE`: a human
    asked for this exact paper, so the score cannot veto it. Any pathway/organism
    mismatch is still computed and recorded in ``warnings``.

    Pass ``observed`` to supply already-known observed context (e.g. merged with
    Stage 0's reading); otherwise it is read from the title/abstract here.
    ``abstract_source`` is recorded verbatim on the persisted screening input so
    an audit can tell a publisher abstract from a derived one. Set
    ``abstract_is_authoritative=False`` for an abstract that was *recovered* rather
    than supplied (the dry-run tool's lead window over a stored full text): the
    score is still taken against the full-abstract bar, because there really is
    more text, but the verdict is marked ``provisional`` and the
    ``needs_manual_review`` rules that apply to provisional verdicts apply here
    too. Passing it after the fact would have set ``provisional`` on a decision
    whose review flag had already been computed as final.
    """
    limits = thresholds or thresholds_from_config()
    title_text = str(title or "").strip()
    screening_input = ScreeningInput.build(
        title_text,
        str(abstract or "").strip(),
        source=abstract_source,
        authoritative=abstract_is_authoritative,
    )
    # The persisted slice is the evidence boundary. Screening beyond it would
    # make a future offline replay incapable of reproducing this decision.
    abstract_text = screening_input.abstract.strip()
    has_abstract = bool(abstract_text)
    provisional = (not has_abstract) or (not abstract_is_authoritative)
    haystack = _haystack(title_text, abstract_text)
    title_hay = _normalize(title_text)
    sentences = _sentences(title_text, abstract_text)

    seen = observe_metadata(title_text, abstract_text, scope)
    if observed is not None:
        seen = seen.merge(observed)
    organism_match = compare_organism(scope, seen)

    decision = EligibilityDecision(
        paper_id=str(paper_id or "").strip(),
        title=title_text,
        score=0.0,
        threshold=limits.threshold_for(title_only=not has_abstract),
        requested_pathway=scope.requested_pathway,
        requested_organism=scope.requested_organism,
        observed_pathways=list(seen.observed_pathways),
        observed_organisms=list(seen.observed_organisms),
        organism_match=organism_match,
        provisional=provisional,
        screening_input=screening_input,
    )
    if screening_input.abstract_truncated:
        decision.warnings.append(
            "abstract truncated before screening at "
            f"{MAX_PERSISTED_ABSTRACT} characters; full input "
            f"length={screening_input.abstract_chars}, "
            f"sha256={screening_input.abstract_sha256}"
        )

    positive, pos_labels, by_category = _score_positive(
        haystack, scope, organism_match
    )

    # --- anchoring -------------------------------------------------------------
    # (a) a pathway-specific enzyme/metabolite term, mentioned outside a screening
    #     hit list; or
    # (b) a pathway mention -- alias or an omics-context term -- WITH strong
    #     reaction/enzyme evidence local to it (same sentence or bounded window).
    #
    # (a) carries the omics qualifier because the three real cholesterol false
    # positives were shaped exactly like a satisfied (a): SQLE, CYP51A1 and DHCR24
    # named as differentially expressed hits from a proteomic or transcriptomic
    # screen. A gene symbol in a hit list is not a paper about that enzyme.
    alias_hits = by_category.get(_POS_PATHWAY_ALIAS) or []
    term_hits = by_category.get(_POS_PATHWAY_TERM) or []
    pathway_mentions = list(term_hits) + list(alias_hits)
    title_names_pathway = _TermSet(
        tuple(scope.pathway_aliases) + tuple(scope.pathway_terms)
    ).hits(title_hay)
    evidence = _local_evidence(
        term_hits,
        alias_hits,
        haystack,
        sentences,
        window_tokens=limits.local_window_tokens,
        title_names_pathway=title_names_pathway,
    )
    local_strong = evidence.strong
    local_omics = evidence.omics
    anchored = evidence.anchored
    if local_strong:
        pos_labels.extend(f"local_mechanism:{t}" for t in local_strong)
    if local_omics:
        pos_labels.extend(f"local_omics_context:{t}" for t in local_omics)

    mechanism_hit = bool(
        by_category.get(_POS_MECHANISM) or by_category.get(_POS_RECONSTRUCTION)
    )
    if anchored:
        classification = CLASS_MECHANISTIC
    elif pathway_mentions and local_omics:
        # The pathway IS named, but only as a screen hit.
        classification = CLASS_OMICS_ONLY
    elif pathway_mentions:
        classification = CLASS_CONTEXT_ONLY
    elif _OMICS_TERMS.hits(haystack):
        classification = CLASS_OMICS_ONLY
    else:
        classification = CLASS_OFF_TOPIC
    decision.classification = classification

    penalty, neg_labels, fired = _score_negative(
        haystack,
        title_hay,
        scope,
        seen,
        organism_match,
        by_category,
        anchored=anchored,
        has_abstract=has_abstract,
        mechanism_hit=mechanism_hit,
        classification=classification,
    )
    decision.score = positive - penalty
    decision.matched_positive = pos_labels
    decision.matched_negative = neg_labels

    if organism_match == ORGANISM_MISMATCH:
        decision.warnings.append(
            "organism mismatch: requested "
            f"{scope.requested_organism or '(none)'}, observed "
            f"{', '.join(seen.observed_organisms) or '(none)'}"
        )
    elif organism_match == ORGANISM_GENUS_LEVEL:
        decision.warnings.append(
            "organism matched at GENUS level only (related taxon, not the "
            f"requested species): requested {scope.requested_organism}, observed "
            f"{', '.join(seen.observed_organisms)} -- no full organism bonus applied"
        )
    if pathway_mentions and not anchored:
        decision.warnings.append(
            f"'{', '.join(pathway_mentions[:4])}' is named but nothing local to that "
            f"mention is a reaction or an enzyme ({classification}"
            + (f"; local screening context: {', '.join(local_omics[:4])}" if local_omics else "")
            + ")"
        )
    elif not anchored:
        decision.warnings.append(
            f"no requested-pathway term found for '{scope.requested_pathway or '(none)'}'"
        )

    # --- outcome resolution, most decisive first ---------------------------
    if not limits.enabled:
        decision.outcome = OUTCOME_ELIGIBLE
        decision.reason = "screening disabled (eligibility_enabled=false)"
        return decision

    if scope.pinned:
        decision.outcome = OUTCOME_PINNED_OVERRIDE
        detail = "; ".join(decision.warnings) if decision.warnings else "no mismatch found"
        decision.reason = f"pinned paper: score bypassed ({detail})"
        return decision

    if len(title_text) < limits.min_title_chars and not has_abstract:
        decision.outcome = OUTCOME_INSUFFICIENT_METADATA
        decision.reason = (
            f"title is {len(title_text)} char(s) (< {limits.min_title_chars}) and no "
            "abstract: not enough metadata to screen on"
        )
        return decision

    if organism_match == ORGANISM_MISMATCH and limits.organism_veto:
        decision.outcome = OUTCOME_INELIGIBLE_ORGANISM
        decision.reason = (
            f"observed organism ({', '.join(seen.observed_organisms)}) does not match "
            f"the requested organism ({scope.requested_organism})"
        )
        decision.needs_manual_review = False
        return decision

    vetoes = [c for c in fired if c in _VETO_NEGATIVES]
    if vetoes and limits.negative_veto:
        decision.outcome = OUTCOME_INELIGIBLE_PATHWAY
        decision.reason = (
            "decisive negative evidence: " + ", ".join(vetoes) + " -- not a "
            "mechanistic pathway paper"
        )
        return decision

    if limits.require_pathway_anchor and not anchored:
        decision.outcome = OUTCOME_INELIGIBLE_PATHWAY
        where = "title" if not has_abstract else "title/abstract"
        if pathway_mentions:
            decision.reason = (
                f"{classification}: '{', '.join(pathway_mentions[:4])}' appears in the "
                f"{where} but no strong reaction/enzyme evidence is local to it"
                + (
                    f" (it sits in screening context: {', '.join(local_omics[:4])})"
                    if local_omics
                    else ""
                )
            )
        else:
            decision.reason = (
                f"{classification}: no mechanistic term for "
                f"'{scope.requested_pathway}' in the {where}"
            )
        decision.needs_manual_review = _has_partial_signal(by_category, decision)
        return decision

    if decision.score < decision.threshold:
        decision.outcome = OUTCOME_INELIGIBLE_PATHWAY
        decision.reason = (
            f"score {round(decision.score, 3)} below the "
            f"{'title-only ' if not has_abstract else ''}threshold "
            f"{decision.threshold}"
        )
        decision.needs_manual_review = _has_partial_signal(by_category, decision)
        return decision

    decision.outcome = OUTCOME_ELIGIBLE
    anchor_terms = term_hits or alias_hits
    local_evidence = local_strong
    decision.reason = (
        f"score {round(decision.score, 3)} >= threshold {decision.threshold}; "
        f"anchored on {', '.join(anchor_terms)}"
        + (f" (local: {', '.join(local_evidence)})" if local_evidence else "")
    )
    # Provisional accepts get flagged for a human when the title alone cannot
    # confirm mechanistic content (no reaction/mechanism language) or when the
    # score only just cleared the bar.
    if provisional and (
        not mechanism_hit or decision.score - decision.threshold <= limits.review_margin
    ):
        decision.needs_manual_review = True
        decision.warnings.append(
            "provisional accept: no reaction/mechanism language, or score within "
            f"{limits.review_margin} of the threshold -- confirm against the "
            "publisher abstract"
        )
    if organism_match == ORGANISM_GENUS_LEVEL:
        decision.needs_manual_review = True
    return decision


def _has_partial_signal(
    by_category: Mapping[str, List[str]], decision: EligibilityDecision
) -> bool:
    """Was there *some* positive signal behind a pathway rejection?

    Used only for ``needs_manual_review`` on a provisional (title-only) reject: a
    paper with mechanism language, an organism match, or the bare pathway
    compound might still be relevant once its abstract is available, whereas one
    with no signal at all is a confident reject.
    """
    if not decision.provisional:
        return False
    return bool(
        by_category.get(_POS_MECHANISM)
        or by_category.get(_POS_RECONSTRUCTION)
        or by_category.get(_POS_ORGANISM)
        or by_category.get(_POS_ORGANISM_GENUS)
        or by_category.get(_POS_PATHWAY_HEAD)
        or by_category.get(_POS_PATHWAY_ALIAS)
        # A pathway-specific enzyme in the title with no abstract to confirm it is
        # the strongest "ask a human" signal there is (PMC13264790 / PPOX).
        or by_category.get(_POS_PATHWAY_TERM)
    )


# ---------------------------------------------------------------------------
# Candidate integration: record requested vs observed on the paper object.
# ---------------------------------------------------------------------------
def screen_candidate(
    candidate: Any,
    scope: RequestedScope,
    *,
    thresholds: Optional[EligibilityThresholds] = None,
    apply: bool = True,
) -> EligibilityDecision:
    """Screen a paper-shaped object (``id``/``title``/``abstract``).

    With ``apply`` (the default) the requested and observed metadata are written
    back onto the object -- ``requested_pathway`` / ``requested_organism`` for
    the request, ``observed_pathways`` / ``observed_organisms`` /
    ``organism_match`` for what the paper reports. The two are kept in separate
    fields on purpose: the requested organism is never written into an
    observed-organism field.
    """
    decision = screen_paper(
        paper_id=str(getattr(candidate, "id", "") or ""),
        title=str(getattr(candidate, "title", "") or ""),
        abstract=str(getattr(candidate, "abstract", "") or ""),
        scope=scope,
        thresholds=thresholds,
        abstract_source="candidate_metadata",
    )
    if apply:
        apply_decision(candidate, decision)
    return decision


def apply_decision(candidate: Any, decision: EligibilityDecision) -> None:
    """Write requested/observed scope from ``decision`` onto ``candidate``.

    Best-effort by design: a caller may pass a plain object without these
    attributes (or a frozen one), and screening must not break on that.
    """
    for attr, value in (
        ("requested_pathway", decision.requested_pathway),
        ("requested_organism", decision.requested_organism),
        ("observed_pathways", list(decision.observed_pathways)),
        ("observed_organisms", list(decision.observed_organisms)),
        ("organism_match", decision.organism_match),
    ):
        try:
            setattr(candidate, attr, value)
        except Exception:  # noqa: BLE001 - a read-only candidate is still screenable
            continue


# ---------------------------------------------------------------------------
# Stage 0 reconciliation (requested scope is immutable).
# ---------------------------------------------------------------------------
#: Stage-0 keys read for observed context. ``pathway_name`` / ``likely_organism``
#: are what the preprocessor emits; the rest are accepted so a richer Stage 0
#: contract does not need a change here.
_STAGE0_PATHWAY_KEYS = ("pathway_name", "pathway", "observed_pathway")
_STAGE0_ORGANISM_KEYS = ("likely_organism", "organism", "observed_organism")
_STAGE0_ALIAS_KEYS = ("aliases", "pathway_aliases", "synonyms")
_STAGE0_AMBIGUITY_KEYS = ("ambiguities", "candidate_examples", "ambiguity_notes")


def apply_stage0_observation(
    scope: RequestedScope,
    stage0_context: Optional[Mapping[str, Any]],
    *,
    observed: Optional[ObservedContext] = None,
) -> Tuple[RequestedScope, ObservedContext, List[str]]:
    """Fold a Stage-0 reading into the observed context, never into the request.

    Returns ``(scope, observed, conflicts)`` where ``scope`` is the **same**
    requested scope that came in -- this function has no path that writes
    ``requested_pathway`` or ``requested_organism``. Stage 0's pathway name,
    organism, aliases and ambiguities land in ``observed``.

    ``conflicts`` is non-empty when Stage 0 read a pathway or organism that
    strongly contradicts the request (a different lexicon pathway, or an organism
    that is a genuine ``mismatch`` -- a related taxon is not a conflict). The
    caller marks such a paper :data:`OUTCOME_SCOPE_CONFLICT` / ineligible -- what
    it must NOT do is adopt the paper's apparent scope as the batch's request,
    which would quietly change what the whole batch is about.
    """
    ctx = dict(stage0_context or {})
    seen = observed or ObservedContext()

    pathways: List[str] = []
    for key in _STAGE0_PATHWAY_KEYS:
        value = str(ctx.get(key) or "").strip()
        if value:
            pathways.append(value)
    organisms: List[str] = []
    for key in _STAGE0_ORGANISM_KEYS:
        value = str(ctx.get(key) or "").strip()
        if value:
            organisms.append(value)
    aliases: List[str] = []
    for key in _STAGE0_ALIAS_KEYS:
        raw = ctx.get(key)
        if isinstance(raw, (list, tuple)):
            aliases.extend(str(item or "").strip() for item in raw)
        elif raw:
            aliases.append(str(raw).strip())
    ambiguities: List[str] = []
    for key in _STAGE0_AMBIGUITY_KEYS:
        raw = ctx.get(key)
        if isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, Mapping):
                    ambiguities.append(str(item.get("name") or item).strip())
                else:
                    ambiguities.append(str(item or "").strip())
        elif raw:
            ambiguities.append(str(raw).strip())

    merged = seen.merge(
        ObservedContext(
            observed_pathways=_dedup(pathways),
            observed_organisms=_dedup(organisms),
            aliases=_dedup(aliases),
            ambiguities=_dedup(ambiguities),
        )
    )

    conflicts: List[str] = []
    requested_key = _pathway_key(scope.requested_pathway)
    requested_aliases = _TermSet(scope.pathway_aliases)
    for observed_pathway in _dedup(pathways):
        observed_key = _pathway_key(observed_pathway)
        if requested_key and observed_key and observed_key != requested_key:
            conflicts.append(
                f"Stage 0 read pathway '{observed_pathway}' but the batch requested "
                f"'{scope.requested_pathway}'"
            )
        elif (
            not requested_key
            and scope.requested_pathway
            and not requested_aliases.hits(_normalize(observed_pathway))
            and not _TermSet(_pathway_aliases(observed_pathway)).hits(
                _normalize(scope.requested_pathway)
            )
        ):
            conflicts.append(
                f"Stage 0 read pathway '{observed_pathway}' which does not match the "
                f"requested '{scope.requested_pathway}'"
            )

    if scope.requested_organism and organisms:
        organism_only = ObservedContext(observed_organisms=_dedup(organisms))
        if compare_organism(scope, organism_only) == ORGANISM_MISMATCH:
            conflicts.append(
                f"Stage 0 read organism '{', '.join(_dedup(organisms))}' but the batch "
                f"requested '{scope.requested_organism}'"
            )

    # ``replace`` with no changes: an explicit, auditable statement that the
    # requested half comes back untouched (and it would raise if a field name
    # ever drifted).
    return replace(scope), merged, _dedup(conflicts)


__all__ = [
    "ACCEPTING_OUTCOMES",
    "CLASS_CONTEXT_ONLY",
    "CLASS_MECHANISTIC",
    "CLASS_OFF_TOPIC",
    "CLASS_OMICS_ONLY",
    "ELIGIBILITY_OUTCOMES",
    "INELIGIBLE_OUTCOMES",
    "MAX_PERSISTED_ABSTRACT",
    "ORGANISM_GENUS_LEVEL",
    "ORGANISM_MATCH",
    "ORGANISM_MATCH_VALUES",
    "ORGANISM_MISMATCH",
    "ORGANISM_UNKNOWN",
    "OUTCOME_DUPLICATE",
    "OUTCOME_ELIGIBLE",
    "OUTCOME_INELIGIBLE_ORGANISM",
    "OUTCOME_INELIGIBLE_PATHWAY",
    "OUTCOME_INSUFFICIENT_METADATA",
    "OUTCOME_NO_FULL_TEXT",
    "OUTCOME_PINNED_OVERRIDE",
    "OUTCOME_SCOPE_CONFLICT",
    "EligibilityDecision",
    "EligibilityThresholds",
    "ObservedContext",
    "OrganismEntry",
    "PathwayLexicon",
    "RequestedScope",
    "ScreeningInput",
    "apply_decision",
    "apply_stage0_observation",
    "compare_organism",
    "observe_metadata",
    "outcome_accepts",
    "outcome_is_ineligible",
    "screen_candidate",
    "screen_paper",
    "thresholds_from_config",
]
