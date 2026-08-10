"""Advisory participant / cofactor / hub / assay-reporter classifier (C-018).

UNWIRED BY DESIGN: nothing calls this and it imports nothing from the pipeline.
Deterministic and offline -- no LLM, no network, no database. It RETURNS A
VERDICT and removes nothing; no verdict here is a licence to delete biology.

  participant           the context makes it this pathway's OWN subject/product
  cofactor_or_currency  ubiquitous currency, no pathway-specific weight here
  hub                   shared by everything, so sharing one is not evidence of
                        a shared pathway -- but it IS often a genuine member
  assay_reporter        used to MEASURE the pathway, not to run it
  unknown               no curated grounds; NOT a rejection, and the deliberate
                        answer for anything this module does not recognise

Reconciles three pre-existing private lists: ``rag.admission._cofactors`` (:322)
and ``bench.semantic._cofactor_names`` (:155) are the SAME 29-name frozenset
(``rag.synthesize.COFACTOR_NAMES``); ``rag.admission._hubs`` (:331) unions that
with the 38-name ``HUB_METABOLITES`` (overlap: ``coa-sh``). Currency and hub stay
SEPARATE verdicts: HUB_METABOLITES mixes true currency (phosphate) with
first-class members of real pathways (glutamate). Each is pinned in the tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["PathwayContext", "Classification", "classify_entity", "VERDICTS",
           "CONFIDENCES", "REASONS", "VOCABULARY"]

PARTICIPANT, COFACTOR_OR_CURRENCY, HUB = "participant", "cofactor_or_currency", "hub"
ASSAY_REPORTER, UNKNOWN = "assay_reporter", "unknown"
VERDICTS: tuple[str, ...] = (PARTICIPANT, COFACTOR_OR_CURRENCY, HUB, ASSAY_REPORTER, UNKNOWN)
CONFIDENCES: tuple[str, ...] = ("high", "moderate", "low")
REASONS: tuple[str, ...] = (
    "no_entity_name", "requested_pathway_not_stated", "not_in_curated_vocabulary",
    "ubiquitous_currency_metabolite", "currency_spelling_reconciled_from_hub_list",
    "connectivity_hub_not_pathway_evidence", "assay_reporter_not_pathway_member",
    "subject_of_the_requested_pathway", "reporter_native_to_organism_pathway_unresolved")

_CURRENCY: dict[str, tuple[str, ...]] = {
    "ATP": ("atp",), "ADP": ("adp",), "AMP": ("amp",), "GTP": ("gtp",), "GDP": ("gdp",),
    "GMP": ("gmp",), "UTP": ("utp",), "UDP": ("udp",), "CTP": ("ctp",), "CDP": ("cdp",),
    "NAD+": ("nad+", "nad"), "NADH": ("nadh",), "NADP+": ("nadp+", "nadp"), "NADPH": ("nadph",),
    "FAD": ("fad",), "FADH2": ("fadh2",), "H2O": ("h2o", "water"), "H+": ("h+", "proton"),
    "O2": ("o2", "oxygen"), "CO2": ("co2", "carbon dioxide"), "H2O2": ("h2o2", "hydrogen peroxide"),
    "NH3": ("nh3", "nh4+", "ammonia", "ammonium"), "electron": ("electron",),
    "Pi": ("phosphate", "orthophosphate", "inorganic phosphate"),  # NOT bare "pi": see below
    "PPi": ("ppi", "pyrophosphate", "diphosphate", "inorganic pyrophosphate")}
_HUB: dict[str, tuple[str, ...]] = {
    "CoA": ("coa", "coa-sh", "coenzyme a"), "acetyl-CoA": ("acetyl-coa",),
    "malonyl-CoA": ("malonyl-coa",), "succinyl-CoA": ("succinyl-coa",), "acyl-CoA": ("acyl-coa",),
    "ACP": ("acp", "holo-acp", "acyl-acp", "acyl carrier protein"),
    "SAM": ("sam", "adomet", "s-adenosyl-l-methionine", "s-adenosylmethionine"),
    "SAH": ("sah", "s-adenosyl-l-homocysteine", "s-adenosylhomocysteine"),
    "glutathione": ("glutathione", "gsh", "gssg"), "tetrahydrofolate": ("tetrahydrofolate", "thf"),
    "thiamine pyrophosphate": ("thiamine pyrophosphate", "thiamine diphosphate", "tpp"),
    "pyridoxal phosphate": ("pyridoxal phosphate", "pyridoxal 5'-phosphate", "plp"),
    "L-glutamate": ("glutamate", "l-glutamate"), "L-glutamine": ("glutamine", "l-glutamine"),
    "2-oxoglutarate": ("2-oxoglutarate", "alpha-ketoglutarate", "2-ketoglutarate", "akg"),
    "quinone pool": ("quinone", "ubiquinone", "ubiquinol", "coenzyme q", "coq")}
#: No pre-existing list held ANY reporter, so PRODUCT_CONTRACT s.2 "no assay reporters" had none.
#: ``lacz`` is the E. coli reporter GENE; the EC 3.2.1.23 ACTIVITY names are deliberately absent --
#: human GLB1/LCT, K. lactis LAC4, Aspergillus, Bacteroides carry them natively (R-003 BLOCKING-1).
_REPORTER: dict[str, tuple[str, ...]] = {"LacZ": ("lacz",),
    "fluorescent protein": ("gfp", "egfp", "green fluorescent protein", "mcherry", "dsred", "rfp"),
    "luciferase": ("luciferase", "firefly luciferase", "renilla luciferase", "luciferin"),
    "viability dye": ("mtt", "xtt", "wst-1", "wst-8", "resazurin", "alamar blue"),
    "detection dye or probe": ("x-gal", "onpg", "propidium iodide", "dapi", "hoechst 33342",
                               "brdu", "edu", "jc-1", "tmrm", "fura-2", "fluo-4", "luminol")}
# --- context: where such a family is the requested pathway's real subject ----
_NAD = ("nad biosynthesis", "nad salvage", "nicotinate", "nicotinamide")
_TCA = ("tca cycle", "citric acid cycle", "krebs cycle", "tricarboxylic acid")
_FAS = ("fatty acid biosynthesis", "fatty acid synthesis", "polyketide")
_MET = ("methionine cycle", "methionine salvage", "s-adenosylmethionine biosynthesis", "polyamine")
_GLN = ("nitrogen assimilation", "glutamine synthetase", "gs-gogat", "glutamate metabolism",
        "glutamine metabolism", "glutamate biosynthesis", "glutamine biosynthesis")
#: ATP is currency in glycolysis and the product in ATP synthesis: this is the
#: context parameter at work. ADDITIVE -- a missing entry means "not evaluated
#: here", never "confirmed disposable".
_SUBJECT_OF: dict[str, tuple[str, ...]] = {
    "ATP": ("atp synthesis", "atp biosynthesis", "oxidative phosphorylation", "photophosphorylation"),
    "ADP": ("atp synthesis", "oxidative phosphorylation"), "NAD+": _NAD, "NADH": _NAD,
    "NADP+": _NAD, "NADPH": _NAD, "FAD": ("riboflavin", "fad biosynthesis"), "FADH2": ("riboflavin",),
    "H2O2": ("reactive oxygen species", "oxidative stress", "peroxide detox"),
    "O2": ("photosynthesis", "water splitting"), "CO2": ("carbon fixation", "calvin cycle"),
    "NH3": ("nitrogen assimilation", "ammonia assimilation", "nitrification", "urea cycle"),
    "Pi": ("phosphate transport", "phosphate uptake", "phosphate starvation"),
    "CoA": ("coenzyme a biosynthesis", "pantothenate"), "SAM": _MET, "SAH": _MET,
    "acetyl-CoA": _TCA + _FAS, "malonyl-CoA": _FAS, "ACP": _FAS,
    "succinyl-CoA": _TCA + ("heme biosynthesis", "porphyrin", "tetrapyrrole"),
    "acyl-CoA": _FAS + ("beta-oxidation", "fatty acid degradation", "fatty acid oxidation"),
    "glutathione": ("glutathione biosynthesis", "glutathione metabolism"),
    "tetrahydrofolate": ("folate biosynthesis", "folate metabolism", "one-carbon metabolism"),
    "thiamine pyrophosphate": ("thiamine biosynthesis", "thiamine metabolism"),
    "pyridoxal phosphate": ("vitamin b6", "pyridoxal", "pyridoxine"),
    "L-glutamate": _GLN, "L-glutamine": _GLN, "2-oxoglutarate": _TCA + _GLN,
    "quinone pool": ("ubiquinone biosynthesis", "coenzyme q biosynthesis", "electron transport",
                     "respiratory chain"), "luciferase": ("bioluminescence", "luciferin"),
    "fluorescent protein": ("bioluminescence",), "LacZ": ("lactose", "lac operon", "galactose")}
#: Organisms genuinely encoding the reporter. Matching the ORGANISM but not the
#: pathway yields ``unknown``, never ``assay_reporter``: LacZ in an E. coli
#: lipid-A paper is probably a reporter, but this module cannot tell.
_REPORTER_NATIVE_ORG: dict[str, tuple[str, ...]] = {
    "fluorescent protein": ("aequorea", "discosoma"), "LacZ": ("escherichia coli",),
    "luciferase": ("photinus", "renilla", "aliivibrio")}
#: Families moved ACROSS the base-SHA class boundary -- flagged ``moderate`` under their own reason
#: code so R-003 adjudicates exactly these: currency already held them under another spelling
#: (Pi/PPi/NH3), or they are not metabolites at all (electron). CoA moved the other way, currency
#: -> hub. Bare ``pi`` is DROPPED from COFACTOR_NAMES: "PI" is also phosphatidylinositol, and an
#: unknown beats a confident wrong call on a two-letter token in a lipid pathway.
_RECONCILED = frozenset({"Pi", "PPi", "NH3", "electron"})
_STRIP_RE = re.compile(r"[^a-z0-9+]")


def _normalize(name: object) -> str:
    """Fold a name to a key. Whole-name only: ``cAMP`` never matches ``AMP``."""
    text = str(name or "").casefold().replace("α", "alpha").replace("β", "beta")
    return _STRIP_RE.sub("", text)


def _has(text: object, markers: tuple[str, ...]) -> str:
    folded = " ".join(str(text or "").casefold().split())
    return next((m for m in markers if folded and m in folded), "")


VOCABULARY: dict[str, tuple[str, str]] = {}
for _v, _table in ((COFACTOR_OR_CURRENCY, _CURRENCY), (HUB, _HUB), (ASSAY_REPORTER, _REPORTER)):
    for _family, _spellings in _table.items():
        for _key in (_normalize(s) for s in _spellings):
            if VOCABULARY.get(_key, (_family, _v)) != (_family, _v):
                raise ValueError(f"cofactor_policy vocabulary collision on {_key!r}")
            VOCABULARY[_key] = (_family, _v)


@dataclass(frozen=True)
class PathwayContext:
    """The stated biological context. An INPUT -- never assumed by this module."""
    requested_pathway: str
    organism: str = ""


@dataclass(frozen=True)
class Classification:
    """One advisory verdict. ``reason`` and ``confidence`` are always populated."""
    entity: str
    family: str
    verdict: str
    reason: str
    confidence: str
    matched_context: str
    context: PathwayContext


def classify_entity(name: object, context: PathwayContext) -> Classification:
    """Classify ``name`` in ``context``. Pure, total and side-effect free."""
    family = ""

    def out(verdict: str, reason: str, confidence: str, marker: str = "") -> Classification:
        return Classification(str(name or ""), family, verdict, reason, confidence, marker, context)

    key, pathway = _normalize(name), str(context.requested_pathway or "").strip()
    if not key or not pathway:
        return out(UNKNOWN, "no_entity_name" if not key else "requested_pathway_not_stated", "low")
    hit = VOCABULARY.get(key)
    if hit is None:
        return out(UNKNOWN, "not_in_curated_vocabulary", "low")
    family, verdict = hit
    marker = _has(pathway, escapes := _SUBJECT_OF.get(family, ()))  # no escapes -> never "high"
    if marker:
        return out(PARTICIPANT, "subject_of_the_requested_pathway", "high", marker)
    if verdict == ASSAY_REPORTER:
        org = _has(context.organism, _REPORTER_NATIVE_ORG.get(family, ()))
        if org:
            return out(UNKNOWN, "reporter_native_to_organism_pathway_unresolved", "low", org)
        return out(ASSAY_REPORTER, "assay_reporter_not_pathway_member", "high")
    if verdict == COFACTOR_OR_CURRENCY:
        if family in _RECONCILED:
            return out(verdict, "currency_spelling_reconciled_from_hub_list", "moderate")
        return out(verdict, "ubiquitous_currency_metabolite", "high" if escapes else "moderate")
    return out(HUB, "connectivity_hub_not_pathway_evidence", "moderate")
