"""ORCH-745 change set B, repair round 1: the species gate's red-team probes, the code
and bio reviews' findings, and support that is never overstated (A3 / ORCH-2).

Every probe sentence the adversarial review marked "admitted" is a parametrised
``test_repair1_745b_*`` row asserting the claim is NOT ``target_species_supported``;
at 467a9faa (and 4f8d33a3) each of those rows reached ``target_species_supported``.
The recall rows (N1, F5) and the support rows (A3) fail there too. The controls the
reviews marked correct are ``test_new_acceptance_745b_r1_*`` and hold at both ends.

The probe shape is the red team's: the probe sentence as a results span ``ev_a``, the
species cited to a plain abstract sentence ``ev_sp``, and the paper typed primary
(review for the review-route rows). Participants are model-cited to ``ev_a`` so F-5
keeps them; positive credit still needs them named in the sentence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import helpers_mining as H  # noqa: E402
from t2pw.mining.admission import ADMITTED_STATUSES  # noqa: E402
from t2pw.mining.claims import ClaimStore  # noqa: E402
from t2pw.mining.controller import MiningController  # noqa: E402
from t2pw.mining.extraction import build_claims  # noqa: E402
from t2pw.mining.models import (  # noqa: E402
    EvidenceSpan,
    ExtractedClaim,
    GraphDelta,
    PaperRecord,
    ReactionParticipant,
    reaction_identity,
)
from t2pw.mining.species import fold_text, split_sentences  # noqa: E402

TARGET = "arabidopsis thaliana"
TSS, IVO, AMBIGUOUS, INFERRED = ("target_species_supported", "in_vitro_or_cell_free",
                                 "species_ambiguous", "inferred_only")
SPECIES_SPAN = EvidenceSpan(evidence_id="ev_sp", paper_id="PMC:PMCX", section="abstract",
                            text="Jasmonate metabolism was studied in Arabidopsis thaliana.")
FIXTURES = ROOT / "tests" / "fixtures" / "mining"


def probe(text, *, sub="jasmonic acid", prod="JA-Ile", enzyme="GH3.10", gene="", section="results",
          review=False, observed=(), species_span=SPECIES_SPAN):
    """The red team's fixture: build_claims on the probe sentence; the claim it returns."""
    span = EvidenceSpan(evidence_id="ev_a", paper_id="PMC:PMCX", text=text, section=section)
    species_span = species_span or span
    row = {"substrates": [{"name": sub, "evidence_ids": ["ev_a"]}],
           "products": [{"name": prod, "evidence_ids": ["ev_a"]}],
           "species": "Arabidopsis thaliana", "species_evidence_ids": [species_span.evidence_id],
           "evidence_ids": ["ev_a"]}
    if enzyme:
        row.update(enzyme=enzyme, enzyme_evidence_ids=["ev_a"])
    if gene:
        row.update(gene=gene, gene_evidence_ids=["ev_a"])
    spans = [span] if species_span is span else [span, species_span]
    claims, _ = build_claims({"claims": [row]}, spans=spans, paper_id="PMC:PMCX",
                             target_species_normalized=TARGET, round_index=1, model="probe",
                             paper_observed_organisms=list(observed),
                             paper_evidence_type="review" if review else "primary")
    return claims[0]


def _id(row):
    return row[0]


# ===========================================================================
# B1 / F2 / BIO-745B-5: a cell-free preparation is in vitro, not in planta
# ===========================================================================
B1 = [
    ("a1b", "Leaf extracts of Arabidopsis thaliana converted jasmonic acid to JA-Ile.", {}),
    ("a2b", "The His-tagged protein converted jasmonic acid to JA-Ile in the leaf assay.", {}),
    ("a3b", "The enzyme produced in bacteria converted jasmonic acid to JA-Ile in wounded leaf material.", {}),
    ("a5", "Desalted flower extracts converted jasmonic acid to JA-Ile.", {}),
    ("a6b", "The soluble fraction of leaves converted jasmonic acid to JA-Ile.", {}),
    ("a7b", "Insect-cell-produced GH3.10 converted jasmonic acid to JA-Ile in leaf discs.", {}),
    ("a10", "Assays with the Strep-tagged enzyme showed conversion of jasmonic acid to JA-Ile in leaf discs.", {}),
    ("f2-leaf-extract", "Conversion of 13-HPOT to 12-OPDA was detected in Arabidopsis thaliana leaf extracts.",
     {"sub": "13-HPOT", "prod": "12-OPDA", "enzyme": ""}),
    ("f2-intact-chloroplasts", "Intact chloroplasts from Arabidopsis thaliana leaves converted 13-HPOT to 12-OPDA.",
     {"sub": "13-HPOT", "prod": "12-OPDA", "enzyme": ""}),
    ("f2-soluble-fraction", "The soluble fraction of Arabidopsis thaliana leaves converted 13-HPOT to 12-OPDA.",
     {"sub": "13-HPOT", "prod": "12-OPDA", "enzyme": ""}),
    ("bio5-cell-suspension", "Arabidopsis cell suspension lines converted jasmonic acid to JA-Ile.", {}),
]


@pytest.mark.parametrize("name,text,kw", B1, ids=[r[0] for r in B1])
def test_repair1_745b_b1_a_cell_free_preparation_is_not_in_planta(name, text, kw):
    claim = probe(text, **kw)
    assert claim.species_state != TSS
    assert claim.species_state == IVO


@pytest.mark.parametrize("text", [
    "Protein extracts of Arabidopsis thaliana leaves converted jasmonic acid to JA-Ile.",
    "Recombinant GH3.10 converted jasmonic acid to JA-Ile in leaves.",
    "The cell lysate converted jasmonic acid to JA-Ile in leaves.",
])
def test_new_acceptance_745b_r1_b1_controls_stay_in_vitro(text):
    assert probe(text).species_state == IVO


# ===========================================================================
# B2 / code F1: a negated or failed conversion earns nothing
# ===========================================================================
B2 = [
    ("n1b", "Jasmonic acid was not converted to JA-Ile in Arabidopsis leaves.", {}),
    ("n2", "No conversion of 12-OH-JA to 12-OH-JA-Ile was detected in Arabidopsis flowers.",
     {"sub": "12-OH-JA", "prod": "12-OH-JA-Ile"}),
    ("n4b", "Arabidopsis leaves lack the ability to convert jasmonic acid to JA-Ile.", {}),
    ("n6", "In wild-type leaves, jasmonic acid was never converted to JA-Ile.", {}),
    ("n7", "Arabidopsis seedlings failed to convert jasmonic acid to JA-Ile.", {}),
    ("f1-not", "In Arabidopsis thaliana wild-type leaves, 13-HPOT is not converted to 12-OPDA.",
     {"sub": "13-HPOT", "prod": "12-OPDA", "enzyme": ""}),
    ("f1-no-conversion", "No conversion of 13-HPOT to 12-OPDA was detected in Arabidopsis thaliana leaves.",
     {"sub": "13-HPOT", "prod": "12-OPDA", "enzyme": ""}),
]


@pytest.mark.parametrize("name,text,kw", B2, ids=[r[0] for r in B2])
def test_repair1_745b_b2_a_negated_conversion_is_not_in_planta(name, text, kw):
    claim = probe(text, **kw)
    assert claim.species_state != TSS
    assert claim.species_state == AMBIGUOUS


def test_new_acceptance_745b_r1_b2_controls_keep_their_credit():
    """A loss-of-function sentence keeps its "no" (T1), and "cyclase, not synthase,
    converts" (the integration fixture's P7) is a positive conversion."""
    knockout = probe("In flowers and wounded leaves of gh3.10 jar1 knockout lines JA-Ile was below "
                     "the detection limit.", gene="GH3.10")
    assert (knockout.species_state, knockout.species_support["grade"]) == (TSS, "in_planta")
    absent = probe("No JA-Ile was detected in gh3.10 jar1 knockout lines.", gene="GH3.10")
    assert absent.species_state == TSS
    p7 = probe("We report that in Arabidopsis thaliana leaves allene oxide cyclase, not allene oxide "
               "synthase, converts 13-HPOT to 12-OPDA.", sub="13-HPOT", prod="12-OPDA",
               enzyme="allene oxide cyclase")
    assert (p7.species_state, p7.species_support["grade"]) == (TSS, "in_planta")


# ===========================================================================
# B3: the hedge list is closed over the common forms
# ===========================================================================
B3 = [
    ("i1-point-towards", "Arabidopsis mutants point towards conversion of 12-OH-JA to 12-OH-JA-Ile in flowers.",
     {"sub": "12-OH-JA", "prod": "12-OH-JA-Ile"}),
    ("i2b-consistent-with", "These data are consistent with conversion of jasmonic acid to JA-Ile in "
                            "Arabidopsis leaves.", {}),
    ("i3b-thought-to", "Jasmonic acid is thought to be converted to JA-Ile in Arabidopsis leaves.", {}),
    ("i4b-speculate", "It is tempting to speculate that jasmonic acid is converted to JA-Ile in "
                      "Arabidopsis leaves.", {}),
    ("i5b-seems-to", "Jasmonic acid seems to be converted to JA-Ile in Arabidopsis leaves.", {}),
    ("i6-presumed", "Jasmonic acid is presumed to be converted to JA-Ile in Arabidopsis leaves.", {}),
]


@pytest.mark.parametrize("name,text,kw", B3, ids=[r[0] for r in B3])
def test_repair1_745b_b3_a_hedged_conversion_is_not_in_planta(name, text, kw):
    claim = probe(text, **kw)
    assert claim.species_state != TSS
    assert "in_planta" not in {s["grade"] for s in claim.species_support["sentences"]}


def test_new_acceptance_745b_r1_b3_control_may_stays_out():
    assert probe("Jasmonic acid may be converted to JA-Ile in Arabidopsis leaves.").species_state != TSS


# ===========================================================================
# B4: homology transfer without the word "ortholog" is an inference
# ===========================================================================
B4 = [
    ("c1b-counterpart", "The Arabidopsis counterpart, AtGH3.10, converts jasmonic acid to JA-Ile in leaves."),
    ("c2b-closest-relative", "Its closest Arabidopsis relative converts jasmonic acid to JA-Ile in leaves."),
    ("c3b-sequence-similarity", "By sequence similarity, AtGH3.10 converts jasmonic acid to JA-Ile in "
                                "Arabidopsis leaves."),
    ("c4b-corresponding", "The corresponding Arabidopsis enzyme performs the same conversion of jasmonic acid "
                          "to JA-Ile in leaves."),
    ("c6-equivalent", "The equivalent reaction in Arabidopsis leaves converts jasmonic acid to JA-Ile."),
]


@pytest.mark.parametrize("name,text", B4, ids=[r[0] for r in B4])
def test_repair1_745b_b4_a_homology_transfer_is_an_inference(name, text):
    claim = probe(text)
    assert claim.species_state != TSS
    assert claim.species_state == INFERRED


# ===========================================================================
# B5: a transgenic host without the old host words
# ===========================================================================
B5 = [
    ("b1-carrying", "Arabidopsis lines carrying MpAOC converted allene oxide to dn-OPDA in leaves.",
     {"sub": "allene oxide", "prod": "dn-OPDA", "enzyme": "MpAOC"}),
    ("b7-carrying-gene", "Arabidopsis plants carrying the MpGH3A gene converted jasmonic acid to JA-Ile in leaves.",
     {"enzyme": "MpGH3A"}),
    ("b5-promoter", "Arabidopsis seedlings that received the gene under a strong promoter converted LA to GLA.",
     {"sub": "LA", "prod": "GLA", "enzyme": ""}),
    ("b6-foreign-complementation", "Complementation of the aos mutant with the MpAOS gene restored 12-OPDA "
                                   "formation from allene oxide in leaves.",
     {"sub": "allene oxide", "prod": "12-OPDA", "enzyme": "MpAOS"}),
    ("b2b-introduction", "Introduction of the bryophyte gene into the jar1 mutant restored JA-Ile formation from "
                         "jasmonic acid in Arabidopsis leaves.", {"enzyme": "JAR1"}),
]


@pytest.mark.parametrize("name,text,kw", B5, ids=[r[0] for r in B5])
def test_repair1_745b_b5_a_transgenic_host_is_not_in_planta(name, text, kw):
    claim = probe(text, **kw)
    assert claim.species_state != TSS
    assert claim.species_state == AMBIGUOUS


def test_new_acceptance_745b_r1_b5_control_transgenic_is_a_host():
    claim = probe("Transgenic Arabidopsis plants converted jasmonic acid to JA-Ile in leaves.")
    assert (claim.species_state, claim.species_support["grade"]) == (AMBIGUOUS, "host")


# ===========================================================================
# B6: a supplied precursor is held back (pending the product-owner hinge)
# ===========================================================================
B6 = [
    ("f1b-supplied", "When Arabidopsis seedlings were supplied with jasmonic acid, it was converted to JA-Ile."),
    ("f2-supplementation", "Supplementation of Arabidopsis leaves with deuterated jasmonic acid led to conversion "
                           "to JA-Ile."),
    ("f3b-infiltration", "After infiltration of Arabidopsis leaves with jasmonic acid, JA-Ile was formed from "
                         "jasmonic acid."),
    ("f6-dosed", "Arabidopsis leaves dosed with jasmonic acid converted it to JA-Ile."),
    ("f7-injected", "Jasmonic acid injected into Arabidopsis leaves was converted to JA-Ile."),
]


@pytest.mark.parametrize("name,text", B6, ids=[r[0] for r in B6])
def test_repair1_745b_b6_a_supplied_precursor_is_not_in_planta(name, text):
    claim = probe(text)
    assert claim.species_state != TSS
    assert "in_planta" not in {s["grade"] for s in claim.species_support["sentences"]}


def test_new_acceptance_745b_r1_b6_control_exogenous_stays_out():
    claim = probe("Exogenous jasmonic acid was converted to JA-Ile in Arabidopsis leaves.")
    assert claim.species_state != TSS


# ===========================================================================
# B7 / code F4: the review route admits no negation, hedge, open question or
# co-occurrence
# ===========================================================================
B7 = [
    ("e1-fails-to", "Arabidopsis fails to convert dn-OPDA to OPC-6:0.", {"sub": "dn-OPDA", "prod": "OPC-6:0"}),
    ("e7-remains-unclear", "It remains unclear whether Arabidopsis converts dn-OPDA to OPC-6:0.",
     {"sub": "dn-OPDA", "prod": "OPC-6:0"}),
    ("e11-open-question", "Whether Arabidopsis converts jasmonic acid to JA-Gln remains an open question.",
     {"prod": "JA-Gln"}),
    ("e6-thought-to", "In Arabidopsis, dn-OPDA is thought to be converted to OPC-6:0.",
     {"sub": "dn-OPDA", "prod": "OPC-6:0"}),
    ("e8-presumed", "In Arabidopsis, 12-OH-JA is presumed to be converted to 12-OH-JA-Ile.",
     {"sub": "12-OH-JA", "prod": "12-OH-JA-Ile"}),
    ("f4-co-occurrence", "In Arabidopsis, OPDA and OPC-8:0 accumulate in wounded leaves.",
     {"sub": "OPDA", "prod": "OPC-8:0", "enzyme": "OPR3"}),
]


@pytest.mark.parametrize("name,text,kw", B7, ids=[r[0] for r in B7])
def test_repair1_745b_b7_the_review_route_needs_a_plain_stated_conversion(name, text, kw):
    claim = probe(text, review=True, species_span=None, **{"enzyme": "", **kw})
    assert claim.species_state != TSS
    assert "review" not in {s["grade"] for s in claim.species_support["sentences"]}


@pytest.mark.parametrize("text,admitted", [
    ("In Arabidopsis, OPDA and dn- cis -OPDA are reduced by OPR3 in the peroxisomes to produce OPC-8:0 and "
     "OPC-6:0, respectively.", True),
    ("In Arabidopsis and Marchantia, dn-OPDA is converted to OPC-6:0.", False),
    ("Conversion of dn-OPDA to OPC-6:0 has not been shown in Arabidopsis.", False),
    ("In Arabidopsis, dn-OPDA would be converted to OPC-6:0.", False),
], ids=["e3-T2-T3-sentence", "e9-two-organisms", "e10-not-shown", "e12-would-be"])
def test_new_acceptance_745b_r1_b7_controls(text, admitted):
    claim = probe(text, sub="dn-OPDA" if "dn-OPDA" in text else "OPDA",
                  prod="OPC-6:0" if "dn-OPDA" in text else "OPC-8:0", enzyme="OPR3", review=True,
                  species_span=None)
    assert (claim.species_state == TSS) is admitted
    if admitted:
        assert claim.species_support["grade"] == "review"


# ===========================================================================
# B8: a gene symbol with another species' prefix names that species
# ===========================================================================
B8 = [
    ("d1-W4-MpFAD5", "Genetic knockout of MpFAD5 nearly abolishes dn-OPDA accumulation after mechanical wounding.",
     {"sub": "HTA (16:3n3)", "prod": "dn-OPDA", "enzyme": "", "gene": "MpFAD5"}),
    ("d5-Mp-gh3a", "In Mp gh3a mutants, dn-iso-OPDA was converted to dn-OPDA-Glu in tissues.",
     {"sub": "dn-iso-OPDA", "prod": "dn-OPDA-Glu", "enzyme": "", "gene": "Mp gh3a"}),
]


@pytest.mark.parametrize("name,text,kw", B8, ids=[r[0] for r in B8])
def test_repair1_745b_b8_a_foreign_gene_prefix_names_another_organism(name, text, kw):
    claim = probe(text, observed=(), **kw)
    assert claim.species_state != TSS
    assert "in_planta" not in {s["grade"] for s in claim.species_support["sentences"]}


def test_new_acceptance_745b_r1_b8_control_with_the_screen_organism():
    claim = probe(B8[0][1], observed=("Marchantia polymorpha",), **B8[0][2])
    assert claim.species_state == AMBIGUOUS


# ===========================================================================
# B9 / code F3: citations the old list missed; N3 sections; discussion/conclusion
# ===========================================================================
B9 = [
    ("g1b-previously-reported", "We confirmed the previously reported conversion of jasmonic acid to JA-Ile in "
                                "Arabidopsis leaves.", {}),
    ("g2b-co-workers", "As shown by Smith and co-workers, jasmonic acid is converted to JA-Ile in Arabidopsis "
                       "leaves.", {}),
    ("f3-numeric", "In Arabidopsis thaliana leaves, 13-HPOT is converted to 12-OPDA (12, 13).",
     {"sub": "13-HPOT", "prod": "12-OPDA", "enzyme": ""}),
    ("f3-superscript", "In Arabidopsis thaliana leaves, 13-HPOT is converted to 12-OPDA.12,13",
     {"sub": "13-HPOT", "prod": "12-OPDA", "enzyme": ""}),
    ("f3-previously-shown", "It was previously shown that in Arabidopsis thaliana leaves 13-HPOT is converted to "
                            "12-OPDA.", {"sub": "13-HPOT", "prod": "12-OPDA", "enzyme": ""}),
    ("f3-discussion", "In Arabidopsis thaliana leaves, 13-HPOT is converted to 12-OPDA.",
     {"sub": "13-HPOT", "prod": "12-OPDA", "enzyme": "", "section": "discussion"}),
    ("conclusion", "In Arabidopsis thaliana leaves, 13-HPOT is converted to 12-OPDA.",
     {"sub": "13-HPOT", "prod": "12-OPDA", "enzyme": "", "section": "conclusion"}),
    ("n3-numbered-methods", "Arabidopsis plants of the gh3.10 mutant line contained no JA-Ile in flowers; JA was "
                            "converted to JA-Ile in leaves of Col-0.",
     {"sub": "JA", "section": "2. Materials and Methods", "gene": "GH3.10"}),
]


@pytest.mark.parametrize("name,text,kw", B9, ids=[r[0] for r in B9])
def test_repair1_745b_b9_n3_background_and_interpretation_earn_nothing(name, text, kw):
    claim = probe(text, **kw)
    assert claim.species_state != TSS
    assert "in_planta" not in {s["grade"] for s in claim.species_support["sentences"]}


def test_new_acceptance_745b_r1_b9_controls():
    assert probe("Jasmonic acid is converted to JA-Ile in Arabidopsis leaves [12].").species_state != TSS
    methods = probe(B9[-1][1], **{**B9[-1][2], "section": "methods"})
    assert methods.species_state != TSS
    # An abstract-only paper is indexed whole as "body": it keeps its credit.
    body = probe("In Arabidopsis thaliana leaves, 13-HPOT is converted to 12-OPDA.", sub="13-HPOT",
                 prod="12-OPDA", enzyme="", section="body")
    assert body.species_state == TSS


# ===========================================================================
# N1 / F5: recall the splitter and the folding cost
# ===========================================================================
@pytest.mark.parametrize("text,kw", [
    ("In the opr3 knockout line, OPC-8:0 was undetectable in leaves.", {"gene": "OPR3"}),
    ("In Arabidopsis leaves, OPDA is converted to OPC-8:0 by OPR3.", {}),
    ("In the fad5 mutant, 16:1n9 was absent from leaves.", {"sub": "16:0", "prod": "16:1n9", "enzyme": "",
                                                           "gene": "FAD5"}),
], ids=["r1-opr3-knockout", "r4-converted-to-OPC-8:0", "r3-fad5-16:1n9"])
def test_repair1_745b_n1_a_colon_inside_a_name_does_not_split_the_clause(text, kw):
    claim = probe(text, **{"sub": "OPDA", "prod": "OPC-8:0", "enzyme": "OPR3", **kw})
    assert (claim.species_state, claim.species_support["grade"]) == (TSS, "in_planta")


def test_repair1_745b_f5_a_single_capital_before_a_sentence_ends_it():
    assert split_sentences("The enzyme belongs to clade A. In opr3 mutant leaves OPDA accumulated. Done.") == [
        "The enzyme belongs to clade A.", "In opr3 mutant leaves OPDA accumulated.", "Done."]
    assert split_sentences("Leaves of A. thaliana were used. Done.") == ["Leaves of A. thaliana were used.",
                                                                         "Done."]


def test_repair1_745b_f5_a_spaced_dash_does_not_glue_tokens():
    assert "ja-ile-the" not in fold_text("JA-Ile — the bioactive form — was absent.")
    claim = probe("In jar1 mutant leaves, JA-Ile — the bioactive form — was below the detection limit.",
                  enzyme="JAR1")
    assert (claim.species_state, claim.species_support["grade"]) == (TSS, "in_planta")


# ===========================================================================
# N2: the species span must name the target plainly and alone
# ===========================================================================
@pytest.mark.parametrize("species_text", [
    "The pathway resembles that of Arabidopsis thaliana.",
    "Arabidopsis thaliana and Marchantia polymorpha both make jasmonates.",
], ids=["j1-resembles", "j2-with-another-organism"])
def test_repair1_745b_n2_a_comparison_species_span_is_not_species_evidence(species_text):
    span = EvidenceSpan(evidence_id="ev_sp2", paper_id="PMC:PMCX", text=species_text, section="abstract")
    claim = probe("In flowers and wounded leaves of gh3.10 jar1 knockout lines JA-Ile was below the detection "
                  "limit.", gene="GH3.10", species_span=span)
    assert claim.species_state == AMBIGUOUS
    assert "plain statement" in claim.species_reason


# ===========================================================================
# BIO-745B-1: bare X of "X (Y)" names a sentence for the negative markers
# ===========================================================================
def test_repair1_745b_bio1_a_bare_name_still_carries_its_in_vitro_veto():
    """Pilot-003 #10 (call 0008 row 0, "ALA (18:3n3)" -> 13-HPOT, the bryophyte review) with
    species "Arabidopsis thaliana" cited to its own spans: at 467a9faa target_species_supported
    (review) on ev_74d8's "In Arabidopsis, JA biosynthesis is initiated by 13-LOX enzymes that
    oxygenate ALA (18:3n3) ...", because "Recombinant enzyme assays ... compared with ALA and LA"
    names only bare ALA. Now in vitro."""
    texts = json.loads((FIXTURES / "orch745a_pilot_texts.json").read_text(encoding="utf-8"))
    recorded = json.loads((FIXTURES / "orch745b_pilot_payloads.json").read_text(encoding="utf-8"))
    papers = {p["paper_id"]: p for p in texts["pilot003"]["papers"]}
    spans = {}
    for row in texts["pilot003"]["spans"]:
        paper = papers[row["paper_id"]]
        text = paper["abstract"] if row["char_start"] is None else paper["full_text"][row["char_start"]:row["char_end"]]
        spans[row["evidence_id"]] = (EvidenceSpan(evidence_id=row["evidence_id"], paper_id=row["paper_id"],
                                                  text=text, section=row["locator"].split("#")[0]), row["role"])
    call = next(c for c in recorded["pilot003"]["calls"] if c["call_id"] == "mc_001_extract_claims_0008")
    payload = json.loads(call["raw_text"])
    payload["claims"][0].update(species="Arabidopsis thaliana",
                                species_evidence_ids=list(payload["claims"][0]["evidence_ids"]))
    shown = [spans[e][0] for e in call["shown"] if spans[e][1] == "prose"]
    claims, _ = build_claims(payload, spans=shown, paper_id=call["paper_id"], target_species_normalized=TARGET,
                             round_index=1, model="recorded", paper_evidence_type="review")
    claim = claims[0]
    assert claim.species_state != TSS
    assert claim.species_state == IVO
    assert any(s["grade"] == "in_vitro" and "Recombinant enzyme assays" in s["text"]
               for s in claim.species_support["sentences"])
