"""ORCH-745 change set B, repair round 2: positive species credit narrowed structurally.

Rule 5(a) (a conversion stated in an organ or genotype context) is gone. A primary paper's
sentence earns ``in_planta`` only from loss-of-function genetics; a review's only from a
strict, linked conversion (rule 7). See ``docs/pathway_mining/species_policy.md``.

Every probe sentence the round-1 reviewers saw admitted at ``fd53a301`` (red team p1a/p1b,
bio conformance N/H/V/T/F/C/RV, code R1-B1..B5) is a parametrised ``test_repair2_745b_*``
row asserting the claim is NOT ``target_species_supported``; each reached
``target_species_supported`` at ``fd53a301``. The ``5a_removed`` rows are the deliberate
policy move: sentences credited by rule 5(a) (correctly, under that rule) that now earn
nothing -- the recall cost, listed in ``known_limitations.md``. The controls the reviewers
marked correct are ``test_new_acceptance_745b_r2_*`` and hold at both ends. The support rows
(R1-B6, R1-N1, R1-N5b) fail at ``fd53a301`` too.

The probe shape is the reviewers': the probe sentence as a results span ``ev_a``, the
species cited to a plain abstract sentence ``ev_sp`` (for the code review's review-route
rows, to ``ev_a`` itself), the paper typed primary (review for the review-route rows), and
the participants model-cited to ``ev_a``.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import helpers_mining as H  # noqa: E402
from t2pw.mining.admission import ADMITTED_STATUSES  # noqa: E402
from t2pw.mining.controller import MiningController  # noqa: E402
from t2pw.mining.extraction import build_claims  # noqa: E402
from t2pw.mining.graph import ResearchReaction  # noqa: E402
from t2pw.mining.models import (  # noqa: E402
    EvidenceSpan,
    ExtractedClaim,
    GraphDelta,
    PaperRecord,
    ReactionParticipant,
    reaction_identity,
)

TARGET = "arabidopsis thaliana"
TSS, IVO, AMBIGUOUS, INFERRED = ("target_species_supported", "in_vitro_or_cell_free",
                                 "species_ambiguous", "inferred_only")
SPECIES_SPAN = EvidenceSpan(evidence_id="ev_sp", paper_id="PMC:PMCX", section="abstract",
                            text="Jasmonate metabolism was studied in Arabidopsis thaliana.")
MP = ("Marchantia polymorpha",)


def probe(text, *, sub="jasmonic acid", prod="JA-Ile", enzyme="GH3.10", gene="", section="results",
          review=False, observed=(), species_span=SPECIES_SPAN):
    """The reviewers' fixture: build_claims on the probe sentence; the claim it returns."""
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


HP = dict(sub="13-HPOT", prod="12-OPDA", enzyme="AOS")
OPR = dict(sub="OPDA", prod="OPC-8:0", enzyme="OPR3")
NO_ENZYME = dict(enzyme="")
REVIEW = dict(review=True)
CODE_REVIEW = dict(review=True, species_span=None)
FI = chr(0xFB01)  # the PDF ligature "fi"


def _rows(*rows):
    return pytest.mark.parametrize("text,kw", [(t, k) for _, t, k in rows], ids=[n for n, _, _ in rows])


# ===========================================================================
# 1. Negation (red team R2-B1, bio BIO-R1-1, code R1-B2)
# ===========================================================================
@_rows(
    ("rt-neg1", "Conversion of jasmonic acid to JA-Ile did not occur in Arabidopsis leaves.", {}),
    ("rt-neg2", "Conversion of jasmonic acid to JA-Ile remained undetectable in wild-type leaves.", {}),
    ("rt-neg3", "Conversion of jasmonic acid to JA-Ile was below the detection limit in wild-type leaves.", {}),
    ("rt-neg4", "Jasmonic acid was converted to JA-Ile in neither wild-type nor jar1 leaves.", {}),
    ("rt-neg5", "In no case was jasmonic acid converted to JA-Ile in wild-type leaves.", {}),
    ("rt-neg6", "Arabidopsis leaves were incapable of converting jasmonic acid to JA-Ile.", {}),
    ("rt-neg7", "Jasmonic acid was converted to JA-Ile in none of the wild-type leaves tested.", {}),
    ("rt-neg8", "We found no evidence that jasmonic acid is converted to JA-Ile in Arabidopsis leaves.", {}),
    ("bio-n-b", "In wild-type Arabidopsis leaves, conversion of jasmonic acid to JA-Ile was below the "
                "detection limit.", {}),
    ("bio-n-h", "Hardly any conversion of jasmonic acid to JA-Ile occurred in Arabidopsis leaves.", {}),
    ("bio-n-l", "Conversion of jasmonic acid to JA-Ile in Arabidopsis leaves was negligible.", {}),
    ("bio-n-s", "Conversion of jasmonic acid into JA-Ile in Arabidopsis wild-type leaves was absent.", {}),
    ("code-neg-does-not-take-place", "Conversion of jasmonic acid to JA-Ile does not take place in Arabidopsis "
                                     "leaves.", {}),
    ("code-neg-neither-nor", "Jasmonic acid was converted to JA-Ile in neither wild-type leaves nor flowers of "
                             "Arabidopsis.", {}),
    ("code-neg-absence-of-conversion", "In Arabidopsis leaves we observed an absence of conversion of jasmonic "
                                       "acid to JA-Ile.", {}),
    ("code-neg-rather-than", "In Arabidopsis leaves, jasmonic acid is converted to 12-OH-JA rather than to "
                             "JA-Ile.", {}),
)
def test_repair2_745b_a_negated_statement_earns_no_credit(text, kw):
    assert probe(text, **kw).species_state != TSS


# ===========================================================================
# 2. Hedge and modality (R2-B2, BIO-R1-3, R1-B5)
# ===========================================================================
@_rows(
    ("rt-h1", "Jasmonic acid is probably converted to JA-Ile in Arabidopsis leaves.", {}),
    ("rt-h2", "Jasmonic acid is believed to be converted to JA-Ile in Arabidopsis leaves.", {}),
    ("rt-h3", "It is unlikely that jasmonic acid is converted to JA-Ile in Arabidopsis leaves.", {}),
    ("rt-h4", "We postulate that jasmonic acid is converted to JA-Ile in Arabidopsis leaves.", {}),
    ("rt-h5", "We cannot exclude that jasmonic acid is converted to JA-Ile in Arabidopsis leaves.", {}),
    ("rt-h6", "It has yet to be demonstrated that jasmonic acid is converted to JA-Ile in Arabidopsis leaves.", {}),
    ("rt-h7", "Jasmonic acid is apparently converted to JA-Ile in Arabidopsis leaves.", {}),
    ("rt-h8", "Jasmonic acid is considered to be converted to JA-Ile in Arabidopsis leaves.", {}),
    ("rt-h9", "A presumptive conversion of jasmonic acid to JA-Ile occurs in Arabidopsis leaves.", {}),
    ("bio-h-a", "In Arabidopsis leaves, jasmonic acid is probably converted to JA-Ile by GH3.10.", {}),
    ("bio-h-c", "It is postulated that jasmonic acid is converted to JA-Ile in Arabidopsis leaves.", {}),
    ("bio-h-d", "Apparently, jasmonic acid is converted to JA-Ile in Arabidopsis leaves.", {}),
    ("code-hedge-indicate", "These results indicate that jasmonic acid is converted to JA-Ile in Arabidopsis "
                            "leaves.", {}),
)
def test_repair2_745b_a_hedged_statement_earns_no_credit(text, kw):
    assert probe(text, **kw).species_state != TSS


# ===========================================================================
# 3. Applied compounds (R2-B3, bio F block, R1-B5)
# ===========================================================================
@_rows(
    ("rt-ap1", "Arabidopsis seedlings pretreated with jasmonic acid converted it to JA-Ile.", {}),
    ("rt-ap2", "In Arabidopsis seedlings exposed to MeJA vapour, MeJA was converted to JA.",
     dict(sub="MeJA", prod="JA", enzyme="MJE")),
    ("rt-ap3", "Arabidopsis seedlings given jasmonic acid converted it to JA-Ile.", {}),
    ("rt-ap4", "Arabidopsis leaves floated on jasmonic acid solution converted it to JA-Ile.", {}),
    ("rt-ap5", "Arabidopsis seedlings grown on medium containing jasmonic acid converted it to JA-Ile.", {}),
    ("rt-ap6", "Upon addition of jasmonic acid, wild-type leaves converted it to JA-Ile.", {}),
    ("rt-ap7", "Arabidopsis seedlings incubated in jasmonic acid solution converted it to JA-Ile.", {}),
    ("rt-ap8", "Following elicitation with jasmonic acid, wild-type seedlings converted it to JA-Ile.", {}),
    ("rt-ap9", "After MeJA vapour, MeJA was converted to JA in Arabidopsis rosettes.",
     dict(sub="MeJA", prod="JA", enzyme="MJE")),
    ("bio-f-a", "Arabidopsis seedlings exposed to jasmonic acid converted it to JA-Ile.", {}),
    ("bio-f-b", "After exposure of Arabidopsis leaves to jasmonic acid, JA-Ile was formed from jasmonic acid.", {}),
    ("bio-f-c", "Arabidopsis leaves floated on jasmonic acid converted it to JA-Ile.", {}),
    ("bio-f-f", "Jasmonic acid added to Arabidopsis leaves was converted to JA-Ile.", {}),
    ("code-app-provided", "Arabidopsis seedlings provided with jasmonic acid converted it to JA-Ile.", {}),
)
def test_repair2_745b_an_applied_compound_earns_no_credit(text, kw):
    assert probe(text, **kw).species_state != TSS


# ===========================================================================
# 4. Cell-free, recombinant and protein-mutant evidence (R2-B4, N-1b, BIO-R1-2, R1-B3)
# ===========================================================================
@_rows(
    ("rt-iv1", "Extracts prepared from wounded Arabidopsis leaves converted jasmonic acid to JA-Ile.", {}),
    ("rt-iv2", "In extracts from Arabidopsis flowers, 12-OH-JA was converted to 12-OH-JA-Ile.",
     dict(sub="12-OH-JA", prod="12-OH-JA-Ile")),
    ("rt-iv3", "Chloroplasts isolated from Arabidopsis leaves converted 13-HPOT to 12-OPDA.",
     dict(sub="13-HPOT", prod="12-OPDA", enzyme="")),
    ("rt-iv4", "Leaf protein preparations converted jasmonic acid to JA-Ile.", {}),
    ("rt-iv5", "In-vitro, the wild-type GH3.10 protein converted jasmonic acid to JA-Ile.", {}),
    ("rt-iv6", "Arabidopsis T87 cell lines converted jasmonic acid to JA-Ile.", {}),
    ("rt-iv7", "Suspension-cultured Arabidopsis cell lines converted jasmonic acid to JA-Ile.", {}),
    ("rt-iv8", "The GH3.10 R117A mutant showed no activity toward jasmonic acid.", {}),
    ("rt-iv9", "The GH3.10 S115A mutant protein produced no JA-Ile from jasmonic acid.", {}),
    ("rt-iv10", "Wild-type GH3.10 converted jasmonic acid to JA-Ile, whereas the R117A variant was inactive.", {}),
    ("rt-iv12", "Recombinant GH3.10 was purified to homogeneity. The enzyme converted jasmonic acid to JA-Ile "
                "with all wild-type and mutant proteins tested.", {}),
    ("rt-iv13", "Puri" + FI + "ed GH3.10 from wounded leaves converted jasmonic acid to JA-Ile.", {}),
    ("rt-iv14", "After in" + FI + "ltration of Arabidopsis leaves with jasmonic acid, JA-Ile was formed from "
                "jasmonic acid.", {}),
    ("bio-v-a", "Extracts of Arabidopsis leaves converted 13-HPOT to 12-OPDA.", HP),
    ("bio-v-b", "Extracts from wounded Arabidopsis leaves converted 13-HPOT to 12-OPDA.", HP),
    ("bio-v-d", "Envelope membranes of Arabidopsis chloroplasts converted 13-HPOT to 12-OPDA in leaves.", HP),
    ("bio-v-e", "A leaf enzyme preparation from Arabidopsis converted 13-HPOT to 12-OPDA.", HP),
    ("bio-v-f", "Peroxisomes isolated from Arabidopsis leaves converted OPDA to OPC-8:0.", OPR),
    ("bio-v-h", "The supernatant of homogenised Arabidopsis leaves converted 13-HPOT to 12-OPDA.", HP),
    ("bio-v-i", "In assays containing Arabidopsis leaf protein, 13-HPOT was converted to 12-OPDA.", HP),
    ("code-iv-crude-enzyme", "A crude enzyme preparation from Arabidopsis leaves converted jasmonic acid to "
                             "JA-Ile.", {}),
    ("code-iv-crude-enzyme-extract", "A crude enzyme extract from Arabidopsis leaves converted jasmonic acid to "
                                     "JA-Ile.", {}),
    ("code-iv-enzyme-extract", "Enzyme extracts of Arabidopsis leaves converted jasmonic acid to JA-Ile.", {}),
    ("code-iv-leaf-slices", "Arabidopsis leaf slices converted jasmonic acid to JA-Ile.", {}),
)
def test_repair2_745b_cell_free_or_protein_evidence_earns_no_credit(text, kw):
    assert probe(text, **kw).species_state != TSS


# ===========================================================================
# 5. Hosts, comparisons and foreign gene symbols (R2-B5..B7, N-1a, BIO-R1-4)
# ===========================================================================
@_rows(
    ("rt-ho1", "Arabidopsis plants that express MpDES6 converted LA to GLA in leaves.",
     dict(sub="LA", prod="GLA", enzyme="MpDES6")),
    ("rt-ho2", "Arabidopsis plants engineered with the MpGH3A gene converted jasmonic acid to JA-Ile in leaves.",
     dict(enzyme="MpGH3A")),
    ("rt-ho3", "Arabidopsis lines with MpGH3A converted jasmonic acid to JA-Ile in leaves.", dict(enzyme="MpGH3A")),
    ("rt-ho4", "Arabidopsis lines producing MpGH3A converted jasmonic acid to JA-Ile in leaves.",
     dict(enzyme="MpGH3A")),
    ("bio-t-a", "Arabidopsis plants containing MpGH3A converted jasmonic acid to JA-Ile in leaves.",
     dict(enzyme="MpGH3A")),
    ("bio-t-b", "Arabidopsis lines with the MpAOC gene converted allene oxide to OPDA in leaves.",
     dict(sub="allene oxide", prod="OPDA", enzyme="MpAOC")),
    ("bio-t-d", "Arabidopsis plants engineered to produce MpGH3A converted jasmonic acid to JA-Ile in leaves.",
     dict(enzyme="MpGH3A")),
    ("bio-t-e", "In Arabidopsis, knockout of MpFAD5 nearly abolishes dn-OPDA accumulation after wounding.",
     dict(sub="16:0", prod="dn-OPDA", enzyme="MpFAD5")),
    ("rt-cmp1", "As in Arabidopsis, knockout of MpAOS abolished 12-OPDA accumulation in wounded tissues.",
     dict(sub="allene oxide", prod="12-OPDA", enzyme="", gene="MpAOS", observed=MP)),
    ("rt-cmp2", "Like its Arabidopsis homologue, MpGH3A converts jasmonic acid to JA-Ile in wounded tissues.",
     dict(enzyme="MpGH3A", observed=MP)),
    ("rt-cmp3", "Unlike in Arabidopsis, Mpaos1 mutants accumulated no 12-OPDA in wounded tissues.",
     dict(sub="allene oxide", prod="12-OPDA", enzyme="", gene="MpAOS1", observed=MP)),
    ("rt-cmp4", "Similar to Arabidopsis, aos mutants accumulated no 12-OPDA in wounded tissues.",
     dict(sub="allene oxide", prod="12-OPDA", enzyme="", gene="AOS", observed=MP)),
    ("rt-gp2", "In Mpgh3a mutants, dn-iso-OPDA was converted to dn-OPDA-Glu in tissues.",
     dict(sub="dn-iso-OPDA", prod="dn-OPDA-Glu", enzyme="", gene="MpGH3A")),
    ("rt-gp3", "In the Mpfad5 mutant, dn-OPDA was undetectable in wounded tissues.",
     dict(sub="HTA (16:3n3)", prod="dn-OPDA", enzyme="", gene="MpFAD5")),
    ("rt-gp4", "In Mp-aos1 mutants, 12-OPDA was undetectable in wounded tissues.",
     dict(sub="allene oxide", prod="12-OPDA", enzyme="", gene="Mp-AOS1")),
    ("rt-gp5", "Knockout of LjAOS abolished 12-OPDA accumulation in wounded leaves.",
     dict(sub="allene oxide", prod="12-OPDA", enzyme="", gene="LjAOS")),
)
def test_repair2_745b_a_host_comparison_or_foreign_gene_earns_no_credit(text, kw):
    assert probe(text, **kw).species_state != TSS


# ===========================================================================
# 6. Citations and background (bio C block, R1-N3)
# ===========================================================================
@_rows(
    ("bio-c-a", "It is well established that jasmonic acid is converted to JA-Ile in Arabidopsis leaves.", {}),
    ("bio-c-b", "As reported earlier, jasmonic acid is converted to JA-Ile in Arabidopsis leaves.", {}),
    ("bio-c-c", "As we reported recently, jasmonic acid is converted to JA-Ile in Arabidopsis leaves.", {}),
    ("bio-c-d", "Jasmonic acid is known to be converted to JA-Ile in Arabidopsis leaves.", {}),
    ("code-cite-ref", "In Arabidopsis leaves, jasmonic acid is converted to JA-Ile (ref. 12).", {}),
    ("code-cite-known", "It is well known that jasmonic acid is converted to JA-Ile in Arabidopsis leaves.", {}),
)
def test_repair2_745b_a_cited_statement_earns_no_credit(text, kw):
    assert probe(text, **kw).species_state != TSS


# ===========================================================================
# 7. A loss-of-function sentence that reports no effect (R1-B4)
# ===========================================================================
@_rows(
    ("code-lof-not-affected", "In gh3.10 mutant leaves, JA-Ile levels were not affected.", dict(gene="GH3.10")),
    ("code-lof-did-not-differ", "In gh3.10 knockout lines, JA-Ile accumulation did not differ from wild type.",
     dict(gene="GH3.10")),
    ("code-lof-not-reduced", "JA-Ile levels were not reduced in gh3.10 mutant leaves.", dict(gene="GH3.10")),
    ("code-lof-wt-levels", "The gh3.10 mutant accumulated wild-type levels of JA-Ile in leaves.",
     dict(gene="GH3.10")),
)
def test_repair2_745b_a_no_effect_loss_of_function_earns_no_credit(text, kw):
    assert probe(text, **kw).species_state != TSS


# ===========================================================================
# 8. The review route: strict, linked, broad-vetoed (R1-B1, R1-N2, R1-B5, bio RV block)
# ===========================================================================
@_rows(
    ("code-rev-required-production", "In Arabidopsis, JAR1 is required for JA-Ile production.",
     dict(CODE_REVIEW, sub="12-OH-JA", enzyme="JAR1")),
    ("code-rev-reduced-levels", "In Arabidopsis, JA-Ile levels are reduced in jar1 mutants.",
     dict(CODE_REVIEW, sub="OPDA", enzyme="JAR1")),
    ("code-rev-formation-requires", "In Arabidopsis, JA-Ile formation requires JAR1.",
     dict(CODE_REVIEW, sub="OPC-8:0", enzyme="JAR1")),
    ("code-rev-catalyzes-step", "In Arabidopsis, JAR1 catalyzes the final step of JA-Ile biosynthesis.",
     dict(CODE_REVIEW, enzyme="JAR1")),
    ("code-rev-co-occur-produced", "In Arabidopsis, OPDA and OPC-8:0 are produced in wounded leaves.",
     dict(CODE_REVIEW, **OPR)),
    ("code-rev-levels-reduced", "In Arabidopsis, OPDA and OPC-8:0 levels are reduced after wounding.",
     dict(CODE_REVIEW, **OPR)),
    ("code-rev-yet-to-be", "In Arabidopsis, the conversion of OPDA to OPC-8:0 is yet to be demonstrated.",
     dict(CODE_REVIEW, **OPR)),
    ("code-rev-yet-to-be-JA", "In Arabidopsis, the conversion of jasmonic acid to JA-Ile is yet to be "
                              "demonstrated.", dict(CODE_REVIEW)),
    ("code-rev-unknown", "In Arabidopsis, the enzyme converting OPDA to OPC-8:0 is unknown.",
     dict(CODE_REVIEW, sub="OPDA", prod="OPC-8:0", enzyme="")),
    ("code-rev-debated", "In Arabidopsis, the conversion of OPDA to OPC-8:0 is debated.",
     dict(CODE_REVIEW, sub="OPDA", prod="OPC-8:0", enzyme="")),
    ("bio-rv-2", "In Arabidopsis, leaves treated with jasmonic acid convert it to JA-Ile.", REVIEW),
    ("bio-rv-3", "Arabidopsis plants were sprayed with jasmonic acid and its conversion to JA-Ile was quantified.",
     REVIEW),
    ("bio-rv-4", "In Arabidopsis, jasmonic acid is probably conjugated to JA-Ile by JAR1.",
     dict(REVIEW, enzyme="JAR1")),
    ("bio-rv-8", "In Arabidopsis, JA-Ile is formed when jasmonic acid is supplied to leaves.", REVIEW),
)
def test_repair2_745b_the_review_route_needs_a_linked_unhedged_conversion(text, kw):
    assert probe(text, **kw).species_state != TSS


# ===========================================================================
# 9. Rule 5(a) removed: a deliberate move (merge rule 4), not a correction
# ===========================================================================
@_rows(
    # Each was credited, as its rule then said, by rule 5(a); the orchestrator's repair-2
    # decision removes the rule. Recall cost, listed in known_limitations.md.
    ("code-pos-plain", "In Arabidopsis leaves, jasmonic acid is converted to JA-Ile.", {}),
    ("code-pos-wt-but-not-mutant", "Jasmonic acid is converted to JA-Ile in wild-type but not in jar1 mutant "
                                   "leaves.", {}),
    ("rt-neg11", "Jasmonic acid was converted to JA-Ile in wild-type leaves but not in roots.", {}),
    ("n1-r4", "In Arabidopsis leaves, OPDA is converted to OPC-8:0 by OPR3.", OPR),
    ("bio-r-7", "In wild-type leaves, OPDA was converted to OPC-8:0, but not in opr3 mutants.", OPR),
    # bio-r-2 ("The aos mutant did not accumulate 12-OPDA after wounding.") left this list
    # under CS-D D1: a negated-verb absence form now earns credit (test_mining_species_cs_d.py).
    ("code-host-native-complement", "Complementation of the jar1 mutant with JAR1 restored conversion of jasmonic "
                                    "acid to JA-Ile in Arabidopsis leaves.", dict(enzyme="JAR1")),
)
def test_repair2_745b_5a_removed_a_stated_conversion_no_longer_earns_credit(text, kw):
    claim = probe(text, **kw)
    assert claim.species_state != TSS
    assert claim.species_state == AMBIGUOUS


# ===========================================================================
# Controls the reviewers marked correct: they hold at fd53a301 and here
# ===========================================================================
@_rows(
    ("T1/bio-r-9", "In flowers and wounded leaves of gh3.10 jar1 knockout lines JA-Ile was below the detection "
                   "limit.", dict(gene="GH3.10")),
    ("absent-no-JA-Ile-detected", "No JA-Ile was detected in gh3.10 jar1 knockout lines.", dict(gene="GH3.10")),
    ("bio-r-1", "In opr3 mutant leaves, OPC-8:0 was not detected.", dict(OPR, gene="OPR3")),
    ("bio-r-4", "JA-Ile levels were strongly reduced in wounded leaves of the jar1 mutant.", dict(enzyme="JAR1")),
    ("bio-r-11", "The jar1 mutant is impaired in JA-Ile formation in wounded leaves.", dict(enzyme="JAR1")),
    ("bio-r-13", "Wounding-induced JA-Ile accumulation was abolished in the jar1 gh3.10 double mutant.",
     dict(enzyme="JAR1, GH3.10")),
    ("n1-r1", "In the opr3 knockout line, OPC-8:0 was undetectable in leaves.", OPR),
)
def test_new_acceptance_745b_r2_loss_of_function_keeps_in_planta_credit(text, kw):
    claim = probe(text, **kw)
    assert (claim.species_state, claim.species_support["grade"]) == (TSS, "in_planta")


@_rows(
    ("bio-rv-1", "In Arabidopsis, 13-HPOT is converted to 12-OPDA by AOS and AOC in plastids.", dict(REVIEW, **HP)),
    ("code-rev-T2T3", "In Arabidopsis, OPDA and dn- cis -OPDA are reduced by OPR3 in the peroxisomes to produce "
                      "OPC-8:0 and OPC-6:0, respectively.", dict(CODE_REVIEW, **OPR)),
)
def test_new_acceptance_745b_r2_a_linked_review_conversion_keeps_review_credit(text, kw):
    claim = probe(text, **kw)
    assert (claim.species_state, claim.species_support["grade"]) == (TSS, "review")


@_rows(
    ("rt-neg9", "Conversion of jasmonic acid to JA-Ile could not be detected in wild-type leaves.", {}),
    ("rt-neg10", "Wild-type leaves did not convert jasmonic acid to JA-Ile.", {}),
    ("rt-h10", "Jasmonic acid is expected to be converted to JA-Ile in Arabidopsis leaves.", {}),
    ("rt-iv11", "GH3.10 and its R117A variant were expressed in E. coli and purified. Wild-type GH3.10 converted "
                "jasmonic acid to JA-Ile, whereas the R117A variant was inactive.", {}),
    ("rt-gp1", "Genetic knockout of Mp FAD5 nearly abolishes dn-OPDA accumulation after mechanical wounding.",
     dict(sub="HTA (16:3n3)", prod="dn-OPDA", enzyme="", gene="MpFAD5")),
    ("bio-n-o", "Conversion of jasmonic acid to JA-Ile was not detected in Arabidopsis leaves.", {}),
    ("bio-n-p", "Arabidopsis leaves did not convert jasmonic acid to JA-Ile.", {}),
    ("bio-n-r", "In Arabidopsis leaves, jasmonic acid conversion to JA-Ile was undetectable.", {}),
    ("bio-h-e", "Jasmonic acid is assumed to be converted to JA-Ile in Arabidopsis leaves.", {}),
    ("bio-v-j", "Leaf extract of Arabidopsis converted 13-HPOT to 12-OPDA.", HP),
    ("bio-rv-5", "In Arabidopsis, both OPDA and dn-OPDA are reduced by OPR3 to produce OPC-8:0.",
     dict(REVIEW, **OPR)),
    ("bio-rv-6", "In Arabidopsis, conversion of jasmonic acid to JA-Ile did not occur in jar1 plants.",
     dict(REVIEW, enzyme="JAR1")),
    ("bio-rv-7", "In Arabidopsis, MpGH3A converts jasmonic acid to JA-Ile.", dict(REVIEW, enzyme="MpGH3A")),
    ("code-neg-not-observed", "Conversion of jasmonic acid into JA-Ile was not observed in Arabidopsis leaves.", {}),
    ("code-app-floated", "Arabidopsis leaf discs floated on jasmonic acid converted it to JA-Ile.", {}),
    ("code-host-transformed-moss", "Arabidopsis plants transformed with moss GH3 converted jasmonic acid to "
                                   "JA-Ile in leaves.", {}),
    ("code-host-bearing", "Arabidopsis lines bearing the MpGH3A transgene converted jasmonic acid to JA-Ile.", {}),
    ("code-cite-reported-by", "In Arabidopsis leaves, jasmonic acid is converted to JA-Ile, as reported by "
                              "Staswick.", {}),
    ("code-iv-desalted-leaf-extract", "Desalted Arabidopsis leaf extract converted jasmonic acid to JA-Ile.", {}),
    ("code-rev-co-occur-form", "In Arabidopsis, OPDA and OPC-8:0 form in wounded leaves.", dict(CODE_REVIEW, **OPR)),
)
def test_new_acceptance_745b_r2_held_back_controls_stay_held_back(text, kw):
    assert probe(text, **kw).species_state != TSS


def test_repair2_745b_n1_ligatures_and_prefixed_alleles_are_normalised():
    """NFKC first: the ligature form of a marker is the marker. The canonical allele
    notations of another species are foreign genes, the target's own are not. At fd53a301
    the ligature survived folding and "Mpgh3a" / "Mp-aos1" / "LjAOS" were not foreign."""
    from t2pw.mining.species import _FOREIGN_GENE, fold_text

    assert fold_text("Puri" + FI + "ed") == fold_text("Purified")
    for symbol in ("MpFAD5", "Mp FAD5", "Mpgh3a", "Mp-aos1", "Mpfad5", "LjAOS", "PhCCD1", "GhOPR3"):
        assert _FOREIGN_GENE.search(symbol), symbol
    for word in ("GH3.10", "jar1", "Calcium", "Most", "Phosphate", "Smaller", "Psoralen"):
        assert not _FOREIGN_GENE.search(word), word


# ===========================================================================
# R1-B6: a lifted node shows its support's attributes, never the lead's
# ===========================================================================
def _claim(paper_id, eid, state, *, gene="", tissue="", conditions="", species_text="Arabidopsis thaliana",
           grade="in_planta"):
    identity = reaction_identity(["jasmonic acid"], ["JA-Ile"], species=TARGET)
    return ExtractedClaim(
        claim_id=f"cl_{paper_id[-4:]}_{eid}", claim_identity=identity, paper_id=paper_id,
        subject="jasmonic acid", relationship="converted_to", object="JA-Ile", gene=gene, tissue=tissue,
        conditions=conditions, species_text=species_text, species_state=state,
        species_reason=f"{state} (test)", species_support={"grade": grade},
        substrates=[ReactionParticipant(name="jasmonic acid", evidence_ids=[eid])],
        products=[ReactionParticipant(name="JA-Ile", evidence_ids=[eid])],
        field_evidence={"substrates": [eid], "products": [eid], "species": [eid], "gene": [eid], "tissue": [eid],
                        "conditions": [eid]},
        evidence_ids=[eid], first_seen_round=1, last_updated_round=1)


def _controller(tmp_path, papers):
    controller = MiningController(H.make_config(tmp_path), enabled_override=True)
    controller.state.papers = {pid: PaperRecord(paper_id=pid, pmcid=pid.split(":")[-1], evidence_type="primary")
                               for pid in papers}
    return controller


def _admit(controller, *claims):
    for claim in claims:
        for eid in claim.evidence_ids:
            controller.state.evidence[eid] = EvidenceSpan(evidence_id=eid, paper_id=claim.paper_id,
                                                          text="(test span)", section="results")
        controller._admit(claim, index=1, delta=GraphDelta(round_index=1), decisions=[], rejections=[])
    return controller.state.graph.by_identity(claims[0].claim_identity)


def test_repair2_745b_r1b6_a_lifted_node_shows_its_supporting_claims_attributes(tmp_path):
    """The A3 lift test extended. At fd53a301 the lifted node kept the in-vitro lead's
    gene 'MpGH3A' and tissue 'flowers'."""
    controller = _controller(tmp_path, ["PMC:PMCP1", "PMC:PMCP2"])
    lead = _claim("PMC:PMCP1", "ev_p1", IVO, gene="MpGH3A", tissue="flowers", conditions="recombinant enzyme",
                  species_text="Marchantia polymorpha", grade="in_vitro")
    support = _claim("PMC:PMCP2", "ev_p2", TSS, gene="GH3.10", tissue="wounded leaves", conditions="wounding")
    node = _admit(controller, lead, support)
    assert node.admitted is True and node.paper_ids == ["PMC:PMCP2"]
    assert (node.gene, node.tissue, node.conditions, node.species_text) == (
        "GH3.10", "wounded leaves", "wounding", "Arabidopsis thaliana")
    assert node.hypothesis is False
    assert (node.substrate_names, node.product_names) == (["jasmonic acid"], ["JA-Ile"])
    assert [r["claim_id"] for r in node.non_supporting_claims] == ["cl_MCP1_ev_p1"]


# ===========================================================================
# R1-N5b: per-claim non-supporting rows under an admitted node
# ===========================================================================
def test_repair2_745b_r1n5b_non_supporting_rows_carry_each_claims_own_paper_and_state(tmp_path):
    """An admitted node with no decision history (the starting-graph edge) and two leads
    from two papers. At fd53a301 the second row read paper 'PMC:PMCP1, PMC:PMCP2'."""
    controller = _controller(tmp_path, ["PMC:PMCP1", "PMC:PMCP2"])
    first, second = (_claim("PMC:PMCP1", "ev_p1", IVO, grade="in_vitro"),
                     _claim("PMC:PMCP2", "ev_p2", AMBIGUOUS, grade="none"))
    controller.state.graph.add_reaction(
        ResearchReaction(reaction_id="", claim_identity=first.claim_identity, substrate_names=["jasmonic acid"],
                         product_names=["JA-Ile"], admitted=True, decision_status="accepted"),
        round_index=0)
    node = _admit(controller, first, second)
    assert node.admitted is True
    rows = {r["claim_id"]: (r["paper_id"], r["species_state"]) for r in node.non_supporting_claims}
    assert rows == {"cl_MCP1_ev_p1": ("PMC:PMCP1", IVO), "cl_MCP2_ev_p2": ("PMC:PMCP2", AMBIGUOUS)}


# ===========================================================================
# R1-N1: a paper's batches are all registered before any is judged
# ===========================================================================
KNOCKOUT = "In Arabidopsis thaliana gh3.10 knockout lines, JA-Ile was undetectable in wounded leaves."
ANALOGY = ("By analogy with the moss enzyme, Arabidopsis thaliana GH3.10 presumably converts jasmonic acid "
           "to JA-Ile.")


def _two_batch_run(tmp_path, order):
    """One primary paper, two passages, one passage per extraction call: batch A cites the
    knockout sentence, batch B the inference sentence, for the same claim."""
    from t2pw.mining.acquisition import OfflinePaperSource
    from t2pw.rag.acquire import CandidatePaper

    texts = {"ev_ko": KNOCKOUT, "ev_an": ANALOGY}
    candidate = CandidatePaper(id="PMC5151515", source="europepmc", title="Jasmonate conjugation",
                               abstract=KNOCKOUT, full_text="Results. " + KNOCKOUT + " " + ANALOGY,
                               source_uri="https://example.org/y", year="2024")
    config = H.make_config(tmp_path, budgets={"max_rounds": 1, "max_passages_per_call": 1})
    index = H.offline_index(config, tmp_path)
    spans = [EvidenceSpan(evidence_id=eid, paper_id="", text=texts[eid], section="results") for eid in order]
    index.spans_for_paper = lambda pid, q, **kw: [dataclasses.replace(s, paper_id=pid) for s in spans]  # noqa: ARG005

    def payload(messages):
        shown = [eid for eid in texts if eid in json.dumps(messages)]
        if len(shown) != 1:
            return {"claims": []}
        eid = shown[0]
        return {"claims": [{"substrates": [{"name": "jasmonic acid", "evidence_ids": [eid]}],
                            "products": [{"name": "JA-Ile", "evidence_ids": [eid]}],
                            "gene": "GH3.10", "gene_evidence_ids": [eid],
                            "species": "Arabidopsis thaliana", "species_evidence_ids": [eid],
                            "evidence_ids": [eid]}]}

    controller = MiningController(config, source=OfflinePaperSource([candidate]), index=index,
                                  enabled_override=True,
                                  chat_fn=lambda messages, **k: SimpleNamespace(text=json.dumps(payload(messages)),
                                                                                diagnostics=None))
    return controller.run(probe_model=False, probe_embeddings=False)


def _assert_never_admitted(outcome):
    state = outcome.state
    claims = list(state.store.claims.values())
    assert len(claims) == 1 and claims[0].species_state == INFERRED
    assert {s["evidence_id"] for s in claims[0].species_support["sentences"]} == {"ev_ko", "ev_an"}
    current = state.store.history(claims[0].claim_identity).current
    assert current.status not in ADMITTED_STATUSES
    for node in state.graph.reactions.values():
        assert node.admitted is False
        assert node.decision_status not in ADMITTED_STATUSES
    accepted = outcome.summary["claims"]["by_latest_status"].get("accepted", 0)
    assert accepted == sum(1 for r in state.graph.reactions.values() if r.decision_status in ADMITTED_STATUSES)
    assert outcome.summary["graph"]["primary_reactions"] == 0
    report = outcome.run_dir.path("report.md").read_text(encoding="utf-8")
    assert "- `accepted`" not in report and "- `candidate`: 1" in report


def test_repair2_745b_r1n1_a_later_batch_of_the_same_paper_never_leaves_a_stale_admission(tmp_path):
    """At fd53a301 batch A (knockout) admitted the reaction; batch B then made the only
    claim inferred_only, the store's latest decision became 'candidate', and the node
    stayed admitted and labelled 'accepted' (summary and report disagreed)."""
    _assert_never_admitted(_two_batch_run(tmp_path, ("ev_ko", "ev_an")))


def test_new_acceptance_745b_r2_n1_the_other_batch_order_gives_the_same_result(tmp_path):
    """The paired order (inference batch first) was already held back at fd53a301."""
    _assert_never_admitted(_two_batch_run(tmp_path, ("ev_an", "ev_ko")))
