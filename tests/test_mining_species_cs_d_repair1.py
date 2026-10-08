"""ORCH-745 change set D, repair round 1: no-effect comparisons after a negated verb (R1),
the effect's polarity attributed to the right participant (R2), and the marker-of tail
(RT-D-4).

Every ``test_cs_d_r1_745_*`` row reached ``target_species_supported`` at ``6454ef98`` (the
not-TSS rows) or failed to (the recall rows); the manifest says which also fail at
``fd7c737c``. The ``test_new_acceptance_cs_d_r1_745_*`` rows are the brief's controls and
hold at both ends; ``..._both_sides_before_the_effect_is_ambiguous`` is the deliberate,
documented recall cost of R2 (ii).

The probe shape is the reviewers': the probe sentence as a results span ``ev_a``, the species
cited to a plain abstract sentence ``ev_sp``, the paper typed primary; most rows also carry
the paper's screen organism (``observed``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.mining.extraction import build_claims  # noqa: E402
from t2pw.mining.models import EvidenceSpan  # noqa: E402

TARGET = "arabidopsis thaliana"
TSS = "target_species_supported"
SPECIES_SPAN = EvidenceSpan(evidence_id="ev_sp", paper_id="PMC:PMCX", section="abstract",
                            text="Jasmonate metabolism was studied in Arabidopsis thaliana.")


def probe(text, *, sub="jasmonic acid", prod="JA-Ile", enzyme="GH3.10", gene="", observed=()):
    """The reviewers' fixture: build_claims on the probe sentence; the claim it returns."""
    span = EvidenceSpan(evidence_id="ev_a", paper_id="PMC:PMCX", text=text, section="results")
    row = {"substrates": [{"name": sub, "evidence_ids": ["ev_a"]}],
           "products": [{"name": prod, "evidence_ids": ["ev_a"]}],
           "species": "Arabidopsis thaliana", "species_evidence_ids": ["ev_sp"], "evidence_ids": ["ev_a"]}
    if enzyme:
        row.update(enzyme=enzyme, enzyme_evidence_ids=["ev_a"])
    if gene:
        row.update(gene=gene, gene_evidence_ids=["ev_a"])
    claims, _ = build_claims({"claims": [row]}, spans=[span, SPECIES_SPAN], paper_id="PMC:PMCX",
                             target_species_normalized=TARGET, round_index=1, model="probe",
                             paper_observed_organisms=list(observed), paper_evidence_type="primary")
    return claims[0]


def _rows(*rows):
    return pytest.mark.parametrize("text,kw", [(t, k) for _, t, k in rows], ids=[n for n, _, _ in rows])


AT = ("Arabidopsis thaliana",)
# the code review's shapes (pm_adv / pm_adv2)
J = dict(gene="", enzyme="JAR1")
AOS = dict(sub="13-HPOT", prod="12-OPDA", gene="", enzyme="AOS")
OPR = dict(sub="OPDA", prod="OPC-8:0", gene="", enzyme="OPR3")
# the bio-auditor's shapes (rev745d-bio probes / set1 / set2)
JA_GH3 = dict(enzyme="", gene="GH3.10", observed=AT)
JA_JAR1 = dict(enzyme="JAR1", observed=AT)
JAP = dict(sub="jasmonic acid (JA)", prod="jasmonoyl-isoleucine (JA-Ile)")
JAS = dict(sub="JA", prod="JA-Ile")
OPR_AT = dict(OPR, observed=AT)
AOS_AT = dict(AOS, observed=AT)
# the red team's shapes (rev745d-redteam p3d2 / p3d3)
RJAR = dict(enzyme="JAR1", observed=AT)
RGH3 = dict(gene="GH3.10", observed=AT)


# ===========================================================================
# R1: a negated comparison or delay is not an absence (CSD-B1, BIO-D-1, RT-D-1)
# ===========================================================================
@_rows(
    ("Y1", "The jar1 mutant did not accumulate significantly less JA-Ile than the wild type after wounding.", J),
    ("Y2", "jar1 mutants did not accumulate significantly lower amounts of JA-Ile than wild-type plants.", J),
    ("Y3", "Unexpectedly, the aos mutant did not accumulate less 12-OPDA than wild-type plants.", AOS),
    ("Y4", "opr3 mutants did not produce less OPC-8:0 than the wild type in wounded leaves.", OPR),
    ("Y5", "In the jar1 mutant, JA-Ile did not accumulate to lower levels than in Col-0.", J),
    ("Y6", "jar1 mutants failed to accumulate more JA-Ile than wild-type plants.", J),
    ("Y7", "Wounded leaves of the jar1 mutant contained no less JA-Ile than wild-type leaves.", J),
    ("X4", "Flowers of jar1 mutants contained no less JA-Ile than wild-type flowers.", J),
    ("X5", "jar1 mutants produced no less JA-Ile than wild-type plants after wounding.", J),
    ("X7", "jar1 mutants did not accumulate less JA-Ile than wild-type plants after wounding.", J),
    ("X8", "The jar1 mutant did not accumulate lower levels of JA-Ile than the wild type.", J),
    ("X9", "Wounded jar1 mutant leaves did not produce less JA-Ile than wild-type leaves.", J),
    ("bio-N1", "The gh3.10 single mutant did not accumulate significantly less JA-Ile than wild-type plants after "
               "wounding.", JA_GH3),
    ("bio-N2", "In wounded leaves, gh3.10 mutants did not accumulate lower amounts of JA-Ile than Col-0.", JA_GH3),
    ("bio-N3", "The opr3 mutant did not produce less OPC-8:0 than wild-type plants.", OPR_AT),
    ("bio-N4-delay", "In jar1 mutants, JA-Ile did not accumulate until 6 h after wounding.", JA_JAR1),
    ("bio-N5", "gh3.10 single mutants did not accumulate significantly lower JA-Ile levels after wounding.", JA_GH3),
    ("bio-N8", "gh3.10 mutants did not accumulate JA-Ile to a lesser extent than Col-0 plants.", JA_GH3),
    ("bio-Bn5", "jar1 mutants did not accumulate significantly less JA-Ile than wild-type plants after wounding.",
     JA_JAR1),
    ("bio-Bn6", "The jar1 mutant did not accumulate less JA-Ile than Col-0 after wounding.", JA_JAR1),
    ("bio-Bn7-delay", "In jar1 mutants, JA-Ile did not accumulate until 2 h after wounding.", JA_JAR1),
    ("bio-Bn8", "The opr3 mutant did not accumulate more OPC-8:0 than the wild type.", OPR_AT),
    ("bio-Bn11", "jar1 mutants did not accumulate lower JA-Ile levels after wounding.", JA_JAR1),
    ("rt-B2", "jar1 mutants produced no less JA-Ile than the wild type.", RJAR),
    ("rt-B3", "The jar1 mutant did not accumulate more JA-Ile than wild-type plants.", RJAR),
    ("rt-B4", "jar1 mutants failed to accumulate more JA-Ile than controls.", RJAR),
    ("rt-B8", "jar1 mutants did not accumulate JA-Ile further after wounding.", RJAR),
    ("rt-N4-failed-further", "jar1 mutants failed to accumulate JA-Ile further.", RJAR),
    # credited at fd7c737c too (via the direction-blind "accumulated")
    ("rt-B1-no-more", "gh3.10 mutants accumulated no more JA-Ile than wild-type plants.", RGH3),
    ("rt-B10-no-greater", "gh3.10 mutants accumulated no greater JA-Ile levels than Col-0.", RGH3),
    ("rt-N5-no-fewer", "gh3.10 mutants contained no fewer JA-Ile molecules than the wild type.", RGH3),
    ("rt-N8-substrate-no-more", "In opr3 mutants, OPDA accumulated no more than in the wild type.", OPR_AT),
)
def test_cs_d_r1_745_r1_a_negated_comparison_or_delay_earns_no_credit(text, kw):
    assert probe(text, **kw).species_state != TSS


# ===========================================================================
# R2: the effect's polarity on the right participant (BIO-D-2, RT-D-2, review N3)
# ===========================================================================
@_rows(
    # (i) "accumulated" with a decrease word is a decrease, never a substrate increase
    ("bio-R1-reversed-reaction", "cyp94b3 mutants accumulated less 12-OH-JA-Ile after wounding.",
     dict(sub="12-OH-JA-Ile", prod="JA-Ile", enzyme="", gene="CYP94B3")),
    ("bio-R2-reversed-lower", "In cyp94b3 mutants, 12-OH-JA-Ile accumulated to lower levels after wounding.",
     dict(sub="12-OH-JA-Ile", prod="JA-Ile", enzyme="", gene="CYP94B3")),
    ("bio-P1-less-JA", "The jar1 mutant accumulated less JA than the wild type after wounding.",
     dict(JAP, enzyme="JAR1", observed=AT)),
    ("bio-P2-lower-JA", "jar1 mutant plants accumulated lower levels of JA after wounding.",
     dict(JAP, enzyme="JAR1", observed=AT)),
    ("bio-P3", "In opr3 mutants, OPDA accumulated to lower levels than in Col-0 after wounding.", OPR_AT),
    ("bio-Cn4", "In opr3 mutants, OPDA accumulated to lower levels after wounding.", OPR_AT),
    ("bio-Cn5", "The jar1 mutant accumulated lower levels of jasmonic acid after wounding.", JA_JAR1),
    ("bio-Cn6", "jar1 mutants accumulated reduced amounts of jasmonic acid in wounded leaves.", JA_JAR1),
    ("bio-Cn7", "In jar1 mutants, jasmonic acid accumulated to reduced levels after wounding.", JA_JAR1),
    ("bio-Cn10", "Opr3 mutants accumulated less OPDA after wounding.", OPR_AT),
    ("bio-Cn11-weakly", "In opr3 mutants, OPDA accumulated only weakly after wounding.", OPR_AT),
    ("bio-Cn12", "The aos mutant accumulated decreased amounts of 13-HPOT.", AOS_AT),
    # (ii) both sides among the subject candidates: ambiguous (the appositives)
    ("bio-S1-conjugate-appositive", "In gh3.10 mutants, the JA conjugate JA-Ile accumulated after wounding.",
     dict(JAP, enzyme="", gene="GH3.10", observed=AT)),
    ("bio-S2-precursor-appositive", "In the jar1 mutant, the JA-Ile precursor JA was reduced after wounding.",
     dict(JAP, enzyme="JAR1", observed=AT)),
    ("bio-S3-product-appositive", "In opr3 mutants, the OPDA product OPC-8:0 accumulated in wounded leaves.", OPR_AT),
    ("bio-Cn8", "In gh3.10 mutants, the JA conjugate JA-Ile accumulated to higher levels after wounding.",
     dict(JAS, enzyme="", gene="GH3.10", observed=AT)),
    ("bio-Cn9", "In gh3.10 mutants, the JA conjugate JA-Ile accumulated after wounding.",
     dict(JAS, enzyme="", gene="GH3.10", observed=AT)),
    ("rt-A2-metabolite-appositive", "In opr3 mutants, the OPDA metabolite OPC-8:0 accumulated.", OPR_AT),
    ("X19-product-up-other-down", "JA-Gln accumulated at higher levels in gh3.10 jar1 knockout lines, which had "
                                  "reduced JA-Ile.", dict(prod="JA-Gln", gene="GH3.10", enzyme="")),
    ("X20-product-up-substrate-down", "In opr3 mutants, OPC-8:0 levels were elevated, with decreased OPDA.", OPR),
    # (iii) a whole-plant phenotype is the subject, or the participant is far after the form
    ("rt-P1-growth", "Root growth of the jar1 mutant was reduced on medium containing JA-Ile.", RJAR),
    ("rt-P2-fertility", "In the jar1 mutant, male fertility was reduced in flowers that accumulate JA-Ile.", RJAR),
    ("rt-P4-resistance", "Herbivore resistance was reduced in jar1 mutants defective in JA-Ile signalling.", RJAR),
    ("rt-P5-senescence", "Leaf senescence was reduced in jar1 mutants, measured alongside JA-Ile.", RJAR),
    ("rt-P6-yield", "Seed yield was lower in gh3.10 mutants grown on JA-Ile medium.", RGH3),
    ("bio-L1-still-fertile", "gh3.10 jar1 mutants lacking the JA-Ile synthesizing enzymes were still fertile.",
     JA_GH3),
    ("bio-L2-root-growth", "jar1 mutants lacking the JA-Ile forming enzyme showed normal root growth.", JA_JAR1),
    # an enzyme noun headed by a participant name is the enzyme (review N1 / N4)
    ("X17-other-cyclases", "In the aoc2 mutant, the other allene oxide cyclases were increased.",
     dict(sub="allene oxide", prod="12-OPDA", enzyme="AOC", gene="AOC2")),
    ("X18-reductase-levels", "In opr3 mutants, OPDA reductase levels were elevated.",
     dict(sub="OPDA", prod="OPC-8:0", enzyme="OPR3", gene="")),
    # a negation after the effect, or a negating adverb (review N4, BIO-D-4)
    ("X11-in-none-of", "JA-Ile levels were reduced in none of the jar1 mutant lines.", J),
    ("X12-neither-nor", "JA-Ile was undetectable in neither the jar1 mutant nor the coi1 mutant.", J),
    ("bio-D8-hardly", "In jar1 mutants, JA-Ile was hardly reduced after wounding.", JA_JAR1),
    ("bio-D11-barely", "In jar1 mutants, JA-Ile was barely decreased after wounding.", JA_JAR1),
    ("bio-M1-scarcely", "In jar1 mutants, JA-Ile was scarcely reduced after wounding.", JA_JAR1),
    ("bio-M2-neither-nor", "In gh3.10 mutants, JA-Ile was reduced in neither flowers nor leaves.", JA_GH3),
)
def test_cs_d_r1_745_r2_an_effect_on_the_wrong_or_an_ambiguous_participant_earns_no_credit(text, kw):
    assert probe(text, **kw).species_state != TSS


# ===========================================================================
# RT-D-4: the marker-of tail
# ===========================================================================
@_rows(
    ("rt-L1-pathway", "Mutants in the JAR1 pathway accumulated no JA-Ile.", RJAR),
    ("rt-L3-background", "The myc2 mutant in the JAR1 background accumulated no JA-Ile.", RJAR),
    ("JAR1-pathway-mutants", "JAR1 pathway mutants accumulated no JA-Ile.", RJAR),
)
def test_cs_d_r1_745_rtd4_a_marker_of_a_pathway_or_background_is_not_of_the_gene(text, kw):
    assert probe(text, **kw).species_state != TSS


@_rows(
    # CS-D's three-token cap dropped it (credited at fd7c737c)
    ("SALK-insertion-line-of", "The SALK_034543 insertion line of GH3.10 accumulated no JA-Ile.", dict(gene="GH3.10")),
    # R2's optional recall gain: "accumulated less <product>" is a product decrease
    ("bio-Rc2-correct-direction", "cyp94b3 mutants accumulated less 12-OH-JA-Ile after wounding.",
     dict(sub="JA-Ile", prod="12-OH-JA-Ile", enzyme="", gene="CYP94B3")),
    ("aos-accumulated-less-12-OPDA", "aos mutants accumulated less 12-OPDA after wounding.", AOS_AT),
)
def test_cs_d_r1_745_recall_earns_credit(text, kw):
    claim = probe(text, **kw)
    assert (claim.species_state, claim.species_support["grade"]) == (TSS, "in_planta")


# ===========================================================================
# Controls (the brief's list): hold at 6454ef98 and here
# ===========================================================================
@_rows(
    ("T1", "In flowers and wounded leaves of gh3.10 jar1 knockout lines JA-Ile was below the detection limit.",
     dict(gene="GH3.10")),
    ("no-JA-Ile-was-detected", "No JA-Ile was detected in gh3.10 jar1 knockout lines.", dict(gene="GH3.10")),
    ("bio-r-1", "In opr3 mutant leaves, OPC-8:0 was not detected.", dict(OPR, gene="OPR3")),
    ("n1-r1", "In the opr3 knockout line, OPC-8:0 was undetectable in leaves.", OPR),
    ("P1-substrate-accumulated", "In opr3 mutants, OPDA accumulated in wounded leaves.", OPR_AT),
    ("Pc1-higher-OPDA-than-Col-0", "The opr3 mutant accumulated higher levels of OPDA than Col-0 after wounding.",
     OPR_AT),
    ("Pc2-lower-JA-Ile", "jar1 mutant plants accumulated lower levels of JA-Ile after wounding.",
     dict(JAP, enzyme="JAR1", observed=AT)),
    ("no-detectable", "gh3.10 knockout lines accumulated no detectable JA-Ile.", dict(gene="GH3.10")),
    ("B1-did-not-accumulate", "The aos mutant did not accumulate 12-OPDA after wounding.", AOS_AT),
    ("B2-failed-to-accumulate", "jar1 mutants failed to accumulate JA-Ile in wounded leaves.", JA_JAR1),
    ("no-longer-detected", "In jar1 mutants, JA-Ile was no longer detected after wounding.", dict(enzyme="JAR1")),
    ("20-fold-lower-than", "JA-Ile was 20-fold lower in jar1 mutants than in wild-type leaves.", RJAR),
    ("Nc1-double-mutant-did-not", "The gh3.10 jar1 double mutant did not accumulate JA-Ile after wounding.", JA_GH3),
    ("Nc2-failed-to-in-flowers", "gh3.10 jar1 knockout lines failed to accumulate JA-Ile in flowers.", JA_GH3),
    ("N6-did-not-detectable", "jar1 mutants did not accumulate detectable JA-Ile.", RJAR),
    ("bio-C2", "JA-Ile levels were reduced in wounded leaves of jar1 mutants.", JA_JAR1),
    ("bio-C3-substrate-elevated", "In the jar1 mutant, jasmonic acid levels were elevated after wounding.", JA_JAR1),
    ("bio-C4-while", "In gh3.10 jar1 knockout lines, JA-Ile was undetectable, while jasmonic acid accumulated.",
     JA_GH3),
    ("T-DNA-insertion-lines-of", "T-DNA insertion lines of GH3.10 accumulated no JA-Ile.", {}),
    ("deficient-in", "Mutants deficient in GH3.10 accumulated no JA-Ile in flowers.", {}),
    ("A7", "GH3.10 knockout plants accumulated no JA-Ile in flowers.", dict(enzyme="GH3.10", observed=AT)),
)
def test_new_acceptance_cs_d_r1_745_controls_keep_in_planta_credit(text, kw):
    claim = probe(text, **kw)
    assert (claim.species_state, claim.species_support["grade"]) == (TSS, "in_planta")


@_rows(
    # R2 (ii), a deliberate recall cost (merge rule 4; known_limitations.md): a clause
    # naming both sides before its effect is ambiguous. The red team listed the first as a
    # control and the bio-auditor named the second (S5) as the accepted cost.
    ("rt-control-formed-from", "In opr3 mutant leaves, OPC-8:0 formed from OPDA was reduced.", OPR_AT),
    ("bio-S5-formation-from", "In gh3.10 mutants, wound-induced JA-Ile formation from JA was abolished.",
     dict(JAP, enzyme="", gene="GH3.10", observed=AT)),
)
def test_cs_d_r1_745_both_sides_before_the_effect_is_ambiguous(text, kw):
    assert probe(text, **kw).species_state == "species_ambiguous"
