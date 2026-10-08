"""ORCH-745 change set D: loss-of-function credit made polarity-, participant- and
genotype-aware, extracts read as cell-free, more plant genera seen.

Every probe the final CS-B reviewers demonstrated at ``fd7c737c`` (code B-1 / B-2, red team
R3-B1 / R3-B2 / R3-N1, bio BIO-R2-1..3) is a ``test_cs_d_745_*`` row asserting NOT
``target_species_supported``; each reached ``target_species_supported`` at ``fd7c737c``. The
BIO-R2-1 rows also hold at ``fd53a301``: they guard a regression the repair-2 wording made.
``test_cs_d_745_d1_negated_verb_absence_forms_earn_credit`` is D1's one recall gain (it fails at
``fd7c737c`` too). The ``test_new_acceptance_cs_d_745_*`` rows are controls: the bio-auditor's
canonical A. thaliana phrasings, the brief's recall controls, and each in-planta veto in a
loss-of-function sentence (code review N-1). They hold at both ends.

The probe shape is the reviewers': the probe sentence as a results span ``ev_a``, the species
cited to a plain abstract sentence ``ev_sp``, the paper typed primary; the bio-auditor's rows
also carry the paper's screen organism (``observed``).
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

from t2pw.mining.extraction import build_claims  # noqa: E402
from t2pw.mining.models import EvidenceSpan  # noqa: E402
from t2pw.mining.species import split_sentences  # noqa: E402

TARGET = "arabidopsis thaliana"
TSS = "target_species_supported"
SPECIES_SPAN = EvidenceSpan(evidence_id="ev_sp", paper_id="PMC:PMCX", section="abstract",
                            text="Jasmonate metabolism was studied in Arabidopsis thaliana.")
FIXTURES = ROOT / "tests" / "fixtures" / "mining"


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


AT = ("Arabidopsis thaliana",)  # the bio-auditor's probes carry the paper's screen organism
GENE_GH3 = dict(enzyme="", gene="GH3.10")  # the code review's probe shape
GENE_JAR1 = dict(enzyme="", gene="JAR1")
AOC = dict(sub="allene oxide", prod="12-OPDA", enzyme="allene oxide cyclase", gene="AOC")
OPR_NAMED = dict(sub="OPDA", prod="OPC-8:0", enzyme="OPDA reductase 3", gene="OPR3")
SMT = dict(sub="sterol", prod="24-methylenecycloartanol", enzyme="sterol methyltransferase", gene="SMT1")


# ===========================================================================
# D1: the effect is a loss-of-function consequence on the RIGHT participant
# ===========================================================================
@_rows(
    # code B-1: a product that still forms or rises, a no-effect statement, an occurrence
    ("B1-JA-Gln-still-accumulated", "In gh3.10 jar1 knockout lines, JA-Gln still accumulated in wounded leaves.",
     dict(GENE_GH3, prod="JA-Gln")),
    ("B1-JA-Gln-increased", "JA-Gln levels increased in wounded leaves of gh3.10 jar1 knockout lines.",
     dict(GENE_GH3, prod="JA-Gln")),
    ("B1-jar1-still-accumulated", "In flowers of jar1 single mutants, JA-Ile still accumulated.", GENE_JAR1),
    ("B1-occurrence", "Upon wounding, gh3.10 mutants accumulated JA-Ile.", GENE_GH3),
    ("B1-normally", "JA-Ile accumulated normally in gh3.10 mutant leaves after wounding.", GENE_GH3),
    ("B1-indistinguishable", "In gh3.10 single mutants, JA-Ile accumulated to levels indistinguishable from the "
                             "wild type.", GENE_GH3),
    ("B1-higher-than-Col-0", "In gh3.10 mutants, JA-Ile levels were higher than in Col-0 plants.", GENE_GH3),
    ("B1-lacking-the-gene", "In mutants lacking GH3.10, wound-induced JA-Ile levels were indistinguishable from "
                            "those of wild-type plants.", GENE_GH3),
    ("B1-still-increased", "JA-Ile still increased in wounded leaves of gh3.10 single mutants.", GENE_GH3),
    ("B1-wild-type-extent", "gh3.10 mutants accumulated JA-Ile to wild-type extent after wounding.", GENE_GH3),
    ("B1-similar-levels", "gh3.10 mutants accumulated similar JA-Ile levels after wounding.", GENE_GH3),
    ("B1-deficient-in-the-enzyme", "In gh3.10 mutants deficient in the enzyme, JA-Ile levels were similar.", GENE_GH3),
    ("B1-in-vitro-only-normally", "Recombinant GH3.5 conjugated jasmonic acid to isoleucine in vitro, yielding "
                                  "JA-Ile. JA-Ile accumulated normally in gh3.5 mutant leaves after wounding.",
     dict(enzyme="", gene="GH3.5")),
    ("B1-in-vitro-only-accumulated", "Recombinant GH3.5 conjugated jasmonic acid to isoleucine in vitro, yielding "
                                     "JA-Ile. Upon wounding, gh3.5 mutants accumulated JA-Ile.",
     dict(enzyme="", gene="GH3.5")),
    # red team R3-B1: an absence form under a negation
    ("R3B1-not-abolished", "Wound-induced JA-Ile accumulation was not abolished in the jar1 mutant.",
     dict(enzyme="JAR1")),
    ("R3B1-not-abolished-ko", "JA-Ile formation was not abolished in gh3.10 knockout lines.", dict(gene="GH3.10")),
    ("R3B1-not-absent", "JA-Ile was not absent from flowers of the gh3.10 mutant.", dict(gene="GH3.10")),
    ("R3B1-not-eliminated", "JA-Ile accumulation was not eliminated in the jar1 mutant.", dict(enzyme="JAR1")),
    ("R3B1-not-depleted", "The pool of JA-Ile was not depleted in jar1 mutant leaves.", dict(enzyme="JAR1")),
    ("R3B1-not-deficient", "The jar1 mutant was not deficient in JA-Ile after wounding.", dict(enzyme="JAR1")),
    ("R3B1-never-abolished", "JA-Ile accumulation was never abolished in jar1 mutant leaves.", dict(enzyme="JAR1")),
    ("R3B1-not-completely-abolished", "Wound-induced JA-Ile accumulation was not completely abolished in the jar1 "
                                      "mutant.", dict(enzyme="JAR1")),
    # red team R3-N1: direction (a product rising in its own synthase's knockout)
    ("R3N1-product-higher", "JA-Ile levels were higher in gh3.10 mutant flowers than in the wild type.",
     dict(gene="GH3.10")),
    ("R3N1-product-accumulated", "JA-Ile accumulated in leaves of the jar1 mutant after wounding.",
     dict(enzyme="JAR1")),
    ("R3N1-unrelated-metabolite", "In the jar1 mutant, salicylic acid accumulated to higher levels than in the wild "
                                  "type.", dict(enzyme="JAR1", sub="salicylate", prod="salicylic acid")),
    # bio BIO-R2-1: "comparable to" (these also hold at fd53a301: a regression guard)
    ("BIO1-N1", "In the jar1 mutant, JA-Ile accumulated to levels comparable to those in the wild type.",
     dict(enzyme="JAR1", observed=AT)),
    ("BIO1-N3", "In opr3 mutants, OPC-8:0 accumulated to levels comparable to control plants.",
     dict(sub="OPDA", prod="OPC-8:0", enzyme="OPR3", observed=AT)),
    ("BIO1-N4", "In the lox2 mutant, 13-HPOT accumulated to concentrations comparable to those of untreated "
                "controls.", dict(sub="alpha-linolenic acid", prod="13-HPOT", enzyme="LOX2", observed=AT)),
    ("BIO1-N5", "JA-Ile accumulated in jar1 mutant leaves at concentrations comparable to the wild type.",
     dict(enzyme="JAR1", observed=AT)),
    ("BIO1-N6", "In aos knockout lines, 12-OPDA accumulated to levels comparable to those of wild-type plants.",
     dict(sub="13-HPOT", prod="12-OPDA", enzyme="AOS", observed=AT)),
    # bio BIO-R2-2: the contrary combinations
    ("BIO2-P2-product-up", "In opr3 mutants, OPC-8:0 accumulated in wounded leaves.",
     dict(sub="OPDA", prod="OPC-8:0", enzyme="OPR3", observed=AT)),
    ("BIO2-P3-product-elevated", "JA-Ile accumulated to elevated levels in gh3.10 knockout lines.",
     dict(enzyme="GH3.10", observed=AT)),
    ("BIO2-P4-substrate-down", "In opr3 mutants, OPDA was reduced in wounded leaves.",
     dict(sub="OPDA", prod="OPC-8:0", enzyme="OPR3", observed=AT)),
    ("BIO2-P5-no-wild-type", "JA-Ile levels were higher in the jar1 mutant than in the aos mutant.",
     dict(enzyme="JAR1", observed=AT)),
)
def test_cs_d_745_d1_an_effect_on_the_wrong_participant_or_negated_earns_no_credit(text, kw):
    assert probe(text, **kw).species_state != TSS


@_rows(
    # bio BIO-R2-4 B1 / B2: the dominant phrasing of a knockout result (recall gain;
    # bio-r-2 leaves the repair-2 5a_removed list)
    ("BIO4-B1-did-not-accumulate", "The aos mutant did not accumulate 12-OPDA after wounding.",
     dict(sub="13-HPOT", prod="12-OPDA", enzyme="AOS", observed=AT)),
    ("BIO4-B2-failed-to-accumulate", "jar1 mutants failed to accumulate JA-Ile in wounded leaves.",
     dict(enzyme="JAR1", observed=AT)),
    ("no-longer-detected", "In jar1 mutants, JA-Ile was no longer detected after wounding.", dict(enzyme="JAR1")),
)
def test_cs_d_745_d1_negated_verb_absence_forms_earn_credit(text, kw):
    claim = probe(text, **kw)
    assert (claim.species_state, claim.species_support["grade"]) == (TSS, "in_planta")


# ===========================================================================
# D2: the participant and the genotype must be the claim's own
# ===========================================================================
def _arc_sentence():
    """The verbatim abstract sentence of PMC:PMC13358708 (pilot-001's screened papers)."""
    payload = json.loads((FIXTURES / "pilot001_screened_papers.json").read_text(encoding="utf-8"))
    paper = next(p for p in payload["papers"] if p["paper_id"] == "PMC:PMC13358708")
    return next(s for s in split_sentences(paper["abstract"]) if "protein abundance of the JA biosynthetic" in s)


def test_cs_d_745_d2_the_verbatim_arc_sentence_earns_no_credit():
    """The mutant is arc, the measure a protein, the "participant" the enzyme's own name."""
    sentence = _arc_sentence()
    assert sentence.startswith("Moreover, protein abundance of the JA biosynthetic enzyme allene oxide cyclase")
    assert probe(sentence, **AOC).species_state != TSS


@_rows(
    ("R3B2-aoc-protein-own-mutant", "Allene oxide cyclase protein was reduced in the aoc1 knockout line.", AOC),
    ("R3B2-aoc-protein-arc", "Protein abundance of the JA biosynthetic enzyme allene oxide cyclase was increased in "
                             "arc mutants.", AOC),
    ("R3B2-opr3-protein", "OPDA reductase 3 protein accumulated in the arc mutant.", OPR_NAMED),
    ("R3B2-smt-transcript", "Sterol methyltransferase transcript levels were reduced in the hydra1 mutant.", SMT),
    ("R3B2-other-gene-by-phrase", "In the myc2 mutant, JA-Ile accumulation by JAR1 was reduced.", dict(enzyme="JAR1")),
    ("R3B2-other-gene-still-expressed", "Wound-induced JA-Ile accumulation was abolished in the aos mutant, in which "
                                        "JAR1 is still expressed.", dict(enzyme="JAR1")),
    ("R3-other-gene-assay", "In the coi1 mutant, JA-Ile accumulation measured with the JAR1 assay was reduced.",
     dict(enzyme="JAR1")),
)
def test_cs_d_745_d2_a_participant_or_genotype_that_is_not_the_claims_own_earns_no_credit(text, kw):
    assert probe(text, **kw).species_state != TSS


# ===========================================================================
# D3: extracts are cell-free (code B-2)
# ===========================================================================
@_rows(
    ("B2-IV1-extracts-of-mutant", "Formation of JA-Ile from jasmonic acid was abolished in extracts of jar1 mutant "
                                  "leaves.", GENE_JAR1),
    ("B2-IV2-crude-enzyme-extracts", "In crude enzyme extracts of jar1 mutant leaves, JA-Ile formation was "
                                     "abolished.", GENE_JAR1),
    ("B2-IV3-extracts-from", "Extracts from jar1 mutant flowers lacked JA-Ile-conjugating activity toward jasmonic "
                             "acid.", GENE_JAR1),
)
def test_cs_d_745_d3_a_loss_of_function_measured_in_an_extract_is_cell_free(text, kw):
    claim = probe(text, **kw)
    assert claim.species_state != TSS
    assert claim.species_state == "in_vitro_or_cell_free"


# ===========================================================================
# D4: plant genera the organism words missed (bio BIO-R2-3)
# ===========================================================================
@_rows(
    ("BIO3-F1-Brachypodium", "In Brachypodium aos mutants, 12-OPDA was undetectable in wounded leaves.",
     dict(sub="13-HPOT", prod="12-OPDA", enzyme="AOS", observed=AT)),
    ("BIO3-F5-Camelina", "In Camelina sativa aos mutants, 12-OPDA was undetectable.",
     dict(sub="13-HPOT", prod="12-OPDA", enzyme="AOS", observed=AT)),
    ("BIO3-Bd-prefix", "In Bdaos1 mutants, 12-OPDA was undetectable in wounded leaves.",
     dict(sub="13-HPOT", prod="12-OPDA", enzyme="", gene="BdAOS1")),
)
def test_cs_d_745_d4_another_plant_genus_earns_no_credit(text, kw):
    assert probe(text, **kw).species_state != TSS


# ===========================================================================
# Controls: hold at fd7c737c and here
# ===========================================================================
@_rows(
    ("T1", "In flowers and wounded leaves of gh3.10 jar1 knockout lines JA-Ile was below the detection limit.",
     dict(gene="GH3.10")),
    ("no-JA-Ile-was-detected", "No JA-Ile was detected in gh3.10 jar1 knockout lines.", dict(gene="GH3.10")),
    ("bio-r-1", "In opr3 mutant leaves, OPC-8:0 was not detected.",
     dict(sub="OPDA", prod="OPC-8:0", enzyme="OPR3", gene="OPR3")),
    ("n1-r1", "In the opr3 knockout line, OPC-8:0 was undetectable in leaves.",
     dict(sub="OPDA", prod="OPC-8:0", enzyme="OPR3")),
    ("BIO2-P1-substrate-up", "In opr3 mutants, OPDA accumulated in wounded leaves.",
     dict(sub="OPDA", prod="OPC-8:0", enzyme="OPR3", observed=AT)),
    # the bio-auditor's canonical A. thaliana loss-of-function prose (rev745b-r2-bio runs/r6, A1-A10)
    ("A1", "In aos knockout lines, 12-OPDA was undetectable in wounded leaves.",
     dict(sub="13-HPOT", prod="12-OPDA", enzyme="", gene="AOS", observed=AT)),
    ("A2", "JA-Ile was not detected in the jar1 mutant.", dict(enzyme="JAR1", observed=AT)),
    ("A3", "OPC-8:0 was absent from opr3 mutant leaves.", dict(sub="OPDA", prod="OPC-8:0", enzyme="OPR3", observed=AT)),
    ("A4", "In the lox2 mutant, wound-induced 13-HPOT accumulation was reduced.",
     dict(sub="alpha-linolenic acid", prod="13-HPOT", enzyme="LOX2", observed=AT)),
    ("A5", "aoc mutants lacked 12-OPDA after wounding.",
     dict(sub="allene oxide", prod="12-OPDA", enzyme="AOC", observed=AT)),
    ("A6", "In acx1 acx5 double mutants, jasmonic acid was strongly reduced after wounding.",
     dict(sub="OPC-8:0", prod="jasmonic acid", enzyme="ACX1", observed=AT)),
    ("A7", "GH3.10 knockout plants accumulated no JA-Ile in flowers.", dict(enzyme="GH3.10", observed=AT)),
    ("A8", "The opr3 mutant accumulated OPDA in wounded leaves.",
     dict(sub="OPDA", prod="OPC-8:0", enzyme="OPR3", observed=AT)),
    ("A9", "JA-Ile biosynthesis was abolished in the gh3.10 jar1 double knockout.", dict(enzyme="GH3.10", observed=AT)),
    ("A10", "T-DNA insertion lines of AOS were deficient in 12-OPDA.",
     dict(sub="13-HPOT", prod="12-OPDA", enzyme="AOS", observed=AT)),
)
def test_new_acceptance_cs_d_745_loss_of_function_on_the_right_participant_keeps_credit(text, kw):
    claim = probe(text, **kw)
    assert (claim.species_state, claim.species_support["grade"]) == (TSS, "in_planta")


@_rows(
    # code review N-1: each in-planta veto, in a sentence that is otherwise a complete
    # loss-of-function result
    ("veto-hedge", "JA-Ile may be reduced in gh3.10 mutants.", dict(gene="GH3.10")),
    ("veto-application", "In jar1 mutants treated with JA, JA-Ile was undetectable.", dict(enzyme="JAR1")),
    ("veto-comparison", "As in jar1 mutants, JA-Ile was undetectable in gh3.10 knockout lines.", dict(gene="GH3.10")),
    ("veto-protein-mutant", "In the GH3.10 R117A mutant, JA-Ile was undetectable.", {}),
    ("veto-citation", "In gh3.10 knockout lines, JA-Ile was undetectable (12).", dict(gene="GH3.10")),
    ("veto-no-effect", "In gh3.10 knockout lines, JA-Ile levels were unchanged.", dict(gene="GH3.10")),
    ("veto-other-organism", "In tomato opr3 mutants, OPC-8:0 was undetectable in wounded leaves.",
     dict(sub="OPDA", prod="OPC-8:0", enzyme="OPR3")),
    # fail closed: the "reduced" clause names no genotype (a recall cost, known_limitations.md)
    ("reduced-but-not-abolished", "JA-Ile accumulation was reduced but not abolished in the jar1 mutant.",
     dict(enzyme="JAR1")),
)
def test_new_acceptance_cs_d_745_vetoes_hold_on_the_loss_of_function_route(text, kw):
    assert probe(text, **kw).species_state != TSS
