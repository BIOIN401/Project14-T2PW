"""ORCH-746 change set E: the two open species-gate blockers of the ORCH-745 CS-D final review.

E1 (BIO-D-R2-1 / RT-D-R2-1 / CSD-R2-N1): an inner comma -- a figure callout "(Fig. 3A, B)", a
stereo descriptor "(3R,7S)", a parenthetical ", which, after wounding, ..." or a second
comma-led clause -- made the 686ae134 subject rule read an open clause as closed, so the head
participant took the opposite predicate: product-up / substrate-down and reversed reactions
were credited. CS-E credits an effect form only when the 686ae134 reading, the bio-auditor's
reading and the red team's reading all credit it.

E2 (BIO-D-R2-2 / CSD-R2-N2): the negated-comparison / delay window stopped at a full stop not
followed by a digit ("(Fig. 3A)", "ca.", "i.e."), so "did not accumulate ... until 6 h" was
credited as an absence. The window now runs to a semicolon or the end of the sentence.

Every ``test_cs_e_746_*`` row is a probe the final review demonstrated as admitted at
``686ae134``; each fails there with an AssertionError (the G9 proof). Rows that several
reviewers wrote identically carry every reviewer id. The ``test_new_acceptance_cs_e_746_*``
rows hold at both ``686ae134`` and the tip: the brief's controls, and the recall cost (the
correct-direction partners the prototypes would credit and CS-E does not, because CS-E may
only subtract credit). The bio-auditor's A1-A10 are already pinned TSS / in_planta by
``test_mining_species_cs_d.py::test_new_acceptance_cs_d_745_loss_of_function_on_the_right_participant_keeps_credit``.

The probe shape is the reviewers': the probe sentence as a results span ``ev_a``, the species
cited to a plain abstract sentence ``ev_sp``, the paper typed primary, the paper's screen
organism A. thaliana.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.mining.admission import ADMITTED_STATUSES, judge  # noqa: E402
from t2pw.mining.claims import ClaimStore  # noqa: E402
from t2pw.mining.extraction import build_claims  # noqa: E402
from t2pw.mining.models import EvidenceSpan, PaperRecord  # noqa: E402
from t2pw.mining.normalize import EntityRegistry  # noqa: E402

TARGET = "arabidopsis thaliana"
TSS = "target_species_supported"
AT = ("Arabidopsis thaliana",)
PID = "PMC:PMCX"
SPECIES_SPAN = EvidenceSpan(evidence_id="ev_sp", paper_id=PID, section="abstract",
                            text="Jasmonate metabolism was studied in Arabidopsis thaliana.")


def _claim(text, *, sub="jasmonic acid", prod="JA-Ile", enzyme="", gene="", observed=AT):
    span = EvidenceSpan(evidence_id="ev_a", paper_id=PID, text=text, section="results")
    row = {"substrates": [{"name": sub, "evidence_ids": ["ev_a"]}],
           "products": [{"name": prod, "evidence_ids": ["ev_a"]}],
           "species": "Arabidopsis thaliana", "species_evidence_ids": ["ev_sp"], "evidence_ids": ["ev_a"]}
    if enzyme:
        row.update(enzyme=enzyme, enzyme_evidence_ids=["ev_a"])
    if gene:
        row.update(gene=gene, gene_evidence_ids=["ev_a"])
    claims, _ = build_claims({"claims": [row]}, spans=[span, SPECIES_SPAN], paper_id=PID,
                             target_species_normalized=TARGET, round_index=1, model="probe",
                             paper_observed_organisms=list(observed), paper_evidence_type="primary")
    return claims[0], {"ev_a": span, "ev_sp": SPECIES_SPAN}


def probe(text, **kw):
    """The reviewers' fixture: build_claims on the probe sentence; the claim it returns."""
    return _claim(text, **kw)[0]


def admitted(text, **kw):
    """The admission path: register the claim in a ClaimStore and judge it as the controller does."""
    claim, evidence = _claim(text, **kw)
    store = ClaimStore()
    stored, _ = store.register(claim)
    outcome = judge(stored, store=store, registry=EntityRegistry(),
                    papers={PID: PaperRecord(paper_id=PID, pmcid="PMCX", evidence_type="primary")},
                    evidence=evidence, round_index=1)
    return stored, outcome.decision


def _rows(*rows):
    return pytest.mark.parametrize("text,kw", [(t, k) for _, t, k in rows], ids=[n for n, _, _ in rows])


CYP = dict(sub="JA-Ile", prod="12-OH-JA-Ile", gene="CYP94B3")
CYP_REV = dict(sub="12-OH-JA-Ile", prod="JA-Ile", gene="CYP94B3")
JAR = dict(enzyme="JAR1")
JAR_REV = dict(sub="JA-Ile", prod="jasmonic acid", enzyme="JAR1")
JAP = dict(sub="jasmonic acid (JA)", prod="jasmonoyl-isoleucine (JA-Ile)", enzyme="JAR1")
JAP_REV = dict(sub="jasmonoyl-isoleucine (JA-Ile)", prod="jasmonic acid (JA)", enzyme="JAR1")
OPR = dict(sub="OPDA", prod="OPC-8:0", enzyme="OPR3")
OPR_REV = dict(sub="OPC-8:0", prod="OPDA", enzyme="OPR3")
AOS = dict(sub="13-HPOT", prod="12-OPDA", enzyme="AOS")
AOS_REV = dict(sub="12-OPDA", prod="13-HPOT", enzyme="AOS")
GH3 = dict(gene="GH3.10")
GLN = dict(prod="JA-Gln", gene="GH3.10")

P1 = "In cyp94b3 mutants, JA-Ile levels were elevated after wounding, with 12-OH-JA-Ile (Fig. 3A, B) strongly reduced."
E1 = "In cyp94b3 mutants, JA-Ile, which is the substrate of CYP94B3, accumulated after wounding, with reduced 12-OH-JA-Ile."
F1 = "JA-Ile accumulated in cyp94b3 mutants, which, after wounding, contained strongly reduced 12-OH-JA-Ile."
E2 = "In the jar1 mutant, JA, which is the substrate of JAR1, accumulated to high levels, with reduced JA-Ile."
P3 = "In the jar1 mutant, JA accumulated after wounding, with JA-Ile (Fig. 2A, B) strongly reduced."
E7 = ("In cyp94b3 mutants, JA-Ile, which is hydroxylated by CYP94B3, accumulated in wounded leaves, which contained "
      "reduced 12-OH-JA-Ile.")
T2 = "In cyp94b3 mutants, JA-Ile accumulated after wounding, which, in turn, resulted in reduced 12-OH-JA-Ile levels."
T3 = "In opr3 mutants, OPDA accumulated after wounding, which, consequently, led to reduced OPC-8:0 levels."
T4 = "In aos mutants, 13-HPOT accumulated, which, in turn, led to strongly reduced 12-OPDA levels."
T5 = "In jar1 mutants, jasmonic acid accumulated after wounding, which, in turn, resulted in reduced JA-Ile."
A4 = "In opr3 mutants, OPDA accumulated to high levels, which, in turn, reduced OPC-8:0 formation."


# ===========================================================================
# E1: an inner comma no longer hands the effect to the participant before the clause
# ===========================================================================
@_rows(
    # BIO-D-R2-1 (C:/t/bt/rev745d-r2-bio, c2 / c3 / e1 / e3); P1r, E1r, F1r, F4r, P2 and F2 are
    # also the review's six regression-fixture rows, verbatim
    ("bio-P1r", P1, CYP_REV),
    ("bio-E1r", E1, CYP_REV),
    ("bio-F1r", F1, CYP_REV),
    ("bio-E2r", E2, JAP_REV),
    ("bio-P3r", P3, JAP_REV),
    ("bio-F4r", "In the jar1 mutant, JA accumulated to high levels, with (3R,7S)-JA-Ile strongly reduced.", JAP_REV),
    ("bio-P2", "In opr3 mutants, OPC-8:0 levels were elevated, with OPDA (Fig. 4A, C) decreased.", OPR),
    ("bio-F2-code-A1", "JA-Gln accumulated at higher levels in gh3.10 jar1 knockout lines, which, after wounding, had "
                       "reduced JA-Ile.", GLN),
    ("bio-E9", "In opr3 mutants, OPC-8:0, which is formed from OPDA, accumulated, with reduced JA.", OPR),
    ("bio-F3-rt-T8-code-A2", "In opr3 mutants, OPC-8:0 levels were elevated, with, after wounding, decreased OPDA.", OPR),
    # the same class in the review's e1 artifact (runs/e1_tip): admitted at 686ae134 too
    ("bio-E7r", E7, CYP_REV),
    ("bio-E8r", "In cyp94b3 mutants, JA-Ile, which is the substrate of CYP94B3, accumulated, although its hydroxylated "
                "product was strongly reduced.", CYP_REV),
    ("bio-F5r", "In cyp94b3 mutants, JA-Ile accumulated after wounding, which, in turn, reduced 12-OH-JA-Ile.", CYP_REV),
)
def test_cs_e_746_e1_bio_d_r2_1_an_inner_comma_does_not_hand_the_effect_to_the_head(text, kw):
    assert probe(text, **kw).species_state != TSS


@_rows(
    # RT-D-R2-1 (C:/t/bt/rev745d-r2-redteam, r2probe / r2probe2 / e2e_r2 / e2e_r2b); T1, T2r, W3r,
    # W6r and T7 are also the review's five regression-fixture rows, verbatim
    ("rt-T1", "JA-Gln accumulated at higher levels in gh3.10 jar1 knockout lines, which, in turn, had reduced JA-Ile.",
     GLN),
    ("rt-T1b", "JA-Gln accumulated at higher levels in gh3.10 jar1 knockout lines, which, upon wounding, had reduced "
               "JA-Ile.", GLN),
    ("rt-T2r", T2, CYP_REV),
    ("rt-T3r", T3, OPR_REV),
    ("rt-T4r", T4, AOS_REV),
    ("rt-T5r", T5, JAR_REV),
    ("rt-T6", "In opr3 mutants, OPC-8:0 accumulated after wounding, which, in turn, resulted in reduced OPDA levels.", OPR),
    ("rt-T7", "In opr3 mutants, OPC-8:0 accumulated, with OPDA, however, strongly decreased.", OPR),
    ("rt-T9", "In jar1 mutants, JA-Ile accumulated after wounding, which, in turn, resulted in reduced jasmonic acid.",
     JAR),
    ("rt-W1r", "In cyp94b3 mutants, JA-Ile accumulated after wounding, with 12-OH-JA-Ile, its hydroxylated derivative, "
               "strongly reduced.", CYP_REV),
    ("rt-W2r", "In cyp94b3 mutants, JA-Ile accumulated after wounding, with 12-OH-JA-Ile (Fig. 3A, B) strongly reduced.",
     CYP_REV),
    ("rt-W3r", "In opr3 mutants, OPDA accumulated to high levels, with OPC-8:0, the reduction product, being "
               "undetectable.", OPR_REV),
    ("rt-W4r", "In opr3 mutants, OPDA accumulated after wounding, although OPC-8:0, its reduction product, was "
               "undetectable.", OPR_REV),
    ("rt-W5r", "In jar1 mutants, jasmonic acid accumulated after wounding, with JA-Ile, the bioactive conjugate, "
               "strongly reduced.", JAR_REV),
    ("rt-W6r", "In aos mutants, 13-HPOT accumulated after wounding, with 12-OPDA (Fig. 2B, C) being undetectable.",
     AOS_REV),
    ("rt-W7r", "In jar1 mutants, jasmonic acid accumulated, with JA-Ile (P < 0.01, Student t-test) reduced by 80%.",
     JAR_REV),
    ("rt-W8", "In jar1 mutants, JA-Ile accumulated after wounding, with jasmonic acid, its precursor, strongly reduced.",
     JAR),
    ("rt-W9", "In opr3 mutants, OPC-8:0 accumulated, with OPDA, its precursor, being undetectable.", OPR),
    ("rt-H4r", "In opr3 mutants, the OPDA reduction product, which is OPC-8:0, was undetectable.", OPR_REV),
)
def test_cs_e_746_e1_rt_d_r2_1_an_inner_comma_does_not_close_the_trailing_clause(text, kw):
    assert probe(text, **kw).species_state != TSS


@_rows(
    # CSD-R2-N1 (C:/t/bt/rev745d-r2-code/scripts/pm_r2.py; A1 and A2 are bio-F2 and bio-F3 above)
    ("code-A4", A4, OPR_REV),
    ("code-A5", "In gh3.10 single mutants, JA-Ile was readily detected in wounded leaves, which, however, had reduced "
                "JA-Val levels.", GH3),
    ("code-B1", "Despite high OPC-8:0 levels in opr3 mutants, which are male sterile, JA was reduced.", OPR),
)
def test_cs_e_746_e1_csd_r2_n1_a_comma_inside_the_clause_does_not_close_it(text, kw):
    assert probe(text, **kw).species_state != TSS


def test_cs_e_746_e1_p1r_reversed_cyp94b3_node_is_not_admitted():
    """End to end on the admission path: at 686ae134 the reversed 12-OH-JA-Ile -> JA-Ile node was
    accepted, in_planta / single_paper."""
    claim, decision = admitted(P1, **CYP_REV)
    assert claim.species_state != TSS
    assert decision.status not in ADMITTED_STATUSES


# ===========================================================================
# E2: the negated-comparison / delay window crosses an abbreviation's full stop
# ===========================================================================
@_rows(
    # BIO-D-R2-2 (the review's regression fixture, lines 211-213)
    ("bio-CF1", "The opr3 mutant did not accumulate OPC-8:0 (Fig. 3A) until 6 h after wounding.", OPR),
    ("bio-CF2", "JA-Ile did not accumulate in gh3.10 single mutants (Fig. 2B) to lower levels than in Col-0.", GH3),
    ("bio-CF4", "gh3.10 mutants did not accumulate JA-Ile (ca. 40 pmol/g) to lower levels than Col-0 after wounding.",
     GH3),
    # CSD-R2-N2 (pm_r2.py CF1-CF4)
    ("code-CF1", "JA-Ile did not accumulate in gh3.10 mutants (Fig. 4B) until 6 h after wounding.", GH3),
    ("code-CF2", "The opr3 mutant did not accumulate OPC-8:0 (Fig. 2) until 6 h after wounding.", OPR),
    ("code-CF3", "opr3 mutants did not accumulate OPC-8:0 (Fig. 3A) to lower levels than the wild type.", OPR),
    ("code-CF4", "jar1 mutants did not accumulate JA-Ile (i.e. the bioactive conjugate) until 6 h after wounding.", JAR),
)
def test_cs_e_746_e2_a_negated_delay_or_comparison_past_an_abbreviation_earns_no_credit(text, kw):
    assert probe(text, **kw).species_state != TSS


def test_cs_e_746_e2_cf1_negated_delay_node_is_not_admitted():
    """End to end on the admission path: at 686ae134 the OPDA -> OPC-8:0 node was accepted on
    "did not accumulate OPC-8:0 (Fig. 3A) until 6 h", in_planta / single_paper."""
    claim, decision = admitted("The opr3 mutant did not accumulate OPC-8:0 (Fig. 3A) until 6 h after wounding.", **OPR)
    assert claim.species_state != TSS
    assert decision.status not in ADMITTED_STATUSES


# ===========================================================================
# Controls: hold at 686ae134 and here
# ===========================================================================
@_rows(
    ("no-JA-Ile-was-detected", "No JA-Ile was detected in gh3.10 jar1 knockout lines.", GH3),
    ("20-fold-lower-than", "JA-Ile was 20-fold lower in jar1 mutants than in wild-type leaves.", JAR),
    # BIO-D-R2-1's control: the P1 sentence with a comma-free callout, correct direction
    ("bio-P4c", "In cyp94b3 mutants, JA-Ile levels were elevated after wounding, with 12-OH-JA-Ile (Fig. 3A) strongly "
                "reduced.", CYP),
)
def test_new_acceptance_cs_e_746_controls_keep_in_planta_credit(text, kw):
    claim, decision = admitted(text, **kw)
    assert (claim.species_state, claim.species_support["grade"]) == (TSS, "in_planta")
    assert decision.status in ADMITTED_STATUSES


def test_new_acceptance_cs_e_746_cf5_a_callout_without_a_full_stop_stays_refused():
    """CF5: "(Figure 3A)" has no full stop, so the window always reached "until"."""
    claim = probe("The opr3 mutant did not accumulate OPC-8:0 (Figure 3A) until 6 h after wounding.", **OPR)
    assert claim.species_state != TSS


@_rows(
    # Each is refused at 686ae134 as well: CS-E may only subtract credit. The prototypes credit
    # them (bio = rev745d_r2_proto, rt = rt745dr2_fix), so each is a recall cost of requiring all
    # three readings to agree.
    ("bio-P1c", P1, CYP),  # not credited at 686ae134; bio credits
    ("bio-E1c", E1, CYP),  # not credited at 686ae134; bio credits
    ("bio-F1c", F1, CYP),  # not credited at 686ae134; bio and rt credit
    ("bio-P3c", P3, JAP),  # not credited at 686ae134; bio credits
    ("bio-E2c", E2, JAP),  # not credited at 686ae134; bio credits
    ("bio-E7c", E7, CYP),  # not credited at 686ae134; bio credits
    ("rt-T2c", T2, CYP),  # not credited at 686ae134; bio and rt credit
    ("rt-T3c", T3, OPR),  # not credited at 686ae134; bio and rt credit
    ("rt-T4c", T4, AOS),  # not credited at 686ae134; bio and rt credit
    ("rt-T5c", T5, JAR),  # not credited at 686ae134; bio and rt credit
    ("code-A4c", A4, OPR),  # not credited at 686ae134; bio and rt credit
)
def test_new_acceptance_cs_e_746_recall_cost_correct_direction_partners_earn_nothing(text, kw):
    assert probe(text, **kw).species_state != TSS


# ###########################################################################
# ORCH-746 CS-E repair round 1 (G9 at 5eca087b). Every ``test_cs_e_r1_746_*`` row is accepted
# at 5eca087b (and at 686ae134) and fails there with an AssertionError; every
# ``test_new_acceptance_cs_e_r1_746_*`` row holds at both ends. Each change of the round only
# removes credit: a reading appended to the all-must-agree ``_READINGS``, a ``_NO_EFFECT`` veto,
# or a narrowing of a credit form or of the marker tail.
# ###########################################################################
AOC2 = dict(sub="12,13-EOT", prod="OPDA", enzyme="AOC2")
LOX = dict(sub="alpha-linolenic acid", prod="13-HPOT", enzyme="LOX2")
B1 = dict(sub="JA-Ile", prod="12-OH-JA-Ile", gene="CYP94B1")
S8 = "Leaves of cyp94b3 mutants hyperaccumulated JA-Ile, with 12-OH-JA-Ile, the major catabolite, reduced by 90%."
D21 = "In wounded leaves of jar1 mutants, JA-Ile was not detected until 30 min after wounding."
S11 = ("In the gh3.10 single mutant, JA-Ile levels were slightly lower than in Col-0 after wounding, but the "
       "difference was not statistically significant.")
T1 = "In flowers and wounded leaves of gh3.10 jar1 knockout lines JA-Ile was below the detection limit."


# ---------------------------------------------------------------------------
# 1. CSE-R0-1: two or more ", " after the opener do not close the clause (the fourth reading)
# ---------------------------------------------------------------------------
@_rows(
    # the code reviewer's adjacent.py (C:/t/bt/rev746e2-code/scripts), reversed claims
    ("code-S1r", "In cyp94b3 mutants, wounding led to a strong accumulation of JA-Ile, with 12-OH-JA-Ile, its hydroxylated "
                 "derivative, strongly reduced.", CYP_REV),
    ("code-S2r", "In opr3 mutants, wounding led to the accumulation of OPDA, with OPC-8:0, the reduction product, being "
                 "undetectable.", OPR_REV),
    ("code-S3r", "In jar1 mutants, wounding triggered the accumulation of jasmonic acid, with JA-Ile, the bioactive "
                 "conjugate, strongly reduced.", JAR_REV),
    ("code-S4r", "In aos mutants, wounding led to the accumulation of 13-HPOT, with 12-OPDA, the cyclization product, being "
                 "undetectable.", AOS_REV),
    ("code-S5r", "In cyp94b3 mutants, wounding led to a strong accumulation of JA-Ile, which, in turn, resulted in reduced "
                 "12-OH-JA-Ile levels.", CYP_REV),
    ("code-S6r", "In opr3 mutants, wounding led to the accumulation of OPDA, which, consequently, resulted in reduced "
                 "OPC-8:0 levels.", OPR_REV),
    ("code-S7r", "In jar1 mutants, wounding led to the accumulation of jasmonic acid, which, in turn, resulted in reduced "
                 "JA-Ile levels.", JAR_REV),
    ("code-S8r", S8, CYP_REV),
)
def test_cs_e_r1_746_1_cse_r0_1_an_appositive_or_inner_comma_pair_does_not_close_the_clause(text, kw):
    assert probe(text, **kw).species_state != TSS


def test_cs_e_r1_746_1_s8r_reversed_cyp94b3_node_is_not_admitted():
    """End to end: at 5eca087b the reversed 12-OH-JA-Ile -> JA-Ile node was accepted on S8."""
    claim, decision = admitted(S8, **CYP_REV)
    assert claim.species_state != TSS
    assert decision.status not in ADMITTED_STATUSES


# ---------------------------------------------------------------------------
# 2. RT-E2-1: a negated delay written with any absence form is not an absence
# ---------------------------------------------------------------------------
@_rows(
    # the red team's hunt2.py; D21, D22 and D24 (with hunt1 D8) are the review's regression fixture
    ("rt-D21", D21, JAR),
    ("rt-D22", "In the opr3 mutant, OPC-8:0 was below the detection limit until 2 h after wounding.", OPR),
    ("rt-D23", "12-OH-JA-Ile was undetectable in cyp94b3 mutant leaves until 1 h after wounding.", CYP),
    ("rt-D24", "jar1 mutants did not accumulate JA-Ile before 1 h after wounding.", JAR),
    ("rt-D25", "In lox2 mutants, 13-HPOT was not detected until 2 h after wounding.", LOX),
    ("rt-D26", "In gh3.10 mutant flowers, JA-Ile was absent until stage 13.", GH3),
    # hunt1.py, every D row accepted at 5eca087b
    ("rt-D1", "JA-Ile was not detected in jar1 mutants until 2 h after wounding.", JAR),
    ("rt-D2", "In opr3 mutants, OPC-8:0 was undetectable until 6 h after wounding.", OPR),
    ("rt-D3", "No 12-OH-JA-Ile was detected in cyp94b3 mutants until 1 h after wounding.", CYP),
    ("rt-D4b", "In cyp94b3 mutants, 12-OH-JA-Ile was below the detection limit until 2 h after wounding.", CYP),
    ("rt-D5", "jar1 mutants did not accumulate JA-Ile before 2 h after wounding.", JAR),
    ("rt-D6", "The opr3 mutant did not accumulate OPC-8:0 prior to 6 h after wounding.", OPR),
    ("rt-D7", "In gh3.10 jar1 knockout lines, JA-Ile was not detected before 2 h after wounding.", GH3),
    ("rt-D8", "In gh3.10 jar1 knockout lines, JA-Ile was not detected until 2 h after wounding.", GH3),
    ("rt-D10", "In jar1 mutant leaves, JA-Ile was absent until 1 h after wounding.", JAR),
    ("rt-D11", "In cyp94b3 mutants, 12-OH-JA-Ile was no longer detectable until 4 h.", CYP),
    ("rt-D12", "JA-Ile was not detectable in jar1 mutants before 30 min.", JAR),
    ("rt-D14", "JA-Ile formation was delayed in jar1 mutants, where it was not detected until 2 h.", JAR),
    # the bio-auditor's semicolon-in-a-reference delays (semi.py S5, resid.py N8): "until" anywhere
    ("bio-S5", "The aos mutant failed to accumulate 12-OPDA (Fig. 1C; n = 4) until 3 h after wounding.", AOS),
    ("bio-N8", "The opr3 mutant did not accumulate OPC-8:0 (Fig. 3A; ca. 2 h) until 6 h after wounding.", OPR),
)
def test_cs_e_r1_746_2_rt_e2_1_a_negated_delay_is_not_an_absence(text, kw):
    assert probe(text, **kw).species_state != TSS


def test_cs_e_r1_746_2_d21_negated_delay_node_is_not_admitted():
    """End to end: at 5eca087b "JA-Ile was not detected until 30 min" admitted jasmonic acid -> JA-Ile."""
    claim, decision = admitted(D21, **JAR)
    assert claim.species_state != TSS
    assert decision.status not in ADMITTED_STATUSES


# ---------------------------------------------------------------------------
# 3. RT-E2-2: a non-significant change, or the genotype joined to the wild type, is no effect
# ---------------------------------------------------------------------------
@_rows(
    # hunt2.py; S11, S12, S16 and N11 (= hunt1 N1) are the review's regression fixture
    ("rt-S11", S11, GH3),
    ("rt-S12", "OPDA levels were reduced in aoc2 mutants, although this reduction was not significant.", AOC2),
    ("rt-S13", "12-OH-JA-Ile levels were lower in cyp94b1 mutant leaves (n.s.).", B1),
    ("rt-S14", "In lox2 mutant roots, 13-HPOT levels were reduced, but this decrease was not significant.", LOX),
    ("rt-S16", "OPDA accumulated to higher levels in opr3 mutants than in Col-0 (P = 0.08).", OPR),
    ("rt-S17", "JA-Ile was reduced in jar1 mutant flowers after wounding (P > 0.05, Student t-test).", JAR),
    ("rt-N11-hunt1-N1", "In unwounded leaves, JA-Ile was below the detection limit in jar1 mutant and wild-type plants.",
     JAR),
    ("rt-N12", "Before wounding, 12-OPDA was undetectable in aos mutant and Col-0 leaves.", AOS),
    ("rt-N13", "JA-Ile was not detected in control jar1 mutant and wild-type seedlings.", JAR),
    ("rt-N14", "Basal JA-Ile levels were below the detection limit in jar1 mutants and Col-0.", JAR),
    ("rt-N15", "Under mock conditions, OPC-8:0 was not detected in opr3 mutants or in the wild type.", OPR),
    # hunt1.py, every S / N row accepted at 5eca087b that the vetoes reach (S10 "within experimental
    # error" is not reached by these vetoes; repair 1 item 10 refuses it, test_cs_e_r1_746_10_*)
    ("rt-S1", "JA-Ile levels were reduced by 20% in jar1 mutants, but the difference was not statistically "
              "significant.", JAR),
    ("rt-S2", "JA-Ile was slightly reduced in jar1 mutants, although the difference was not significant.", JAR),
    ("rt-S3", "In jar1 mutants, JA-Ile levels were reduced (n.s.).", JAR),
    ("rt-S4", "In jar1 mutants, JA-Ile levels were reduced (P > 0.05).", JAR),
    ("rt-S5", "jar1 mutants showed a non-significant reduction, with lower JA-Ile levels after wounding.", JAR),
    ("rt-S6", "In jar1 mutants, JA-Ile was reduced only insignificantly after wounding.", JAR),
    ("rt-S7", "JA-Ile levels tended to be lower in jar1 mutants after wounding.", JAR),
    ("rt-S8", "OPC-8:0 levels were lower in opr3 mutants, but this decrease was statistically insignificant.", OPR),
    ("rt-S9", "OPC-8:0 levels were reduced in opr3 mutants and not statistically different from Col-0.", OPR),
    ("rt-N2", "OPC-8:0 was undetectable in unwounded leaves of opr3 mutant and Col-0 plants.", OPR),
    ("rt-N3", "JA-Ile was not detected in unwounded jar1 mutants or wild-type plants.", JAR),
    ("rt-N5", "Basal JA-Ile was undetectable in jar1 mutants and in wild-type seedlings.", JAR),
    ("rt-N6", "In unwounded leaves, OPC-8:0 was undetectable in opr3 mutants and Col-0 alike.", OPR),
    ("rt-N7", "After 2 h, JA-Ile was reduced in jar1 mutants and in wild-type plants.", JAR),
    ("rt-N8", "In unwounded rosettes, 12-OPDA was undetectable in aos mutants and also in Col-0.", AOS),
)
def test_cs_e_r1_746_3_rt_e2_2_a_non_significant_or_shared_result_is_no_effect(text, kw):
    assert probe(text, **kw).species_state != TSS


def test_cs_e_r1_746_3_s11_non_significant_node_is_not_admitted():
    """End to end: at 5eca087b the GH3.10 single-mutant non-significant result admitted jasmonic acid -> JA-Ile."""
    claim, decision = admitted(S11, **GH3)
    assert claim.species_state != TSS
    assert decision.status not in ADMITTED_STATUSES


# ---------------------------------------------------------------------------
# 4. RT-E2-4: a concessive clause (although / though / despite / in spite of)
# ---------------------------------------------------------------------------
@_rows(
    ("rt-C1-I4-code-B1c", "Despite high OPC-8:0 levels in opr3 mutants, JA was reduced.", OPR),
    ("rt-C3", "Although OPC-8:0 accumulated in opr3 mutants, JA levels were strongly reduced.", OPR),
    ("rt-C4", "Although JA-Ile was present in jar1 mutants, JA-Ile-dependent responses were reduced.", JAR),
    ("rt-C7", "Even though jar1 mutants accumulated JA-Ile, JA-responsive gene induction was reduced.", JAR),
    ("rt-C8", "Although opr3 mutants produced OPC-8:0 after infection, JA was reduced.", OPR),
)
def test_cs_e_r1_746_4_rt_e2_4_a_concessive_clause_earns_nothing(text, kw):
    assert probe(text, **kw).species_state != TSS


# ---------------------------------------------------------------------------
# 5. RT-E2-3: a participant inside the "accumulated ... less" gap is not the decreased one
# ---------------------------------------------------------------------------
@_rows(
    ("rt-R11-hunt1-R4", "opr3 mutants accumulated OPDA with lower OPC-8:0 levels.", OPR_REV),
    # the implementer's: the same shape on JAR1 ("ja" is a name variant of "jasmonic acid (JA)")
    ("m746e-R14", "jar1 mutants accumulated JA with lower JA-Ile levels.", JAP_REV),
)
def test_cs_e_r1_746_5_rt_e2_3_a_participant_in_the_accumulated_less_gap_earns_nothing(text, kw):
    assert probe(text, **kw).species_state != TSS


# ---------------------------------------------------------------------------
# 6. CSE-R0-N1: "with" written without its comma opens a clause (the fifth reading)
# ---------------------------------------------------------------------------
@_rows(
    ("m746e-X3r-rt-I1-bio-I1r", "In cyp94b3 mutants, JA-Ile accumulated after wounding with reduced 12-OH-JA-Ile.",
     CYP_REV),
    ("m746e-X5r", "In cyp94b3 mutants, JA-Ile, which is the substrate of CYP94B3 (Fig. 3A, B), accumulated after "
                  "wounding with reduced 12-OH-JA-Ile.", CYP_REV),
    ("m746e-X4-rt-I2-bio-I2", "In opr3 mutants, OPC-8:0 levels were elevated with decreased OPDA.", OPR),
    # the brief lists R12 and R13 under item 5, but no participant starts inside an "accumulated ...
    # less" gap there ("ja-ile" is hyphenated, "jasmonic acid" is two words before "with"): the
    # bare-"with" reading is what refuses them
    ("rt-R12", "cyp94b3 mutants accumulated JA-Ile with reduced 12-OH-JA-Ile levels.", CYP_REV),
    ("rt-R13", "jar1 mutants accumulated jasmonic acid with reduced JA-Ile.", JAR_REV),
    ("rt-hunt1-R2", "cyp94b3 mutant leaves accumulated JA-Ile with strongly reduced levels of 12-OH-JA-Ile.", CYP_REV),
    ("bio-V7r", "In cyp94b3 mutants JA-Ile accumulated to higher levels with reduced 12-OH-JA-Ile (Fig. 3).", CYP_REV),
    ("bio-V9r", "The aos mutant accumulated 13-HPOT with no detectable 12-OPDA.", AOS_REV),
)
def test_cs_e_r1_746_6_cse_r0_n1_with_without_its_comma_opens_a_clause(text, kw):
    assert probe(text, **kw).species_state != TSS


# ---------------------------------------------------------------------------
# 7. CSE-R0-N5 / 8. BIO-D-R2-3 / RT-D-R2-3: "nowhere" in the gap; a regulator's mutant
# ---------------------------------------------------------------------------
def test_cs_e_r1_746_7_cse_r0_n5_accumulated_nowhere_near_less_is_not_a_decrease():
    assert probe("jar1 mutants accumulated nowhere near less JA-Ile after wounding.", **JAR).species_state != TSS


@_rows(
    ("bio-K8-rt-M1", "Mutants lacking a regulator of AOS accumulated no 12-OPDA.", AOS),
    ("bio-K9-rt-M2", "Mutants with defects upstream of JAR1 accumulated no JA-Ile.", JAR),
    ("rt-M3", "Mutants for regulators of JAR1 accumulated no JA-Ile.", JAR),
)
def test_cs_e_r1_746_8_a_regulators_mutant_is_not_the_claims_genotype(text, kw):
    assert probe(text, **kw).species_state != TSS


# ---------------------------------------------------------------------------
# Controls: hold at 5eca087b and here
# ---------------------------------------------------------------------------
@_rows(
    ("T1", T1, GH3),
    ("aos-accumulated-less", "aos mutants accumulated less 12-OPDA after wounding.", AOS),
    ("accumulated-much-less", "jar1 mutants accumulated much less JA-Ile after wounding.", JAR),
    ("SALK-insertion-line-of", "The SALK_034543 insertion line of GH3.10 accumulated no JA-Ile.", GH3),
    ("T-DNA-insertion-lines-of-AOS", "T-DNA insertion lines of AOS were deficient in 12-OPDA.", AOS),
    ("deficient-in-GH3.10", "Mutants deficient in GH3.10 accumulated no JA-Ile in flowers.", dict(enzyme="GH3.10")),
    ("knockout-of-GH3.10", "Knockout of GH3.10 abolished JA-Ile accumulation in flowers.", GH3),
    # the head right before a relative clause with one ", " after its opener keeps its credit
    ("rt-R5-which-is-formed-from", "In opr3 mutants, OPC-8:0, which is formed from OPDA, was undetectable.", OPR),
)
def test_new_acceptance_cs_e_r1_746_controls_keep_in_planta_credit(text, kw):
    claim, decision = admitted(text, **kw)
    assert (claim.species_state, claim.species_support["grade"]) == (TSS, "in_planta")
    assert decision.status in ADMITTED_STATUSES


# ---------------------------------------------------------------------------
# 9. (addendum) a participle or consequence clause opens a clause (the sixth reading)
# ---------------------------------------------------------------------------
RESULTING = "In cyp94b3 mutants, JA-Ile accumulated after wounding, resulting in reduced 12-OH-JA-Ile levels."


@_rows(
    # the implementer's runs/suspect_r1 rows, reversed claims, and the contrary "yet" row
    ("m746e-resulting-in", RESULTING, CYP_REV),
    ("m746e-leading-to", "In opr3 mutants, OPDA accumulated, leading to strongly reduced OPC-8:0 levels.", OPR_REV),
    ("m746e-accompanied-by", "In cyp94b3 mutants, JA-Ile accumulated after wounding, accompanied by reduced "
                             "12-OH-JA-Ile levels.", CYP_REV),
    ("m746e-yet-contrary", "In jar1 mutants, JA-Ile was formed, yet 12-OH-JA-Ile was reduced.", JAR),
)
def test_cs_e_r1_746_9_a_participle_or_consequence_clause_opens_a_clause(text, kw):
    assert probe(text, **kw).species_state != TSS


def test_cs_e_r1_746_9_resulting_in_reversed_node_is_not_admitted():
    """End to end: at 5eca087b the reversed 12-OH-JA-Ile -> JA-Ile node was accepted on ', resulting in'."""
    claim, decision = admitted(RESULTING, **CYP_REV)
    assert claim.species_state != TSS
    assert decision.status not in ADMITTED_STATUSES


# ---------------------------------------------------------------------------
# 10. (addendum) "within experimental error" is no effect
# ---------------------------------------------------------------------------
def test_cs_e_r1_746_10_a_change_within_experimental_error_is_no_effect():
    """The red team's hunt1 S10."""
    claim = probe("In jar1 mutants, JA-Ile was reduced, but within experimental error.", **JAR)
    assert claim.species_state != TSS


# ###########################################################################
# ORCH-746 CS-E repair round 2 (G9 at 01d436cf): the delay and no-difference classes closed by
# SHAPE, the union of the three round-1 prototypes, plus the round-1 non-blocking residuals.
# Every ``test_cs_e_r2_746_*`` row is accepted at 01d436cf and fails there with an
# AssertionError; every ``test_new_acceptance_cs_e_r2_746_*`` row holds at both ends.
# ###########################################################################
DL11 = "In gh3.10 mutant flowers, JA-Ile was absent at stage 12 but present at stage 13."
NS1 = ("In the gh3.10 single mutant, JA-Ile levels were slightly lower than in Col-0, but the difference did not "
       "reach statistical significance.")
WT3 = "In unwounded leaves, JA-Ile was not detected in jar1 mutants and control plants."


# ---------------------------------------------------------------------------
# 1. A time course (RT-R1-1, CSE-R1-C3, BIO-E746R1-2)
# ---------------------------------------------------------------------------
@_rows(
    # the code reviewer's probes_r1.py
    ("code-DL1", "In jar1 mutants, JA-Ile was not detected during the first hour after wounding.", JAR),
    ("code-DL2", "In wounded jar1 mutant leaves, JA-Ile was not detected at early time points.", JAR),
    ("code-DL3", "In jar1 mutants, JA-Ile was not detected at 15 min but accumulated at 2 h after wounding.", JAR),
    ("code-DL4", "In opr3 mutants, OPC-8:0 was undetectable within the first 30 min after wounding.", OPR),
    ("code-DL7-rt-DL3", "In jar1 mutants, JA-Ile was not detected till 2 h after wounding.", JAR),
    ("code-DL8", "In jar1 mutants, JA-Ile was not detected at 30 min, but was detected at 2 h after wounding.", JAR),
    # the red team's hunt3.py
    ("rt-DL1", "In jar1 mutants, JA-Ile was not detected at 5 min after wounding but accumulated at later time "
               "points.", JAR),
    ("rt-DL2", "In wounded jar1 mutant leaves, JA-Ile was below the detection limit for the first hour.", JAR),
    ("rt-DL4", "JA-Ile was not detected in jar1 mutants before the 2-h time point.", JAR),
    ("rt-DL5", "JA-Ile was not detectable in jar1 mutants earlier than 2 h after wounding.", JAR),
    ("rt-DL7", "In jar1 mutants, JA-Ile was undetectable within the first 30 min and was detected only after 2 h.",
     JAR),
    ("rt-DL8", "In opr3 mutants, OPC-8:0 was not detected 30 min after wounding; it was detected at 6 h.", OPR),
    ("rt-DL9", "In jar1 mutants, JA-Ile was not detected before one hour after wounding.", JAR),
    ("rt-DL10", "JA-Ile accumulation was delayed in jar1 mutants, in which it was undetectable at 15 min.", JAR),
    ("rt-DL11", DL11, GH3),
    ("rt-DL12", "In jar1 mutant leaves, JA-Ile was undetectable during a lag of about 1 h after wounding.", JAR),
    # the bio-auditor's bat.py (Z) and bat3.py (Y)
    ("bio-Z9", "In jar1 mutants, JA-Ile was not detected 15 min after wounding but accumulated at later time points.",
     JAR),
    ("bio-Z10", "JA-Ile was undetectable in jar1 mutants 15 min after wounding, but reached 50 percent of the Col-0 "
                "level by 2 h.", JAR),
    ("bio-Z11", "JA-Ile was below the detection limit in jar1 mutants at 30 min and became detectable only at 2 h "
                "after wounding.", JAR),
    ("bio-Z12", "In jar1 mutants, JA-Ile accumulation was reduced at early time points and delayed.", JAR),
    ("bio-Z13", "In lox2 mutants, 13-HPOT was not detected at the earliest time points after wounding.", LOX),
    ("bio-Z14", "In wounded leaves of jar1 mutants, JA-Ile was not detected during the first 30 min.", JAR),
    ("bio-Y9", "OPDA was undetectable in aoc2 mutants at 15 min, and recovered to Col-0 levels at 1 h.", AOC2),
    ("bio-Y11", "OPDA was reduced in aoc2 mutants at 15 min only.", AOC2),
    ("bio-Y12", "OPDA was undetectable in aoc2 mutants at 15 min but not at 1 h after wounding.", AOC2),
)
def test_cs_e_r2_746_1_a_time_course_is_not_an_absence(text, kw):
    assert probe(text, **kw).species_state != TSS


def test_cs_e_r2_746_1_dl11_stage_time_course_node_is_not_admitted():
    """End to end: at 01d436cf "absent at stage 12 but present at stage 13" admitted jasmonic acid -> JA-Ile."""
    claim, decision = admitted(DL11, **GH3)
    assert claim.species_state != TSS
    assert decision.status not in ADMITTED_STATUSES


# ---------------------------------------------------------------------------
# 2. Any statement of significance, a trend or variability (RT-R1-2 (i), CSE-R1-C1, BIO-E746R1-1)
# ---------------------------------------------------------------------------
@_rows(
    ("code-NS1", NS1, GH3),
    ("code-NS2", "JA-Ile levels were reduced in jar1 mutants, but there was no statistically significant difference.",
     JAR),
    ("code-NS3", "In aoc2 mutants, OPDA levels were reduced, but this failed to reach significance.", AOC2),
    ("code-NS6", "In gh3.10 single mutants, JA-Ile levels were marginally reduced and the reduction lacked "
                 "statistical support.", GH3),
    ("rt-NS1", "JA-Ile levels were reduced in gh3.10 single mutants after wounding, but the reduction did not reach "
               "statistical significance.", GH3),
    ("rt-NS3", "JA-Ile levels were lower in gh3.10 mutants, but the difference lacked statistical significance.", GH3),
    ("rt-NS4", "The gh3.10 single mutant showed a trend towards lower JA-Ile levels after wounding.", GH3),
    ("rt-NS5", "cyp94b1 mutants showed a tendency towards reduced 12-OH-JA-Ile levels.", B1),
    ("rt-NS6", "In aoc2 mutants, OPDA levels were slightly reduced (ns).", AOC2),
    ("rt-NS7", "OPDA levels were reduced in aoc2 mutants (p-value > 0.05).", AOC2),
    ("rt-NS8", "JA-Ile was reduced in gh3.10 mutant flowers, with a P value of 0.12.", GH3),
    ("rt-NS10", "In lox2 mutant roots, 13-HPOT levels were reduced (n.s).", LOX),
    ("rt-NS11", "JA-Ile levels were reduced in gh3.10 mutants, with no statistically significant difference from "
                "Col-0.", GH3),
    ("rt-NS14", "JA-Ile levels were lower in gh3.10 mutants but stayed within the range of Col-0.", GH3),
    ("bio-Z1", "In gh3.10 single mutants, JA-Ile levels were reduced by 30 percent, but the difference did not reach "
               "statistical significance.", GH3),
    ("bio-Z2", "JA-Ile levels were lower in jar1 mutant leaves after wounding, but this did not reach significance.",
     JAR),
    ("bio-Z3", "JA-Ile was reduced in gh3.10 single mutants, but not to a statistically significant extent.", GH3),
    ("bio-Z4", "JA-Ile levels were reduced in gh3.10 single mutants without reaching statistical significance.", GH3),
    ("bio-Z5", "JA-Ile levels were slightly lower in the gh3.10 single mutant, but the differences were within the "
               "range of biological variation.", GH3),
    ("bio-Y1", "JA-Ile levels showed a trend towards lower values in gh3.10 single mutants after wounding.", GH3),
    ("bio-Y2", "JA-Ile was reduced in gh3.10 single mutants in one of three independent experiments.", GH3),
    ("bio-Y3", "In gh3.10 single mutants, JA-Ile was reduced, but the effect was weak and variable.", GH3),
    ("bio-Y4", "OPDA levels were lower in aoc2 mutants, but there was no statistically significant difference.", AOC2),
    ("bio-Y5", "OPDA levels were reduced in aoc2 mutants but failed to reach statistical significance.", AOC2),
)
def test_cs_e_r2_746_2_any_statement_of_significance_or_trend_earns_nothing(text, kw):
    assert probe(text, **kw).species_state != TSS


def test_cs_e_r2_746_2_ns1_non_significant_node_is_not_admitted():
    """End to end: at 01d436cf "did not reach statistical significance" admitted jasmonic acid -> JA-Ile [GH3.10]."""
    claim, decision = admitted(NS1, **GH3)
    assert claim.species_state != TSS
    assert decision.status not in ADMITTED_STATUSES


# ---------------------------------------------------------------------------
# 3. A result shared with a control or with every genotype (RT-R1-2 (ii), CSE-R1-C2, BIO-E746R1-1)
# ---------------------------------------------------------------------------
@_rows(
    ("code-WT3", "In unwounded leaves, JA-Ile was below the detection limit in jar1 mutant leaves and in wild-type "
                 "leaves.", JAR),
    ("code-WT5", "OPC-8:0 was not detected in unwounded opr3 mutant seedlings or Col-0 seedlings.", OPR),
    ("code-WT8", "In unwounded leaves of Col-0 and jar1 mutants, JA-Ile was below the detection limit.", JAR),
    ("code-WT9", "In unwounded leaves of wild-type and jar1 mutant plants, JA-Ile was below the detection limit.", JAR),
    ("rt-WT1", "In unwounded leaves, JA-Ile was below the detection limit in jar1 mutants and their wild-type "
               "siblings.", JAR),
    ("rt-WT2", "Before wounding, OPC-8:0 was undetectable in the opr3 mutant and its wild type.", OPR),
    ("rt-WT3", WT3, JAR),
    ("rt-WT4", "In unwounded leaves, OPC-8:0 was below the detection limit in opr3 mutant and Ws plants.", OPR),
    ("rt-WT5", "Before wounding, JA-Ile was undetectable in jar1 mutants, and also in Col-0.", JAR),
    ("rt-WT6", "In unwounded leaves, JA-Ile was not detected in jar1 mutant, wild-type and complemented plants.", JAR),
    ("rt-WT7", "JA-Ile was not detected in unwounded leaves of any genotype, including jar1 mutants.", JAR),
    ("rt-WT8", "JA-Ile was not detected in unwounded leaves of jar1 mutants, nor in Col-0.", JAR),
    ("rt-WT9", "JA-Ile was below the detection limit in unwounded jar1 mutant leaves; the same was true for Col-0.",
     JAR),
    ("rt-WT10", "In unwounded leaves, JA-Ile was undetectable in jar1 mutants, matching Col-0.", JAR),
    ("rt-WT11", "Basal 12-OPDA was undetectable in aos mutants and the parental line.", AOS),
    ("rt-WT12", "Before wounding, 12-OPDA was undetectable in aos mutant and Col plants.", AOS),
    ("rt-WT13", "Basal JA-Ile was undetectable in jar1 mutants and the corresponding wild type.", JAR),
    ("bio-Z15", "JA-Ile was undetectable in unwounded leaves of all genotypes, including jar1 mutants.", JAR),
    # the bio-auditor's no-effect rows with an unlisted comparison word, refused here because the
    # comparator is a control genotype and the sentence has no "than"
    ("bio-U4c", "gh3.10 mutants did not accumulate JA-Ile to levels different from those of Col-0.", GH3),
    ("bio-U4d", "gh3.10 single mutants did not accumulate JA-Ile to levels below those of Col-0.", GH3),
)
def test_cs_e_r2_746_3_a_result_shared_with_a_control_earns_nothing(text, kw):
    assert probe(text, **kw).species_state != TSS


def test_cs_e_r2_746_3_wt3_shared_absence_node_is_not_admitted():
    """End to end: at 01d436cf "not detected in jar1 mutants and control plants" admitted jasmonic acid -> JA-Ile."""
    claim, decision = admitted(WT3, **JAR)
    assert claim.species_state != TSS
    assert decision.status not in ADMITTED_STATUSES


# ---------------------------------------------------------------------------
# 4-8. The round-1 non-blocking residuals
# ---------------------------------------------------------------------------
@_rows(
    ("code-CO1-rt-CC7", "Notwithstanding high OPC-8:0 levels in opr3 mutants, JA was reduced.", OPR),
    ("code-CO2", "OPC-8:0 accumulated in opr3 mutants, albeit JA was reduced.", OPR),
    ("code-CO3", "While OPC-8:0 accumulated in opr3 mutants, JA was reduced.", OPR),
    ("code-CO4", "Even if OPC-8:0 accumulated in opr3 mutants, JA levels were reduced.", OPR),
    ("rt-CC1", "While OPC-8:0 accumulated in opr3 mutants, JA levels were strongly reduced.", OPR),
    ("rt-CC2", "Whilst OPC-8:0 accumulated in opr3 mutants, JA levels were strongly reduced.", OPR),
    ("rt-CC3", "While jar1 mutants accumulated JA-Ile, JA-responsive gene induction was reduced.", JAR),
    ("rt-CC6", "Even if JA-Ile accumulated in jar1 mutants, JA-dependent defence was reduced.", JAR),
    ("rt-CC8", "Regardless of high OPC-8:0 levels in opr3 mutants, JA was reduced after infection.", OPR),
    ("rt-CC10", "While opr3 mutants produced OPC-8:0 after Botrytis infection, JA levels were reduced.", OPR),
)
def test_cs_e_r2_746_4_a_concessive_clause_earns_nothing(text, kw):
    assert probe(text, **kw).species_state != TSS


@_rows(
    ("code-N7a", "jar1 mutants accumulated nowhere near lower JA-Ile levels after wounding.", JAR),
    ("code-N7b", "jar1 mutants accumulated nowhere near reduced JA-Ile after wounding.", JAR),
)
def test_cs_e_r2_746_5_nowhere_near_is_not_a_decrease(text, kw):
    assert probe(text, **kw).species_state != TSS


@_rows(
    ("rt-BG1", "jar1 mutants are known to contain reduced JA-Ile levels.", JAR),
    ("rt-BG2", "jar1 mutants were shown previously to contain reduced JA-Ile levels.", JAR),
    ("rt-BG4", "That JA-Ile is reduced in jar1 mutants is well documented.", JAR),
)
def test_cs_e_r2_746_6_background_in_a_results_sentence_earns_nothing(text, kw):
    assert probe(text, **kw).species_state != TSS


@_rows(
    ("rt-PO1", "In jar1 mutant plants, JA-Ile was detected mainly in the lower rosette leaves after wounding.", JAR),
    ("rt-PO3", "In opr3 mutants, OPC-8:0 accumulated in the lower epidermis.", OPR),
    ("rt-PO6", "Wounded jar1 mutant leaves accumulated JA-Ile in their lower part.", JAR),
)
def test_cs_e_r2_746_7_a_position_is_not_a_change(text, kw):
    assert probe(text, **kw).species_state != TSS


@_rows(
    ("rt-RG1", "Mutants lacking a suppressor of AOS accumulated no 12-OPDA.", AOS),
    ("rt-RG2-code-RG1", "Mutants lacking an inducer of AOS accumulated no 12-OPDA.", AOS),
    ("code-RG2", "Mutants lacking a co-activator of AOS accumulated no 12-OPDA.", AOS),
    ("bio-U3-code-SU3", "Mutants lacking a modulator of AOS accumulated no 12-OPDA.", AOS),
    ("rt-RG4", "In the jar1 suppressor mutant, JA-Ile was reduced.", JAR),
    ("rt-RG5", "In jar1 enhancer mutants, JA-Ile was undetectable.", JAR),
    ("rt-RG7", "Knockout of the AOS activator abolished 12-OPDA.", AOS),
)
def test_cs_e_r2_746_8_a_regulators_mutant_is_not_the_claims_genotype(text, kw):
    assert probe(text, **kw).species_state != TSS


# ---------------------------------------------------------------------------
# 9. The true subject after the effect word (BIO-E746R1-4 (i))
# ---------------------------------------------------------------------------
@_rows(
    ("bio-U1-code-SU1", "In cyp94b3 mutants, JA-Ile accumulated after wounding at the expense of reduced "
                        "12-OH-JA-Ile.", CYP_REV),
    ("bio-U1b", "In cyp94b3 mutants, JA-Ile accumulated after wounding alongside reduced 12-OH-JA-Ile.", CYP_REV),
    ("bio-U1c", "In cyp94b3 mutants, JA-Ile accumulated after wounding plus reduced 12-OH-JA-Ile.", CYP_REV),
    ("bio-U1g", "opr3 mutants accumulated OPDA alongside strongly reduced OPC-8:0 levels after wounding.", OPR_REV),
    ("code-SU2", "In cyp94b3 mutants, JA-Ile accumulated after wounding, alongside reduced 12-OH-JA-Ile.", CYP_REV),
    ("m746e-comma-plus", "In cyp94b3 mutants, JA-Ile accumulated after wounding, plus reduced 12-OH-JA-Ile.", CYP_REV),
)
def test_cs_e_r2_746_9_an_attributive_participant_after_the_effect_is_a_candidate(text, kw):
    assert probe(text, **kw).species_state != TSS


# ---------------------------------------------------------------------------
# Controls: hold at 01d436cf and here
# ---------------------------------------------------------------------------
@_rows(
    ("T1", T1, GH3),
    ("bio-W3-up-to", "In aos mutants, 12-OPDA was not detected at any time point up to 2 h after wounding.", AOS),
    ("bio-W4-throughout", "OPC-8:0 was undetectable in opr3 mutants throughout the time course (0.5 to 4 h after "
                          "wounding).", OPR),
    ("bio-W5-at-1-h", "At 1 h after wounding, JA-Ile was below the detection limit in gh3.10 jar1 knockout lines.", GH3),
    ("bio-W8-up-to-6-h", "In opr3 mutants, OPC-8:0 was not detected at any time point examined (up to 6 h).", OPR),
    ("bio-W9-all-time-points", "Wound-induced 12-OPDA accumulation was abolished in aos mutants at all time points "
                               "tested.", AOS),
    ("rt-RC3", "No JA-Ile was detected in jar1 mutants 1 h after wounding.", JAR),
    ("significantly-reduced", "JA-Ile was significantly reduced in jar1 mutants after wounding.", JAR),
    ("reduced-by-90%", "JA-Ile was reduced by 90% in jar1 mutants after wounding.", JAR),
    # the red team's RC4 survives: "significantly" is the adverb, and "P < 0.01" names no non-significance
    ("rt-RC4", "JA-Ile levels were significantly reduced in jar1 mutants (P < 0.01).", JAR),
    ("K2-20-fold-lower-than", "JA-Ile was 20-fold lower in jar1 mutants than in wild-type leaves.", JAR),
    ("rt-RC6", "Wound-induced JA-Ile was 90% lower in jar1 mutants than in Col-0 at 1 h.", JAR),
    ("no-JA-Ile-was-detected", "No JA-Ile was detected in gh3.10 jar1 knockout lines.", GH3),
    ("opr3-did-not-accumulate", "opr3 mutants did not accumulate OPC-8:0.", OPR),
    ("aos-accumulated-no", "aos mutants accumulated no 12-OPDA.", AOS),
    ("SALK-insertion-line-of", "The SALK_034543 insertion line of GH3.10 accumulated no JA-Ile.", GH3),
    ("T-DNA-insertion-lines-of-AOS", "T-DNA insertion lines of AOS were deficient in 12-OPDA.", AOS),
    ("deficient-in-GH3.10", "Mutants deficient in GH3.10 accumulated no JA-Ile in flowers.", dict(enzyme="GH3.10")),
    ("knockout-of-GH3.10", "Knockout of GH3.10 abolished JA-Ile accumulation in flowers.", GH3),
)
def test_new_acceptance_cs_e_r2_746_controls_keep_in_planta_credit(text, kw):
    claim, decision = admitted(text, **kw)
    assert (claim.species_state, claim.species_support["grade"]) == (TSS, "in_planta")
    assert decision.status in ADMITTED_STATUSES
