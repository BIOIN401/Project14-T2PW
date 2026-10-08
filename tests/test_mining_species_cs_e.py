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
