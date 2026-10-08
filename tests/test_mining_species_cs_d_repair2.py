"""ORCH-745 change set D, repair round 2: the four one-line fixes.

1. A participant INSIDE a comma-led relative clause that closes before the effect is not
   its subject; the head before the clause stays one (CSD-R1-B1, RT-D-R1-1, BIO-D-R1-1).
2. A negator inside the "accumulated ... less" gap refuses the form (CSD-R1-B2).
3. The gap tokens before "of" / "in" in the marker tail may not be "of" / "in" (RT-D-R1-2).
4. The negated-comparison window does not stop at "OPC-8:0" / "gh3.10" (BIO-D-R1-2).

Every ``test_cs_d_r2_745_*`` row fails at ``fd7ea63a``; the manifest says which also fail at
``6454ef98`` / ``fd7c737c``. The ``test_new_acceptance_cs_d_r2_745_*`` rows are the brief's
controls and hold at both ends.
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
JAP = dict(sub="jasmonic acid (JA)", prod="jasmonoyl-isoleucine (JA-Ile)")
OPR_AT = dict(sub="OPDA", prod="OPC-8:0", gene="", enzyme="OPR3", observed=AT)
AOS_AT = dict(sub="13-HPOT", prod="12-OPDA", gene="", enzyme="AOS", observed=AT)
RJAR = dict(gene="", enzyme="JAR1", observed=AT)
RGH3 = dict(gene="GH3.10", enzyme="", observed=AT)
CYP = dict(enzyme="", gene="CYP94B3", observed=AT)


# ===========================================================================
# 1. a participant inside a closed relative clause is not the subject
# ===========================================================================
@_rows(
    # RT-D-R1-1's regression fixtures
    ("rt-R2-cyp94b3-reversed", "In cyp94b3 mutants, JA-Ile, which is hydroxylated to 12-OH-JA-Ile by CYP94B3, "
                               "accumulated after wounding.", dict(CYP, sub="12-OH-JA-Ile", prod="JA-Ile")),
    ("rt-R3-gh3.10-product-up", "In gh3.10 mutants, JA-Ile, which is formed from jasmonic acid, accumulated after "
                                "wounding.", RGH3),
    ("rt-R4-jar1-substrate-down", "In the jar1 mutant, jasmonic acid, which is conjugated to JA-Ile, was strongly "
                                  "reduced after wounding.", RJAR),
    ("rt-R6-opr3-substrate-down", "In opr3 mutants, OPDA, which is the precursor of OPC-8:0, was reduced.", OPR_AT),
    # BIO-D-R1-1's regression fixtures
    ("bio-W6-reversed", "In cyp94b3 mutants, 12-OH-JA-Ile, which is formed from JA-Ile, was strongly reduced after "
                        "wounding.", dict(CYP, sub="12-OH-JA-Ile", prod="JA-Ile")),
    ("bio-W7-reversed", "In jar1 mutants, JA-Ile, which is synthesized from JA, was strongly reduced after wounding.",
     dict(sub="jasmonoyl-isoleucine (JA-Ile)", prod="jasmonic acid (JA)", **RJAR)),
    ("bio-W2-product-up", "In the jar1 mutant, JA-Ile, which is formed from JA, accumulated after wounding.",
     dict(JAP, **RJAR)),
    ("bio-W12-product-up", "In opr3 mutants, OPC-8:0, which is the reduction product of OPDA, accumulated after "
                           "wounding.", OPR_AT),
    ("bio-W10-reversed-undetectable", "In cyp94b3 mutants, 12-OH-JA-Ile, which is the hydroxylated form of JA-Ile, "
                                      "was undetectable.", dict(CYP, sub="12-OH-JA-Ile", prod="JA-Ile")),
)
def test_cs_d_r2_745_1_a_participant_inside_a_closed_relative_clause_is_not_the_subject(text, kw):
    assert probe(text, **kw).species_state != TSS


@_rows(
    ("rt-R1-cyp94b3-correct-direction", "In cyp94b3 mutants, JA-Ile, which is hydroxylated to 12-OH-JA-Ile by "
                                        "CYP94B3, accumulated after wounding.",
     dict(CYP, sub="JA-Ile", prod="12-OH-JA-Ile")),
    ("rt-R5-which-is-formed-from", "In opr3 mutants, OPC-8:0, which is formed from OPDA, was undetectable.", OPR_AT),
    ("rt-R7-which-is-the-bioactive-form", "In jar1 mutants, JA-Ile, which is the bioactive form, was undetectable.",
     RJAR),
)
def test_cs_d_r2_745_1_the_head_before_a_closed_relative_clause_keeps_its_credit(text, kw):
    claim = probe(text, **kw)
    assert (claim.species_state, claim.species_support["grade"]) == (TSS, "in_planta")


# ===========================================================================
# 2. a negator inside the "accumulated ... less" gap
# ===========================================================================
@_rows(
    ("H1-hardly-less", "jar1 mutants accumulated hardly less JA-Ile than wild-type plants after wounding.", RJAR),
    ("H2-barely-less", "The jar1 mutant accumulated barely less JA-Ile than the wild type.", RJAR),
    ("H3-scarcely-lower", "gh3.10 mutants accumulated scarcely lower JA-Ile levels after wounding.", RGH3),
    ("F1-not-less", "The jar1 mutant accumulated not less JA-Ile than the wild type.", RJAR),
)
def test_cs_d_r2_745_2_a_negator_inside_the_accumulated_less_gap_earns_no_credit(text, kw):
    assert probe(text, **kw).species_state != TSS


# ===========================================================================
# 3. the marker tail does not bind through another "of" / "in"
# ===========================================================================
@_rows(
    ("genes-upstream-of", "Mutants in genes upstream of JAR1 accumulated no JA-Ile.", RJAR),
    ("targets-of", "Knockouts of transcriptional targets of JAR1 accumulated no JA-Ile.", RJAR),
    ("a-regulator-of", "Mutants in a regulator of JAR1 accumulated less JA-Ile after wounding.", RJAR),
)
def test_cs_d_r2_745_3_a_marker_of_another_genes_mutant_is_not_of_the_claims_gene(text, kw):
    assert probe(text, **kw).species_state != TSS


# ===========================================================================
# 4. the comparative window passes "OPC-8:0" / "gh3.10"
# ===========================================================================
@_rows(
    ("CL1-until", "The opr3 mutant did not accumulate OPC-8:0 until 6 h after wounding.", OPR_AT),
    ("CL2-lower-than", "opr3 mutants did not accumulate OPC-8:0 to lower levels than the wild type.", OPR_AT),
    ("CL3-gh3.10-lower-than", "JA-Ile did not accumulate in gh3.10 single mutants to lower levels than in Col-0.",
     RGH3),
    ("CL4-gh3.10-until", "JA-Ile did not accumulate in gh3.10 mutants until 6 h after wounding.", RGH3),
    ("CL5-further", "The acx1 mutant failed to accumulate OPC-6:0 further after wounding.",
     dict(sub="OPC-8:0", prod="OPC-6:0", gene="", enzyme="ACX1", observed=AT)),
    ("NX3-any-later-than", "The opr3 mutant did not form OPC-8:0 any later than Col-0.", OPR_AT),
)
def test_cs_d_r2_745_4_a_negated_comparison_past_a_dotted_name_earns_no_credit(text, kw):
    assert probe(text, **kw).species_state != TSS


# ===========================================================================
# Controls (the brief's list): hold at fd7ea63a and here
# ===========================================================================
@_rows(
    ("H4-much-less", "jar1 mutants accumulated much less JA-Ile after wounding.", RJAR),
    ("Pc2", "jar1 mutant plants accumulated lower levels of JA-Ile after wounding.", dict(JAP, **RJAR)),
    ("aos-accumulated-less", "aos mutants accumulated less 12-OPDA after wounding.", AOS_AT),
    ("SALK-insertion-line-of", "The SALK_034543 insertion line of GH3.10 accumulated no JA-Ile.", dict(gene="GH3.10")),
    ("T-DNA-insertion-lines-of", "T-DNA insertion lines of GH3.10 accumulated no JA-Ile.", {}),
    ("deficient-in", "Mutants deficient in GH3.10 accumulated no JA-Ile in flowers.", {}),
    ("opr3-did-not-accumulate", "The opr3 mutant did not accumulate OPC-8:0 after wounding.", OPR_AT),
    ("gh3.10-jar1-did-not-accumulate", "JA-Ile did not accumulate in gh3.10 jar1 double mutants after wounding.", RGH3),
    ("flowers-did-not-accumulate", "In flowers of gh3.10 jar1 knockout lines, JA-Ile did not accumulate.", RGH3),
    ("B1", "The aos mutant did not accumulate 12-OPDA after wounding.", AOS_AT),
    ("B2", "jar1 mutants failed to accumulate JA-Ile in wounded leaves.", RJAR),
    ("B3", "In jar1 mutants, JA-Ile was no longer detected after wounding.", RJAR),
    ("B4", "Wounded leaves of the opr3 mutant failed to produce OPC-8:0.", OPR_AT),
    ("B5", "The jar1 gh3.10 double mutant does not form JA-Ile in flowers.", dict(gene="", enzyme="GH3.10",
                                                                               observed=AT)),
    ("B6", "In wounded leaves of opr3 mutants, OPC-8:0 did not accumulate.", OPR_AT),
    ("Nc1", "The gh3.10 jar1 double mutant did not accumulate JA-Ile after wounding.", RGH3),
    ("Nc2", "gh3.10 jar1 knockout lines failed to accumulate JA-Ile in flowers.", RGH3),
    ("20-fold-lower-than", "JA-Ile was 20-fold lower in jar1 mutants than in wild-type leaves.", RJAR),
    ("appositive", "In jar1 mutants, JA-Ile, the bioactive form, was absent after wounding.", RJAR),
)
def test_new_acceptance_cs_d_r2_745_controls_keep_in_planta_credit(text, kw):
    claim = probe(text, **kw)
    assert (claim.species_state, claim.species_support["grade"]) == (TSS, "in_planta")


@_rows(
    ("X19", "JA-Gln accumulated at higher levels in gh3.10 jar1 knockout lines, which had reduced JA-Ile.",
     dict(prod="JA-Gln", gene="GH3.10", enzyme="")),
    ("X20", "In opr3 mutants, OPC-8:0 levels were elevated, with decreased OPDA.",
     dict(sub="OPDA", prod="OPC-8:0", gene="", enzyme="OPR3")),
)
def test_new_acceptance_cs_d_r2_745_an_open_clause_still_cuts_its_subject_off(text, kw):
    assert probe(text, **kw).species_state != TSS
