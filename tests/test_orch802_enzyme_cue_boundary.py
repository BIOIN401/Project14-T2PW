"""ORCH-802 -- ``enzyme_cues.cue_near_name`` matched actor names as bare substrings.

ORCH-801, PMC12184417, cdadc1 leg: reaction R1 ``dCTP -> dUTP + NH3`` is
catalysed by CDADC1 alone, yet the shipped PWML carried DUT as a second enzyme,
because the case-insensitive scan found "DUT" inside "dUTP" and "catalyz" sat in
the window (``runs_papers/ORCH-801/ACCURACY_AUDIT.md`` F1,
``product_contract_violation``). Both name-based attachers route through this one
function, so the fix is a word-boundary-aware match there.

G9: the ``test_dut_*`` cases are the correction proof (fail on base d2e68b87,
pass at the tip). The positive controls pass on both and guard that the
boundary does not refuse real mentions.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.enzyme_cues import cue_near_name  # noqa: E402
from t2pw.pipeline.pipeline import _inject_name_based_modifiers  # noqa: E402
from t2pw.pipeline.process_normalizer import attach_enzymes_from_reaction_evidence  # noqa: E402

# The injector's haystack for cdadc1 R1, verbatim from merged_payload.json as
# quoted in ACCURACY_AUDIT.md F1: reaction name, a space, reaction evidence.
CDADC1_R1_NAME = "CDADC1-catalyzed dCTP deamination"
CDADC1_R1_EVIDENCE = "CDADC1 catalyzes the conversion of dCTP into dUTP"
CDADC1_R1_HAYSTACK = f"{CDADC1_R1_NAME} {CDADC1_R1_EVIDENCE}"


def _cdadc1_payload() -> dict:
    return {
        "entities": {"proteins": [{"name": "CDADC1"}, {"name": "DUT"}], "protein_complexes": []},
        "processes": {
            "reactions": [
                {
                    "name": CDADC1_R1_NAME,
                    "inputs": ["dCTP"],
                    "outputs": ["dUTP"],
                    "enzymes": [{"protein": "CDADC1", "evidence": CDADC1_R1_EVIDENCE}],
                    "evidence": CDADC1_R1_EVIDENCE,
                }
            ],
            "transports": [],
        },
    }


def _actor_names(reaction: dict) -> list:
    return [
        (row.get("entity") or row.get("protein") or row.get("protein_complex"))
        for bucket in ("enzymes", "modifiers")
        for row in reaction.get(bucket) or []
    ]


# --- correction proof: fails on base ------------------------------------------------


def test_dut_does_not_match_inside_dutp():
    assert cue_near_name(CDADC1_R1_HAYSTACK, "DUT") is None


def test_dut_not_injected_by_stage2_injector():
    payload = _cdadc1_payload()
    _inject_name_based_modifiers(payload)
    assert _actor_names(payload["processes"]["reactions"][0]) == ["CDADC1"]


def test_dut_not_attached_by_stage5_normalizer():
    payload = _cdadc1_payload()
    attach_enzymes_from_reaction_evidence(payload)
    assert _actor_names(payload["processes"]["reactions"][0]) == ["CDADC1"]


def test_short_symbol_not_matched_inside_longer_gene_name():
    # "CYP1" is not a mention inside "CYP11A1".
    assert cue_near_name("CYP11A1-catalyzed side-chain cleavage", "CYP1") is None


# --- positive controls: pass on base and tip ----------------------------------------


def test_real_dut_mention_still_matches():
    assert cue_near_name("dUTP hydrolysis catalyzed by DUT", "DUT")


def test_real_dut_still_injected_by_stage2_injector():
    payload = {
        "entities": {"proteins": [{"name": "DUT"}], "protein_complexes": []},
        "processes": {
            "reactions": [{"name": "dUTP hydrolysis", "evidence": "dUTP hydrolysis is catalyzed by DUT"}],
            "transports": [],
        },
    }
    _inject_name_based_modifiers(payload)
    assert _actor_names(payload["processes"]["reactions"][0]) == ["DUT"]


def test_name_at_start_of_text():
    assert cue_near_name("LpxC catalyzes the committed step", "LpxC")


def test_name_at_end_of_text():
    assert cue_near_name("the committed step is catalyzed by LpxC", "LpxC")


def test_name_followed_by_punctuation():
    assert cue_near_name("catalyzed by LpxC, the deacetylase.", "LpxC")
    assert cue_near_name("LpxC-catalyzed deacetylation", "LpxC")
    assert cue_near_name("an enzyme (LpxC) deacetylates it", "LpxC")


def test_name_ending_in_plus():
    assert cue_near_name("the NAD+ dependent dehydrogenase", "NAD+")
    assert cue_near_name("a reaction dependent on NAD+.", "NAD+")


def test_name_with_plus_and_slash_inside():
    assert cue_near_name("Na+/K+-ATPase activity drives the gradient", "Na+/K+-ATPase")


def test_multi_word_name():
    assert cue_near_name(
        "catalyzed by the enterobactin synthase complex", "enterobactin synthase complex"
    )


def test_name_with_internal_hyphen_and_digits():
    assert cue_near_name("UDP-glucose 4-epimerase catalyzes the interconversion", "UDP-glucose 4-epimerase")
    assert cue_near_name("CYP11A1-catalyzed side-chain cleavage", "CYP11A1")


def test_case_insensitive_hit():
    assert cue_near_name("lpxc catalyzes the committed step", "LpxC")
