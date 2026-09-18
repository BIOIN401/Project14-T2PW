"""The species guardrail.

The property under test throughout: **nothing but an explicitly evidenced
target-species claim is admissible**, and an ortholog / analogy never transfers a
reaction between species however confidently it is asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.mining.species import (  # noqa: E402
    ADMISSIBLE_SPECIES_STATES,
    SPECIES_AMBIGUOUS,
    SPECIES_INFERRED_ONLY,
    SPECIES_IN_VITRO,
    SPECIES_MULTIPLE,
    SPECIES_NON_TARGET,
    SPECIES_NOT_STATED,
    SPECIES_STATES,
    SPECIES_TARGET_SUPPORTED,
    classify_species,
    normalize_species,
    species_matches_target,
)

TARGET = "arabidopsis thaliana"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Arabidopsis thaliana", "arabidopsis thaliana"),
        ("  ARABIDOPSIS   THALIANA  ", "arabidopsis thaliana"),
        ("Arabidopsis thaliana (Col-0)", "arabidopsis thaliana"),
        ("A. thaliana", "arabidopsis thaliana"),
        ("human", "homo sapiens"),
        ("mice", "mus musculus"),
        ("E. coli", "escherichia coli"),
        ("Escherichia coli K-12 MG1655", "escherichia coli"),
        ("", ""),
        (None, ""),
    ],
)
def test_local_normalization(raw, expected):
    assert normalize_species(raw) == expected


def test_normalization_is_local_and_not_a_database_claim():
    """An unknown organism normalizes to itself rather than being 'resolved'."""
    assert normalize_species("Quercus robur") == "quercus robur"
    assert normalize_species("X. unknownii") == "x. unknownii"


def test_genus_alone_does_not_match_a_binomial():
    assert species_matches_target("Arabidopsis lyrata", TARGET) is False
    assert species_matches_target("Zea mays", TARGET) is False
    assert species_matches_target("Arabidopsis thaliana", TARGET) is True


def test_target_supported_requires_a_cited_evidence_span():
    verdict = classify_species(
        target_species_normalized=TARGET,
        observed=["Arabidopsis thaliana"],
        species_evidence_ids=["ev_1"],
        evidence_text="Measured in Arabidopsis thaliana leaves.",
    )
    assert verdict.state == SPECIES_TARGET_SUPPORTED
    assert verdict.admissible is True
    assert verdict.species_evidence_ids == ["ev_1"]


def test_named_but_uncited_target_is_ambiguous_not_supported():
    verdict = classify_species(
        target_species_normalized=TARGET,
        observed=["Arabidopsis thaliana"],
        species_evidence_ids=[],
        evidence_text="The conversion was observed.",
    )
    assert verdict.state == SPECIES_AMBIGUOUS
    assert verdict.admissible is False


def test_other_species_is_non_target():
    verdict = classify_species(
        target_species_normalized=TARGET,
        observed=["Zea mays"],
        species_evidence_ids=["ev_1"],
        evidence_text="Assayed in Zea mays seedlings.",
    )
    assert verdict.state == SPECIES_NON_TARGET
    assert verdict.admissible is False


def test_several_species_including_the_target_is_multiple_not_supported():
    verdict = classify_species(
        target_species_normalized=TARGET,
        observed=["Arabidopsis thaliana", "Nicotiana tabacum"],
        species_evidence_ids=["ev_1"],
        evidence_text="Both systems were used.",
    )
    assert verdict.state == SPECIES_MULTIPLE
    assert verdict.admissible is False


def test_nothing_observed_is_not_stated():
    verdict = classify_species(
        target_species_normalized=TARGET, observed=[], evidence_text="It is converted."
    )
    assert verdict.state == SPECIES_NOT_STATED
    assert verdict.admissible is False


def test_in_vitro_is_its_own_state():
    verdict = classify_species(
        target_species_normalized=TARGET,
        observed=["Arabidopsis thaliana"],
        species_evidence_ids=["ev_1"],
        evidence_text="The reaction was reconstituted in vitro with purified enzyme.",
    )
    assert verdict.state == SPECIES_IN_VITRO
    assert verdict.admissible is False


@pytest.mark.parametrize(
    "text",
    [
        "The Arabidopsis thaliana ortholog presumably catalyses the same step.",
        "By analogy with the tomato enzyme, Arabidopsis thaliana is expected to convert it.",
        "The homolog of the rice protein in Arabidopsis thaliana likely catalyzes this.",
        "Inferred from the mouse pathway by similarity.",
    ],
)
def test_ortholog_and_analogy_never_transfer_between_species(text):
    """The single most important rule in this module.

    Even with the target species named AND an evidence span cited, an inference
    marker demotes the claim to ``inferred_only``. A shared enzyme name is not a
    reaction in a new organism.
    """
    verdict = classify_species(
        target_species_normalized=TARGET,
        observed=["Arabidopsis thaliana"],
        species_evidence_ids=["ev_1"],
        evidence_text=text,
    )
    assert verdict.state == SPECIES_INFERRED_ONLY
    assert verdict.admissible is False
    assert verdict.inference_markers


def test_explicit_inferred_flag_is_honoured_without_a_marker():
    verdict = classify_species(
        target_species_normalized=TARGET,
        observed=["Arabidopsis thaliana"],
        species_evidence_ids=["ev_1"],
        evidence_text="A clean sentence with no marker at all.",
        claimed_inferred=True,
    )
    assert verdict.state == SPECIES_INFERRED_ONLY


def test_exactly_one_state_is_admissible():
    assert ADMISSIBLE_SPECIES_STATES == {SPECIES_TARGET_SUPPORTED}
    assert set(SPECIES_STATES) - ADMISSIBLE_SPECIES_STATES == {
        SPECIES_NON_TARGET,
        SPECIES_MULTIPLE,
        SPECIES_AMBIGUOUS,
        SPECIES_NOT_STATED,
        SPECIES_IN_VITRO,
        SPECIES_INFERRED_ONLY,
    }


def test_every_verdict_carries_a_reason():
    for observed, ids, text in (
        (["Arabidopsis thaliana"], ["ev"], "in Arabidopsis thaliana"),
        (["Zea mays"], ["ev"], "in Zea mays"),
        ([], [], "no species"),
        (["Arabidopsis thaliana", "Zea mays"], ["ev"], "both"),
    ):
        verdict = classify_species(
            target_species_normalized=TARGET,
            observed=observed,
            species_evidence_ids=ids,
            evidence_text=text,
        )
        assert verdict.reason.strip(), verdict
        assert verdict.state in SPECIES_STATES
