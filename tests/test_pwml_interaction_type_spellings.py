"""ORCH-749 -- the relationship spellings the PWML exporter maps to PathBank's types.

ORCH-747 built the PathBank-shaped ``<interaction>`` export but mapped only the exact
nouns ``activation`` / ``inhibition``. No committed interaction is spelled that way (they
are verb forms: ``inhibits``, ``represses``, ``activates``), so the exporter withheld
everything. These tests pin the extended table in ``t2pw.pwml.ir.PWML_INTERACTION_TYPES``:
every verb form whose sign is in the word reaches its PathBank type; every word that
carries no sign is still withheld with a reason and its spelling preserved; case,
whitespace and punctuation variants resolve; and an unknown or near-miss word is withheld
rather than defaulted or stemmed.

The G9 base-SHA proof (the base withholds a verb-form payload, the tip exports it) is
``docs/pwml_recovery_sprint/evidence/orch749_verb_form_export_base_proof.py``. Nothing here
resolves or invents biology: the fixture declares its endpoints, and the sign comes from
the word the payload already carries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.ir import PWML_INTERACTION_TYPES, build_pwml_ir  # noqa: E402
from t2pw.pwml.validate import discover_structure_signature  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402

REFERENCE = ROOT / "reference" / "PW000001.pwml"

#: The documented table, word for word. The test below pins the module to exactly this.
ACTIVATION = [
    "activation",
    "activates",
    "stimulates",
    "induces",
    "upregulates",
    "up-regulates",
    "enhances",
    "promotes",
]
INHIBITION = [
    "inhibition",
    "inhibits",
    "represses",
    "repression",
    "blocks",
    "suppresses",
    "downregulates",
    "down-regulates",
    "antagonizes",
    "competes with",
    "competes_with",
    "prevents",
]

#: Words the pipeline really writes whose sign is NOT in the word. PathBank has no neutral
#: interaction type, so each must stay withheld rather than be guessed into one.
UNSIGNED = [
    "binds",
    "binding",
    "binds_to",
    "binds to",
    "interacts",
    "interacts_with",
    "interacts with",
    "interaction",
    "associates with",
    "SAME_AS",
    "promoter_of",
    "modifies",
    "degrades",
    "regulates",
    "modulates",
    "catalyzes",
    "cofactor_of",
    "component_of",
    "transports",
    "inhibited_by",
    "allosteric_inhibitor_of",
    "recruits",
    "stabilizes",
    "reduces",
]


def _payload(*relationships: str) -> dict:
    """One interaction per spelling, complex -> promoter, so no row is withheld for a
    bare-protein endpoint and the only question left is the word."""
    return {
        "entities": {
            "species": [{"name": "Escherichia coli", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [],
            "proteins": [
                {"name": "AraC", "species": "Escherichia coli", "pathbank_protein_id": 301, "mapped_ids": {"uniprot": "P0A9E0"}},
            ],
            "protein_complexes": [
                {
                    "name": "AraC complex",
                    "species": "Escherichia coli",
                    "components": ["AraC"],
                    "generated": True,
                    "generation_reason": "single_protein_pathwhiz_wrapper",
                }
            ],
            "nucleic_acids": [{"name": "araBAD promoter", "class": "nucleic_acid"}],
        },
        "biological_states": [
            {"name": "cytosol", "species": "Escherichia coli", "subcellular_location": "cytosol", "compartment_canonical": "cytosol"},
        ],
        "processes": {
            "reactions": [],
            "transports": [],
            "interactions": [
                {
                    "name": f"row {index}",
                    "entity_1": "AraC complex",
                    "relationship": relationship,
                    "entity_2": "araBAD promoter",
                    "biological_state": "cytosol",
                    "evidence": "q",
                }
                for index, relationship in enumerate(relationships)
            ],
        },
    }


def _rows(*relationships: str):
    ir, report = build_pwml_ir(_payload(*relationships), strict_db=True, name_index=None)
    assert report["errors"] == [], report["errors"]
    return ir["processes"]["interactions"], report


# ── the table itself ─────────────────────────────────────────────────────────


def test_the_table_is_exactly_the_documented_words_and_nothing_else() -> None:
    assert set(PWML_INTERACTION_TYPES) == set(ACTIVATION) | set(INHIBITION)
    assert {PWML_INTERACTION_TYPES[w] for w in ACTIVATION} == {"Activation"}
    assert {PWML_INTERACTION_TYPES[w] for w in INHIBITION} == {"Inhibition"}
    # Keys are already in lookup form, so a reader can match a payload word to a row by eye.
    assert all(key == key.casefold().strip() for key in PWML_INTERACTION_TYPES)
    assert not set(UNSIGNED) & set(PWML_INTERACTION_TYPES)


# ── each mapped verb reaches its PathBank type ──────────────────────────────


@pytest.mark.parametrize("word", ACTIVATION)
def test_each_activation_spelling_exports_as_activation(word: str) -> None:
    (row,), report = _rows(word)
    assert row["interaction_type"] == "Activation"
    assert row["export_status"] == "exportable"
    assert "withheld" not in row
    assert not [w for w in report["warnings"] if w["code"] == "interaction_withheld_from_pwml"]


@pytest.mark.parametrize("word", INHIBITION)
def test_each_inhibition_spelling_exports_as_inhibition(word: str) -> None:
    (row,), report = _rows(word)
    assert row["interaction_type"] == "Inhibition"
    assert row["export_status"] == "exportable"
    assert "withheld" not in row
    assert not [w for w in report["warnings"] if w["code"] == "interaction_withheld_from_pwml"]


# ── every unsigned word stays withheld, with a reason and its spelling ──────


@pytest.mark.parametrize("word", UNSIGNED)
def test_a_word_whose_sign_is_not_in_it_is_withheld_not_guessed(word: str) -> None:
    (row,), report = _rows(word)
    assert row["export_status"] == "withheld"
    assert row["interaction_type"] == word, "the word is kept as written, no sign read into it"
    assert row["withheld"]["reason"] == "untyped_interaction"
    assert row["withheld"]["raw_interaction_type"] == word
    assert row["withheld"]["pointer"] == "/processes/interactions/0"
    warnings = [w for w in report["warnings"] if w["code"] == "interaction_withheld_from_pwml"]
    assert len(warnings) == 1
    assert warnings[0]["reason"] == "untyped_interaction"
    assert warnings[0]["raw_interaction_type"] == word


# ── case, whitespace and punctuation variants map; nothing else is tolerated ─


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("INHIBITS", "Inhibition"),
        ("Inhibits", "Inhibition"),
        ("REPRESSES", "Inhibition"),
        ("  inhibits  ", "Inhibition"),
        ('"inhibits"', "Inhibition"),
        ("'represses'", "Inhibition"),
        ("“inhibits”", "Inhibition"),
        ("inhibits.", "Inhibition"),
        ('"Activates".', "Activation"),
        ("Competes  With", "Inhibition"),
        ("UP-REGULATES", "Activation"),
        ("Down-Regulates", "Inhibition"),
        ("Activation", "Activation"),
    ],
)
def test_case_whitespace_quote_and_period_variants_map(spelling: str, expected: str) -> None:
    (row,), _ = _rows(spelling)
    assert row["interaction_type"] == expected
    assert row["export_status"] == "exportable"


def test_an_unknown_word_or_a_missing_relationship_is_withheld_rather_than_defaulted() -> None:
    rows, _ = _rows("frobnicates", "")
    unknown, missing = rows
    assert unknown["export_status"] == "withheld"
    assert unknown["interaction_type"] == "frobnicates"
    assert unknown["withheld"]["reason"] == "untyped_interaction"
    # No relationship at all: the exporter's placeholder word is itself unsigned.
    assert missing["export_status"] == "withheld"
    assert missing["interaction_type"] == "interaction"
    assert missing["withheld"]["raw_interaction_type"] == "interaction"


@pytest.mark.parametrize(
    "near_miss",
    ["inhibit", "inhibitor", "inhibiting", "inhibited", "activate", "activated", "activator",
     "promoter", "promoter_of", "represse", "inhibits strongly", "competes", "un-inhibits"],
)
def test_no_stemming_a_near_miss_of_a_table_word_is_withheld(near_miss: str) -> None:
    (row,), _ = _rows(near_miss)
    assert row["export_status"] == "withheld"
    assert row["interaction_type"] == near_miss
    assert row["withheld"]["reason"] == "untyped_interaction"


# ── end to end: the PWML carries the signed rows and the side list the rest ──


def test_the_pwml_carries_the_signed_rows_and_withholds_the_unsigned_ones() -> None:
    words = ["inhibits", "binds", "activates", "SAME_AS", "REPRESSES", "promoter_of", '"competes with".']
    ir, report = build_pwml_ir(_payload(*words), strict_db=True, name_index=None)
    assert report["errors"] == []
    signature = discover_structure_signature(REFERENCE)
    args = SimpleNamespace(
        name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000",
        height=1400, width=3200, background_color="#FFFFFF", ref=str(REFERENCE),
    )
    build = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=args).build()

    exported = [node.findtext("interaction-type") for node in build.root.iter("interaction")]
    assert exported == ["Inhibition", "Activation", "Inhibition", "Inhibition"]
    assert len(list(build.root.iter("interaction-visualization"))) == 4
    withheld = {row["raw_interaction_type"]: row["reason"] for row in build.withheld_interactions}
    assert withheld == {
        "binds": "untyped_interaction",
        "SAME_AS": "untyped_interaction",
        "promoter_of": "untyped_interaction",
    }
