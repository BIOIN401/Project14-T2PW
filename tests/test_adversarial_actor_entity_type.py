"""ADVERSARIAL GROUP 3 — reconstruction must not guess what kind of actor it saw.

Reconstructing a missing registry row for a compound is bookkeeping: in this
schema a reaction's ``inputs`` and ``outputs`` ARE its compounds, so putting a
shell in ``entities.compounds`` restates a role the payload already asserted and
adds no claim.

Reconstructing one for an ENZYME is not. "Enzyme" spans two buckets, and they are
not interchangeable: ``entities.proteins`` is one gene product with a UniProt
identity, ``entities.protein_complexes`` is an assembly with components and
stoichiometry. They hit different identity gates, produce different PWML, and
resolve against different PathBank tables. Sending a complex to ``proteins``
because that is the commoner case asserts that it is a single gene product —
a *biological* claim, made by a function whose entire premise is that it makes
none.

So the bucket is read off what the payload states, and an actor that states
nothing is skipped with a reason. A reported gap is recoverable; a silently
mistyped entity is not.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.localized_repair import (  # noqa: E402
    SHELL_PROVENANCE,
    reconstruct_registry_shells,
)


def _reaction(**overrides: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "name": "r",
        "inputs": ["ATP"],
        "outputs": ["ADP"],
        "evidence": "ATP becomes ADP.",
    }
    row.update(overrides)
    return {"entities": {}, "processes": {"reactions": [row]}}


def _buckets(result: Any) -> Dict[str, List[str]]:
    return {
        bucket: [row["name"] for row in rows]
        for bucket, rows in result.payload["entities"].items()
        if isinstance(rows, list) and rows
    }


def _skips(result: Any) -> List[str]:
    return [item["reason"] for item in result.skipped]


# ---------------------------------------------------------------------------
# The type the payload states is the type the shell gets.
# ---------------------------------------------------------------------------
def test_a_protein_typed_enzyme_lands_in_proteins() -> None:
    result = reconstruct_registry_shells(_reaction(enzymes=[{"protein": "ATPase"}]))

    assert _buckets(result)["proteins"] == ["ATPase"]
    assert "protein_complexes" not in _buckets(result)


def test_a_protein_complex_typed_enzyme_lands_in_protein_complexes() -> None:
    """THE regression. Before the fix this complex was filed as a protein."""

    result = reconstruct_registry_shells(
        _reaction(enzymes=[{"protein_complex": "ATP synthase complex"}])
    )

    assert _buckets(result)["protein_complexes"] == ["ATP synthase complex"]
    assert "proteins" not in _buckets(result)


def test_a_reaction_modifier_keeps_its_complex_type() -> None:
    """``modifiers`` is the pre-cleaning key and takes the typed ``entity`` form."""

    result = reconstruct_registry_shells(
        _reaction(
            modifiers=[{"entity": "pyruvate dehydrogenase complex", "entity_type": "protein_complex"}]
        )
    )

    assert _buckets(result)["protein_complexes"] == ["pyruvate dehydrogenase complex"]


def test_a_reaction_modifier_typed_protein_stays_a_protein() -> None:
    result = reconstruct_registry_shells(
        _reaction(modifiers=[{"entity": "LpxA", "entity_type": "protein"}])
    )

    assert _buckets(result)["proteins"] == ["LpxA"]


def test_a_transporter_keeps_its_complex_type() -> None:
    """Transports have their own actor key and the same rule applies to it."""

    payload = {
        "entities": {},
        "processes": {
            "transports": [
                {
                    "name": "efflux",
                    "cargo": "glutathione",
                    "transporters": [{"protein_complex": "ABC transporter complex"}],
                }
            ]
        },
    }

    result = reconstruct_registry_shells(payload)

    buckets = _buckets(result)
    assert buckets["protein_complexes"] == ["ABC transporter complex"]
    assert buckets["compounds"] == ["glutathione"]
    assert "proteins" not in buckets


def test_a_transporter_typed_protein_stays_a_protein() -> None:
    payload = {
        "entities": {},
        "processes": {
            "transports": [
                {"name": "efflux", "cargo": "glutathione", "transporters": [{"protein": "MRP1"}]}
            ]
        },
    }

    result = reconstruct_registry_shells(payload)

    assert _buckets(result)["proteins"] == ["MRP1"]


def test_a_reaction_coupled_transport_enzyme_keeps_its_type() -> None:
    payload = {
        "entities": {},
        "processes": {
            "reaction_coupled_transports": [
                {
                    "name": "rct",
                    "reaction": "r",
                    "transport": "t",
                    "enzymes": [{"protein_complex": "V-ATPase"}],
                }
            ]
        },
    }

    result = reconstruct_registry_shells(payload)

    assert _buckets(result)["protein_complexes"] == ["V-ATPase"]


def test_both_actor_kinds_on_one_reaction_are_filed_separately() -> None:
    result = reconstruct_registry_shells(
        _reaction(
            enzymes=[
                {"protein": "LpxA"},
                {"protein_complex": "LpxA-LpxD complex"},
            ]
        )
    )

    buckets = _buckets(result)
    assert buckets["proteins"] == ["LpxA"]
    assert buckets["protein_complexes"] == ["LpxA-LpxD complex"]


# ---------------------------------------------------------------------------
# No statement, no guess.
# ---------------------------------------------------------------------------
def test_an_untyped_actor_is_skipped_with_a_reason() -> None:
    """``{"name": ...}`` states no type. Nothing is invented and the gap is named.

    This is also the shape ``_clean_enzymes`` deletes outright, so the skip and
    the ``actor_missing_protein_reference`` cleaning discard are two views of one
    upstream problem — and a reviewer now gets both.
    """

    result = reconstruct_registry_shells(_reaction(enzymes=[{"name": "ATPase"}]))

    assert "proteins" not in _buckets(result)
    assert "protein_complexes" not in _buckets(result)
    assert "actor_entity_type_unknown" in _skips(result)
    skip = next(item for item in result.skipped if item["reason"] == "actor_entity_type_unknown")
    assert skip["name"] == "ATPase"
    assert skip["role"] == "enzymes"


def test_a_bare_string_actor_is_skipped() -> None:
    result = reconstruct_registry_shells(_reaction(enzymes=["ATPase"]))

    assert "actor_entity_type_unknown" in _skips(result)
    assert "proteins" not in _buckets(result)


@pytest.mark.parametrize("declared", ["", "   ", "compound", "cofactor", "ion", "enzyme", "complex"])
def test_an_actor_with_an_unusable_entity_type_is_skipped(declared: str) -> None:
    """Only ``protein`` and ``protein_complex`` name an actor bucket.

    ``"enzyme"`` and ``"complex"`` are the interesting ones: both read like a
    type and neither names a bucket. Treating ``"complex"`` as
    ``protein_complex`` would be a guess about spelling standing in for a
    statement about biology.
    """

    result = reconstruct_registry_shells(
        _reaction(modifiers=[{"entity": "X", "entity_type": declared}])
    )

    assert "actor_entity_type_unknown" in _skips(result)
    assert "proteins" not in _buckets(result)
    assert "protein_complexes" not in _buckets(result)


def test_a_normalizable_entity_type_is_still_accepted() -> None:
    """Whitespace and case are formatting, not ambiguity."""

    result = reconstruct_registry_shells(
        _reaction(modifiers=[{"entity": "LpxA", "entity_type": "  Protein "}])
    )

    assert _buckets(result)["proteins"] == ["LpxA"]


def test_a_skipped_actor_does_not_block_the_compounds_on_the_same_row() -> None:
    """One unknowable actor must not cost the reaction its participant rows."""

    result = reconstruct_registry_shells(_reaction(enzymes=[{"name": "mystery"}]))

    assert _buckets(result)["compounds"] == ["ATP", "ADP"]
    assert "actor_entity_type_unknown" in _skips(result)


# ---------------------------------------------------------------------------
# Static compound reconstruction is unchanged.
# ---------------------------------------------------------------------------
def test_reaction_inputs_and_outputs_still_reconstruct_as_compounds() -> None:
    """The role IS the type here, so nothing about it is a guess.

    Pinned separately because the actor fix works by removing a static mapping,
    and removing the wrong one would silently stop recovering the participants
    this whole feature exists for.
    """

    result = reconstruct_registry_shells(_reaction())

    compounds = result.payload["entities"]["compounds"]
    assert [row["name"] for row in compounds] == ["ATP", "ADP"]
    for row in compounds:
        assert row["provenance"] == SHELL_PROVENANCE
        assert row["resolution_status"] == "unresolved"


@pytest.mark.parametrize("role", ["inputs", "outputs", "left", "right", "substrates", "products"])
def test_every_static_compound_role_still_reconstructs(role: str) -> None:
    payload = {
        "entities": {},
        "processes": {"reactions": [{"inputs": ["ATP"], "outputs": ["ADP"], role: ["Novel"]}]},
    }

    result = reconstruct_registry_shells(payload)

    assert "Novel" in _buckets(result)["compounds"]


def test_transport_cargo_still_reconstructs_as_a_compound() -> None:
    payload = {
        "entities": {},
        "processes": {"transports": [{"name": "t", "cargo": "glutathione"}]},
    }

    result = reconstruct_registry_shells(payload)

    assert _buckets(result)["compounds"] == ["glutathione"]


def test_a_shell_for_a_complex_carries_no_components_or_identity() -> None:
    """A complex shell must not acquire a components list either.

    Inventing components would be the same class of error as inventing the
    bucket, one level down.
    """

    result = reconstruct_registry_shells(
        _reaction(enzymes=[{"protein_complex": "ATP synthase complex"}])
    )

    row = result.payload["entities"]["protein_complexes"][0]
    assert set(row) == {"name", "provenance", "resolution_status", "reconstructed_from"}
    for forbidden in ("components", "mapped_ids", "uniprot", "pathbank_complex_id", "generated"):
        assert forbidden not in row


def test_an_actor_already_registered_under_either_bucket_is_not_duplicated() -> None:
    payload = {
        "entities": {"protein_complexes": [{"name": "ATP synthase complex"}]},
        "processes": {
            "reactions": [
                {
                    "inputs": ["ATP"],
                    "outputs": ["ADP"],
                    "enzymes": [{"protein": "ATP synthase complex"}],
                }
            ]
        },
    }

    result = reconstruct_registry_shells(payload)

    assert "proteins" not in _buckets(result)
    assert len(result.payload["entities"]["protein_complexes"]) == 1
