"""C-099 -- a functional wrapper keeps the species that was already resolved for it.

``hydrate_species_references`` resolves species per row and stamps them through
``_stamp_entity_species`` long before Stage 6's PathBank ``Unknown`` fallback
builds its one-protein functional wrapper. Both wrapper-construction sites in
``_apply_pathbank_unknown_enzyme_fallback`` used to ``dict.update()`` the
sentinel record's OWN species (PathBank 9659 is *Arabidopsis thaliana*, species
id 4) over that result unconditionally, producing rows that contradicted
themselves: ``species: "Arabidopsis thaliana"`` beside
``species_name: "Escherichia coli"``, ``taxonomy_id: "562"`` and an *E. coli*
``species_ref`` at confidence 1.0. A released payload shipped that shape --
``runs_verify/2026-08-04_1754/papers/PMC12856317/strict`` put *Arabidopsis* on a
human ALAS2 wrapper.

Ruling: D-070 § O-1. The sixteen generated wrappers keep TRAP-3 protection
(O-1b), the five PathBank sentinel protein rows are untouched (O-1a), and only
an *already resolved, source-supported* species is protected from the clobber.

Every payload here is built by hand and stamped with the production
``_stamp_entity_species``. Nothing in this module reads the PathBank database or
any network resource, so its verdicts do not depend on whether a ``.env`` is
present in the tree under measurement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from t2pw.mapping import map_ids
from t2pw.mapping.map_ids import (
    _apply_pathbank_unknown_enzyme_fallback,
    _stamp_entity_species,
)
from t2pw.pipeline.entity_identity import (
    PATHBANK_UNKNOWN_PROTEIN_ID,
    PATHBANK_UNKNOWN_PROTEIN_NAME,
    PATHBANK_UNKNOWN_PROTEIN_UNIPROT,
    is_generated_complex_wrapper,
    is_pathbank_unknown_protein,
    protein_species_context,
)
from t2pw.pwml.ir import validate_required_pwml_contract
from t2pw.pipeline.process_normalizer import GateValidationError

# The PathBank ``Unknown`` sentinel record's own species. A true fact about
# record 9659, and never a fact about the entity a wrapper names.
SENTINEL_SPECIES = "Arabidopsis thaliana"
SENTINEL_SPECIES_ID = 4
SENTINEL_TAXONOMY_ID = "3702"

# A distinct third species, used to prove the wrapper never falls back to the
# pathway's dominant/"requested" organism or to an export-time default.
DEFAULT_EXPORT_SPECIES = "Camellia sinensis"
DEFAULT_EXPORT_SPECIES_ID = 77


def _ref(
    name: str,
    *,
    source: str,
    species_id: Optional[int],
    taxonomy_id: str,
    status: str = "matched",
    confidence: float = 1.0,
) -> Dict[str, Any]:
    ref: Dict[str, Any] = {
        "name": name,
        "source": source,
        "status": status,
        "confidence": confidence,
    }
    if species_id is not None:
        ref["pathbank_species_id"] = species_id
        ref["species_id"] = species_id
    if taxonomy_id:
        ref["taxonomy_id"] = taxonomy_id
    return ref


ECOLI_REF = _ref(
    "Escherichia coli", source="explicit_entity_species", species_id=9, taxonomy_id="562"
)
HUMAN_REF = _ref(
    "Homo sapiens", source="explicit_entity_species", species_id=1, taxonomy_id="9606"
)
YEAST_PATHWAY_REF = _ref(
    "Saccharomyces cerevisiae",
    source="single_pathway_species",
    species_id=11,
    taxonomy_id="4932",
)
MOUSE_STATE_REF = _ref(
    "Mus musculus", source="biological_state_species", species_id=2, taxonomy_id="10090"
)
# Inference, not source evidence: preserving it would launder a guess into a
# confident answer, so it must behave exactly as today.
LLM_REF = _ref(
    "Bacillus subtilis", source="gap_resolver_llm", species_id=13, taxonomy_id="1423"
)
# The 25-wrapper population: nothing was resolved. TRAP-3 protection stands.
NOVEL_REF: Dict[str, Any] = {
    "name": "Unknown species",
    "source": "novel_species",
    "status": "novel",
    "reason": "no_species_source",
    "confidence": 0.0,
}


def _hydrated_complex(name: str, ref: Dict[str, Any]) -> Dict[str, Any]:
    """A protein_complex row exactly as ``hydrate_species_references`` leaves it."""

    row: Dict[str, Any] = {"name": name, "components": []}
    _stamp_entity_species(row, ref)
    return row


def _enzyme_payload(
    specs: List[Tuple[str, Optional[Dict[str, Any]]]],
    *,
    pathway_species: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Reaction-enzyme payload that reaches wrapper-construction site #1.

    ``(name, ref)`` with a ref builds a hydrated protein_complex the fallback
    REUSES. ``(name, None)`` builds an unmapped protein instead, so the fallback
    CREATES a fresh wrapper that never had a resolution of its own.
    """

    complexes: List[Dict[str, Any]] = []
    proteins: List[Dict[str, Any]] = []
    reactions: List[Dict[str, Any]] = []
    for name, ref in specs:
        if ref is None:
            proteins.append({"name": name})
            entity_type = "protein"
        else:
            complexes.append(_hydrated_complex(name, ref))
            entity_type = "protein_complex"
        reactions.append(
            {
                "name": f"{name} reaction",
                "enzymes": [{"entity": name, "entity_type": entity_type, "role": "catalyst"}],
                "inputs": [],
                "outputs": [],
            }
        )
    return {
        "entities": {
            "proteins": proteins,
            "protein_complexes": complexes,
            "compounds": [],
            "species": list(pathway_species or []),
        },
        "processes": {"reactions": reactions, "transports": []},
    }


def _transport_payload(
    specs: List[Tuple[str, Optional[Dict[str, Any]]]],
    *,
    pathway_species: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Transporter payload that reaches wrapper-construction site #2."""

    complexes: List[Dict[str, Any]] = []
    proteins: List[Dict[str, Any]] = []
    transports: List[Dict[str, Any]] = []
    for name, ref in specs:
        if ref is None:
            proteins.append({"name": name})
            entity_type = "protein"
        else:
            complexes.append(_hydrated_complex(name, ref))
            entity_type = "protein_complex"
        transports.append(
            {
                "cargo": {"entity": "ferric enterobactin", "entity_type": "compound"},
                "transporters": [
                    {"entity": name, "entity_type": entity_type, "role": "transporter"}
                ],
            }
        )
    return {
        "entities": {
            "proteins": proteins,
            "protein_complexes": complexes,
            "compounds": [],
            "species": list(pathway_species or []),
        },
        "processes": {"reactions": [], "transports": transports},
    }


def _wrapper(payload: Dict[str, Any], name: str) -> Dict[str, Any]:
    rows = [
        row
        for row in payload["entities"]["protein_complexes"]
        if str(row.get("name") or "").casefold() == name.casefold()
    ]
    assert len(rows) == 1, f"expected exactly one wrapper named {name!r}, got {len(rows)}"
    row = rows[0]
    # Every case below is about a row the fallback actually built. If the
    # fallback silently skipped it, the species assertions would be vacuous.
    assert row.get("generated") is True
    assert row.get("generation_reason") == "single_protein_pathwhiz_wrapper"
    return row


def _sentinel(payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = [
        row
        for row in payload["entities"]["proteins"]
        if is_pathbank_unknown_protein(row)
    ]
    assert len(rows) == 1, f"expected exactly one sentinel protein row, got {len(rows)}"
    return rows[0]


def _note(row: Dict[str, Any]) -> Dict[str, Any]:
    return dict(row.get("mapping_meta", {}).get("species_preservation") or {})


def _assert_species_preserved(row: Dict[str, Any], ref: Dict[str, Any]) -> None:
    """The whole preservation contract, asserted together.

    Preserving one field and clobbering another is how the shipped contradiction
    was born, so every visible species field, the reference and the resolution
    provenance are checked in one place.
    """

    assert row["species"] == ref["name"]
    assert row["species_name"] == ref["name"]
    assert row["organism"] == ref["name"]
    assert row["taxonomy_id"] == ref["taxonomy_id"]
    assert row["species_id"] == ref["species_id"]
    assert row["pathbank_species_id"] == ref["pathbank_species_id"]
    assert row["species_ref"]["name"] == ref["name"]
    assert row["species_ref"]["source"] == ref["source"]
    assert row["species_ref"]["confidence"] == ref["confidence"]
    resolution = row["mapping_meta"]["species_resolution"]
    assert resolution["name"] == ref["name"]
    assert resolution["source"] == ref["source"]
    assert resolution["confidence"] == ref["confidence"]
    # Nothing anywhere on the row may carry the sentinel record's species.
    assert SENTINEL_SPECIES not in {
        row.get("species"),
        row.get("species_name"),
        row.get("organism"),
    }
    assert SENTINEL_SPECIES_ID not in {
        row.get("species_id"),
        row.get("pathbank_species_id"),
    }


def _assert_sentinel_species_applied(row: Dict[str, Any]) -> None:
    """Behaviour for a wrapper with nothing resolved -- unchanged from today."""

    assert row["species"] == SENTINEL_SPECIES
    assert row["organism"] == SENTINEL_SPECIES
    assert row["species_id"] == SENTINEL_SPECIES_ID
    assert row["pathbank_species_id"] == SENTINEL_SPECIES_ID


# ── 1. explicit entity species survives wrapper construction ────────────────


@pytest.mark.parametrize(
    "builder", [_enzyme_payload, _transport_payload], ids=["enzyme_site", "transporter_site"]
)
def test_explicit_entity_species_survives_wrapper_construction(builder: Any) -> None:
    payload = builder([("Enterobactin synthase", ECOLI_REF)])

    _apply_pathbank_unknown_enzyme_fallback(payload)

    row = _wrapper(payload, "Enterobactin synthase")
    assert row["species"] == "Escherichia coli"
    assert row["species"] != SENTINEL_SPECIES


# ── 2. name, taxonomy id and reference survive together, consistently ───────


@pytest.mark.parametrize(
    "builder", [_enzyme_payload, _transport_payload], ids=["enzyme_site", "transporter_site"]
)
def test_species_name_taxonomy_and_reference_survive_together(builder: Any) -> None:
    payload = builder([("Enterobactin synthase", ECOLI_REF)])

    _apply_pathbank_unknown_enzyme_fallback(payload)

    _assert_species_preserved(_wrapper(payload, "Enterobactin synthase"), ECOLI_REF)


# ── 3. the placeholder cannot clobber a stronger source-supported species ───


@pytest.mark.parametrize(
    "ref",
    [ECOLI_REF, YEAST_PATHWAY_REF, MOUSE_STATE_REF],
    ids=["explicit_entity_species", "single_pathway_species", "biological_state_species"],
)
@pytest.mark.parametrize(
    "builder", [_enzyme_payload, _transport_payload], ids=["enzyme_site", "transporter_site"]
)
def test_placeholder_species_cannot_clobber_a_source_supported_species(
    builder: Any, ref: Dict[str, Any]
) -> None:
    payload = builder([("Enterobactin synthase", ref)])

    _apply_pathbank_unknown_enzyme_fallback(payload)

    row = _wrapper(payload, "Enterobactin synthase")
    _assert_species_preserved(row, ref)
    assert _note(row)["decision"] == "resolved_species_preserved"
    assert _note(row)["resolution_source"] == ref["source"]


# ── 4. a wrapper with no resolved species retains CURRENT behaviour ─────────


@pytest.mark.parametrize(
    "builder", [_enzyme_payload, _transport_payload], ids=["enzyme_site", "transporter_site"]
)
@pytest.mark.parametrize("ref", [NOVEL_REF, None], ids=["novel_species", "never_hydrated"])
def test_wrapper_without_resolved_species_keeps_current_behaviour(
    builder: Any, ref: Optional[Dict[str, Any]]
) -> None:
    payload = builder([("HepPPS", ref)])

    _apply_pathbank_unknown_enzyme_fallback(payload)

    _assert_sentinel_species_applied(_wrapper(payload, "HepPPS"))


@pytest.mark.parametrize(
    "builder", [_enzyme_payload, _transport_payload], ids=["enzyme_site", "transporter_site"]
)
def test_llm_inferred_species_is_not_source_supported_and_is_clobbered(builder: Any) -> None:
    """``gap_resolver_llm`` is inference. Preserving it would launder a guess."""

    payload = builder([("HepPPS", LLM_REF)])

    _apply_pathbank_unknown_enzyme_fallback(payload)

    row = _wrapper(payload, "HepPPS")
    _assert_sentinel_species_applied(row)
    assert row["species"] != "Bacillus subtilis"
    assert _note(row)["reason"] == "species_source_is_inference_not_evidence"


# ── 5 & 6. no fallback to the requested organism or the export default ──────


@pytest.mark.parametrize(
    "builder", [_enzyme_payload, _transport_payload], ids=["enzyme_site", "transporter_site"]
)
@pytest.mark.parametrize("ref", [ECOLI_REF, NOVEL_REF], ids=["resolved", "unresolved"])
def test_no_fallback_to_the_requested_or_default_export_species(
    builder: Any, ref: Dict[str, Any]
) -> None:
    """The pathway's own organism is the requested organism AND, downstream, the
    exporter's ``default_species_id``. Neither may reach the wrapper."""

    payload = builder(
        [("Enterobactin synthase", ref)],
        pathway_species=[
            {
                "name": DEFAULT_EXPORT_SPECIES,
                "pathbank_species_id": DEFAULT_EXPORT_SPECIES_ID,
                "species_id": DEFAULT_EXPORT_SPECIES_ID,
            }
        ],
    )

    _apply_pathbank_unknown_enzyme_fallback(payload)

    row = _wrapper(payload, "Enterobactin synthase")
    assert DEFAULT_EXPORT_SPECIES not in {
        row.get("species"),
        row.get("species_name"),
        row.get("organism"),
    }
    assert DEFAULT_EXPORT_SPECIES_ID not in {
        row.get("species_id"),
        row.get("pathbank_species_id"),
    }
    if ref is ECOLI_REF:
        _assert_species_preserved(row, ECOLI_REF)
    else:
        _assert_sentinel_species_applied(row)


# ── 7. a mixed-organism case ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "builder", [_enzyme_payload, _transport_payload], ids=["enzyme_site", "transporter_site"]
)
def test_mixed_organism_case_resolves_each_wrapper_independently(builder: Any) -> None:
    payload = builder(
        [
            ("Enterobactin synthase", ECOLI_REF),
            ("ALAS2 homodimer", HUMAN_REF),
            ("HepPPS", NOVEL_REF),
        ]
    )

    _apply_pathbank_unknown_enzyme_fallback(payload)

    _assert_species_preserved(_wrapper(payload, "Enterobactin synthase"), ECOLI_REF)
    _assert_species_preserved(_wrapper(payload, "ALAS2 homodimer"), HUMAN_REF)
    _assert_sentinel_species_applied(_wrapper(payload, "HepPPS"))


# ── 8 & 9. the two pinned wrappers ──────────────────────────────────────────


def test_pinned_enterobactin_synthase_wrapper_keeps_escherichia_coli() -> None:
    payload = _enzyme_payload([("Enterobactin synthase", ECOLI_REF)])

    _apply_pathbank_unknown_enzyme_fallback(payload)

    row = _wrapper(payload, "Enterobactin synthase")
    _assert_species_preserved(row, ECOLI_REF)
    assert row["taxonomy_id"] == "562"


def test_pinned_alas2_homodimer_wrapper_keeps_homo_sapiens() -> None:
    """PMC12856317 shipped ``pathway.pwml`` with *Arabidopsis* on this human row."""

    payload = _enzyme_payload([("ALAS2 homodimer", HUMAN_REF)])

    _apply_pathbank_unknown_enzyme_fallback(payload)

    row = _wrapper(payload, "ALAS2 homodimer")
    _assert_species_preserved(row, HUMAN_REF)
    assert row["taxonomy_id"] == "9606"


# ── 10. contradictions are surfaced, never silently resolved ────────────────


def test_preserved_species_contradicting_the_placeholder_record_is_surfaced() -> None:
    payload = _enzyme_payload([("ALAS2 homodimer", HUMAN_REF)])

    _apply_pathbank_unknown_enzyme_fallback(payload)

    note = _note(_wrapper(payload, "ALAS2 homodimer"))
    kinds = {item["kind"] for item in note["contradictions"]}
    assert "preserved_species_differs_from_placeholder_record" in kinds
    assert note["placeholder_record"] == {
        "name": SENTINEL_SPECIES,
        "species_id": SENTINEL_SPECIES_ID,
    }
    assert note["resolved_species"] == "Homo sapiens"
    assert note["decision"] == "resolved_species_preserved"


def test_row_contradicting_its_own_resolution_is_surfaced_not_arbitrated() -> None:
    """The exact shape the defect shipped: a false *Arabidopsis* surface field
    beside internal *E. coli* evidence. No winner is picked silently."""

    row = _hydrated_complex("Enterobactin synthase", ECOLI_REF)
    row["species"] = SENTINEL_SPECIES
    row["species_id"] = SENTINEL_SPECIES_ID
    payload = _enzyme_payload([])
    payload["entities"]["protein_complexes"].append(row)
    payload["processes"]["reactions"].append(
        {
            "name": "enterobactin reaction",
            "enzymes": [
                {
                    "entity": "Enterobactin synthase",
                    "entity_type": "protein_complex",
                    "role": "catalyst",
                }
            ],
            "inputs": [],
            "outputs": [],
        }
    )

    _apply_pathbank_unknown_enzyme_fallback(payload)

    note = _note(_wrapper(payload, "Enterobactin synthase"))
    assert note["reason"] == "row_species_fields_disagree_with_resolution"
    disagreement = next(
        item
        for item in note["contradictions"]
        if item["kind"] == "row_species_fields_disagree_with_resolution"
    )
    assert disagreement["fields"]["species"] == SENTINEL_SPECIES
    assert disagreement["fields"]["species_id"] == SENTINEL_SPECIES_ID
    # Both sides stay on the record: the internal E. coli evidence is not erased.
    assert note["resolved_species"] == "Escherichia coli"
    assert _wrapper(payload, "Enterobactin synthase")["species_ref"]["name"] == "Escherichia coli"


# ── 11. the five sentinel protein rows are unchanged ────────────────────────


@pytest.mark.parametrize(
    "builder", [_enzyme_payload, _transport_payload], ids=["enzyme_site", "transporter_site"]
)
def test_all_five_sentinel_protein_rows_are_unchanged(builder: Any) -> None:
    """D-070 § O-1a: one sentinel per affected leg, and *Arabidopsis* is that
    record's own species. Asserted on the sentinel builders' output directly,
    across five legs whose wrappers carry five different species."""

    legs = [ECOLI_REF, HUMAN_REF, YEAST_PATHWAY_REF, MOUSE_STATE_REF, NOVEL_REF]
    for index, ref in enumerate(legs):
        payload = builder([(f"Wrapped enzyme {index}", ref)])

        _apply_pathbank_unknown_enzyme_fallback(payload)

        sentinel = _sentinel(payload)
        assert sentinel["name"] == PATHBANK_UNKNOWN_PROTEIN_NAME
        assert sentinel["species"] == SENTINEL_SPECIES
        assert sentinel["organism"] == SENTINEL_SPECIES
        assert sentinel["species_id"] == SENTINEL_SPECIES_ID
        assert sentinel["pathbank_species_id"] == SENTINEL_SPECIES_ID
        assert sentinel["taxonomy_id"] == SENTINEL_TAXONOMY_ID
        assert sentinel["pathbank_protein_id"] == PATHBANK_UNKNOWN_PROTEIN_ID
        assert sentinel["uniprot_id"] == PATHBANK_UNKNOWN_PROTEIN_UNIPROT
        assert sentinel["identity_status"] == "placeholder"
        # The wrapper's own species never leaks onto the shared record.
        assert "species_preservation" not in sentinel.get("mapping_meta", {})


# ── 12. non-vacuity: restore the clobber, prove preservation goes red ───────


@pytest.mark.parametrize(
    "builder", [_enzyme_payload, _transport_payload], ids=["enzyme_site", "transporter_site"]
)
def test_restoring_the_unconditional_clobber_turns_preservation_red(
    builder: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard is demonstrated against a case that can actually exercise it.

    ``raising=False`` is deliberate: with it this test is meaningful on a tree
    that has no guard at all, where it simply confirms the pre-C-099 behaviour
    the preservation cases are written against.
    """

    monkeypatch.setattr(
        map_ids,
        "_wrapper_species_fields",
        lambda complex_row: {
            "species": SENTINEL_SPECIES,
            "organism": SENTINEL_SPECIES,
            "species_id": SENTINEL_SPECIES_ID,
            "pathbank_species_id": SENTINEL_SPECIES_ID,
        },
        raising=False,
    )
    payload = builder([("Enterobactin synthase", ECOLI_REF)])

    _apply_pathbank_unknown_enzyme_fallback(payload)

    row = _wrapper(payload, "Enterobactin synthase")
    # The clobber is back, and it is exactly the shipped contradiction.
    assert row["species"] == SENTINEL_SPECIES
    assert row["species_name"] == "Escherichia coli"
    assert row["taxonomy_id"] == "562"
    # ...and every preservation assertion in this module now fails.
    with pytest.raises(AssertionError):
        _assert_species_preserved(row, ECOLI_REF)


# ── 13. preservation: valid wrapper biology is still serialized ─────────────


@pytest.mark.parametrize(
    "builder", [_enzyme_payload, _transport_payload], ids=["enzyme_site", "transporter_site"]
)
@pytest.mark.parametrize("ref", [ECOLI_REF, NOVEL_REF], ids=["resolved", "unresolved"])
def test_valid_wrapper_biology_is_still_serialized(
    builder: Any, ref: Dict[str, Any]
) -> None:
    """Nothing is stripped or suppressed: the wrapper is still in the payload,
    still a generated wrapper, still carries the sentinel component, and still
    satisfies the species precondition the strict PWML contract enforces."""

    payload = builder([("Enterobactin synthase", ref)])

    _apply_pathbank_unknown_enzyme_fallback(payload)

    row = _wrapper(payload, "Enterobactin synthase")
    assert is_generated_complex_wrapper(row) is True
    assert [component["name"] for component in row["components"]] == [
        PATHBANK_UNKNOWN_PROTEIN_NAME
    ]
    assert row["components"][0]["pathbank_protein_id"] == PATHBANK_UNKNOWN_PROTEIN_ID
    # ``protein_complex_missing_species`` is an ERROR in the strict contract;
    # this is the value that gate reads.
    assert protein_species_context(row)

    try:
        result = validate_required_pwml_contract(payload, strict_db=False)
        errors = list((result or {}).get("errors") or [])
    except GateValidationError as exc:
        errors = list(exc.details.get("errors") or [])
    offending = [
        error
        for error in errors
        if error.get("issue") == "protein_complex_missing_species"
        and error.get("entity_name") == "Enterobactin synthase"
    ]
    assert offending == []
