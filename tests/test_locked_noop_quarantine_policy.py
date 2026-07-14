from __future__ import annotations

from copy import deepcopy

import pytest

from t2pw.mapping.map_ids import _apply_pathbank_unknown_enzyme_fallback
from t2pw.pipeline.process_normalizer import (
    GateValidationError,
    dedupe_processes,
    drop_process_orphan_proteins,
    normalize_process_payload,
    prune_disconnected_proteins,
    run_strict_post_normalization_gates,
)
from t2pw.pipeline.reaction_preservation_validator import validate_reaction_preservation


def _base_payload() -> dict:
    return {
        "metadata": {"organism": "Test organism"},
        "entities": {
            "compounds": [{"name": "A"}, {"name": "B"}],
            "proteins": [],
            "protein_complexes": [],
            "species": [{"name": "Test organism"}],
        },
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }


def test_klpd_locked_same_label_reaction_is_quarantined_and_orphan_is_pruned() -> None:
    payload = _base_payload()
    payload["entities"]["compounds"] = [{"name": "klebsazolicin"}]
    payload["entities"]["proteins"] = [{"name": "KlpD", "species": "Test organism"}]
    reaction = {
        "locked_reaction_id": "rxn_lock_009",
        "locked": True,
        "essential": True,
        "name": "KlpD-catalysed azoline and macroamide formation",
        "inputs": ["klebsazolicin"],
        "outputs": ["klebsazolicin"],
        "enzymes": [{"entity": "KlpD", "entity_type": "protein", "role": "catalyst"}],
        "evidence": (
            "bifunctional YcaO protein, KlpD, was characterized in the biosynthesis "
            "of the ribosome inhibitor klebsazolicin"
        ),
    }
    payload["processes"]["reactions"] = [reaction]
    manifest = [deepcopy(reaction)]

    normalized, report = normalize_process_payload(payload)

    assert normalized["processes"]["reactions"] == []
    assert normalized["entities"]["proteins"] == []
    assert len(normalized["quarantined_locked_reactions"]) == 1
    quarantine = normalized["quarantined_locked_reactions"][0]
    assert quarantine["locked_reaction_id"] == "rxn_lock_009"
    assert quarantine["reaction_name"] == "KlpD-catalysed azoline and macroamide formation"
    assert quarantine["reason"] == "coarse_grained_same_entity_transformation"
    assert quarantine["action"] == "quarantined_from_active_reactions"
    assert quarantine["json_pointer"] == "/processes/reactions/0"
    assert quarantine["original_reaction"]["essential"] is True
    assert quarantine["original_reaction"]["evidence"] == reaction["evidence"]
    assert report["summary"]["locked_no_op_quarantined_count"] == 1
    assert report["summary"]["orphan_proteins_dropped"] == 1
    preservation = validate_reaction_preservation(
        normalized,
        manifest,
        stage="after_normalization",
    )
    assert preservation["locked_reaction_count"] == 1
    assert preservation["preserved_count"] == 0
    assert preservation["quarantined_count"] == 1
    assert preservation["missing_count"] == 0


def test_unlocked_unsupported_noop_is_dropped_without_quarantine() -> None:
    payload = _base_payload()
    payload["processes"]["reactions"] = [
        {"name": "unsupported self-loop", "inputs": ["A"], "outputs": ["A"]}
    ]
    report = {"summary": {}, "actions": []}

    dedupe_processes(payload, report=report)

    assert payload["processes"]["reactions"] == []
    assert payload.get("quarantined_locked_reactions") is None
    assert report["summary"]["unsupported_no_op_dropped_count"] == 1
    assert report["actions"][0]["reason"] == "unsupported_no_op"


def test_evidenced_output_subset_is_quarantined_with_stable_generic_reason() -> None:
    payload = _base_payload()
    payload["processes"]["reactions"] = [
        {
            "name": "coarse multi-substrate claim",
            "inputs": ["A", "B"],
            "outputs": ["A"],
            "evidence": "The paper reports a transformation but does not identify a distinct product.",
            "enzymes": [
                "legacy enzyme actor",
                {"entity": "second enzyme", "entity_type": "protein", "role": "catalyst"},
            ],
        }
    ]
    report = {"summary": {}, "actions": []}

    dedupe_processes(payload, report=report)

    assert payload["processes"]["reactions"] == []
    quarantine = payload["quarantined_locked_reactions"][0]
    assert quarantine["reason"] == "output_subset_of_input_without_distinct_product"
    assert quarantine["original_reaction"]["enzymes"] == [
        "legacy enzyme actor",
        {"entity": "second enzyme", "entity_type": "protein", "role": "catalyst"},
    ]
    assert "locked_reaction_id" not in quarantine
    assert report["summary"]["evidenced_no_op_quarantined_count"] == 1


def test_cleanup_after_quarantine_keeps_every_surviving_process_reference() -> None:
    payload = _base_payload()
    payload["entities"]["proteins"] = [
        {"name": name, "species": "Test organism"}
        for name in ("orphan", "reaction enzyme", "interaction protein", "transporter", "complex member")
    ]
    payload["entities"]["protein_complexes"] = [
        {
            "name": "surviving complex",
            "components": [{"name": "complex member", "stoichiometry": 1}],
        }
    ]
    payload["processes"] = {
        "reactions": [
            {
                "name": "rejected",
                "inputs": ["A"],
                "outputs": ["A"],
                "evidence": "orphan catalyses an unresolved coarse transformation",
                "enzymes": [{"entity": "orphan", "entity_type": "protein", "role": "catalyst"}],
            },
            {
                "name": "valid",
                "inputs": ["A"],
                "outputs": ["B"],
                "enzymes": [
                    {"entity": "reaction enzyme", "entity_type": "protein", "role": "catalyst"},
                    {"entity": "surviving complex", "entity_type": "protein_complex", "role": "catalyst"},
                ],
            },
        ],
        "transports": [
            {
                "cargo": "A",
                "from_biological_state": "inside",
                "to_biological_state": "outside",
                "transporters": [{"entity": "transporter", "entity_type": "protein"}],
            }
        ],
        "interactions": [
            {"entity_1": "interaction protein", "entity_2": "A", "relationship": "binds"}
        ],
    }
    report = {"summary": {}, "actions": []}

    dedupe_processes(payload, report=report)
    drop_process_orphan_proteins(payload, report=report)
    prune_disconnected_proteins(payload, report=report)

    names = {row["name"] for row in payload["entities"]["proteins"]}
    assert "orphan" not in names
    assert names == {"reaction enzyme", "interaction protein", "transporter", "complex member"}


def test_valid_surviving_unmapped_enzyme_remains_eligible_for_stage6_unknown_fallback() -> None:
    payload = _base_payload()
    payload["entities"]["proteins"] = [{"name": "KlpD", "species": "Test organism"}]
    payload["processes"]["reactions"] = [
        {
            "name": "source-supported KlpD transformation",
            "inputs": ["A"],
            "outputs": ["B"],
            "enzymes": [{"entity": "KlpD", "entity_type": "protein", "role": "catalyst"}],
            "evidence": "KlpD converts the identified precursor A into product B.",
        }
    ]

    normalized, _report = normalize_process_payload(payload)
    fallback_report = _apply_pathbank_unknown_enzyme_fallback(normalized)

    assert len(normalized["processes"]["reactions"]) == 1
    actor = normalized["processes"]["reactions"][0]["enzymes"][0]
    assert actor["entity"] == "KlpD"
    assert actor["entity_type"] == "protein_complex"
    assert actor["role"] == "catalyst"
    assert normalized["entities"]["protein_complexes"][0]["components"][0]["name"] == "Unknown"
    assert normalized["entities"]["proteins"][0]["pathbank_protein_id"] == 9659
    assert fallback_report["summary"]["reaction_enzyme_unknown_fallbacks"] == 1


def test_distinct_locked_duplicates_are_accounted_as_active_plus_quarantined() -> None:
    payload = _base_payload()
    payload["processes"]["reactions"] = [
        {"locked_reaction_id": "rxn_lock_001", "name": "duplicate", "inputs": ["A"], "outputs": ["B"]},
        {"locked_reaction_id": "rxn_lock_002", "name": "duplicate", "inputs": ["A"], "outputs": ["B"]},
    ]
    manifest = deepcopy(payload["processes"]["reactions"])
    report = {"summary": {}, "actions": []}

    dedupe_processes(payload, report=report)

    assert len(payload["processes"]["reactions"]) == 1
    assert len(payload["quarantined_locked_reactions"]) == 1
    assert payload["quarantined_locked_reactions"][0]["reason"] == "duplicate_locked_reaction"
    assert payload["locked_reaction_filter_report"] == {
        "locked_reactions_found": 2,
        "exported_locked_reactions": 1,
        "quarantined_locked_reactions": 1,
        "unaccounted_locked_reactions": 0,
    }
    preservation = validate_reaction_preservation(payload, manifest, stage="after_normalization")
    assert preservation["preserved_count"] + preservation["quarantined_count"] == 2
    assert preservation["missing_count"] == 0


def test_strict_gate_blocks_unaccounted_locks_at_stable_pointer() -> None:
    payload = _base_payload()
    payload["locked_reaction_filter_report"] = {
        "locked_reactions_found": 2,
        "exported_locked_reactions": 1,
        "quarantined_locked_reactions": 0,
        "unaccounted_locked_reactions": 1,
    }

    try:
        run_strict_post_normalization_gates(payload)
    except GateValidationError as exc:
        errors = exc.details["errors"]
    else:  # pragma: no cover - the assertion below documents a hard boundary
        raise AssertionError("unaccounted locked reaction did not fail the strict gate")

    assert {
        "path": "/locked_reaction_filter_report/unaccounted_locked_reactions",
        "reason": (
            "Locked reaction accounting failed: 1 locked reaction(s) are neither active nor quarantined."
        ),
    } in errors


def test_strict_gate_accepts_active_plus_quarantined_lock_accounting() -> None:
    payload = _base_payload()
    payload["locked_reaction_filter_report"] = {
        "locked_reactions_found": 2,
        "exported_locked_reactions": 1,
        "quarantined_locked_reactions": 1,
        "unaccounted_locked_reactions": 0,
    }
    payload["quarantined_locked_reactions"] = [
        {
            "locked_reaction_id": "rxn_lock_002",
            "reason": "coarse_grained_same_entity_transformation",
            "original_reaction": {"inputs": ["A"], "outputs": ["A"]},
        }
    ]

    details = run_strict_post_normalization_gates(payload)

    assert details["errors"] == []


@pytest.mark.parametrize("bad_value", [-1, True, 1.5, "1", None])
def test_strict_gate_rejects_malformed_lock_accounting(bad_value: object) -> None:
    payload = _base_payload()
    payload["locked_reaction_filter_report"] = {
        "unaccounted_locked_reactions": bad_value,
    }

    with pytest.raises(GateValidationError) as exc_info:
        run_strict_post_normalization_gates(payload)

    assert exc_info.value.details["errors"][0]["path"] == (
        "/locked_reaction_filter_report/unaccounted_locked_reactions"
    )
    assert "must be a non-negative integer" in exc_info.value.details["errors"][0]["reason"]
