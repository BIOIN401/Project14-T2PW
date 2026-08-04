"""A substitution that renames an entity must repoint the references naming it.

THE DEFECT, from ``runs/2026-08-02_2130/papers/PMC12856317/strict/final_mapped.json``::

    entities.proteins          : ['Unknown']          uniprot="Unknown", pathbank_protein_id=9659
    entities.protein_complexes : ['ALAS2 homodimer']  components=['Unknown']
    processes.interactions     : 4 rows, every entity_2 == 'ALAS2'

    GATE ERROR /processes | Registry validation failed:
      /processes/interactions/{0,1,2,3}/entity_2 unknown entity: ALAS2

TRACED AND REPRODUCED, not assumed. The writer is
``_rewrite_reaction_protein_enzymes_to_complexes`` (``map_ids.py:6010``). That
leg's merged payload gives the ALAS2 reaction *two* enzyme actors --
``{protein: ALAS2}`` and ``{protein_complex: ALAS2 homodimer}`` -- so
``resolved_protein_count`` counts the complex as resolved and the rule "drop an
unresolved protein actor when a resolved one remains in the same list" deletes
``ALAS2`` from ``entities.proteins``. Complex components have had a repair pass
since ``map_ids.py:7178``; interactions, reaction inputs/outputs and transport
cargo were only ever read.

Two readings that look right and are not, recorded so they are not re-derived:
it is NOT the Unknown fallback skipping a check (its
``_has_disqualifying_reference`` guard does read interactions, does retain a
protein they reference, and never sees ``ALAS2`` because the actor is already
gone); and ``ALAS2`` is neither renamed nor quarantined, which is why the
artifacts alone cannot answer "what removed it?".

``test_the_dropped_enzyme_actors_references_are_repaired`` runs the real
normalizer and the real ``map_payload`` over that leg's own
``merged_payload.json`` with the resolvers stubbed unmapped, so the defect and
its repair are measured rather than described.

THE CORRECT TRANSFORMATION::

    protein entity      : ALAS2 -> Unknown
    complex component   : ALAS2 -> Unknown
    functional process  : ALAS2 -> ALAS2 homodimer      (the wrapper, NOT the sentinel)

The last line is the one this file exists for. ``Unknown`` is a single shared
row: send ``heme inhibits ALAS2`` there and the claim no longer names which
unresolved actor it meant, and silently merges with every other actor backed by
the same sentinel.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.process_normalizer import validate_registry_references  # noqa: E402
from t2pw.pipeline.reference_repair import (  # noqa: E402
    ReferenceRepairError,
    ReplacementRecord,
    repair_entity_references,
)


SENTINEL = "Unknown"
REASON = "pathbank_unknown_protein_fallback"


def _alas2_payload() -> dict:
    """The real shape: the protein gone, the wrapper holding the sentinel, and
    four interactions still naming the protein."""

    return {
        "entities": {
            "compounds": [
                {"name": "glycine"},
                {"name": "succinyl-CoA"},
                {"name": "5-aminolevulinate"},
                {"name": "heme"},
                {"name": "hemin"},
            ],
            "proteins": [
                {
                    "name": SENTINEL,
                    "uniprot_id": SENTINEL,
                    "pathbank_protein_id": 9659,
                    "mapping_meta": {"chosen_rule": REASON},
                }
            ],
            "protein_complexes": [
                {
                    "name": "ALAS2 homodimer",
                    "generated": True,
                    "components": [{"name": SENTINEL, "pathbank_protein_id": 9659}],
                }
            ],
        },
        "processes": {
            "reactions": [
                {
                    "name": "ALAS2 reaction",
                    "inputs": ["glycine", "succinyl-CoA"],
                    "outputs": ["5-aminolevulinate"],
                    "enzymes": [
                        {"entity": "ALAS2 homodimer", "entity_type": "protein_complex"}
                    ],
                }
            ],
            "transports": [],
            "interactions": [
                {
                    "entity_1": "heme",
                    "entity_2": "ALAS2",
                    "name": "heme inhibits ALAS2",
                    "relationship": "inhibits",
                },
                {
                    "entity_1": "heme",
                    "entity_2": "ALAS2",
                    "name": "heme binding to ALAS2",
                    "relationship": "binds_to",
                },
                {
                    "entity_1": "hemin",
                    "entity_2": "ALAS2",
                    "name": "hemin binds ALAS2",
                    "relationship": "binds_to",
                },
                {
                    "entity_1": "hemin",
                    "entity_2": "ALAS2",
                    "name": "hemin inhibits ALAS2",
                    "relationship": "inhibits",
                },
            ],
        },
    }


def _alas2_record(**overrides) -> ReplacementRecord:
    values = {
        "old_name": "ALAS2",
        "sentinel_name": SENTINEL,
        "functional_replacement": "ALAS2 homodimer",
        "reason": REASON,
    }
    values.update(overrides)
    return ReplacementRecord(**values)


# ── The headline case ──────────────────────────────────────────────────────


def test_the_alas2_interactions_are_repointed_at_the_wrapper() -> None:
    payload = _alas2_payload()

    with pytest.raises(ValueError, match="unknown entity: ALAS2"):
        validate_registry_references(payload)

    report = repair_entity_references(payload, [_alas2_record()])

    assert [row["entity_2"] for row in payload["processes"]["interactions"]] == [
        "ALAS2 homodimer"
    ] * 4
    assert report["summary"]["references_rewritten"] == 4
    # And the payload the gate refused now passes it.
    validate_registry_references(payload)


def test_a_functional_reference_is_never_repointed_at_the_sentinel() -> None:
    """The doctrine, stated as a test.

    ``Unknown`` is one shared row. Sending a process reference there does not
    merely lose which actor was meant -- with two unresolved proteins backed by
    the same sentinel it silently merges two distinct claims into one.
    """

    payload = _alas2_payload()

    repair_entity_references(payload, [_alas2_record()])

    entity_2_values = {row["entity_2"] for row in payload["processes"]["interactions"]}
    assert SENTINEL not in entity_2_values


def test_the_substitution_event_is_recorded_with_every_field() -> None:
    payload = _alas2_payload()

    report = repair_entity_references(payload, [_alas2_record()])

    first = report["rewrites"][0]
    assert first["json_pointer"] == "/processes/interactions/0/entity_2"
    assert first["old_value"] == "ALAS2"
    assert first["replacement_value"] == "ALAS2 homodimer"
    assert first["replacement_reason"] == REASON


# ── Several sentinel-backed actors ─────────────────────────────────────────


def test_two_unresolved_proteins_sharing_the_sentinel_keep_separate_wrappers() -> None:
    """The case that makes "never point at Unknown" load-bearing rather than tidy."""

    payload = {
        "entities": {
            "compounds": [{"name": "heme"}],
            "proteins": [{"name": SENTINEL, "pathbank_protein_id": 9659}],
            "protein_complexes": [
                {"name": "ALAS2 homodimer", "components": [{"name": SENTINEL}]},
                {"name": "FECH homodimer", "components": [{"name": SENTINEL}]},
            ],
        },
        "processes": {
            "interactions": [
                {"entity_1": "heme", "entity_2": "ALAS2"},
                {"entity_1": "heme", "entity_2": "FECH"},
            ]
        },
    }

    repair_entity_references(
        payload,
        [
            _alas2_record(),
            _alas2_record(old_name="FECH", functional_replacement="FECH homodimer"),
        ],
    )

    assert [row["entity_2"] for row in payload["processes"]["interactions"]] == [
        "ALAS2 homodimer",
        "FECH homodimer",
    ]
    validate_registry_references(payload)


def test_an_unresolved_protein_with_no_wrapper_is_left_alone() -> None:
    """Not repaired, and deliberately not sent to the sentinel either.

    Leaving the reference dangling makes strict validation fail on it clearly --
    "this actor is unresolved" -- which is a better outcome than a payload that
    validates while meaning something it does not say.
    """

    payload = _alas2_payload()
    before = deepcopy(payload)

    report = repair_entity_references(
        payload, [_alas2_record(functional_replacement="")]
    )

    assert payload == before
    assert report["summary"]["references_rewritten"] == 0
    with pytest.raises(ValueError, match="unknown entity: ALAS2"):
        validate_registry_references(payload)


def test_a_resolved_real_protein_produces_no_rewrite_at_all() -> None:
    """No substitution, no repair. The pass must be inert on a healthy payload."""

    payload = {
        "entities": {
            "compounds": [{"name": "heme"}],
            "proteins": [{"name": "ALAS2", "uniprot_id": "P22557"}],
            "protein_complexes": [],
        },
        "processes": {"interactions": [{"entity_1": "heme", "entity_2": "ALAS2"}]},
    }
    before = deepcopy(payload)

    report = repair_entity_references(payload, [])

    assert payload == before
    assert report["rewrites"] == []


# ── Every field the registry validator reads ───────────────────────────────


def test_reaction_actor_and_participant_references_are_repaired() -> None:
    payload = {
        "entities": {
            "compounds": [{"name": "substrate"}, {"name": "product"}],
            "proteins": [{"name": SENTINEL}],
            "protein_complexes": [
                {"name": "LpxK holoenzyme", "components": [{"name": SENTINEL}]}
            ],
        },
        "processes": {
            "reactions": [
                {
                    "name": "r",
                    "inputs": ["substrate", "LpxK"],
                    "outputs": ["product"],
                    "enzymes": [{"entity": "LpxK"}],
                    "modifiers": [{"protein": "LpxK", "role": "catalyst"}],
                }
            ]
        },
    }

    repair_entity_references(
        payload,
        [_alas2_record(old_name="LpxK", functional_replacement="LpxK holoenzyme")],
    )

    reaction = payload["processes"]["reactions"][0]
    assert reaction["inputs"] == ["substrate", "LpxK holoenzyme"]
    assert reaction["enzymes"][0]["entity"] == "LpxK holoenzyme"
    # Written back to the key it was READ from: an actor spelled ``protein``
    # must not acquire a second, conflicting ``entity``.
    assert reaction["modifiers"][0]["protein"] == "LpxK holoenzyme"
    assert "entity" not in reaction["modifiers"][0]
    validate_registry_references(payload)


def test_transport_cargo_and_transporter_references_are_repaired() -> None:
    payload = {
        "entities": {
            "compounds": [],
            "proteins": [{"name": SENTINEL}],
            "protein_complexes": [
                {"name": "SFXN1 dimer", "components": [{"name": SENTINEL}]},
                {"name": "SFXN3 dimer", "components": [{"name": SENTINEL}]},
            ],
        },
        "processes": {
            "transports": [
                {"cargo": "SFXN1", "transporters": [{"entity": "SFXN3"}]},
                {"cargo_complex": "SFXN1", "cargo": "ignored", "transporters": []},
            ]
        },
    }

    repair_entity_references(
        payload,
        [
            _alas2_record(old_name="SFXN1", functional_replacement="SFXN1 dimer"),
            _alas2_record(old_name="SFXN3", functional_replacement="SFXN3 dimer"),
        ],
    )

    transports = payload["processes"]["transports"]
    assert transports[0]["cargo"] == "SFXN1 dimer"
    assert transports[0]["transporters"][0]["entity"] == "SFXN3 dimer"
    # ``cargo_complex`` wins when present, exactly as the validator reads it, so
    # the shadowed ``cargo`` is left untouched rather than double-written.
    assert transports[1]["cargo_complex"] == "SFXN1 dimer"
    assert transports[1]["cargo"] == "ignored"


@pytest.mark.parametrize(
    ("left_key", "right_key"),
    [("left", "right"), ("source", "target"), ("entity_1", "entity_2")],
)
def test_the_interaction_alias_spellings_are_all_repaired(
    left_key: str, right_key: str
) -> None:
    """A pass that rewrites ``entity_1`` while the payload carries ``left``
    validates clean and repairs nothing: the validator reads whichever alias is
    present, so the repair must write whichever alias is present."""

    payload = {
        "entities": {
            "compounds": [{"name": "heme"}],
            "proteins": [{"name": SENTINEL}],
            "protein_complexes": [
                {"name": "ALAS2 homodimer", "components": [{"name": SENTINEL}]}
            ],
        },
        "processes": {"interactions": [{left_key: "ALAS2", right_key: "heme"}]},
    }

    repair_entity_references(payload, [_alas2_record()])

    assert payload["processes"]["interactions"][0][left_key] == "ALAS2 homodimer"
    validate_registry_references(payload)


# ── Idempotence and refusal ────────────────────────────────────────────────


def test_running_the_repair_twice_changes_nothing() -> None:
    payload = _alas2_payload()

    repair_entity_references(payload, [_alas2_record()])
    once = deepcopy(payload)
    second = repair_entity_references(payload, [_alas2_record()])

    assert payload == once
    assert second["rewrites"] == []


def test_two_records_disagreeing_about_one_name_fail_rather_than_guess() -> None:
    payload = _alas2_payload()
    payload["entities"]["protein_complexes"].append(
        {"name": "ALAS2 tetramer", "components": [{"name": SENTINEL}]}
    )

    with pytest.raises(ReferenceRepairError, match="ambiguous replacement"):
        repair_entity_references(
            payload,
            [_alas2_record(), _alas2_record(functional_replacement="ALAS2 tetramer")],
        )


def test_a_replacement_that_names_no_registry_entry_fails() -> None:
    """Trading one unknown reference for another is not a repair."""

    payload = _alas2_payload()

    with pytest.raises(ReferenceRepairError, match="names 0 registry entries"):
        repair_entity_references(
            payload, [_alas2_record(functional_replacement="ALAS2 octamer")]
        )


def test_a_replacement_that_names_two_registry_entries_fails() -> None:
    payload = _alas2_payload()
    payload["entities"]["compounds"].append({"name": "ALAS2 homodimer"})

    with pytest.raises(ReferenceRepairError, match="names 2 registry entries"):
        repair_entity_references(payload, [_alas2_record()])


def test_a_replacement_that_is_the_sentinel_itself_fails() -> None:
    payload = _alas2_payload()

    with pytest.raises(ReferenceRepairError, match="refusing to repoint"):
        repair_entity_references(
            payload, [_alas2_record(functional_replacement=SENTINEL)]
        )


def test_a_wrapper_that_kept_the_protein_name_needs_no_rewrite() -> None:
    """The direct-wrap case: the generated wrapper takes the protein's own name,
    so a reference to it already resolves -- to the complex."""

    payload = {
        "entities": {
            "compounds": [{"name": "heme"}],
            "proteins": [{"name": SENTINEL}],
            "protein_complexes": [{"name": "ALAS2", "components": [{"name": SENTINEL}]}],
        },
        "processes": {"interactions": [{"entity_1": "heme", "entity_2": "ALAS2"}]},
    }
    before = deepcopy(payload)

    report = repair_entity_references(
        payload, [_alas2_record(functional_replacement="ALAS2")]
    )

    assert payload == before
    assert report["rewrites"] == []
    validate_registry_references(payload)


#: The archived leg the defect was measured on. Its ``merged_payload.json`` is
#: pre-mapping, so the whole failure reproduces offline from it.
ARCHIVED_LEG = (
    ROOT / "runs" / "2026-08-02_2130" / "papers" / "PMC12856317" / "strict" / "merged_payload.json"
)


def _map_offline(payload: dict, tmp_path: Path) -> dict:
    """The real ``map_payload``, with every resolver stubbed "unmapped".

    Stubbing the resolvers rather than the passes is what makes this a
    reproduction: every structural pass -- the enzyme->complex conversion that
    drops the protein, the Unknown fallback, the structural cleanup -- runs
    exactly as it does in production. Only the network and the database are
    replaced, and "nothing matched" is the state this leg was actually in.
    """

    from unittest.mock import patch  # noqa: PLC0415

    from t2pw.mapping.map_ids import map_payload  # noqa: PLC0415

    unmapped = {
        "status": "unmapped",
        "reason": "no_match",
        "provider": "UniProt",
        "source": "api",
        "confidence": 0.0,
        "candidates": [],
        "resolution": {"status": "unresolved", "issue": "no_match", "order_step": "api_uniprot"},
    }
    with (
        patch(
            "t2pw.mapping.map_ids.hydrate_species_references",
            return_value={"hydrated": 0, "matched": 0, "novel": 0},
        ),
        patch("t2pw.mapping.map_ids._map_protein_with_strategy", return_value=unmapped),
        patch("t2pw.mapping.map_ids.map_protein_uniprot", return_value=unmapped),
        patch("t2pw.mapping.map_ids._map_compound_with_strategy", return_value=unmapped),
    ):
        return map_payload(
            payload,
            cache_path=tmp_path / "mapping-cache.json",
            id_source="api",
            use_cache=False,
        )


@pytest.mark.skipif(not ARCHIVED_LEG.exists(), reason="archived leg not present")
def test_the_dropped_enzyme_actors_references_are_repaired(tmp_path: Path) -> None:
    """The real leg, end to end: normalize then map, resolvers stubbed unmapped.

    Verified to reproduce the defect exactly before the repair existed --
    ``proteins: ['Unknown']``, ``components: ['Unknown']``, and all four
    interactions still reading ``entity_2: 'ALAS2'``.
    """

    import json  # noqa: PLC0415

    from t2pw.pipeline.process_normalizer import normalize_process_payload  # noqa: PLC0415

    merged = json.loads(ARCHIVED_LEG.read_text(encoding="utf-8"))
    normalized, _report = normalize_process_payload(merged)

    result = _map_offline(normalized, tmp_path)
    mapped = result["payload"]

    # The substitution: the protein row is gone and the wrapper holds the shared
    # sentinel. That part was never in dispute and is unchanged.
    assert [row["name"] for row in mapped["entities"]["proteins"]] == [SENTINEL]
    wrapper = mapped["entities"]["protein_complexes"][0]
    assert wrapper["name"] == "ALAS2 homodimer"
    assert [component["name"] for component in wrapper["components"]] == [SENTINEL]

    # The half that was missing: the four interactions follow the FUNCTION into
    # the wrapper. Not left dangling, and not sent to the shared sentinel.
    assert [row["entity_2"] for row in mapped["processes"]["interactions"]] == [
        "ALAS2 homodimer"
    ] * 4
    validate_registry_references(mapped)

    conversion = result["report"]["enzyme_complex_conversion"]
    assert conversion["summary"]["dropped_protein_process_references_repaired"] == 4
    repairs = [
        action
        for action in conversion["actions"]
        if action["type"] == "dropped_protein_process_reference_repaired"
    ]
    assert [row["json_pointer"] for row in repairs] == [
        f"/processes/interactions/{index}/entity_2" for index in range(4)
    ]
    assert {row["old_value"] for row in repairs} == {"ALAS2"}
    assert {row["replacement_value"] for row in repairs} == {"ALAS2 homodimer"}
    assert {row["replacement_reason"] for row in repairs} == {
        "reaction_enzyme_protein_dropped_no_external_id"
    }


@pytest.mark.skipif(not ARCHIVED_LEG.exists(), reason="archived leg not present")
def test_the_repaired_leg_clears_the_gate_error_it_used_to_fail_on(tmp_path: Path) -> None:
    """The verdict, not just the mechanism.

    ``gate_fail_report.json`` for this leg records exactly one blocking finding:
    ``/processes | Registry validation failed: /processes/interactions/{0,1,2,3}
    /entity_2 unknown entity: ALAS2``. It is gone.
    """

    import json  # noqa: PLC0415

    from t2pw.pipeline.process_normalizer import (  # noqa: PLC0415
        GateValidationError,
        normalize_process_payload,
        run_strict_post_normalization_gates,
    )

    merged = json.loads(ARCHIVED_LEG.read_text(encoding="utf-8"))
    normalized, _report = normalize_process_payload(merged)
    mapped = _map_offline(normalized, tmp_path)["payload"]

    try:
        run_strict_post_normalization_gates(mapped, enforce_all_proteins_connected=True)
        remaining = []
    except GateValidationError as exc:
        remaining = [
            str(error.get("reason") or "")
            for error in (getattr(exc, "details", {}) or {}).get("errors", [])
        ]

    assert not [reason for reason in remaining if "unknown entity: ALAS2" in reason], remaining


def test_a_nucleic_acid_wrapper_is_a_legitimate_target() -> None:
    """``entities.nucleic_acids`` is in the validator's registry, so a rewrite
    that lands there is valid rather than an unknown name."""

    payload = {
        "entities": {
            "compounds": [{"name": "heme"}],
            "proteins": [{"name": SENTINEL}],
            "protein_complexes": [],
            "nucleic_acids": [{"name": "pmrHFIJKLM operon"}],
        },
        "processes": {"interactions": [{"entity_1": "heme", "entity_2": "pmrH"}]},
    }

    repair_entity_references(
        payload,
        [_alas2_record(old_name="pmrH", functional_replacement="pmrHFIJKLM operon")],
    )

    assert payload["processes"]["interactions"][0]["entity_2"] == "pmrHFIJKLM operon"
    validate_registry_references(payload)
