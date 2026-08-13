"""C-050 acceptance tests -- compound canonicalization runs BEFORE the freeze.

**G9 labelling, stated up front.** Everything in this file except
``test_canonical_graph_hash_moves_when_biology_moves_prefreeze`` is an
**explicitly labelled NEW acceptance test** for a new module
(``t2pw.pwml.prefreeze_resolution``) and a new capability. No base failure is
fabricated for them, because there is no pre-existing behaviour here to preserve.

The **correction** half of C-050 -- the five-category measurement that fails
behaviourally at ``bcc0bfe`` -- is not a pytest: it is
``docs/pwml_recovery_sprint/evidence/probe_c050_prefreeze_measurement.py``, whose
``--leg auto`` replicates whatever the enrichment->freeze region of the tree it
runs in does. At base that is "no pre-freeze resolution" and it exits 1 with
1/1/1/8/16; here it exits 0 with five zeros and ``RESULT: MEASURED``.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional

import pytest

from t2pw.pipeline.canonical_hash import canonical_graph_sha256, graph_projection
from t2pw.pwml.prefreeze_resolution import (
    PrefreezeResolutionError,
    resolve_compounds_prefreeze,
    run_prefreeze_resolution,
)


class _OfflineResolver:
    """A DB resolver that is reachable but reports itself unavailable.

    Passed explicitly so no test ever falls through to
    ``PathBankDbResolver.from_env()`` and opens a real connection.
    """

    last_error = "db_not_configured_in_test"

    def available(self) -> bool:
        return False


class _StubNameIndex:
    """Offline id -> canonical name index. ``chebi`` is the only key used here."""

    def __init__(self, by_chebi: Dict[str, Dict[str, Any]]) -> None:
        self._by_chebi = by_chebi
        self.queried: List[str] = []

    def compound_canonical(self, **ids: Any) -> Optional[Dict[str, Any]]:
        chebi = ids.get("chebi")
        self.queried.append(str(chebi))
        return self._by_chebi.get(str(chebi)) if chebi else None


def _compound(name: str, chebi: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "name": name,
        "mapping_meta": {"query": {"name": name}},
        **extra,
    }
    if chebi:
        row["chebi_id"] = chebi
    return row


def _payload() -> Dict[str, Any]:
    """One compound referenced from five places in four distinct roles.

    ``gly`` is a reactant (bare string), a product (``compound`` key), an
    enzyme-adjacent modifier (``entity`` key), transport cargo (``element`` key)
    and an interaction endpoint (scalar). A3 needs more than one role and more
    than one process; this is deliberately more than the minimum, because the
    reference table is the part that can silently miss a shape.
    """

    return {
        "entities": {
            "compounds": [
                _compound("gly", "CHEBI:15428"),
                _compound("succinyl-CoA"),
            ],
            "proteins": [_compound("ALAS2")],
        },
        "processes": {
            "reactions": [
                {
                    "name": "R1",
                    "inputs": ["gly", "succinyl-CoA"],
                    "outputs": [{"compound": "gly", "stoichiometry": 2}],
                    "modifiers": [{"entity": "gly", "role": "activator"}],
                    "enzymes": [{"entity": "ALAS2", "role": "catalyst"}],
                },
                {"name": "R2", "inputs": ["succinyl-CoA"], "outputs": ["gly"]},
            ],
            "transports": [
                {"name": "T1", "transport_elements": [{"element": "gly"}],
                 "transporters": [{"protein": "ALAS2"}]},
            ],
            "interactions": [{"left": "gly", "right": "ALAS2"}],
        },
    }


def _run(payload: Dict[str, Any], index: _StubNameIndex) -> Dict[str, Any]:
    return resolve_compounds_prefreeze(
        payload,
        db_resolver=_OfflineResolver(),
        strict_db=False,
        name_index=index,
    )


def _glycine_index() -> _StubNameIndex:
    return _StubNameIndex({"15428": {"id": 78, "name": "Glycine", "matched_on": "chebi"}})


def _all_strings(value: Any) -> List[str]:
    """Every string anywhere in ``value`` -- keys excluded, values included."""

    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for element in value for item in _all_strings(element)]
    if isinstance(value, dict):
        return [item for element in value.values() for item in _all_strings(element)]
    return []


# --------------------------------------------------------------------------
# A2 -- NEW ACCEPTANCE: an explicit, unambiguous rename map.
# --------------------------------------------------------------------------


def test_new_acceptance_rename_map_is_explicit_and_recorded() -> None:
    payload = _payload()
    summary = _run(payload, _glycine_index())

    assert summary["applied"] is True
    assert summary["rename_map"] == {"gly": "Glycine"}
    assert summary["renamed"] == 1
    assert payload["entities"]["compounds"][0]["name"] == "Glycine"
    # An unrenamed compound must not appear in the map at all -- "no implicit
    # renames" cuts both ways.
    assert "succinyl-CoA" not in summary["rename_map"]


# --------------------------------------------------------------------------
# A3 -- NEW ACCEPTANCE: atomic propagation to EVERY participant reference.
# --------------------------------------------------------------------------


def test_new_acceptance_every_participant_reference_is_propagated() -> None:
    payload = _payload()
    summary = _run(payload, _glycine_index())

    processes = payload["processes"]
    assert processes["reactions"][0]["inputs"] == ["Glycine", "succinyl-CoA"]
    assert processes["reactions"][0]["outputs"][0]["compound"] == "Glycine"
    assert processes["reactions"][0]["modifiers"][0]["entity"] == "Glycine"
    assert processes["reactions"][1]["outputs"] == ["Glycine"]
    assert processes["transports"][0]["transport_elements"][0]["element"] == "Glycine"
    assert processes["interactions"][0]["left"] == "Glycine"

    # Five references, five recorded updates, each with its pointer.
    pointers = {update["pointer"] for update in summary["references_updated"]}
    assert pointers == {
        "/processes/reactions/0/inputs/0",
        "/processes/reactions/0/outputs/0/compound",
        "/processes/reactions/0/modifiers/0/entity",
        "/processes/reactions/1/outputs/0",
        "/processes/transports/0/transport_elements/0/element",
        "/processes/interactions/0/left",
    }
    # And nothing anywhere still says the old name.
    assert "gly" not in _all_strings(payload["processes"])
    # References to other entities are untouched.
    assert processes["reactions"][0]["enzymes"][0]["entity"] == "ALAS2"
    assert processes["interactions"][0]["right"] == "ALAS2"


def test_new_acceptance_propagation_is_all_or_nothing() -> None:
    """A6: a failure must not leave a half-propagated payload behind."""

    payload = _payload()
    # A second compound already answers to the rename target, so redirecting
    # references to it would change which entity they resolve to.
    payload["entities"]["compounds"].append(_compound("Glycine"))
    original = deepcopy(payload)

    with pytest.raises(PrefreezeResolutionError) as excinfo:
        _run(payload, _glycine_index())

    assert excinfo.value.code == "PREFREEZE_CONNECTIVITY_BROKEN"
    assert payload == original, "the payload was mutated despite a named failure"


# --------------------------------------------------------------------------
# A4 -- NEW ACCEPTANCE: connectivity preserved, edges as well as counts.
# --------------------------------------------------------------------------


def test_new_acceptance_reaction_count_and_participant_edges_are_preserved() -> None:
    payload = _payload()
    before = deepcopy(payload["processes"])
    _run(payload, _glycine_index())
    after = payload["processes"]

    assert len(after["reactions"]) == len(before["reactions"]) == 2
    assert len(after["transports"]) == len(before["transports"]) == 1
    assert len(after["interactions"]) == len(before["interactions"]) == 1

    def edges(processes: Dict[str, Any], renamed: str) -> List[tuple]:
        """Edges keyed by role and position, with the renamed name folded back."""

        out: List[tuple] = []
        for index, reaction in enumerate(processes["reactions"]):
            for role in ("inputs", "outputs", "modifiers", "enzymes"):
                for position, member in enumerate(reaction.get(role) or []):
                    name = member if isinstance(member, str) else (
                        member.get("compound") or member.get("entity") or member.get("name")
                    )
                    stoich = member.get("stoichiometry") if isinstance(member, dict) else None
                    out.append((index, role, position, renamed if name == renamed else name, stoich))
        return out

    # The edge set is identical once the rename is folded back: same processes,
    # same roles, same positions, same stoichiometry, same partners.
    assert edges(after, "Glycine") == [
        (index, role, position, "Glycine" if name == "gly" else name, stoich)
        for index, role, position, name, stoich in edges(before, "gly")
    ]


# --------------------------------------------------------------------------
# A5 -- NEW ACCEPTANCE: the supported name survives as a synonym.
# --------------------------------------------------------------------------


def test_new_acceptance_original_name_is_preserved_as_a_synonym() -> None:
    payload = _payload()
    summary = _run(payload, _glycine_index())

    row = payload["entities"]["compounds"][0]
    assert "gly" in row["synonyms"]
    assert row["raw_name"] == "gly"
    assert summary["aliases_preserved"] == [{"name": "Glycine", "synonym": "gly"}]


def test_new_acceptance_a_pure_case_change_adds_no_redundant_synonym() -> None:
    """``glycine -> Glycine`` needs no synonym: name matching is case-folded."""

    payload = _payload()
    payload["entities"]["compounds"][0]["name"] = "glycine"
    payload["entities"]["compounds"][0]["mapping_meta"]["query"]["name"] = "glycine"
    payload["processes"]["reactions"][0]["inputs"][0] = "glycine"
    payload["processes"]["reactions"][0]["outputs"][0]["compound"] = "glycine"
    payload["processes"]["reactions"][0]["modifiers"][0]["entity"] = "glycine"
    payload["processes"]["reactions"][1]["outputs"][0] = "glycine"
    payload["processes"]["transports"][0]["transport_elements"][0]["element"] = "glycine"
    payload["processes"]["interactions"][0]["left"] = "glycine"

    summary = _run(payload, _glycine_index())
    assert summary["rename_map"] == {"glycine": "Glycine"}
    assert summary["aliases_preserved"] == []
    assert "synonyms" not in payload["entities"]["compounds"][0]


# --------------------------------------------------------------------------
# A6 -- NEW ACCEPTANCE: the failure paths, proven, not just the happy one.
# --------------------------------------------------------------------------


def test_new_acceptance_two_compounds_canonicalizing_to_one_name_is_fatal() -> None:
    payload = _payload()
    payload["entities"]["compounds"][1] = _compound("aminoacetic acid", "CHEBI:99999")
    payload["processes"]["reactions"][0]["inputs"][1] = "aminoacetic acid"
    payload["processes"]["reactions"][1]["inputs"][0] = "aminoacetic acid"
    original = deepcopy(payload)

    index = _StubNameIndex({
        "15428": {"id": 78, "name": "Glycine", "matched_on": "chebi"},
        "99999": {"id": 78, "name": "Glycine", "matched_on": "chebi"},
    })
    with pytest.raises(PrefreezeResolutionError) as excinfo:
        _run(payload, index)

    assert excinfo.value.code == "AMBIGUOUS_RENAME_TARGET"
    assert sorted(excinfo.value.details["sources"]) == ["aminoacetic acid", "gly"]
    assert payload == original


def test_new_acceptance_a_name_shared_with_another_entity_is_fatal() -> None:
    """The rename source is also a protein's name: references cannot be split."""

    payload = _payload()
    payload["entities"]["proteins"].append(_compound("gly"))
    original = deepcopy(payload)

    with pytest.raises(PrefreezeResolutionError) as excinfo:
        _run(payload, _glycine_index())

    assert excinfo.value.code == "AMBIGUOUS_REFERENCE"
    assert "proteins#1" in excinfo.value.details["conflicting_entities"]
    assert payload == original


def test_new_acceptance_a_failure_is_never_silently_skipped() -> None:
    """``run_prefreeze_resolution`` propagates; it does not swallow and continue."""

    payload = _payload()
    payload["entities"]["proteins"].append(_compound("gly"))
    with pytest.raises(PrefreezeResolutionError):
        run_prefreeze_resolution(
            payload,
            db_resolver=_OfflineResolver(),
            name_index=_glycine_index(),
        )


# --------------------------------------------------------------------------
# A7 -- NEW ACCEPTANCE: nothing invented.
# --------------------------------------------------------------------------


def test_new_acceptance_a_compound_with_no_id_hit_gains_no_identity() -> None:
    payload = _payload()
    _run(payload, _glycine_index())

    row = payload["entities"]["compounds"][1]  # succinyl-CoA: no chebi, no index hit
    assert row["name"] == "succinyl-CoA"
    assert "db_row" not in row
    assert "pathwhiz_id" not in row
    assert "pathbank_compound_id" not in row
    assert row["db_status"] == "unmatched"


def test_new_acceptance_the_db_identity_projection_invents_no_value() -> None:
    """``pathwhiz_id`` is only ever a projection of an id the row already carried."""

    payload = _payload()
    payload["entities"]["compounds"][1]["pathbank_compound_id"] = 808
    summary = _run(payload, _glycine_index())

    row = payload["entities"]["compounds"][1]
    assert row["pathwhiz_id"] == 808
    assert row["pathbank_compound_id"] == 808
    # 78 is NOT projected: it lives in ``db_row``, which ``ir._entity_record``
    # does not read either. Projecting it would be inventing an identity the
    # exporter never claimed.
    assert {entry["value"] for entry in summary["identity_projected"]} == {808}
    assert "pathwhiz_id" not in payload["entities"]["compounds"][0]


# --------------------------------------------------------------------------
# A8 / A9 -- NEW ACCEPTANCE: resolution is finished, not merely started.
# --------------------------------------------------------------------------


def test_new_acceptance_resolution_reaches_its_fixed_point_before_returning() -> None:
    payload = _payload()
    summary = _run(payload, _glycine_index())
    assert summary["resolution_passes"] >= 2, "one pass is not a fixed point"

    # Re-running the whole operation is a no-op: nothing left for an exporter.
    frozen = deepcopy(payload)
    again = _run(payload, _glycine_index())
    assert again["rename_map"] == {}
    assert payload == frozen


def test_new_acceptance_no_lookup_is_deferred_past_this_call() -> None:
    """A8: the resolution a later stage would perform has already been performed.

    The claim is about *this* stage's end state, not about anyone else -- so it
    is stated as "a repeat of the same resolution finds nothing left to do",
    which stays true when C-051 removes the downstream call.
    """

    payload = _payload()
    index = _glycine_index()
    _run(payload, index)
    assert "15428" in index.queried

    second = _StubNameIndex({"15428": {"id": 78, "name": "Glycine", "matched_on": "chebi"}})
    frozen = deepcopy(payload)
    summary = _run(payload, second)
    # The resolved compound is never looked up again: it already carries the
    # canonical name and the db_row that produced it.
    assert "15428" not in second.queried
    assert summary["rename_map"] == {}
    assert payload == frozen


# --------------------------------------------------------------------------
# A12 -- NEW ACCEPTANCE: one object, mutated in place, no stale alias.
# --------------------------------------------------------------------------


def test_new_acceptance_the_caller_s_object_is_the_resolved_object() -> None:
    payload = _payload()
    entities = payload["entities"]
    processes = payload["processes"]
    compounds = entities["compounds"]
    reaction = processes["reactions"][0]

    run_prefreeze_resolution(
        payload,
        db_resolver=_OfflineResolver(),
        name_index=_glycine_index(),
    )

    # Every container a downstream consumer could already be holding is the same
    # object it was, and it holds resolved content.
    assert payload["entities"] is entities
    assert payload["processes"] is processes
    assert entities["compounds"] is compounds
    assert processes["reactions"][0] is reaction
    assert compounds[0]["name"] == "Glycine"
    assert reaction["inputs"][0] == "Glycine"


def test_new_acceptance_an_empty_or_compoundless_payload_is_left_alone() -> None:
    for payload in ({}, {"entities": {"compounds": []}, "processes": {}}):
        original = deepcopy(payload)
        summary = resolve_compounds_prefreeze(payload, db_resolver=_OfflineResolver())
        assert summary["applied"] is False
        assert summary["skipped_reason"]
        assert payload == original


# --------------------------------------------------------------------------
# A11 -- hashing. C-030's projection is asserted, never modified.
# --------------------------------------------------------------------------


def test_canonical_graph_hash_moves_when_biology_moves_prefreeze() -> None:
    payload = _payload()
    before = canonical_graph_sha256(deepcopy(payload))
    _run(payload, _glycine_index())
    after = canonical_graph_sha256(payload)

    # The point of the card: the rename is inside the graph hash because it
    # happened before the freeze, where PRODUCT_CONTRACT section 5 requires it.
    assert before != after
    # And the projection is still an allowlist -- the fields resolution attaches
    # for provenance are not silently admitted into the graph hash.
    projection = graph_projection(payload)
    compound = projection["entities"]["compounds"][0]
    assert "db_match" not in compound
    assert compound["name"] == "Glycine"
