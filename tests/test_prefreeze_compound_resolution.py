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
from t2pw.pwml import compound_resolution
from t2pw.pwml.compound_resolution import _resolve_compound_rows, ensure_resolution_report
from t2pw.pwml.name_index import default_name_index
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
    # These four carry A7 on their own: with no id hit and no reachable DB, no
    # identity is attached.
    assert "db_row" not in row
    assert "pathwhiz_id" not in row
    assert "pathbank_compound_id" not in row
    # This one does NOT. ``unmatched`` here is the recorded consequence of
    # ``_OfflineResolver`` reporting itself unavailable -- it is a property of
    # the stub, not of the product, and on production wiring against a reachable
    # PathBank DB this same row could legitimately record something else. What
    # it does pin, post-B2, is that the status the FIRST pass recorded is the one
    # that survives; before the fix the fixed-point loop overwrote it.
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


def test_new_acceptance_a_rename_propagates_into_element_locations() -> None:
    """B1: ``element_locations`` is a name-keyed reference section too.

    ``canonical._parse_json`` reads ``compound_locations[].compound`` as an entity
    reference and ``compound`` is inside the graph-hash allowlist, so a location
    left holding the pre-rename name dangles -- and the dangling name is hashed
    into ``canonical_graph_sha256``. The walk used to visit ``processes`` only,
    which also meant ``_assert_fully_propagated`` could not see its own output.
    """

    payload = _payload()
    payload["element_locations"] = {
        "compound_locations": [
            {"compound": "gly", "biological_state": "cytosol"},
            {"compound": "succinyl-CoA", "biological_state": "cytosol"},
            {"entity": "gly", "biological_state": "mitochondrion"},
        ],
        "protein_locations": [{"protein": "ALAS2", "biological_state": "cytosol"}],
    }
    summary = _run(payload, _glycine_index())

    locations = payload["element_locations"]["compound_locations"]
    assert [row.get("compound") or row.get("entity") for row in locations] == [
        "Glycine", "succinyl-CoA", "Glycine",
    ]
    # An entity of another kind is not touched by a compound rename.
    assert payload["element_locations"]["protein_locations"][0]["protein"] == "ALAS2"

    # Recorded with a pointer, like every other propagated reference.
    pointers = {update["pointer"] for update in summary["references_updated"]}
    assert "/element_locations/compound_locations/0/compound" in pointers
    assert "/element_locations/compound_locations/2/entity" in pointers

    # And the direct statement of the defect: nothing anywhere in the frozen
    # sections still names a compound that no longer exists.
    names = {row["name"] for row in payload["entities"]["compounds"]}
    referenced = {row.get("compound") or row.get("entity") for row in locations}
    assert referenced <= names, f"dangling location references: {referenced - names}"


def test_new_acceptance_a_dangling_location_cannot_survive_the_commit() -> None:
    """B1, from the other side: the module can now SEE a dangling location.

    ``_assert_fully_propagated`` reads the world through the same generator as
    the propagation, so it used to be structurally incapable of catching one.
    """

    from t2pw.pwml.prefreeze_resolution import _assert_fully_propagated, _iter_refs

    payload = _payload()
    payload["element_locations"] = {
        "compound_locations": [{"compound": "gly", "biological_state": "cytosol"}]
    }
    _run(payload, _glycine_index())
    _assert_fully_propagated(payload, {"gly": "Glycine"})  # does not raise

    pointer = "/element_locations/compound_locations/0/compound"
    payload["element_locations"]["compound_locations"][0]["compound"] = "gly"
    with pytest.raises(PrefreezeResolutionError) as excinfo:
        _assert_fully_propagated(payload, {"gly": "Glycine"})
    assert excinfo.value.code == "PREFREEZE_RENAME_NOT_PROPAGATED"
    assert excinfo.value.details["pointer"] == pointer
    assert any(ref.pointer == pointer for ref in _iter_refs(payload))


def test_new_acceptance_a_compound_rename_cannot_rewrite_another_kind_s_location() -> None:
    """DEF-3: ``_propagate`` matches on the name string alone. Is that a hazard?

    ``_iter_refs`` walks all four ``element_locations`` buckets and ``_propagate``
    rewrites purely by name, so on the face of it a compound rename ``X -> Y``
    also rewrites a protein, nucleic-acid or element-collection location row named
    ``X``. **Measured: it cannot, whenever that row names a different entity.**
    Two independent mechanisms already foreclose it, and neither is new here:

    1. the other kind carries ``X`` in a primary alias field --
       ``_reject_ambiguous_renames`` finds it as a rogue owner in
       ``_alias_index``'s primary index and raises ``AMBIGUOUS_REFERENCE`` before
       anything is written;
    2. it carries ``X`` only as a synonym -- the primary index does not see it,
       but ``_connectivity_signature`` resolves location refs through the synonym
       index, so the ref moves from ``<<compounds#0|proteins#1>>`` to
       ``<<compounds#0>>`` and ``PREFREEZE_CONNECTIVITY_BROKEN`` raises.

    The residual case, recorded rather than hidden (arm 3): when **no** entity but
    the compound answers to ``X``, the row is rewritten -- and that is correct, not
    corruption. Name resolution is kind-blind in ``canonical`` and ``ir`` too, so
    the row resolved to this compound before the rename and resolves to the same
    compound after; skipping it would leave a name no entity carries, which
    ``_assert_fully_propagated`` raises on by design. Pinning it here means a
    future kind filter cannot introduce that failure silently.
    """

    kinds = {
        "protein_locations": ("proteins", "protein"),
        "nucleic_acid_locations": ("nucleic_acids", "nucleic_acid"),
        "element_collection_locations": ("element_collections", "element_collection"),
    }

    def _staged() -> Dict[str, Any]:
        payload = _payload()
        payload["element_locations"] = {
            "compound_locations": [{"compound": "gly", "biological_state": "cytosol"}],
            **{bucket: [{field: "gly", "biological_state": "cytosol"}]
               for bucket, (_, field) in kinds.items()},
        }
        return payload

    for bucket, (entity_bucket, field) in kinds.items():
        payload = _staged()
        payload["entities"].setdefault(entity_bucket, []).append(_compound("gly"))
        original = deepcopy(payload)
        with pytest.raises(PrefreezeResolutionError) as excinfo:
            _run(payload, _glycine_index())
        assert excinfo.value.code == "AMBIGUOUS_REFERENCE", bucket
        assert payload == original, bucket
        assert payload["element_locations"][bucket][0][field] == "gly", bucket

    payload = _staged()
    payload["entities"]["proteins"].append(_compound("Glycine receptor", synonyms=["gly"]))
    original = deepcopy(payload)
    with pytest.raises(PrefreezeResolutionError) as excinfo:
        _run(payload, _glycine_index())
    assert excinfo.value.code == "PREFREEZE_CONNECTIVITY_BROKEN"
    assert payload == original

    payload = _staged()
    assert _run(payload, _glycine_index())["applied"] is True
    for bucket, (_, field) in kinds.items():
        assert payload["element_locations"][bucket][0][field] == "Glycine", bucket


# --------------------------------------------------------------------------
# B2 -- NEW ACCEPTANCE: the loop settles identity, it does not invent a decision.
# --------------------------------------------------------------------------


class _SubThresholdCompoundResolver:
    """``PathWhizCompoundResolver``'s measured live answer for OPDA, without a DB.

    ``status='matched'``, ``chosen_rule='fuzzy_name'``, ``confidence=0.65`` --
    below C-040's own ``>= 0.85`` gate and applied anyway by the fall-through.
    Whether it *should* apply is escalated separately (B3); this fake pins what
    the payload RECORDS about it, which is this card's business.
    """

    OPDA_ROW = {"id": 104723, "name": "Dinor-12-oxo-phytodienoate"}

    def __init__(self, db_resolver: Any) -> None:
        self.db_resolver = db_resolver

    def resolve(self, row: Dict[str, Any]) -> Dict[str, Any]:
        name = str(row.get("name") or "")
        if name != "OPDA":
            return {"status": "unmatched", "raw_name": name, "chosen": None,
                    "candidates": [], "chosen_rule": "", "confidence": 0.0,
                    "reason": "No PathWhiz DB match by IDs or name"}
        return {"status": "matched", "raw_name": name, "chosen": dict(self.OPDA_ROW),
                "candidates": [], "chosen_rule": "fuzzy_name", "confidence": 0.65}


class _AvailableDbResolver:
    last_error = ""

    def available(self) -> bool:
        return True


def _opda_payload() -> Dict[str, Any]:
    return {
        "entities": {"compounds": [_compound("OPDA"), _compound("OPC-8:0")]},
        "element_locations": {
            "compound_locations": [
                {"compound": "OPDA", "biological_state": "cytosol"},
                {"compound": "OPC-8:0", "biological_state": "cytosol"},
            ]
        },
        "processes": {
            "reactions": [
                {"name": "peripheral OPDA reduction", "inputs": ["OPDA"], "outputs": ["OPC-8:0"]}
            ]
        },
    }


def _fuzzy(monkeypatch: pytest.MonkeyPatch, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve ``payload`` against the sub-threshold OPDA answer, no DB needed."""

    monkeypatch.setattr(compound_resolution, "PathWhizCompoundResolver", _SubThresholdCompoundResolver)
    return resolve_compounds_prefreeze(
        payload, db_resolver=_AvailableDbResolver(), strict_db=False, name_index=None)


def test_new_acceptance_the_rule_that_chose_the_identity_survives_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C-040a's refusal is now the observed behaviour; this case pins the refusal.

    **This fixture no longer demonstrates provenance drift, and is not offered as
    if it did.** It was written when a sub-threshold ``fuzzy_name`` hit was applied
    anyway: pass 1 stamped a ``pathwhiz_id``, pass 2 read it back through the
    legacy-id branch and restated the decision as ``legacy_id_unverified`` at 0.85
    with the rule that chose it erased. **C-040a** (D-028 rule 1) now refuses a
    fuzzy match outright -- nothing is renamed, no identifier is stamped, and pass
    2 has nothing to read back -- so this row now observes the same values with and
    without the provenance mechanism. The discriminating case moved to the
    offline-index path, which D-028's admission does not gate; it is the test
    directly below.

    What this pins instead is D-028's refusal arriving intact at the pre-freeze
    caller. It is deliberately NOT "repaired" by lowering a threshold, raising a
    confidence or reshaping the fixture until the match is admitted again: that
    would weaken a biological gate to obtain a green test.
    """

    payload = _opda_payload()
    summary = _fuzzy(monkeypatch, payload)
    row = payload["entities"]["compounds"][0]

    assert summary["resolution_passes"] >= 2
    assert (row["db_status"], row["chosen_rule"], row["confidence"]) == (
        "identity_refused_review_required", "fuzzy_name", 0.65)
    # D-028 rule 5: record-only is no rename AND no identifier stamp, never a
    # partial apply -- so the location keeps the extracted name too.
    assert row["name"] == "OPDA"
    assert "pathwhiz_id" not in row
    assert summary["rename_map"] == {}
    assert payload["element_locations"]["compound_locations"][0]["compound"] == "OPDA"

    # The invariant is still asserted -- it just no longer discriminates here.
    # One pass is what the exporter runs, so one pass is what the provenance must
    # say. Computed rather than hard-coded, so it survives a change to the rules.
    single = _resolve_compound_rows(
        _opda_payload()["entities"]["compounds"], db_resolver=_AvailableDbResolver(),
        strict_db=False, report=ensure_resolution_report({}),
        pointer_prefix="/entities/compounds", name_index=None)
    fields = ("db_status", "chosen_rule", "confidence")
    assert [{f: r.get(f) for f in fields} for r in payload["entities"]["compounds"]] == [
        {f: r.get(f) for f in fields} for r in single]

    # And a re-run must not re-decide either: a row arriving with a ``db_status``
    # was decided by an earlier call, and this one has nothing to add to it.
    frozen = deepcopy(payload)
    _fuzzy(monkeypatch, payload)
    assert payload == frozen


def test_new_acceptance_the_offline_index_decision_survives_convergence() -> None:
    """B-4 replacement: the discriminating provenance case, on the ungated path.

    The offline name-index path is not subject to D-028's DB-match admission, so
    the drift the snapshot/restore mechanism exists to prevent is still observable
    there -- in the other direction. Pass 1 records
    ``matched_offline_name_index``; pass 2 finds no ``pathwhiz_id`` to read back,
    re-consults an unavailable resolver and overwrites the status with
    ``unmatched``, while ``_canonicalize_compound_offline`` returns early on the
    ``db_row`` pass 1 attached and so cannot restore it.

    **Behavioural G9 contrast, measured, everything but the module held at the
    tip:** pre-correction (``768be75``, before ``e7f28e7`` added
    ``_snapshot_provenance``/``_authoritative_provenance``/``_restore_provenance``)
    this same row ends ``unmatched``; here it ends
    ``matched_offline_name_index``. Reproduce with
    ``evidence/probe_c050e_offline_provenance.py``; result committed as
    ``evidence/c050e_offline_provenance.json``.
    """

    payload = _payload()
    summary = _run(payload, _glycine_index())
    row = payload["entities"]["compounds"][0]

    assert summary["resolution_passes"] >= 2, "one pass would not exercise the drift"
    assert row["db_status"] == "matched_offline_name_index"
    # The identity the loop settled on is untouched: only the account of who
    # decided it is restored.
    assert row["name"] == "Glycine"
    assert summary["rename_map"] == {"gly": "Glycine"}

    # Computed, not hard-coded: what one pass recorded is what the converged
    # operation must record. This is the assertion the pre-correction module
    # fails, and it covers the unmatched row beside the matched one.
    single = _resolve_compound_rows(
        _payload()["entities"]["compounds"], db_resolver=_OfflineResolver(),
        strict_db=False, report=ensure_resolution_report({}),
        pointer_prefix="/entities/compounds", name_index=_glycine_index())
    fields = ("db_status", "chosen_rule", "confidence")
    assert [{f: r.get(f) for f in fields} for r in payload["entities"]["compounds"]] == [
        {f: r.get(f) for f in fields} for r in single]


def test_new_acceptance_a_repeat_pass_is_not_a_second_consultation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The report may not claim the resolver failed a row more often than it did.

    Every pass appends one entry per row to ``db_resolution.compounds``,
    ``unresolved.db_identities`` and the issue lists, so an unfixed three-pass
    convergence over two rows records six consultations of a resolver that made
    two decisions."""

    payload = _opda_payload()
    summary = _fuzzy(monkeypatch, payload)
    report = summary["resolution_report"]

    assert summary["resolution_passes"] >= 2
    assert len(report["db_resolution"]["compounds"]) == 2
    # Both rows are unresolved: OPC-8:0 outright, OPDA because 0.65 is below
    # C-040's >= 0.85 gate -- the sub-threshold apply that C-040a owns under
    # D-028 and that this card deliberately leaves alone. One entry each.
    assert len(report["unresolved"]["db_identities"]) == 2
    assert len(report.get("warnings") or []) == 2
    assert [i["chosen_rule"] for i in report["unresolved"]["db_identities"]] == ["fuzzy_name", ""]


# --------------------------------------------------------------------------
# D5 -- NEW ACCEPTANCE: the wiring production actually uses.
# --------------------------------------------------------------------------


def test_new_acceptance_the_production_wiring_holds_its_invariants() -> None:
    """``db_resolver=None`` and the real ``default_name_index()``.

    Every other test here supplies ``_OfflineResolver`` and a stub index.
    ``streamlit_app`` supplies neither: ``db_resolver=None`` makes
    ``_resolve_compound_rows`` construct ``PathBankDbResolver.from_env()``. That
    gap is why B1 and B2 were invisible to this suite.

    The assertions hold on **both** a machine with a reachable PathBank DB and one
    without -- that is the difference between the environments this branch has
    been measured in, and a test that held on only one of them is exactly how a
    vacuously clean measurement gets recorded as a result.
    """

    payload = _opda_payload()
    payload["entities"]["compounds"].append(_compound("L-glutamate"))
    payload["element_locations"]["compound_locations"].append(
        {"compound": "L-glutamate", "biological_state": "cytosol"})
    before_rows = deepcopy(payload["entities"]["compounds"])

    summary = resolve_compounds_prefreeze(
        payload, db_resolver=None, strict_db=False, name_index=default_name_index())
    assert summary["applied"] is True

    # 1. No dangling reference, whatever the DB decided to rename.
    rows = payload["entities"]["compounds"]
    reachable = {row["name"] for row in rows}
    for row in rows:
        reachable.update(str(value) for value in row.get("synonyms") or [])
        reachable.add(str(row.get("raw_name") or ""))
    for location in payload["element_locations"]["compound_locations"]:
        assert location["compound"] in reachable, location
    for name in payload["processes"]["reactions"][0]["inputs"]:
        assert name in reachable, name

    # 2. The recorded decision is a single pass's, not the loop's re-observation.
    single = _resolve_compound_rows(
        before_rows, db_resolver=None, strict_db=False,
        report=ensure_resolution_report({}), pointer_prefix="/entities/compounds",
        name_index=default_name_index())
    fields = ("db_status", "chosen_rule", "confidence")
    assert [{f: r.get(f) for f in fields} for r in rows] == [
        {f: r.get(f) for f in fields} for r in single]

    # 3. Say which environment produced the run, so a clean result can never be
    #    read as clean-because-nothing-was-consulted.
    assert "available" in summary["resolution_report"]["db_resolution"]


# --------------------------------------------------------------------------
# D2 -- NEW ACCEPTANCE: ``ok`` is a verdict; an unreachable DB is review, not death.
# --------------------------------------------------------------------------


def _unresolvable_payload() -> Dict[str, Any]:
    return {
        "entities": {"compounds": [_compound("a compound no database carries")]},
        "processes": {"reactions": [{"name": "R", "inputs": ["a compound no database carries"]}]},
    }


def test_new_acceptance_a_failed_resolution_report_is_not_reported_as_ok() -> None:
    """``report["ok"]`` used to be set True before anything ran and never fell, so
    a nested report carrying ``ok=False`` and error-severity
    ``compound_db_resolution_failed`` entries came back as a success -- the record
    saying something the run did not establish."""

    def _failed(payload: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {"applied": True, "resolution_report": {"ok": False}}

    report = run_prefreeze_resolution({}, canonicalizers=(("compounds", _failed),))
    assert report["ok"] is False
    assert report["failures"] == {"compounds": "resolution_report_not_ok"}
    # A resolver that WAS consulted and rejected the row is not review-deferred.
    assert report["review_required"] == {}

    # A skip that is not a clean no-op falsifies it too -- for EVERY registered
    # canonicalizer, not only the first. C-045 registered ``species`` beside
    # ``compounds`` (D-016), and a verdict that named only one of them would be
    # the same partial record this test exists to forbid.
    skipped = run_prefreeze_resolution([], db_resolver=_OfflineResolver(), name_index=None)
    assert skipped["ok"] is False
    assert skipped["failures"] == {
        "compounds": "payload_not_a_mapping",
        "species": "payload_not_a_mapping",
    }

    # A payload with nothing to canonicalize is still a clean run.
    clean = run_prefreeze_resolution(
        {"entities": {"compounds": []}}, db_resolver=_OfflineResolver())
    assert clean["ok"] is True
    assert clean["failures"] == {}


def test_new_acceptance_an_unreachable_db_is_review_required_not_fatal() -> None:
    """An identity the DB was never reachable to establish is incomplete, not wrong.

    ``_resolve_compound_rows`` records its failures at severity ``error`` under
    ``strict_db``, so with no PathBank DB a defect-free payload arrives here with
    ``ok=False``. Merge rule 7 keeps that as ``review_required``; stopping on it
    would turn a database outage into a total export failure, and D-015 clause 6
    is about ambiguous or dangling *references*, which this is not.
    """

    report = run_prefreeze_resolution(
        _unresolvable_payload(), strict_db=True,
        db_resolver=_OfflineResolver(), name_index=None)

    assert report["ok"] is False, "it must still not claim success"
    assert report["failures"] == {"compounds": "resolution_report_not_ok:db_unavailable"}
    assert report["review_required"] == report["failures"]


def test_new_acceptance_a_structural_failure_still_raises_under_the_same_call() -> None:
    """The half of D-015 clause 6 that IS a stop, proven beside the half that is not.

    Same entry point, same ``strict_db=True``: an ambiguous reference raises where
    an unreachable database only reports, so the softer verdict above cannot be
    read as this module having stopped failing visibly.
    """

    payload = _payload()
    payload["entities"]["proteins"].append(_compound("gly"))
    with pytest.raises(PrefreezeResolutionError) as excinfo:
        run_prefreeze_resolution(
            payload, strict_db=True,
            db_resolver=_OfflineResolver(), name_index=_glycine_index())
    assert excinfo.value.code == "AMBIGUOUS_REFERENCE"


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


# --------------------------------------------------------------------------
# C-050f -- the rewrite rule and the audit rule are the same rule.
# CORRECTION of pre-existing behaviour (G9); the base-vs-tip measurement is
# ``evidence/probe_c050f_propagation_match_rule.py`` (base a81b1d65) and these
# are the durable guards. ``PREFREEZE_RENAME_MAP_COLLISION`` is the one **new**
# capability, labelled as such, so it carries no base failure.
# --------------------------------------------------------------------------


def _variant_payload(entity: str, chebi: str, reference: str) -> Dict[str, Any]:
    return {
        "entities": {"compounds": [_compound(entity, chebi)]},
        "processes": {"reactions": [{"name": "R1", "inputs": [reference], "outputs": []}]},
    }


@pytest.mark.parametrize(
    "entity,chebi,reference,expected",
    [
        ("gly", "15428", "gly", "Glycine"),                                  # control
        ("gly", "15428", "GLY", "Glycine"),                                  # case
        ("gly", "15428", "Gly", "Glycine"),                                  # case
        ("succinyl-CoA", "15380", "succinyl CoA", "Succinyl coenzyme A"),    # punctuation
        ("glycine", "15428", "glycine", "Glycine"),                          # pure case change
        ("glycine", "15428", "GLYCINE", "Glycine"),                          # ... its variant
    ],
)
def test_c050f_a_variant_reference_is_rewritten_not_aborted_on(
    entity: str, chebi: str, reference: str, expected: str,
) -> None:
    """A1/A2: every spelling that resolves to the renamed entity is propagated.

    None of these is dangling -- each resolves, by the same ``_norm`` rule
    ``_alias_index`` and ``ir.resolve_entity`` use, to the very entity being
    renamed -- so ``PRODUCT_CONTRACT`` section 1 forbids aborting an export on
    it. Asserting on the **live** payload covers the committed pass too, and the
    rename map pins that widening the match did not widen the rename (A5).
    """

    payload = _variant_payload(entity, f"CHEBI:{chebi}", reference)
    index = _StubNameIndex({chebi: {"id": 78, "name": expected, "matched_on": "chebi"}})

    report = run_prefreeze_resolution(
        payload, db_resolver=_OfflineResolver(), strict_db=False, name_index=index)

    assert payload["processes"]["reactions"][0]["inputs"] == [expected]
    assert payload["entities"]["compounds"][0]["name"] == expected
    assert report["compounds"]["rename_map"] == {entity: expected}


def test_c050f_a_genuinely_stale_reference_still_raises() -> None:
    """A3: the rewriter widened; the audit did not narrow. D-015 clause 6 holds.

    Each stale spelling names something no entity carries once the row is
    ``Glycine``, so each stays fatal. The third is the one the base could not see
    at all: after a pure case change the audit skipped the rename entirely.
    """

    from t2pw.pwml.prefreeze_resolution import _assert_fully_propagated

    for entity, stale, rename in (
        ("gly", "gly", {"gly": "Glycine"}),
        ("gly", "GLY", {"gly": "Glycine"}),
        ("glycine", "GLYCINE", {"glycine": "Glycine"}),
    ):
        payload = _variant_payload(entity, "CHEBI:15428", entity)
        _run(payload, _glycine_index())
        _assert_fully_propagated(payload, rename)  # the propagated payload is fine

        payload["processes"]["reactions"][0]["inputs"][0] = stale
        with pytest.raises(PrefreezeResolutionError) as excinfo:
            _assert_fully_propagated(payload, rename)
        assert excinfo.value.code == "PREFREEZE_RENAME_NOT_PROPAGATED", stale
        assert excinfo.value.details["pointer"] == "/processes/reactions/0/inputs/0"


def test_c050f_a_rename_map_colliding_under_norm_is_refused_not_guessed() -> None:
    """NEW acceptance (3b): one rename source must not silently shadow another.

    ``gly`` and ``Gly`` are one key to everything that reads the frozen payload,
    so a reference spelled either way cannot be rewritten to two targets.
    Fail-closed, like ``_reject_ambiguous_renames`` -- which already refuses this
    through the real entry point one stage earlier, so the guard protects the
    next caller of ``_propagate`` (C-045's species map), not a hole in this one.
    """

    from t2pw.pwml.prefreeze_resolution import _propagate

    payload = _variant_payload("gly", "CHEBI:15428", "gly")
    with pytest.raises(PrefreezeResolutionError) as excinfo:
        _propagate(payload, {"gly": "Glycine", "Gly": "Glycinate"})
    assert excinfo.value.code == "PREFREEZE_RENAME_MAP_COLLISION"
    assert payload["processes"]["reactions"][0]["inputs"] == ["gly"], "nothing was written"

    # Two spellings of one source agreeing on one target is not a collision.
    payload = _variant_payload("gly", "CHEBI:15428", "GLY")
    _propagate(payload, {"gly": "Glycine", "Gly": "Glycine"})
    assert payload["processes"]["reactions"][0]["inputs"] == ["Glycine"]


def test_c050f_a_widened_match_still_stops_at_a_cross_kind_collision() -> None:
    """A4: C-050e's synonym-only gate, reached by a VARIANT spelling for the first
    time. At the base this aborted on the un-rewritten ``GLY`` and never got
    there; reaching ``PREFREEZE_CONNECTIVITY_BROKEN`` proves the variant was
    rewritten and that the gate behind propagation is not weakened.
    """

    payload = _variant_payload("gly", "CHEBI:15428", "GLY")
    payload["entities"]["proteins"] = [_compound("Glycine receptor", synonyms=["GLY"])]
    original = deepcopy(payload)

    with pytest.raises(PrefreezeResolutionError) as excinfo:
        _run(payload, _glycine_index())
    assert excinfo.value.code == "PREFREEZE_CONNECTIVITY_BROKEN"
    assert payload == original


@pytest.mark.parametrize("name", ["---", "α", "-", "??"])
def test_c050f_an_empty_norm_rename_source_is_matched_not_discarded(name: str) -> None:
    """B-1 (REV-050f): a name normalizing to "" must not fall out of BOTH sets.

    ``_norm`` keeps only ``[a-z0-9:+ ]``, so these normalize to ``""``. Round 1
    discarded such keys from the rewriter and the audit at once, so the row was
    renamed while its reference was left behind: ``applied: True`` with a
    reference resolving to nothing on reload (``PRODUCT_CONTRACT`` section 5,
    non-atomic under D-015 clause 3). Two assertions, because neither alone
    catches the round-1 shape: the reference is **matched and rewritten**, and
    end to end the payload is still **refused**.
    """

    from t2pw.pwml.prefreeze_resolution import _propagate

    direct = _variant_payload(name, "CHEBI:15428", name)
    assert _propagate(direct, {name: "Glycine"}), "the reference was not matched"
    assert direct["processes"]["reactions"][0]["inputs"] == ["Glycine"]

    payload = _variant_payload(name, "CHEBI:15428", name)
    original = deepcopy(payload)
    with pytest.raises(PrefreezeResolutionError) as excinfo:
        _run(payload, _glycine_index())
    assert excinfo.value.code == "PREFREEZE_CONNECTIVITY_BROKEN"
    assert payload == original, "nothing may be committed on a refusal"


def test_c050f_the_rewrite_set_and_the_detection_set_are_one_set() -> None:
    """The invariant B-1 broke, asserted directly rather than by example.

    Both key on ``_match_key``, so this compares the two key sets over a map
    spanning both classes: ordinary names, a pure case change, and two that
    normalize to ``""``.
    """

    from t2pw.pwml.prefreeze_resolution import _canonical, _match_key, _rename_targets

    rename = {"gly": "Glycine", "glycine": "Glycine", "---": "Serine", "α": "Alanine"}
    rewrite_keys = set(_rename_targets(rename))
    detect_keys = {_match_key(old) for old in rename if _canonical(old)}

    assert rewrite_keys == detect_keys
    assert len(rewrite_keys) == 4, "the empty-_norm names must not share one bucket"


# --------------------------------------------------------------------------
# C-050g -- the source-collision comparison collapses interior whitespace.
#
# G9 LABELLING. ``test_c050g_two_spellings_of_one_molecule_are_one_source`` is a
# **behavioural correction**: it FAILS at the base SHA 9cc40286, where the guard
# raises ``AMBIGUOUS_RENAME_TARGET`` on two spellings of one molecule, and passes
# here. Symbol absence is not used anywhere. The over-fire controls beside it are
# labelled NEW ACCEPTANCE -- they pass at base too, and that is the point: they
# pin that the relaxation did not take the gate with it (merge rule 6).
# --------------------------------------------------------------------------


def _guard(rename_map: Dict[str, str], before: List[str], after: List[str]) -> Optional[str]:
    """Call the guard alone and return the code it raised, or ``None``.

    ``primary_before`` is derived from ``before`` exactly as
    ``resolve_compounds_prefreeze`` derives it, so the rogue-owner half is
    satisfied and the **source-collision** half is the only thing under test.
    """

    from t2pw.pwml.prefreeze_resolution import _norm, _reject_ambiguous_renames

    primary: Dict[str, tuple] = {}
    for index, name in enumerate(before):
        key = _norm(name)
        primary[key] = primary.get(key, ()) + (f"compounds#{index}",)
    try:
        _reject_ambiguous_renames(rename_map, before, after, primary)
    except PrefreezeResolutionError as stop:
        return stop.code
    return None


def test_c050g_two_spellings_of_one_molecule_are_one_source() -> None:
    """BEHAVIOURAL CORRECTION -- fails at 9cc40286, passes here.

    ``_norm`` is ``_canonical`` (which collapses whitespace) followed by
    ``[^a-z0-9:+ ]+ -> " "``, and that substitution can put a second space back
    where a separator sat next to one::

        _norm('sn -glycerol 3-phosphate') == 'sn  glycerol 3 phosphate'
        _norm('sn-glycerol 3-phosphate')  == 'sn glycerol 3 phosphate'

    Nothing collapses it again, so the guard counted one molecule as two and
    aborted a real committed 27-reaction leg at both production entry points in
    both ``strict_db`` modes (``PRODUCT_CONTRACT`` section 1, merge rule 7).
    """

    assert _guard(
        {"sn -glycerol 3-phosphate": "Glycerol 3-phosphate",
         "sn-glycerol 3-phosphate": "Glycerol 3-phosphate"},
        ["sn -glycerol 3-phosphate", "sn-glycerol 3-phosphate"],
        ["Glycerol 3-phosphate", "Glycerol 3-phosphate"],
    ) is None


@pytest.mark.parametrize(
    "label,sources",
    [
        ("different molecules", ["D-glucose", "D-fructose"]),
        ("one phosphate position apart", ["glycerol 2-phosphate", "glycerol 3-phosphate"]),
        ("the second pinned leg shape", ["PEtN-lipid A", "modified Lipid A"]),
        # Both sources carry the SAME double-space artefact. A fix keyed on
        # "this name contains a collapsible run" rather than on the collapsed
        # VALUE would wave these through; they are different molecules.
        ("same artefact, different tokens",
         ["sn -glycerol 3-phosphate", "sn -glycerol 1-phosphate"]),
    ],
)
def test_c050g_new_acceptance_genuinely_distinct_sources_still_abort(
    label: str, sources: List[str],
) -> None:
    """NEW ACCEPTANCE, and the merge-rule-6 control on this card.

    C-050g makes a biological gate fire less often. If it stopped catching a real
    merge it would have traded a dropped pathway for invented biology, which is
    strictly worse. Each pair below is two **different compounds** sent to one
    target; every one must still raise.

    The relaxation cannot reach them, and the reason is structural rather than
    empirical: ``_norm`` already maps every character outside ``[a-z0-9:+ ]`` to a
    space, so it already read ``beta-D-glucose`` and ``beta D glucose`` as one
    source before this card. The equivalence has always been "the same token
    sequence"; the double space was the one place the *count* of separator
    characters leaked through. Collapsing it adds no name that a single separator
    did not already make equivalent.
    """

    target = "Merged target"
    assert _guard(
        {source: target for source in sources}, list(sources), [target] * len(sources),
    ) == "AMBIGUOUS_RENAME_TARGET", label


def test_c050g_new_acceptance_the_collapsed_spellings_are_recorded() -> None:
    """NEW ACCEPTANCE. Reading two names as one is a fact the operator gets told.

    Recorded in the shape the module already uses for such facts --
    ``aliases_preserved``, ``identity_projected``: a list of flat dicts on the
    compound summary. No new report schema.
    """

    payload = {
        "entities": {"compounds": [
            _compound("sn -glycerol 3-phosphate", "CHEBI:15428"),
            _compound("sn-glycerol 3-phosphate", "CHEBI:15428"),
        ]},
        # Neither row is referenced from ``processes``; see the connectivity test
        # below for why that is load-bearing rather than laziness.
        "processes": {"reactions": [{"name": "R1", "inputs": [], "outputs": []}]},
    }
    index = _StubNameIndex(
        {"15428": {"id": 78, "name": "Glycerol 3-phosphate", "matched_on": "chebi"}})

    summary = _run(payload, index)

    assert summary["applied"] is True
    assert summary["rename_sources_collapsed"] == [{
        "target": "glycerol 3 phosphate",
        "sources": ["sn -glycerol 3-phosphate", "sn-glycerol 3-phosphate"],
        "source_key": "sn glycerol 3 phosphate",
    }]
    assert [row["name"] for row in payload["entities"]["compounds"]] == [
        "Glycerol 3-phosphate", "Glycerol 3-phosphate"]


def test_c050g_new_acceptance_the_record_is_empty_when_nothing_merged() -> None:
    """NEW ACCEPTANCE. The key is always present, and empty when nothing merged."""

    payload = _payload()
    summary = _run(payload, _glycine_index())
    assert summary["rename_map"] == {"gly": "Glycine"}
    assert summary["rename_sources_collapsed"] == []


def test_c050g_new_acceptance_merged_rows_that_are_referenced_still_stop() -> None:
    """NEW ACCEPTANCE, and the honest record of what this card does NOT fix.

    The guard no longer miscounts the spellings -- but the two rows now share one
    name, so ``_alias_index`` maps that name to **both** tokens and a participant
    reference that resolved to ``compounds#0`` starts resolving to
    ``compounds#0|compounds#1``. The connectivity signature moves and the run
    stops, correctly: D-015 clause 5 requires participant connectivity to be
    preserved, and merging duplicate ROWS is a policy this codebase does not have
    -- pre-freeze may not (``PREFREEZE_ROW_COUNT_CHANGED``) and post-freeze may
    not (permanent merge rule 8). It is routed as a separate card.

    So the practical effect of C-050g on a *referenced* duplicate is to replace a
    **false** diagnosis with a **true** one, not to produce PWML. That is the
    ratified position, and this test is where a future reader meets it rather
    than rediscovering it from a digest that moved.
    """

    payload = {
        "entities": {"compounds": [
            _compound("sn -glycerol 3-phosphate", "CHEBI:15428"),
            _compound("sn-glycerol 3-phosphate", "CHEBI:15428"),
        ]},
        "processes": {"reactions": [
            {"name": "R1", "inputs": ["sn -glycerol 3-phosphate"], "outputs": []},
        ]},
    }
    original = deepcopy(payload)
    index = _StubNameIndex(
        {"15428": {"id": 78, "name": "Glycerol 3-phosphate", "matched_on": "chebi"}})

    with pytest.raises(PrefreezeResolutionError) as excinfo:
        _run(payload, index)

    assert excinfo.value.code == "PREFREEZE_CONNECTIVITY_BROKEN"
    assert payload == original, "nothing may be committed on a refusal"


def test_c050g_new_acceptance_a_collision_with_an_untouched_row_is_invisible_here() -> None:
    """NEW ACCEPTANCE. The finding this card surfaced, pinned as behaviour.

    ``_reject_ambiguous_renames`` groups only over ``rename_map`` **sources**, so
    a target whose ``_norm`` equals the ``_norm`` of a row that is *not* being
    renamed is invisible to both halves of it. Below, ``glycerol-3-phosphate``
    (row 0) is never renamed, yet the rename of row 1 lands on its normalized
    name. The guard returns cleanly; only the connectivity check two stages later
    catches it, and it reports a diff-string rather than a named cause.

    This is the shape that stops ``PMC12444477…/strict`` at production defaults.
    Recorded, not fixed: naming that condition is outside this card's boundary.
    """

    before = ["glycerol-3-phosphate", "sn-glycerol 3-phosphate"]
    after = ["glycerol-3-phosphate", "Glycerol 3-phosphate"]
    assert _guard({"sn-glycerol 3-phosphate": "Glycerol 3-phosphate"}, before, after) is None
