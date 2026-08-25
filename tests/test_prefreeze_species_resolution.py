"""C-045 acceptance -- species canonicalization runs BEFORE the freeze (D-016).

**G9 labelling, stated up front.** C-045 *corrects pre-existing observable
behaviour*: at the base SHA the species rename happens inside ``build_pwml_ir``,
after the canonical payload is frozen. That correction's behavioural base proof
is **not** in this file, because a pytest cannot fail at a SHA where the symbol
it would call does not exist. It is
``docs/pwml_recovery_sprint/evidence/probe_c045_species_prefreeze.py``, whose
``--mode g9`` runs the identical production region at both SHAs and exits 1 at
``0ec64d2c`` (frozen payload carries the strain-qualified name, exporter emits
the binomial) and 0 here, and whose ``--mode census`` shows the rename set and
the at-risk set are byte-identical across all 152 committed payloads.

Everything **in this file** is an explicitly labelled NEW acceptance test for the
new entry point ``resolve_species_prefreeze``. No base failure is fabricated for
any of them.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

import pytest

from helpers_prefreeze import prefrozen
from t2pw.pwml import ir as ir_module
from t2pw.pwml.ir import SPECIES_CANONICALIZATION_FIELD, build_pwml_ir
from t2pw.pwml.prefreeze_resolution import (
    PrefreezeResolutionError,
    resolve_species_prefreeze,
    run_prefreeze_resolution,
)

#: Strain-qualified and taxonomy-identified, so step 4 of the ladder collapses it
#: offline. No database, no name index -- which is what makes every assertion here
#: valid in a worktree with no ``.env`` (P4-01).
STRAIN = "Lactococcus lactis subsp. lactis KF147"
BINOMIAL = "Lactococcus lactis"


def _payload(*species: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metadata": {"pathway_name": "P", "pathway_subject": "Metabolic"},
        "entities": {
            "species": [dict(row) for row in species],
            "compounds": [{"name": "glycine"}],
            "proteins": [],
        },
        "processes": {"reactions": [{"name": "R1", "inputs": ["glycine"], "outputs": ["glycine"]}]},
        "biological_states": [{"name": "cytosol", "species": STRAIN}],
    }


def _resolve(payload: Dict[str, Any]) -> Dict[str, Any]:
    return resolve_species_prefreeze(payload, name_index=None)


def _build(payload: Dict[str, Any]) -> tuple:
    # C-051 / D-015 (LOCKED): ``build_pwml_ir`` no longer resolves compound
    # identity after the canonical freeze and refuses a compound row that
    # carries no verdict. ``_payload`` above carries one ``glycine`` row, so the
    # copy handed to the exporter goes through the pre-freeze stage first --
    # which is what production does at both entry points, and what these tests
    # already assert for the *species* half. ``prefrozen`` asserts the stage
    # ruled on every compound row, so this cannot pass on a tree where the
    # pre-freeze sequence is absent or inert.
    return build_pwml_ir(
        prefrozen(deepcopy(payload), name_index=None),
        pathway_name="P", pathway_subject="Metabolic",
        strict_db=False, name_index=None,
    )


def test_new_acceptance_the_rename_is_on_the_payload_before_the_exporter_sees_it() -> None:
    """A1/A2: the payload the freeze would hash already carries the canonical name.

    This is the whole card in one assertion. The exporter is handed a payload
    that is *already* canonical, so there is nothing left for it to rewrite --
    merge rule 8's "no exporter repairs biology after the freeze", measured on the
    object rather than argued from the call graph.
    """

    payload = _payload({"name": STRAIN, "taxonomy_id": "1091041"})
    summary = _resolve(payload)

    assert summary["applied"] is True
    assert summary["rename_map"] == {STRAIN: BINOMIAL}
    assert payload["entities"]["species"][0]["name"] == BINOMIAL
    # D-015 clause 4: the supported name stays reachable.
    assert payload["entities"]["species"][0]["raw_name"] == STRAIN
    assert STRAIN in payload["entities"]["species"][0]["aliases"]

    ir, _ = _build(payload)
    assert [row["name"] for row in ir["species"]] == [BINOMIAL]


def test_new_acceptance_the_exporter_does_not_enter_the_ladder_a_second_time() -> None:
    """A1: moved, not duplicated. Counted, not reasoned about.

    A second pass over an already-canonical row is the non-idempotence trap that
    produced B-1 on the compound side, so the count is asserted at zero rather
    than assumed to be harmless.
    """

    payload = _payload({"name": STRAIN, "taxonomy_id": "1091041"})
    _resolve(payload)

    calls: List[Any] = []
    original = ir_module._canonicalize_species_offline

    def _counting(record: Any, **kwargs: Any) -> str:
        calls.append(record.get("name"))
        return original(record, **kwargs)

    ir_module._canonicalize_species_offline = _counting
    try:
        ir, report = _build(payload)
    finally:
        ir_module._canonicalize_species_offline = original

    assert calls == [], "build_pwml_ir re-entered the species ladder after the freeze"
    assert [row["name"] for row in ir["species"]] == [BINOMIAL]
    # ...and the report it used to compute is still the report it hands back.
    assert report["name_canonicalization"]["species"] == [
        {"from": STRAIN, "to": BINOMIAL, "taxonomy_id": "1091041",
         "source": "deterministic_strain_normalization"},
    ]


def test_new_acceptance_the_preflight_still_receives_the_at_risk_species() -> None:
    """A7: ``_species_at_risk`` survives the move, and its consumer still warns.

    ``_emit_canonicalization_preflight`` is C-040's and unchanged; what changed is
    where its input is computed. A taxonomy-identified species the offline index
    does not cover is ``deterministic`` -- it may still collide on import -- and
    that is what the operator has to be told.
    """

    payload = _payload({"name": STRAIN, "taxonomy_id": "1091041"})
    summary = _resolve(payload)
    assert summary["at_risk"] == [BINOMIAL]

    _, report = _build(payload)
    assert report["preflight"]["species"] == [BINOMIAL]
    assert report["preflight"]["db_available"] is False
    codes = [issue["code"] for issue in report["warnings"]]
    assert "noncanonical_names_collision_risk" in codes


def test_new_acceptance_a_species_the_ladder_refuses_is_not_renamed() -> None:
    """A6: the move admits nothing the precedence ladder refuses.

    No taxonomy id is the ladder's ``novel`` rung: a genuinely unidentified
    organism keeps its extraction name verbatim, strain qualifier included. The
    gate decides whether to APPLY, not merely whether to log (D-028).
    """

    payload = _payload({"name": STRAIN})
    summary = _resolve(payload)

    assert summary["renamed"] == 0
    assert summary["rename_map"] == {}
    assert payload["entities"]["species"][0]["name"] == STRAIN
    assert "raw_name" not in payload["entities"]["species"][0]
    assert summary["statuses"] == {STRAIN: "novel"}
    assert summary["at_risk"] == []

    _, report = _build(payload)
    assert report.get("preflight", {}).get("species", []) == []


def test_new_acceptance_an_ambiguous_species_rename_still_fails_visibly() -> None:
    """A4 / D-015 clause 6: a name another entity answers to cannot be redirected.

    The structural codes are C-050's and still raise through this stage, and the
    payload is left exactly as it was -- there is no partially propagated state to
    observe.
    """

    payload = _payload({"name": STRAIN, "taxonomy_id": "1091041"})
    payload["entities"]["proteins"].append({"name": STRAIN})
    before = deepcopy(payload)

    with pytest.raises(PrefreezeResolutionError) as excinfo:
        _resolve(payload)
    assert excinfo.value.code == "AMBIGUOUS_REFERENCE"
    assert payload == before, "the payload must be untouched on a refusal"


def test_new_acceptance_a_duplicate_row_follows_its_group_and_adds_no_species() -> None:
    """The exporter drops ``_norm``-duplicate rows; canonicalizing them apart would
    turn a dropped duplicate into a **second organism** the exporter never emitted.

    So the group leader is canonicalized and the group follows it, and only when
    the leader's rename moved the group's ``_norm`` -- which is exactly when the
    duplicates would otherwise stop deduplicating.
    """

    shouty = STRAIN.upper()
    payload = _payload(
        {"name": STRAIN, "taxonomy_id": "1091041"},
        {"name": shouty, "taxonomy_id": "1091041"},
    )
    summary = _resolve(payload)

    assert [row["name"] for row in payload["entities"]["species"]] == [BINOMIAL, BINOMIAL]
    # One decision, one log entry -- the follower is not a second consultation.
    assert summary["name_canonicalization"] == [
        {"from": STRAIN, "to": BINOMIAL, "taxonomy_id": "1091041",
         "source": "deterministic_strain_normalization"},
    ]
    assert payload["entities"]["species"][1][SPECIES_CANONICALIZATION_FIELD][
        "followed_leader"
    ] == shouty

    ir, report = _build(payload)
    assert [row["name"] for row in ir["species"]] == [BINOMIAL]
    assert len(report["name_canonicalization"]["species"]) == 1


def test_new_acceptance_the_group_leader_rule_forecloses_the_rename_map_collision() -> None:
    """C-050f's ``PREFREEZE_RENAME_MAP_COLLISION`` is unreachable from this ladder.

    That guard refuses a map whose sources share a ``_match_key`` but target
    different names, and species names are far more collision-prone than compound
    names -- ``LACTOCOCCUS LACTIS SUBSP. LACTIS KF147`` and its mixed-case twin
    share a ``_match_key`` and, canonicalized **independently**, would produce
    ``LACTOCOCCUS LACTIS`` and ``Lactococcus lactis``, which is exactly the map
    the guard exists to refuse. Canonicalizing the group leader and letting the
    group follow it means at most one target per ``_match_key`` is ever produced,
    so the guard is satisfied by construction rather than worked around.
    """

    from t2pw.pwml.prefreeze_resolution import _match_key, _rename_targets

    shouty = STRAIN.upper()
    assert _match_key(STRAIN) == _match_key(shouty), "the premise: one match key"

    payload = _payload(
        {"name": STRAIN, "taxonomy_id": "1091041"},
        {"name": shouty, "taxonomy_id": "1091041"},
    )
    summary = _resolve(payload)

    assert set(summary["rename_map"]) == {STRAIN, shouty}
    assert set(summary["rename_map"].values()) == {BINOMIAL}
    # The guard is run over the map this stage produced, and accepts it.
    assert _rename_targets(summary["rename_map"]) == {_match_key(STRAIN): BINOMIAL}


def test_new_acceptance_two_strains_of_one_species_are_declined_not_merged() -> None:
    """Distinct organisms collapsing onto one name are refused (D-015 clause 6).

    ``subsp.`` and ``subsp`` normalize differently -- ``_norm`` turns punctuation
    into a space and does not collapse the run -- so these are two rename
    *sources*, not one group, and they target the same binomial. Merging two
    named organisms into one is inventing biology, so the stage refuses to apply
    either rename and leaves both rows under the names they arrived with.

    **Measured behavioural delta, reported not landed silently:** at the base SHA
    this payload does not raise. ``_dedupe_named_rows`` keeps both rows, the
    exporter renames both, and the IR comes out carrying **two species rows with
    the same name** -- the import collision ``_deterministic_species_name`` exists
    to prevent. Neither shape occurs in the 152 committed payloads.

    **C-082 / F-115 moved the DISPOSITION of this pin, deliberately.** It used to
    assert ``pytest.raises(...)`` and ``payload == before``. The refusal it pins
    is unchanged and is still asserted, in the form that matters: the two rows are
    not merged and neither name moves. What changed is that the refusal no longer
    terminates the leg -- on T-106 that cost a ten-reaction pathway with a passing
    audit, which is the shape merge rule 7 exists to prevent. The payload is now
    ``before`` plus the declination markers, and nothing else; the marker is the
    record ``PRODUCT_CONTRACT`` §3 requires and the reason the equality is on the
    rows rather than on the whole object.
    """

    other = STRAIN.replace("subsp.", "subsp")
    payload = _payload(
        {"name": STRAIN, "taxonomy_id": "1091041"},
        {"name": other, "taxonomy_id": "1091041"},
    )
    before = deepcopy(payload)

    summary = _resolve(payload)

    # Refused: neither rename applied, so nothing merged.
    assert summary["rename_map"] == {}
    assert [row["name"] for row in payload["entities"]["species"]] == [STRAIN, other]
    assert len({row["name"] for row in payload["entities"]["species"]}) == 2
    for index, row in enumerate(payload["entities"]["species"]):
        stripped = dict(row)
        marker = stripped.pop(SPECIES_CANONICALIZATION_FIELD)
        assert stripped == before["entities"]["species"][index]
        assert marker["status"] == "rename_declined"
        assert marker["declined_rename"]["code"] == "AMBIGUOUS_RENAME_TARGET"
    assert {record["source"] for record in summary["renames_declined"]} == {STRAIN, other}


def test_new_acceptance_a_rename_onto_an_existing_organism_is_declined() -> None:
    """REV-045 B-1. Renaming onto a name a **different** species row keeps is a
    merge, and for species nothing else catches it.

    ``_LOCATION_MEMBER_FIELDS`` has no species bucket, so ``_iter_refs``,
    ``_propagate``, ``_assert_fully_propagated`` and ``_connectivity_signature``
    are all blind to species rows; the row-count check sees 2 == 2 because the
    loss happens later. The compound stage fails closed on this exact shape only
    because compounds are participants.

    **Base delta, disclosed:** at ``0ec64d2c`` the ladder ran *after*
    ``_dedupe_named_rows``, so dedupe never saw the renamed name and **both**
    rows survived -- the IR carried taxonomy ``1091041`` and ``1358``, and the
    preflight listed the organism twice. Unrefused, dedupe collapses them and
    taxonomy ``1358`` is **deleted**, leaving a row named ``Lactococcus lactis``
    carrying the strain's id ``1091041``: a wrong organism identity, not a lossy
    one.

    **C-082 / F-115 moved the DISPOSITION of this pin, deliberately** -- see the
    test above for the full account. The merge is still refused and is asserted
    here all the way into the IR, which is the strongest form this pin has ever
    had: both organisms reach the exporter, under their own names, carrying their
    own taxonomy ids. What no longer happens is the leg ending on it.
    """

    payload = _payload(
        {"name": STRAIN, "taxonomy_id": "1091041"},
        {"name": BINOMIAL, "taxonomy_id": "1358"},
    )
    before = deepcopy(payload)

    summary = _resolve(payload)

    (record,) = summary["renames_declined"]
    assert record["code"] == "AMBIGUOUS_RENAME_TARGET"
    assert record["source"] == STRAIN
    assert record["collides_with"] == [BINOMIAL]
    assert summary["rename_map"] == {}
    stripped = dict(payload["entities"]["species"][0])
    stripped.pop(SPECIES_CANONICALIZATION_FIELD)
    assert stripped == before["entities"]["species"][0]
    # The occupant is untouched too. Its own marker is the ladder's ordinary
    # record of having consulted the row -- taxonomy-identified, not covered by
    # the offline index, so ``deterministic`` -- and predates this card.
    occupant = dict(payload["entities"]["species"][1])
    assert occupant.pop(SPECIES_CANONICALIZATION_FIELD) == {"status": "deterministic"}
    assert occupant == before["entities"]["species"][1]

    # The merge the guard exists to prevent, refused where it would have happened.
    ir, _ = _build(payload)
    assert [row["name"] for row in ir["species"]] == [STRAIN, BINOMIAL]
    assert sorted(str(row.get("taxonomy_id") or "") for row in ir["species"]) == ["1091041", "1358"]


def test_new_acceptance_the_existing_row_guard_does_not_over_fire() -> None:
    """The control for the guard above: it must refuse a merge and nothing else.

    A refusal on a path every species payload crosses is how a card turns a
    working export into a dead one, so the two shapes that look like the merge
    and are not are pinned here.

    1. **A group renaming to one target.** ``_norm``-duplicates legitimately land
       on the same name -- that is the leader/follower rule -- and the occupant
       of the target is the leader itself, not a distinct organism.
    2. **A target inside the source's own ``_norm`` group.** A pure spelling
       change merges nothing, so it is skipped before the occupancy scan. The
       ladder cannot currently emit one (both rungs guard on ``_norm``), so it is
       exercised at the guard directly rather than through a payload that cannot
       exist.
    3. **A row that vacates the target.** If the occupant is itself renamed away,
       there is nothing left to collide with.
    """

    from t2pw.pwml.prefreeze_resolution import _alias_index, _screen_ambiguous_species_renames

    # 1 -- the leader/follower group still succeeds, and emits one organism.
    payload = _payload(
        {"name": STRAIN, "taxonomy_id": "1091041"},
        {"name": STRAIN.upper(), "taxonomy_id": "1091041"},
    )
    summary = _resolve(payload)
    assert set(summary["rename_map"].values()) == {BINOMIAL}
    ir, _ = _build(payload)
    assert [row["name"] for row in ir["species"]] == [BINOMIAL]

    primary, _ = _alias_index(_payload({"name": STRAIN, "taxonomy_id": "1091041"}))
    # 2 -- target inside the source's own group: accepted.
    _screen_ambiguous_species_renames(
        {STRAIN: STRAIN.upper()}, [STRAIN], [STRAIN.upper()], primary,
    )
    # 3 -- the occupant vacates the name, so nothing collides.
    _screen_ambiguous_species_renames(
        {STRAIN: BINOMIAL, BINOMIAL: "Lactococcus cremoris"},
        [STRAIN, BINOMIAL],
        [BINOMIAL, "Lactococcus cremoris"],
        {},
    )


def test_new_acceptance_the_stage_is_idempotent() -> None:
    """Running it twice must not re-decide, re-log, or re-rename.

    ``run_prefreeze_resolution`` is called once per export today, but a payload
    that has already been through it is exactly what a resumed or replayed run
    hands back, and B-1 is what a second pass costs when it re-derives.
    """

    payload = _payload({"name": STRAIN, "taxonomy_id": "1091041"})
    _resolve(payload)
    once = deepcopy(payload)
    second = _resolve(payload)

    assert payload == once, "the second pass re-decided something"
    assert second["renamed"] == 0
    # The summary describes THIS pass, which consulted nothing and renamed
    # nothing; the row still carries the account of the pass that did.
    assert second["name_canonicalization"] == []
    assert payload["entities"]["species"][0][SPECIES_CANONICALIZATION_FIELD]["entry"]["to"] == (
        BINOMIAL
    )

    _, report = _build(payload)
    assert len(report["name_canonicalization"]["species"]) == 1


def test_new_acceptance_species_is_registered_and_an_absent_bucket_is_a_clean_run() -> None:
    """A5 / D-029: this stage has no database to be unreachable, and says so.

    It is registered at C-050's declared seam, it reports through the same verdict
    machinery, and a payload with no species is a benign skip rather than a
    failure -- which is what keeps ``report["ok"]`` a statement about the payload.
    """

    from t2pw.pwml.prefreeze_resolution import PREFREEZE_CANONICALIZERS

    assert [name for name, _ in PREFREEZE_CANONICALIZERS] == ["compounds", "species"]

    report = run_prefreeze_resolution({"entities": {"compounds": [], "species": []}})
    assert report["canonicalizers"] == ["compounds", "species"]
    assert report["species"]["skipped_reason"] == "no_species_rows"
    assert report["ok"] is True
    assert report["failures"] == {}
    assert report["review_required"] == {}
