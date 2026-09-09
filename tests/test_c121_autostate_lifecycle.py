"""C-121 / F-192 -- the auto-state lifecycle across the quarantine sweep.

``ensure_autostates`` runs **once**, from ``normalize_process_payload``. The strict
quarantine sweep runs later and ``_prune_biological_states`` drops every state
nothing surviving references -- correctly, by its own policy -- and ``audit_repair``
can add an element-location row *after* the one pass that would have assigned a
state to it. On the base tree nothing re-established the placeholder afterwards, so
a payload whose biology, identity and reaction support were all sound was refused at
the **last** gate and the whole pathway was lost.

Five archived production legs did exactly that. Two shapes, both reachable:

======  =========================================================  ==============================================
shape   what happens                                               gate error
======  =========================================================  ==============================================
A       content survives, every state swept as unreferenced        ``no_biological_states``
B       a location row created after ``ensure_autostates`` never    ``visible_entity_missing_location_state``
        received an assignment; a real state survives because
        other rows reference it
======  =========================================================  ==============================================

A third shape -- a surviving row dangling at a REMOVED state -- is **impossible** and
is deliberately not fixtured: ``_prune_biological_states`` removes a state only when
nothing references it, so the removal is conditioned on the absence of exactly that
referent (``D-099`` § 4).

What each test in this module is, stated so a reviewer never has to infer it:

* ``test_shape_a_*`` and ``test_shape_b_*`` are the two **G9 behavioural proofs**.
  Both FAIL on base SHA ``c4a3a92c`` on VALUES -- the required-field gate's own
  verdict and error-code set, and the surviving state list -- and pass at the tip.
  Neither asserts that a symbol exists.
* ``test_fixture_c_*`` is the **preservation** arm and is the most important test
  here. A genuinely unreferenced ``__auto_state__`` that no surviving row requires
  must still be removed and must **not** be recreated. It passes on base and at the
  tip, and it is what stops the fix from making the placeholder immortal.
* ``test_the_guard_*`` and ``test_an_empty_graph_*`` are **new acceptance arms**
  for the guard itself (labelled as such under G9: new capability, no fabricated
  base failure). The guard is the entire difference between this fix and a
  corpus-wide mutation -- an unguarded re-run changes 40 archived production legs,
  six of them legs that export today.
"""

from __future__ import annotations

import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest  # noqa: E402

from t2pw.pipeline.process_normalizer import ensure_autostates  # noqa: E402
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402
from t2pw.pwml.ir import validate_required_pwml_contract  # noqa: E402

# THE G9 PROOFS BELOW IMPORT NONE OF THE NEW SYMBOLS, ON PURPOSE. A base-tree run
# must fail on VALUES -- the gate verdict, the error-code set, the state list -- and
# a module-level import of something the base tree does not have would fail
# COLLECTION instead, which is symbol absence dressed up as a behavioural proof and
# is explicitly not evidence. So the seam's own API is imported optionally, and only
# the arms that test the API directly -- all of them labelled NEW acceptance arms --
# depend on it.
try:
    from t2pw.pipeline.process_normalizer import (  # noqa: E402
        _VISIBLE_LOCATION_BUCKETS,
        autostate_restoration_required,
        restore_autostates_if_required,
    )

    _RESTORATION_API = True
except ImportError:  # pragma: no cover -- taken on the base tree only
    _VISIBLE_LOCATION_BUCKETS = ()
    autostate_restoration_required = None  # type: ignore[assignment]
    restore_autostates_if_required = None  # type: ignore[assignment]
    _RESTORATION_API = False

#: New capability, so no base failure is claimed for these arms (G9): on a tree
#: without the seam they SKIP with this reason instead of erroring in collection.
new_capability = pytest.mark.skipif(
    not _RESTORATION_API,
    reason="new acceptance arm for the C-121 restoration API; no base failure claimed",
)

AUTO_STATE = "__auto_state__"

#: Exactly what the exporter puts on ``metadata`` before it calls the
#: required-field gate. Reproduced rather than approximated: the gate errors on a
#: missing pathway name, subject, width or height, and a fixture that omitted them
#: would report a metadata omission of its own making as a payload defect.
EXPORT_METADATA = {
    "pathway_name": "C121 fixture",
    "name": "C121 fixture",
    "pathway_subject": "Metabolic",
    "subject": "Metabolic",
    "width": 3200,
    "height": 1400,
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _gate_codes(payload: Dict[str, Any]) -> Set[str]:
    """The required-field gate's error codes for ``payload``, as the exporter sees them."""

    staged = deepcopy(payload)
    metadata = staged.setdefault("metadata", {})
    for key, value in EXPORT_METADATA.items():
        metadata.setdefault(key, value)
    report = validate_required_pwml_contract(staged, strict_db=True)
    return {
        str(entry.get("code"))
        for entry in (report.get("errors") or [])
        if isinstance(entry, dict)
    }


def _gate_ok(payload: Dict[str, Any]) -> bool:
    staged = deepcopy(payload)
    metadata = staged.setdefault("metadata", {})
    for key, value in EXPORT_METADATA.items():
        metadata.setdefault(key, value)
    return bool(validate_required_pwml_contract(staged, strict_db=True).get("ok"))


def _state_names(payload: Dict[str, Any]) -> List[str]:
    return [
        str(row.get("name"))
        for row in (payload.get("biological_states") or [])
        if isinstance(row, dict)
    ]


def _removed_state_reasons(result) -> Dict[str, str]:
    """``state name -> removal reason``, from the removed-entity report.

    Read from the artifact a reviewer reads, not from a recomputation: this is the
    same ``removed_biological_states`` list the five archived legs carry.
    """

    return {
        str(row.get("name")): str(row.get("reason"))
        for row in result.removed_entity_report.get("removed_biological_states") or []
    }


def _reaction_names(payload: Dict[str, Any]) -> List[str]:
    return [
        str(row.get("name"))
        for row in ((payload.get("processes") or {}).get("reactions") or [])
        if isinstance(row, dict)
    ]


# ── Fixtures ────────────────────────────────────────────────────────────────


def _exportable_payload() -> Dict[str, Any]:
    """Two-step glutathione biosynthesis: small, valid, fully declared, exportable.

    Deliberately clears the required-field gate on its own -- taxonomy id AND
    classification on the species row, an external identity and a species on every
    protein -- so a gate verdict measured below is attributable to the ONE thing the
    fixture that derives from it broke, and not to species metadata the archived
    payloads happen to be missing for unrelated reasons.
    """

    return {
        "metadata": dict(EXPORT_METADATA, organism="Homo sapiens"),
        "entities": {
            "species": [
                {
                    "name": "Homo sapiens",
                    "taxonomy_id": "9606",
                    "classification": "Eukaryote",
                }
            ],
            "subcellular_locations": [{"name": "cytosol"}],
            "compounds": [
                {"name": "L-glutamate"},
                {"name": "L-cysteine"},
                {"name": "gamma-glutamylcysteine"},
                {"name": "glycine"},
                {"name": "glutathione"},
            ],
            "proteins": [
                {
                    "name": "glutamate-cysteine ligase",
                    "species": "Homo sapiens",
                    "mapped_ids": {"uniprot": "P48506"},
                },
                {
                    "name": "glutathione synthetase",
                    "species": "Homo sapiens",
                    "mapped_ids": {"uniprot": "P48637"},
                },
            ],
            "protein_complexes": [
                {
                    "name": "glutamate-cysteine ligase complex",
                    "species": "Homo sapiens",
                    "components": ["glutamate-cysteine ligase"],
                },
                {
                    "name": "glutathione synthetase complex",
                    "species": "Homo sapiens",
                    "components": ["glutathione synthetase"],
                },
            ],
            "nucleic_acids": [],
        },
        "biological_states": [
            {
                "name": "cytosol state",
                "species": "Homo sapiens",
                "subcellular_location": "cytosol",
            }
        ],
        "element_locations": {
            "compound_locations": [
                {"compound": name, "biological_state": "cytosol state"}
                for name in (
                    "L-glutamate",
                    "L-cysteine",
                    "gamma-glutamylcysteine",
                    "glycine",
                    "glutathione",
                )
            ],
            "protein_locations": [
                {"protein": "glutamate-cysteine ligase", "biological_state": "cytosol state"},
                {"protein": "glutathione synthetase", "biological_state": "cytosol state"},
            ],
        },
        "processes": {
            "reactions": [
                {
                    "name": "gamma-glutamylcysteine synthesis",
                    "inputs": ["L-glutamate", "L-cysteine"],
                    "outputs": ["gamma-glutamylcysteine"],
                    "enzymes": [
                        {
                            "entity": "glutamate-cysteine ligase complex",
                            "entity_type": "protein_complex",
                        }
                    ],
                    "biological_state": "cytosol state",
                },
                {
                    "name": "glutathione synthesis",
                    "inputs": ["gamma-glutamylcysteine", "glycine"],
                    "outputs": ["glutathione"],
                    "enzymes": [
                        {
                            "entity": "glutathione synthetase complex",
                            "entity_type": "protein_complex",
                        }
                    ],
                    "biological_state": "cytosol state",
                },
            ],
            "transports": [],
            "interactions": [],
        },
    }


def _shape_a_payload() -> Dict[str, Any]:
    """Shape A, as the archive shows it: content survives, every state is swept.

    Reproduces ``PMC11961743`` and ``PMC4471609`` in structure rather than in
    content -- **zero** element-location rows, no reaction carrying a state, and a
    ``__auto_state__`` that Stage 3 created with
    ``n_entities_assigned_to_autostate: 0``. ``_prune_biological_states`` then finds
    nothing referencing either state and removes both, correctly. The reactions and
    every entity they need survive.

    This is why the authorized invariant is a disjunction and not a row-scoped
    rule: there is no row here for a row-scoped invariant to repair.
    """

    payload = _exportable_payload()
    payload["element_locations"] = {"compound_locations": [], "protein_locations": []}
    for reaction in payload["processes"]["reactions"]:
        reaction.pop("biological_state", None)
    payload["biological_states"].append(
        {
            "name": AUTO_STATE,
            "species": "Homo sapiens",
            "subcellular_location": "cell",
        }
    )
    payload["entities"]["subcellular_locations"].append({"name": "cell"})
    return payload


def _shape_b_payload() -> Dict[str, Any]:
    """Shape B: a location row added AFTER ``ensure_autostates`` ran.

    ``audit_repair``'s own mechanism, and the shape ``PMC12312563`` (7 rows) and
    ``PMC13231680`` (5 rows) failed on. The row's reference is **absent**, not
    dangling: the real state survives because the other rows reference it, so the
    sweep never touches it, and nothing ever assigns a state to the new row.
    """

    payload = _exportable_payload()
    payload["entities"]["compounds"].append({"name": "glutathione disulfide"})
    payload["processes"]["reactions"].append(
        {
            "name": "glutathione oxidation",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "biological_state": "cytosol state",
        }
    )
    # The repaired row, exactly as a post-normalization pass leaves it: an entity
    # and no state.
    payload["element_locations"]["compound_locations"].append(
        {"compound": "glutathione disulfide"}
    )
    return payload


def _fixture_c_payload() -> Dict[str, Any]:
    """Fixture C: a genuinely unreferenced ``__auto_state__`` that must still die.

    Every surviving row already carries the real state, so nothing the payload needs
    is missing and the placeholder is pure litter. The sweep must remove it, and the
    restoration must **not** bring it back. Getting this wrong is how a narrow fix
    turns into ``__auto_state__`` being immortal, and it is the reason the guard's
    second clause is row-scoped rather than "an auto-state is absent".
    """

    payload = _exportable_payload()
    payload["biological_states"].append(
        {
            "name": AUTO_STATE,
            "species": "Homo sapiens",
            "subcellular_location": "cell",
        }
    )
    payload["entities"]["subcellular_locations"].append({"name": "cell"})
    return payload


# ── G9 proof, shape A ───────────────────────────────────────────────────────


def test_shape_a_a_swept_payload_with_surviving_reactions_still_exports() -> None:
    """G9, shape A. On base ``c4a3a92c`` this fails on VALUES, twice over.

    Base measures ``states == []`` and ``codes == {'no_biological_states'}``, so a
    payload carrying two supported reactions is refused at the last gate and the
    pathway is lost. Neither assertion mentions a symbol.
    """

    result = quarantine_and_close(_shape_a_payload(), strict_db=True)

    # The sweep did its job and is unchanged: both states genuinely were
    # unreferenced, and both were removed for the reason it has always given.
    assert _removed_state_reasons(result) == {
        "cytosol state": "state_unreferenced_after_quarantine",
        AUTO_STATE: "state_unreferenced_after_quarantine",
    }

    # Exportable content survived the sweep. This is the clause that makes the
    # restoration legitimate rather than cosmetic.
    assert _reaction_names(result.payload) == [
        "gamma-glutamylcysteine synthesis",
        "glutathione synthesis",
    ]

    # THE BEHAVIOURAL CLAIM. Base: []. Tip: the placeholder, re-established.
    assert _state_names(result.payload) == [AUTO_STATE]

    # And the gate the leg used to die at. Base: {'no_biological_states'}.
    assert _gate_codes(result.payload) == set()
    assert _gate_ok(result.payload) is True


def test_shape_a_the_gate_stops_refusing_on_no_biological_states() -> None:
    """G9, shape A, the gate verdict on its own line.

    Separated from the state-list proof above so a base-tree run names the error
    code in the failure text rather than stopping at the first assertion. Base:
    ``ok is False`` and ``{'no_biological_states'}``. This is the error that
    destroyed 10 reactions on ``PMC11961743`` and 4 on ``PMC4471609``.
    """

    result = quarantine_and_close(_shape_a_payload(), strict_db=True)

    assert _gate_codes(result.payload) == set()
    assert _gate_ok(result.payload) is True


def test_shape_b_the_gate_stops_refusing_on_a_missing_location_state() -> None:
    """G9, shape B, the gate verdict on its own line.

    Base: ``ok is False`` and ``{'visible_entity_missing_location_state'}`` -- the
    only error code either of the two oldest archived instances carried, 7 rows on
    ``PMC12312563`` and 5 on ``PMC13231680``.
    """

    result = quarantine_and_close(_shape_b_payload(), strict_db=True)

    assert _gate_codes(result.payload) == set()
    assert _gate_ok(result.payload) is True


@new_capability
def test_shape_a_restoration_adds_no_process_and_no_participant() -> None:
    """The placeholder is presentation scaffolding, not biology -- measured.

    Reaction, transport and interaction counts, and every reaction's participants,
    enzymes, direction and stoichiometry, are identical to what closure left. A fix
    that grew any of them would be admitting biology and is a reject.
    """

    payload = _shape_a_payload()
    result = quarantine_and_close(payload, strict_db=True)

    before = deepcopy(result.payload["processes"])
    restored = deepcopy(result.payload)
    # Idempotent: the seam already ran this once, so a second call must be a no-op.
    assert restore_autostates_if_required(restored) is False
    assert restored["processes"] == before

    assert {
        bucket: len(rows)
        for bucket, rows in result.payload["processes"].items()
        if isinstance(rows, list)
    } == {"reactions": 2, "transports": 0, "interactions": 0}
    assert len(result.payload["entities"]["compounds"]) == 5
    assert len(result.payload["entities"]["proteins"]) == 2


# ── G9 proof, shape B ───────────────────────────────────────────────────────


def test_shape_b_a_row_added_after_the_one_autostate_pass_still_exports() -> None:
    """G9, shape B. On base ``c4a3a92c`` this fails on VALUES.

    Base measures ``codes == {'visible_entity_missing_location_state'}`` and the
    repaired row still carrying no ``biological_state``. The real state survives
    here -- the sweep removes nothing -- so this is an **absent** reference, and a
    fix keyed on auto-state *removal* cannot see it. That is exactly why the two
    oldest archived instances went unattributed for three weeks.
    """

    result = quarantine_and_close(_shape_b_payload(), strict_db=True)

    # Nothing was swept: the real state is referenced by seven other rows.
    assert _removed_state_reasons(result) == {}
    assert "cytosol state" in _state_names(result.payload)

    rows = result.payload["element_locations"]["compound_locations"]
    repaired = next(row for row in rows if row.get("compound") == "glutathione disulfide")

    # THE BEHAVIOURAL CLAIM. Base: the row carries no biological_state at all.
    assert repaired.get("biological_state") == AUTO_STATE

    # Base: {'visible_entity_missing_location_state'}.
    assert _gate_codes(result.payload) == set()
    assert _gate_ok(result.payload) is True

    # The state the surviving payload required, and nothing more: the real state is
    # untouched and keeps its own compartment.
    assert _state_names(result.payload) == ["cytosol state", AUTO_STATE]
    real = next(
        row for row in result.payload["biological_states"] if row["name"] == "cytosol state"
    )
    assert real["subcellular_location"] == "cytosol"


def test_shape_b_the_rows_that_already_had_a_state_keep_it() -> None:
    """Restoration fills the gap and rewrites nothing. Base fails on the same values.

    A fix that reassigned every row to the placeholder would flatten a multi-compartment
    pathway into one compartment while still clearing the gate -- silently wrong,
    and invisible to a test that only checked the gate verdict.
    """

    result = quarantine_and_close(_shape_b_payload(), strict_db=True)
    locations = result.payload["element_locations"]

    # ``.get`` rather than ``[...]``: on the base tree the repaired row has no
    # ``biological_state`` key at all, and a KeyError would be an ERROR where this
    # gate needs a failure on VALUES.
    assert [row.get("biological_state") for row in locations["compound_locations"]] == [
        "cytosol state",
        "cytosol state",
        "cytosol state",
        "cytosol state",
        "cytosol state",
        AUTO_STATE,
    ]
    assert [row.get("biological_state") for row in locations["protein_locations"]] == [
        "cytosol state",
        "cytosol state",
    ]


# ── Fixture C: the placeholder must NOT become immortal ─────────────────────


def test_fixture_c_a_genuinely_unreferenced_autostate_is_still_removed() -> None:
    """PRESERVATION arm, and the most important test in this module.

    Passes on base ``c4a3a92c`` and at the tip: it is not a G9 proof, it is the
    thing the G9 proofs must not have broken. Every surviving row already carries
    the real state, so the payload requires nothing -- the guard is quiet, the sweep
    removes the placeholder, and it stays removed.
    """

    result = quarantine_and_close(_fixture_c_payload(), strict_db=True)

    assert _removed_state_reasons(result) == {
        AUTO_STATE: "state_unreferenced_after_quarantine"
    }
    assert _state_names(result.payload) == ["cytosol state"]
    assert AUTO_STATE not in _state_names(result.payload)

    # The payload was exportable before and after: nothing about it needed repair,
    # which is precisely why nothing may be added to it.
    assert _gate_codes(result.payload) == set()
    assert _gate_ok(result.payload) is True


@new_capability
def test_fixture_c_the_guard_is_quiet_and_the_payload_is_byte_identical() -> None:
    """Quiet means "not one byte", not "nothing important".

    The inertness property the corpus depends on: an unguarded ``ensure_autostates``
    re-run changes 40 archived production legs, six of them legs that export today,
    each gaining a spurious placeholder and a ``cell`` compartment and moving its
    graph hash. This asserts the guarded entry point does none of that.
    """

    result = quarantine_and_close(_fixture_c_payload(), strict_db=True)
    settled = deepcopy(result.payload)

    assert autostate_restoration_required(settled) is False
    before = json.dumps(settled, sort_keys=True, ensure_ascii=False)
    assert restore_autostates_if_required(settled) is False
    assert json.dumps(settled, sort_keys=True, ensure_ascii=False) == before

    # And the comparand: an UNGUARDED re-run on the same payload does change it.
    # This is the regression the guard prevents, measured rather than described.
    unguarded = deepcopy(result.payload)
    ensure_autostates(unguarded)
    assert json.dumps(unguarded, sort_keys=True, ensure_ascii=False) != before
    assert AUTO_STATE in _state_names(unguarded)


# ── New acceptance arms for the guard itself ────────────────────────────────


@new_capability
def test_the_guard_fires_on_both_reachable_shapes_and_on_neither_control() -> None:
    """NEW acceptance arm (G9: new capability, no base failure claimed).

    The predicate itself, over the four payloads above: it is a disjunction, so
    clause 1 must catch shape A and clause 2 must catch shape B, and neither may
    catch a payload that is already correct.
    """

    swept = quarantine_and_close(_shape_a_payload(), strict_db=True)
    unassigned = quarantine_and_close(_shape_b_payload(), strict_db=True)

    # Measured on the payload as closure left it, with the restoration undone, so
    # the predicate is evaluated on the input it is meant to judge.
    swept_input = deepcopy(swept.payload)
    swept_input["biological_states"] = []
    assert autostate_restoration_required(swept_input) is True  # clause 1

    unassigned_input = deepcopy(unassigned.payload)
    for row in unassigned_input["element_locations"]["compound_locations"]:
        if row.get("compound") == "glutathione disulfide":
            row.pop("biological_state", None)
    assert autostate_restoration_required(unassigned_input) is True  # clause 2

    assert autostate_restoration_required(_exportable_payload()) is False
    assert autostate_restoration_required(_fixture_c_payload()) is False


@new_capability
def test_an_empty_graph_does_not_get_a_manufactured_state() -> None:
    """NEW acceptance arm. The content clause is checked FIRST and is not a formality.

    A payload with nothing to export is a refusal, and manufacturing a state for it
    would dress a dead run as a serializable one -- raising the PWML count by
    exporting nothing, which is the merge-rule-6 direction.
    """

    empty = _exportable_payload()
    empty["processes"] = {"reactions": [], "transports": [], "interactions": []}
    empty["biological_states"] = []

    assert autostate_restoration_required(empty) is False
    assert restore_autostates_if_required(empty) is False
    assert empty["biological_states"] == []

    # And the seam still refuses it, for the reason it always gave.
    result = quarantine_and_close(empty, strict_db=True)
    assert result.ok is False
    assert result.payload.get("biological_states") == []
    assert "no_biological_states" in _gate_codes(result.payload)


@new_capability
def test_a_payload_with_content_but_no_states_is_the_only_clause_one_case() -> None:
    """NEW acceptance arm. Clause 1 keys on zero states, not on the auto-state's name.

    Both ORCH-734 blockers had zero states and zero rows. A guard that asked "is
    ``__auto_state__`` missing?" would fire on every payload that never had one,
    including the 40 quiet legs; a guard that asked "is a row dangling?" would fire
    on neither blocker. Clause 1 is neither of those.
    """

    payload = _exportable_payload()
    payload["biological_states"] = []
    assert autostate_restoration_required(payload) is True

    payload["processes"] = {"reactions": [], "transports": [], "interactions": []}
    assert autostate_restoration_required(payload) is False

    # A payload whose only content is an interaction still counts as content: the
    # two shape-B legs export one reaction and four and two interactions.
    payload["processes"] = {
        "reactions": [],
        "transports": [],
        "interactions": [{"name": "x", "participants": []}],
    }
    assert autostate_restoration_required(payload) is True


@new_capability
def test_the_visible_location_buckets_are_the_gates_own_buckets() -> None:
    """NEW acceptance arm. A drift lock, because the list is duplicated on purpose.

    ``process_normalizer`` imports nothing from the exporter, so clause 2's bucket
    list is a copy of the gate's ``location_fields`` table. A copy that silently
    fell behind would make the guard blind to a whole bucket of rows the gate still
    refuses -- so the copy is compared against the gate's own source.
    """

    source = (SRC / "t2pw" / "pwml" / "ir.py").read_text(encoding="utf-8")
    match = re.search(
        r"location_fields\s*=\s*\{(.*?)\}", source, re.DOTALL
    )
    assert match, "the gate's location_fields table moved; clause 2 cannot be verified"
    gate_buckets = re.findall(r'"([a-z_]+)"\s*:', match.group(1))

    assert list(_VISIBLE_LOCATION_BUCKETS) == gate_buckets
