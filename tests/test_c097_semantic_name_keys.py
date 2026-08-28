"""C-097 / F-131 -- ``bench.semantic._names`` reads the SCHEMA keys, not the legacy tail.

**These are CORRECTION tests, offered as the G9 proof.** C-089 unified two readers
onto ``participant_schema.PARTICIPANT_NAME_KEYS``. Its charter scoped the legacy
``ref``/``id`` tail to ``identity_admission``, but both readers consume the union,
so ``_names`` began treating ``ref`` and ``id`` as entity-name keys -- which it
never did before C-089. ``test_an_id_only_participant_row_yields_no_name``,
``test_a_ref_only_participant_row_yields_no_name`` and
``test_an_id_only_participant_is_not_a_referential_orphan`` FAIL on the base SHA
(``ea688e0``) and pass at this tip. They are behavioural, not symbol-absence.

``ref`` and ``id`` appear in **no** participant model. They exist for
``identity_admission``, where a reader that stops seeing them can only start
stripping identities -- so ``test_identity_admission_still_reads_ref_and_id`` is
the preservation obligation, asserted THROUGH ``identity_admission``'s own entry
point rather than by re-reading a constant.

No key list is defined in this file, and none was added to production: the card
consumes the right half of the canonical definition.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench import semantic as S  # noqa: E402
from t2pw.mapping import identity_admission as ia  # noqa: E402

#: A real name, so a failure cannot be blamed on an unrenderable string.
NAME = "Glucose 6-phosphate"


def _reaction_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    """SYNTHETIC, schema-legal: one reaction whose single input is ``row``.

    ``entities`` is deliberately EMPTY, so every name ``_names`` yields from the
    input slot is a referential orphan and the orphan finding count is a direct,
    scorer-visible readout of what the key list saw.
    """

    return {"entities": {"compounds": []},
            "processes": {"reactions": [{"name": "R", "inputs": [dict(row)]}]}}


# ---------------------------------------------------------------------------
# 1 + 2. The legacy tail is not an entity name.  G9 BASE FAILURES.
# ---------------------------------------------------------------------------
def test_an_id_only_participant_row_yields_no_name() -> None:
    """CORRECTION. Base SHA: ``["CHEBI:4167"]``. An identifier is not a name."""

    assert S._names([{"id": "CHEBI:4167"}]) == []


def test_a_ref_only_participant_row_yields_no_name() -> None:
    """CORRECTION. Base SHA: ``["/entities/compounds/0"]`` -- a JSON pointer."""

    assert S._names([{"ref": "/entities/compounds/0"}]) == []


def test_an_id_only_participant_is_not_a_referential_orphan() -> None:
    """CORRECTION, at the acceptance gate rather than the helper.

    Priority 3 counts references to names no entity bucket declares. On the base
    SHA an ``id``-only input produced one orphan finding, so an accession sitting
    in an ``id`` slot was scored as a dangling entity edge. It is not one.
    """

    findings = S._orphaned_references(_reaction_payload({"id": "CHEBI:4167"}))
    assert findings == [], f"an id was scored as an entity reference: {findings}"


# ---------------------------------------------------------------------------
# 3. Every schema key still yields its name -- all eight, asserted.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", [
    "name", "entity", "compound", "protein", "protein_complex", "element",
    "element_collection", "nucleic_acid",
])
def test_every_schema_name_key_still_yields_its_name(key: str) -> None:
    """The narrowing removed the legacy tail and NOTHING else.

    The eight keys are spelled out rather than imported, so this file states the
    contract instead of asserting the constant against itself.
    """

    assert S._names([{key: NAME}]) == [NAME], f"the reader stopped seeing {key!r}"


def test_the_reader_covers_exactly_the_eight_schema_keys() -> None:
    """Count guard: eight, so a ninth silently added is a visible failure here."""

    from t2pw.pipeline.participant_schema import PARTICIPANT_SCHEMA_NAME_KEYS

    assert len(PARTICIPANT_SCHEMA_NAME_KEYS) == 8
    assert set(PARTICIPANT_SCHEMA_NAME_KEYS) == {
        "name", "entity", "compound", "protein", "protein_complex", "element",
        "element_collection", "nucleic_acid"}


# ---------------------------------------------------------------------------
# 4 + 6. First-key-present precedence, unchanged.
# ---------------------------------------------------------------------------
def test_first_key_present_precedence_is_unchanged() -> None:
    """A row carrying several schema keys still resolves by the tuple's order."""

    assert S._names([{"name": "A", "entity": "B", "compound": "C"}]) == ["A"]
    assert S._names([{"entity": "B", "compound": "C", "protein": "D"}]) == ["B"]
    assert S._names([{"compound": "C", "protein": "D"}]) == ["C"]
    assert S._names([{"protein": "D", "protein_complex": "E"}]) == ["D"]
    assert S._names([{"element": "F", "nucleic_acid": "G"}]) == ["F"]


def test_a_row_carrying_name_and_id_still_yields_name() -> None:
    """Precedence is unaffected by the removal: ``name`` won before, and wins now.

    This is the shape the corpus actually has -- an entry naming its participant
    AND carrying its accession -- and the one that must not move.
    """

    assert S._names([{"name": NAME, "id": "CHEBI:4167"}]) == [NAME]
    assert S._names([{"ref": "/x", "protein": NAME}]) == [NAME]


# ---------------------------------------------------------------------------
# 5. PRESERVATION. identity_admission must keep the full union.
# ---------------------------------------------------------------------------
def test_identity_admission_still_reads_ref_and_id() -> None:
    """The obligation that matters, asserted through the reader, not the constant.

    A reader that stops seeing ``ref``/``id`` here can only start stripping
    identities, so ``identity_admission`` must NOT narrow with ``_names``.
    """

    payload = {"processes": {"reactions": [{
        "name": "R",
        "inputs": [{"id": "CHEBI:4167"}, {"ref": "/entities/compounds/0"}],
        "outputs": [{"name": NAME}],
    }]}}
    names = ia.reaction_participant_names(payload)
    assert names is not None
    assert ia.normalize_name_key("CHEBI:4167") in names, (
        "identity_admission stopped reading 'id'; identities will be stripped")
    assert ia.normalize_name_key("/entities/compounds/0") in names, (
        "identity_admission stopped reading 'ref'; identities will be stripped")
    assert ia.normalize_name_key(NAME) in names


# ---------------------------------------------------------------------------
# Derived view, not a copy: no new key list was introduced.
# ---------------------------------------------------------------------------
def test_names_is_a_derived_view_of_the_shared_schema_constant(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """NON-VACUITY for the whole file, and the no-new-key-list guard.

    Every assertion above is equally satisfied by a private tuple hard-coded in
    ``semantic.py``. Perturbing the canonical constant must change what the
    reader sees; if it does not, the reader is a copy and C-089's Ruling 7 was
    undone rather than narrowed.
    """

    from t2pw.pipeline import participant_schema

    monkeypatch.setattr(participant_schema, "PARTICIPANT_SCHEMA_NAME_KEYS",
                        ("invented_slot_key",))
    assert S._names([{"invented_slot_key": NAME}]) == [NAME]
    assert S._names([{"name": NAME}]) == []
