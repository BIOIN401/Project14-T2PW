"""PROBE 2 — a row repairer that ADDS what the row never had.

The deletion guard closed one direction: a repair may not drop an extracted fact.
It walked the ORIGINAL row's non-empty fields, which is exactly the wrong place to
stand to see the other direction. Four things slipped past it:

* adding an immutable field the row never carried — a fabricated ``provenance``
  or ``source_refs`` is worse than a missing one, because it is *citable*: it
  claims the row came from somewhere, and every tiering, review and export path
  downstream believes it;
* populating an immutable or unrelated field that was present but empty, which
  is the same fabrication wearing a key that already existed;
* deleting a field whose value was empty, since a walk over non-empty fields
  skipped it entirely;
* introducing any brand-new top-level key, which a walk over the original's keys
  is structurally incapable of noticing.

The rule is now exact shape: outside the fields the contract explicitly named as
repairable, every key must be present if it was present, absent if it was absent,
and equal if it was equal — empty values included. And "explicitly named" is read
from the contract itself, including the ``required_any`` an error carries, so a
participants error cannot license writing an organism.

This file probes each of the four, on every field that matters, and checks the
legitimate repairs still get through.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.localized_repair import (  # noqa: E402
    IMMUTABLE_ROW_FIELDS,
    REPAIR_OK,
    preserves_original_values,
    repair_invalid_rows,
)


class _Reply:
    def __init__(self, text: str) -> None:
        self.text = text
        self.diagnostics = {
            "model": "test-model",
            "finish_reason": "stop",
            "attempts": 1,
            "response_status": "ok",
            "terminal_reason": "",
            "attempt_log": [],
        }


class _Provider:
    def __init__(self, *replies: _Reply) -> None:
        self._replies = list(replies)
        self.prompts: List[str] = []

    def __call__(self, messages: List[Dict[str, str]], **_: Any) -> _Reply:
        self.prompts.append(messages[-1]["content"])
        if not self._replies:
            raise AssertionError("more repair draws were made than were scripted")
        return self._replies.pop(0)


POINTER = "/processes/reactions/0"

#: A SPARSE row: the failing field is empty and the whole audit trail is ABSENT —
#: no provenance, no source refs, no scope label, no confidence. That absence is
#: the blind spot: a guard that walks the original's fields has almost nothing to
#: walk, so every one of those keys could be conjured into existence unnoticed.
#:
#: ``evidence`` is present and supports the honest fill, so the probes below
#: isolate the SHAPE guard rather than tripping the grounding guard first. The
#: empty-value cases are probed through :func:`preserves_original_values`
#: directly, where "empty" can be stated exactly.
SPARSE_ROW: Dict[str, Any] = {
    "name": "gamma-glutamylcysteine synthesis",
    "inputs": ["L-glutamate"],
    "outputs": [],
    "evidence": "L-glutamate is converted to gamma-glutamylcysteine.",
}

#: The contract's own words. ``required_any`` is what ``_validate_reaction_structure``
#: attaches to this code, and it is what narrows the repair's licence to two fields.
PARTICIPANTS_ERROR = {
    "code": "reaction_missing_participants",
    "message": "Reaction must include at least one usable input or output participant.",
    "pointer": POINTER,
    "required_any": ["inputs", "outputs"],
}


def _payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "entities": {"compounds": [{"name": "L-glutamate"}]},
        "processes": {"reactions": [json.loads(json.dumps(row))]},
    }


def _repair(returned_row: Dict[str, Any], *, row: Dict[str, Any] = None) -> Any:
    provider = _Provider(
        _Reply(json.dumps({"repaired_rows": [{"pointer": POINTER, "row": returned_row}]}))
    )
    return repair_invalid_rows(
        _payload(row if row is not None else SPARSE_ROW), [PARTICIPANTS_ERROR], chat_fn=provider
    )


def _rejection(result: Any) -> Dict[str, Any]:
    assert result.rejected, "the repair was accepted when it should have been refused"
    return result.rejected[0]


#: The honest repair of SPARSE_ROW: fill the empty ``outputs`` and touch nothing
#: else. Every probe below is this plus one forbidden addition.
def _honest(**extra: Any) -> Dict[str, Any]:
    row = dict(SPARSE_ROW, outputs=["gamma-glutamylcysteine"])
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# The control.
# ---------------------------------------------------------------------------
def test_the_honest_repair_of_the_sparse_row_is_accepted() -> None:
    """Fill the field the contract named. Change nothing else. That is a repair."""

    result = _repair(_honest())

    assert result.outcome == REPAIR_OK
    assert result.repaired_pointers == [POINTER]
    row = result.payload["processes"]["reactions"][0]
    assert row["outputs"] == ["gamma-glutamylcysteine"]
    assert row["evidence"] == SPARSE_ROW["evidence"]  # untouched


# ---------------------------------------------------------------------------
# Attack 1: inventing an ABSENT immutable field.
# ---------------------------------------------------------------------------
def test_invented_evidence_is_refused() -> None:
    """A row's evidence is rewritten into a sentence that supports the repair.

    This is the most consequential single fabrication in the pipeline: evidence
    is what every later grounding check compares against, so an invented passage
    does not merely add one false fact — it licenses every future repair of the
    row to add more.
    """

    result = _repair(_honest(evidence="Glutamate-cysteine ligase joins them."))

    assert result.repaired_pointers == []
    finding = _rejection(result)
    assert finding["reason"] == "immutable_field_altered"
    assert finding["lost"][0]["field"] == "evidence"


def test_evidence_conjured_onto_a_row_that_had_none_is_refused() -> None:
    """The absent-key form of the same attack, stated exactly.

    A row carrying no ``evidence`` at all is the one a fabricated passage helps
    most, and it is the one a walk over the original's fields cannot see.
    """

    row = {key: value for key, value in SPARSE_ROW.items() if key != "evidence"}
    returned = dict(row, outputs=["x"], evidence="A sentence nobody wrote.")

    ok, findings = preserves_original_values(returned, row, {"inputs", "outputs"})

    assert ok is False
    assert findings[0]["field"] == "evidence"
    assert findings[0]["reason"] == "immutable_field_added"


def test_invented_provenance_is_refused() -> None:
    """"extracted" claims a human-traceable origin the row does not have."""

    result = _repair(_honest(provenance="extracted"))

    finding = _rejection(result)
    assert finding["reason"] == "immutable_field_added"
    assert finding["lost"][0]["field"] == "provenance"
    assert "extracted" in finding["lost"][0]["added"]


def test_invented_source_refs_are_refused() -> None:
    """A fabricated citation is the failure mode a reviewer cannot catch by eye."""

    result = _repair(_honest(source_refs=["PMC13231680"]))

    finding = _rejection(result)
    assert finding["reason"] == "immutable_field_added"
    assert finding["lost"][0]["field"] == "source_refs"


def test_invented_rag_provenance_is_refused() -> None:
    """It would make a single-paper row look like multi-paper evidence."""

    result = _repair(_honest(rag_provenance={"source_id": "PMC1", "tier": "A"}))

    finding = _rejection(result)
    assert finding["reason"] == "immutable_field_added"
    assert finding["lost"][0]["field"] == "rag_provenance"


def test_invented_scope_membership_is_refused() -> None:
    """``scope_membership`` decides whether the reaction survives at all.

    Writing ``out_of_scope`` deletes it downstream, from a filter whose removals
    surface only as a log line — a deletion disguised as a label.
    """

    result = _repair(_honest(scope_membership="out_of_scope"))

    finding = _rejection(result)
    assert finding["reason"] == "immutable_field_added"
    assert finding["lost"][0]["field"] == "scope_membership"


def test_invented_confidence_is_refused() -> None:
    """A number nobody computed, which downstream ranking will believe."""

    result = _repair(_honest(confidence=0.99))

    finding = _rejection(result)
    assert finding["reason"] == "immutable_field_added"
    assert finding["lost"][0]["field"] == "confidence"


@pytest.mark.parametrize("field_name", IMMUTABLE_ROW_FIELDS)
def test_no_immutable_field_can_be_conjured_from_absence(field_name: str) -> None:
    """Every entry in the constant, not just the six with named probes above.

    Parameterized so adding a field to :data:`IMMUTABLE_ROW_FIELDS` without it
    actually being defended fails here.
    """

    row = {key: value for key, value in SPARSE_ROW.items() if key != field_name}
    returned = dict(row, outputs=["x"])
    returned[field_name] = "invented" if field_name != "confidence" else 0.5

    ok, findings = preserves_original_values(returned, row, {"inputs", "outputs"})

    assert ok is False
    assert any(item["field"] == field_name for item in findings)


# ---------------------------------------------------------------------------
# Attack 2: populating an EMPTY field.
# ---------------------------------------------------------------------------
def test_populating_an_empty_immutable_field_is_refused() -> None:
    """Same fabrication, wearing a key that already existed but held nothing.

    The old guard skipped empty originals entirely, so this was the easiest way
    through it: the key is already there, so nothing looks "added", and the value
    was empty, so nothing looks "lost".
    """

    row = dict(SPARSE_ROW, evidence="")
    returned = dict(row, outputs=["x"], evidence="A sentence that was not there.")

    ok, findings = preserves_original_values(returned, row, {"inputs", "outputs"})

    assert ok is False
    assert findings[0]["field"] == "evidence"
    assert findings[0]["reason"] == "immutable_field_altered"


def test_populating_an_empty_unrelated_field_is_refused() -> None:
    row = dict(SPARSE_ROW, biological_state="")
    returned = dict(row, outputs=["gamma-glutamylcysteine"], biological_state="mitochondrion")

    ok, findings = preserves_original_values(returned, row, {"inputs", "outputs"})

    assert ok is False
    assert findings[0]["field"] == "biological_state"
    assert findings[0]["reason"] == "unrelated_field_altered"


# ---------------------------------------------------------------------------
# Attack 3: deleting an EMPTY field.
# ---------------------------------------------------------------------------
def test_deleting_an_empty_immutable_field_is_refused() -> None:
    """Exact shape means exact, and an empty string is a shape.

    ``evidence: ""`` and no ``evidence`` key are different states: the first says
    somebody looked and found nothing, the second says nobody looked. A guard
    that only inspects non-empty originals cannot tell them apart, so tidying the
    key away was free.
    """

    row = dict(SPARSE_ROW, evidence="")
    returned = {
        key: value for key, value in dict(row, outputs=["x"]).items() if key != "evidence"
    }

    ok, findings = preserves_original_values(returned, row, {"inputs", "outputs"})

    assert ok is False
    assert findings[0]["field"] == "evidence"
    assert findings[0]["reason"] == "immutable_field_removed"


def test_deleting_a_populated_immutable_field_is_still_refused_end_to_end() -> None:
    """The same rule through the full repair path, on a non-empty value."""

    returned = {key: value for key, value in _honest().items() if key != "evidence"}

    result = _repair(returned)

    finding = _rejection(result)
    assert finding["reason"] == "immutable_field_removed"
    assert finding["lost"][0]["field"] == "evidence"


def test_deleting_an_empty_unrelated_field_is_refused() -> None:
    row = dict(SPARSE_ROW, aliases=[])
    returned = {
        key: value
        for key, value in dict(row, outputs=["gamma-glutamylcysteine"]).items()
        if key != "aliases"
    }

    ok, findings = preserves_original_values(returned, row, {"inputs", "outputs"})

    assert ok is False
    assert findings[0]["field"] == "aliases"
    assert findings[0]["reason"] == "unrelated_field_removed"


# ---------------------------------------------------------------------------
# Attack 4: introducing a brand-new top-level field.
# ---------------------------------------------------------------------------
def test_a_newly_invented_top_level_field_is_refused() -> None:
    """A key nobody has ever heard of. A walk over the original cannot see it."""

    result = _repair(_honest(reaction_class="ligation"))

    finding = _rejection(result)
    assert finding["reason"] == "unrelated_field_added"
    assert finding["lost"][0]["field"] == "reaction_class"


def test_a_scientific_field_outside_the_contracts_remit_is_refused() -> None:
    """THE narrowing that ``required_any`` buys.

    ``organism`` is a scientific field, so a blanket "any scientific field may
    grow" rule would let a *participants* error license writing one. The contract
    named ``inputs`` and ``outputs``; everything else on the row is frozen,
    whether or not it is biological.
    """

    result = _repair(_honest(organism="Homo sapiens"))

    finding = _rejection(result)
    assert finding["reason"] == "unrelated_field_added"
    assert finding["lost"][0]["field"] == "organism"


def test_a_row_level_error_with_no_hint_still_permits_the_scientific_fields() -> None:
    """The fallback is unchanged for errors that name nothing.

    Not every contract error carries ``required_any``; those still get the wider
    licence, because the alternative is a repair that cannot repair.
    """

    bare_error = {"code": "reaction_missing_participants", "pointer": POINTER}
    provider = _Provider(
        _Reply(
            json.dumps(
                {
                    "repaired_rows": [
                        {
                            "pointer": POINTER,
                            "row": dict(
                                SPARSE_ROW,
                                outputs=["gamma-glutamylcysteine"],
                                biological_state="cytosol",
                                evidence="gamma-glutamylcysteine forms in the cytosol.",
                            ),
                        }
                    ]
                }
            )
        )
    )
    payload = _payload(dict(SPARSE_ROW, evidence="gamma-glutamylcysteine forms in the cytosol."))

    result = repair_invalid_rows(payload, [bare_error], chat_fn=provider)

    assert result.outcome == REPAIR_OK
    assert result.payload["processes"]["reactions"][0]["biological_state"] == "cytosol"


# ---------------------------------------------------------------------------
# The legitimate repairs must survive all of this.
# ---------------------------------------------------------------------------
def test_products_to_outputs_migration_still_works() -> None:
    """Preserved explicitly, because it is the repair most likely to be lost.

    ``products`` and ``outputs`` are both named by the contract here, so moving a
    participant between them keeps the fact and changes only the key. A per-field
    rule would call that a deletion; the union check does not.
    """

    row = {
        "name": "r",
        "inputs": ["ATP"],
        "outputs": [],
        "products": ["ADP"],
        "evidence": "ATP becomes ADP.",
    }
    returned = {"name": "r", "inputs": ["ATP"], "outputs": ["ADP"], "evidence": "ATP becomes ADP."}

    ok, findings = preserves_original_values(returned, row, {"inputs", "outputs", "products"})

    assert ok is True, findings


def test_products_to_outputs_migration_survives_the_full_repair_path() -> None:
    """End to end, not just the predicate: the guard order must not block it."""

    row = {
        "name": "r",
        "inputs": ["ATP"],
        "outputs": [],
        "products": ["ADP"],
        "evidence": "ATP becomes ADP.",
    }
    returned = {"name": "r", "inputs": ["ATP"], "outputs": ["ADP"], "evidence": "ATP becomes ADP."}
    provider = _Provider(
        _Reply(json.dumps({"repaired_rows": [{"pointer": POINTER, "row": returned}]}))
    )
    error = dict(PARTICIPANTS_ERROR, required_any=["inputs", "outputs", "products"])

    result = repair_invalid_rows(_payload(row), [error], chat_fn=provider)

    assert result.outcome == REPAIR_OK
    assert result.payload["processes"]["reactions"][0]["outputs"] == ["ADP"]
    assert "products" not in result.payload["processes"]["reactions"][0]


def test_growing_a_named_field_with_supported_evidence_still_works() -> None:
    row = dict(SPARSE_ROW, evidence="L-glutamate and L-cysteine form gamma-glutamylcysteine.")
    returned = dict(
        row,
        inputs=["L-glutamate", "L-cysteine"],
        outputs=["gamma-glutamylcysteine"],
    )
    provider = _Provider(
        _Reply(json.dumps({"repaired_rows": [{"pointer": POINTER, "row": returned}]}))
    )

    result = repair_invalid_rows(_payload(row), [PARTICIPANTS_ERROR], chat_fn=provider)

    assert result.outcome == REPAIR_OK
    assert result.payload["processes"]["reactions"][0]["inputs"] == [
        "L-glutamate",
        "L-cysteine",
    ]


def test_the_contract_named_fields_reach_the_prompt() -> None:
    """The model is told which fields it may touch; the guard checks it anyway."""

    provider = _Provider(_Reply(json.dumps({"repaired_rows": []})))

    repair_invalid_rows(_payload(SPARSE_ROW), [PARTICIPANTS_ERROR], chat_fn=provider)

    assert "required_any" in provider.prompts[0]
    assert "_target_fields" not in provider.prompts[0]
