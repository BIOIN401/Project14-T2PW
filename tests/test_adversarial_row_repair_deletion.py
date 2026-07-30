"""ADVERSARIAL GROUP 2 — a row repairer that fixes the contract by deleting facts.

Grounding is one-sided. :func:`evidence_supports` asks whether the values a
repair ADDED are carried by the row's evidence, and says nothing whatsoever
about the values it REMOVED. So the cheapest way for a model to satisfy a
structural contract is to delete until the row stops failing — drop the two
participants it is unsure about, drop the evidence field it cannot reconcile,
drop the provenance it does not understand. Every one of those passes a
grounding check *trivially*, because nothing was added.

That is the easier failure mode for a repair prompt to produce and it was the
unguarded one. Deleting an extracted input is worse than refusing the repair: the
compound was in the paper, the extractor found it, and now no error anywhere says
it went missing.

:func:`preserves_original_values` closes it with two rules — the fields the
contract complained about may grow but not shrink, and every other field must be
byte-identical. This file attacks both, and checks that the several *legitimate*
repairs still get through.
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

from t2pw.pipeline.extraction_diagnostics import deactivate  # noqa: E402
from t2pw.pipeline.localized_repair import (  # noqa: E402
    IMMUTABLE_ROW_FIELDS,
    REPAIR_INCOMPLETE,
    REPAIR_OK,
    preserves_original_values,
    repair_invalid_rows,
)


@pytest.fixture(autouse=True)
def _clean_recorder():
    deactivate()
    yield
    deactivate()


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

#: A reaction that is RICH: four participants, a modifier, a compartment, an
#: organism, evidence, provenance, source refs, a scope label and a lock id. Every
#: one of those is a fact somebody extracted, and every one is a thing a lazy
#: repair can drop.
RICH_ROW: Dict[str, Any] = {
    "name": "gamma-glutamylcysteine synthesis",
    "inputs": ["L-glutamate", "L-cysteine", "ATP"],
    "outputs": ["gamma-glutamylcysteine", "ADP"],
    "enzymes": [{"protein": "glutamate-cysteine ligase", "evidence": "GCL joins them."}],
    "biological_state": "cytosol",
    "organism": "Homo sapiens",
    "evidence": (
        "Glutamate-cysteine ligase joins L-glutamate and L-cysteine to form "
        "gamma-glutamylcysteine, consuming ATP and releasing ADP in the cytosol "
        "of Homo sapiens cells."
    ),
    "provenance": "extracted",
    "source_refs": ["PMC13231680"],
    "source_papers": ["PMC13231680"],
    "rag_provenance": {"source_id": "PMC13231680", "tier": "A"},
    "scope_membership": "in_scope",
    "locked_reaction_id": "LR-0001",
    "preservation_status": "preserved",
    "inference": "extracted",
    "confidence": 0.9,
}


def _payload() -> Dict[str, Any]:
    return {
        "entities": {"compounds": [{"name": "L-glutamate"}]},
        "processes": {"reactions": [json.loads(json.dumps(RICH_ROW))]},
    }


ROW_LEVEL_ERROR = {
    "code": "reaction_missing_participants",
    "message": "Reaction must include at least one usable input or output participant.",
    "pointer": POINTER,
}


def _repair(returned_row: Dict[str, Any], *, attempts: int = 1) -> Any:
    provider = _Provider(
        *[
            _Reply(json.dumps({"repaired_rows": [{"pointer": POINTER, "row": returned_row}]}))
            for _ in range(attempts)
        ]
    )
    return repair_invalid_rows(_payload(), [ROW_LEVEL_ERROR], chat_fn=provider)


def _rejection(result: Any) -> Dict[str, Any]:
    assert result.rejected, "the repair was accepted when it should have been refused"
    return result.rejected[0]


# ---------------------------------------------------------------------------
# Attack 1: deleting participants.
# ---------------------------------------------------------------------------
def test_a_repair_that_drops_an_input_is_refused() -> None:
    """ATP disappears. Nothing was added, so grounding alone sees nothing wrong."""

    thinned = dict(RICH_ROW, inputs=["L-glutamate", "L-cysteine"])

    result = _repair(thinned)

    assert result.repaired_pointers == []
    assert result.outcome == REPAIR_INCOMPLETE
    finding = _rejection(result)
    assert finding["reason"] == "original_value_removed"
    assert "ATP" in json.dumps(finding["lost"])
    # And the payload still carries the original row, untouched.
    assert result.payload["processes"]["reactions"][0]["inputs"] == RICH_ROW["inputs"]


def test_a_repair_that_drops_an_output_is_refused() -> None:
    result = _repair(dict(RICH_ROW, outputs=["gamma-glutamylcysteine"]))

    assert _rejection(result)["reason"] == "original_value_removed"
    assert "ADP" in json.dumps(_rejection(result)["lost"])


def test_a_repair_that_drops_the_modifier_is_refused() -> None:
    """The enzyme is the part of a reaction most often "simplified" away."""

    result = _repair(dict(RICH_ROW, enzymes=[]))

    assert _rejection(result)["reason"] == "original_value_removed"
    assert "glutamate-cysteine ligase" in json.dumps(_rejection(result)["lost"])


def test_a_repair_that_drops_the_compartment_is_refused() -> None:
    stripped = {key: value for key, value in RICH_ROW.items() if key != "biological_state"}

    result = _repair(stripped)

    assert _rejection(result)["reason"] == "original_value_removed"
    assert "cytosol" in json.dumps(_rejection(result)["lost"])


def test_a_repair_that_drops_the_organism_is_refused() -> None:
    stripped = {key: value for key, value in RICH_ROW.items() if key != "organism"}

    result = _repair(stripped)

    assert _rejection(result)["reason"] == "original_value_removed"
    assert "Homo sapiens" in json.dumps(_rejection(result)["lost"])


def test_a_repair_that_drops_a_transporter_is_refused() -> None:
    """Same rule on the transport side, which has its own actor key."""

    payload = {
        "entities": {},
        "processes": {
            "transports": [
                {
                    "name": "glutathione efflux",
                    "cargo": "glutathione",
                    "transporters": [{"protein": "MRP1"}],
                    "evidence": "MRP1 exports glutathione.",
                }
            ]
        },
    }
    pointer = "/processes/transports/0"
    provider = _Provider(
        _Reply(
            json.dumps(
                {
                    "repaired_rows": [
                        {
                            "pointer": pointer,
                            "row": {
                                "name": "glutathione efflux",
                                "cargo": "glutathione",
                                "transporters": [],
                                "evidence": "MRP1 exports glutathione.",
                            },
                        }
                    ]
                }
            )
        )
    )

    result = repair_invalid_rows(
        payload,
        [{"code": "transport_missing_cargo", "pointer": pointer}],
        chat_fn=provider,
    )

    assert result.repaired_pointers == []
    assert result.rejected[0]["reason"] == "original_value_removed"
    assert "MRP1" in json.dumps(result.rejected[0]["lost"])


def test_a_repair_that_empties_the_row_entirely_is_refused() -> None:
    """The degenerate case: satisfy nothing by containing nothing."""

    result = _repair({"name": "gamma-glutamylcysteine synthesis"})

    assert result.repaired_pointers == []
    assert result.rejected


# ---------------------------------------------------------------------------
# Attack 2: removing provenance and the audit trail.
# ---------------------------------------------------------------------------
def test_a_repair_that_deletes_the_evidence_is_refused() -> None:
    """Without evidence the row is unciteable, and the grounding guard that
    protects every FUTURE repair of this row has nothing left to check against."""

    stripped = {key: value for key, value in RICH_ROW.items() if key != "evidence"}

    result = _repair(stripped)

    assert _rejection(result)["reason"] == "immutable_field_removed"
    assert _rejection(result)["lost"][0]["field"] == "evidence"


def test_a_repair_that_deletes_the_provenance_is_refused() -> None:
    stripped = {key: value for key, value in RICH_ROW.items() if key != "provenance"}

    result = _repair(stripped)

    assert _rejection(result)["reason"] == "immutable_field_removed"


def test_a_repair_that_deletes_the_source_references_is_refused() -> None:
    """In research mode these decide the citation tier of the row."""

    stripped = {key: value for key, value in RICH_ROW.items() if key != "source_refs"}

    result = _repair(stripped)

    assert _rejection(result)["reason"] == "immutable_field_removed"


def test_a_repair_that_changes_the_scope_label_is_refused() -> None:
    """``scope_membership`` decides whether the reaction is kept at all.

    A repair that flips it to ``out_of_scope`` deletes the reaction downstream,
    from a stage whose removal surfaces only as a log line.
    """

    result = _repair(dict(RICH_ROW, scope_membership="out_of_scope"))

    assert _rejection(result)["reason"] == "immutable_field_altered"


def test_a_repair_that_changes_the_lock_id_is_refused() -> None:
    result = _repair(dict(RICH_ROW, locked_reaction_id="LR-9999"))

    assert _rejection(result)["reason"] == "immutable_field_altered"


def test_a_repair_that_rewrites_the_name_is_refused_when_the_name_was_not_the_fault() -> None:
    """The contract complained about participants. The name is not in play."""

    result = _repair(dict(RICH_ROW, name="ATP-dependent ligation"))

    assert _rejection(result)["reason"] == "immutable_field_altered"


def test_a_repair_that_alters_an_unrelated_field_is_refused() -> None:
    result = _repair(dict(RICH_ROW, confidence=1.0))

    assert _rejection(result)["reason"] in {"immutable_field_altered", "unrelated_field_altered"}


@pytest.mark.parametrize("field_name", IMMUTABLE_ROW_FIELDS)
def test_every_immutable_field_is_actually_defended(field_name: str) -> None:
    """A list of names in a constant is not protection; this proves each entry.

    Parameterized so adding a field to :data:`IMMUTABLE_ROW_FIELDS` without
    wiring it up fails here rather than in a run six weeks later.
    """

    if field_name not in RICH_ROW:
        pytest.skip(f"{field_name} is not on this fixture row")
    stripped = {key: value for key, value in RICH_ROW.items() if key != field_name}

    ok, findings = preserves_original_values(stripped, RICH_ROW, {"inputs", "outputs"})

    assert ok is False
    assert any(item["field"] == field_name for item in findings)


# ---------------------------------------------------------------------------
# The guard must not break legitimate repairs.
# ---------------------------------------------------------------------------
def test_adding_a_participant_the_evidence_supports_is_accepted() -> None:
    """The repair this pass exists to perform: the row grows, nothing is lost."""

    grown = dict(RICH_ROW, outputs=["gamma-glutamylcysteine", "ADP", "phosphate"])
    grown["evidence"] = RICH_ROW["evidence"] + " Phosphate is released."

    provider = _Provider(
        _Reply(json.dumps({"repaired_rows": [{"pointer": POINTER, "row": grown}]}))
    )
    payload = _payload()
    payload["processes"]["reactions"][0]["evidence"] = grown["evidence"]

    result = repair_invalid_rows(payload, [ROW_LEVEL_ERROR], chat_fn=provider)

    assert result.outcome == REPAIR_OK
    assert result.repaired_pointers == [POINTER]
    assert "phosphate" in result.payload["processes"]["reactions"][0]["outputs"]


def test_moving_a_value_between_target_fields_is_accepted() -> None:
    """``products`` -> ``outputs`` is normalization, not deletion.

    The fact survives; only the key it sits under changed. A field-by-field rule
    would read this as a loss, which is why the check is over the union of the
    fields the contract complained about.
    """

    original = {
        "name": "r",
        "inputs": ["ATP"],
        "outputs": [],
        "products": ["ADP"],
        "evidence": "ATP becomes ADP.",
    }
    ok, findings = preserves_original_values(
        {"name": "r", "inputs": ["ATP"], "outputs": ["ADP"], "evidence": "ATP becomes ADP."},
        original,
        {"inputs", "outputs", "products"},
    )

    assert ok is True, findings


def test_filling_the_field_that_was_empty_is_accepted() -> None:
    """The commonest real repair: the failing field had nothing to lose."""

    original = {"name": "", "evidence": "LpxA acetylates UDP-GlcNAc."}
    ok, _ = preserves_original_values(
        {"name": "LpxA", "evidence": "LpxA acetylates UDP-GlcNAc."}, original, {"name"}
    )

    assert ok is True


def test_the_guard_bounds_what_it_reports() -> None:
    """A row that lost fifty values must not put fifty of them in an artifact."""

    original = {"inputs": [f"compound {index}" for index in range(50)]}
    ok, findings = preserves_original_values({"inputs": []}, original, {"inputs"})

    assert ok is False
    assert len(findings) == 1
    assert len(findings[0]["missing"]) <= 8


def test_the_repair_prompt_never_carries_the_guard_bookkeeping() -> None:
    """``_target_fields`` is the guard's input, not the model's.

    Telling the model which fields it is allowed to change would be handing it
    the answer key to the check that is supposed to be independent of it.
    """

    provider = _Provider(_Reply(json.dumps({"repaired_rows": []})))

    repair_invalid_rows(_payload(), [ROW_LEVEL_ERROR], chat_fn=provider)

    assert "_target_fields" not in provider.prompts[0]
