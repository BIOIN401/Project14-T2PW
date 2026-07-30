"""PROBE 1 — a JSON syntax repairer that fixes the syntax by throwing content away.

The first version of the content guard tested that the repaired literal stream
was an ordered SUBSEQUENCE of the malformed input's. That half is about
invention: a subsequence cannot contain a token the input never had, cannot
reorder, cannot duplicate. It says nothing at all about deletion.

So a repair could return one of two complete reactions, or drop a middle entity
row, and pass — silently, from an extraction that was correct about them, with
the fault surfacing three stages later as "Payload must include a processes
object" or simply as a smaller pathway nobody questioned. That is the same class
of loss as the row-repair deletion gap, one level up, and worse: it happens
before the row even exists.

The rule is now equality, not containment. The repaired stream must be exactly
the input's complete literals, **optionally minus the final one, and only when
the document was cut off mid-value and that literal is the key the value belonged
to**. That is the single case the repair prompt asks the model to handle by
deleting, and nothing else authorizes a deletion: not a trailing comma, not a
missing comma, not a code fence, not a missing closing brace.

This file probes both directions — every deletion that must be refused, and every
syntax fault that must still be repairable.
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
    REPAIR_OK,
    REPAIR_SEMANTIC_GUARD_FAILED,
    json_repair_preserves_content,
    repair_json_text,
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


# ---------------------------------------------------------------------------
# The headline case: two complete reactions, one trailing comma.
# ---------------------------------------------------------------------------
#: Both reactions are COMPLETE. The only fault is the trailing comma after the
#: second one — a syntax error that authorizes nothing.
TWO_REACTIONS_TRAILING_COMMA = (
    '{"entities": {"compounds": [{"name": "ATP"}, {"name": "ADP"}, {"name": "AMP"}]}, '
    '"processes": {"reactions": ['
    '{"name": "r1", "inputs": ["ATP"], "outputs": ["ADP"]}, '
    '{"name": "r2", "inputs": ["ADP"], "outputs": ["AMP"]},'
    "]}}"
)

BOTH_REACTIONS = (
    '{"entities": {"compounds": [{"name": "ATP"}, {"name": "ADP"}, {"name": "AMP"}]}, '
    '"processes": {"reactions": ['
    '{"name": "r1", "inputs": ["ATP"], "outputs": ["ADP"]}, '
    '{"name": "r2", "inputs": ["ADP"], "outputs": ["AMP"]}'
    "]}}"
)

ONLY_FIRST_REACTION = (
    '{"entities": {"compounds": [{"name": "ATP"}, {"name": "ADP"}, {"name": "AMP"}]}, '
    '"processes": {"reactions": ['
    '{"name": "r1", "inputs": ["ATP"], "outputs": ["ADP"]}'
    "]}}"
)


def _repair(malformed: str, reply: str) -> Any:
    provider = _Provider(_Reply(reply))
    return repair_json_text(malformed, "JSONDecodeError", chat_fn=provider)


def test_a_trailing_comma_repair_that_drops_a_reaction_is_refused() -> None:
    """THE probe. Two complete reactions in, one out, because of a comma.

    A trailing comma is a one-character fault. Answering it by deleting an
    evidenced reaction is not a repair, and a subsequence-only guard accepted it.
    """

    result = _repair(TWO_REACTIONS_TRAILING_COMMA, ONLY_FIRST_REACTION)

    assert not result.ok
    assert result.outcome == REPAIR_SEMANTIC_GUARD_FAILED
    assert result.guard["divergence"] == "removed"
    assert "r2" in result.guard["removed_sample"]
    assert "AMP" in json.dumps(result.guard["removed_sample"])


def test_the_same_trailing_comma_repaired_honestly_is_accepted() -> None:
    """The control. Without this the probe above only proves the guard says no."""

    result = _repair(TWO_REACTIONS_TRAILING_COMMA, BOTH_REACTIONS)

    assert result.ok
    assert result.outcome == REPAIR_OK
    assert len(result.payload["processes"]["reactions"]) == 2


# ---------------------------------------------------------------------------
# Deleting a row from each position.
# ---------------------------------------------------------------------------
#: Three entity rows and two reactions, one missing comma. Every literal complete.
THREE_ROWS = (
    '{"entities": {"compounds": [{"name": "ATP"}, {"name": "ADP"}, {"name": "AMP"}]}, '
    '"processes": {"reactions": ['
    '{"name": "r1" "inputs": ["ATP"], "outputs": ["ADP"]}, '
    '{"name": "r2", "inputs": ["ADP"], "outputs": ["AMP"]}'
    "]}}"
)


def _entities(*names: str) -> str:
    rows = ", ".join(f'{{"name": "{name}"}}' for name in names)
    return (
        f'{{"entities": {{"compounds": [{rows}]}}, '
        '"processes": {"reactions": ['
        '{"name": "r1", "inputs": ["ATP"], "outputs": ["ADP"]}, '
        '{"name": "r2", "inputs": ["ADP"], "outputs": ["AMP"]}'
        "]}}"
    )


@pytest.mark.parametrize(
    "position, reply, dropped",
    [
        ("first", _entities("ADP", "AMP"), "ATP"),
        ("middle", _entities("ATP", "AMP"), "ADP"),
        ("last", _entities("ATP", "ADP"), "AMP"),
    ],
)
def test_deleting_an_entity_row_from_any_position_is_refused(
    position: str, reply: str, dropped: str
) -> None:
    """First, middle and last, because they fail differently.

    Dropping the FIRST or MIDDLE row also shifts everything after it, so an
    order-sensitive check notices immediately. Dropping the LAST one does not
    shift anything — the remaining stream is a clean prefix — and that is exactly
    the deletion a subsequence test cannot see. All three must be refused for the
    same reason.
    """

    result = _repair(THREE_ROWS, reply)

    assert result.outcome == REPAIR_SEMANTIC_GUARD_FAILED, position
    assert result.guard["divergence"] == "removed"
    assert dropped in result.guard["removed_sample"]


def _reactions(*names: str) -> str:
    bodies = {
        "r1": '{"name": "r1", "inputs": ["ATP"], "outputs": ["ADP"]}',
        "r2": '{"name": "r2", "inputs": ["ADP"], "outputs": ["AMP"]}',
        "r3": '{"name": "r3", "inputs": ["AMP"], "outputs": ["adenosine"]}',
    }
    rows = ", ".join(bodies[name] for name in names)
    return (
        '{"entities": {"compounds": [{"name": "ATP"}, {"name": "ADP"}, '
        '{"name": "AMP"}, {"name": "adenosine"}]}, '
        f'"processes": {{"reactions": [{rows}]}}}}'
    )


THREE_REACTIONS_MALFORMED = _reactions("r1", "r2", "r3").replace('"r1",', '"r1"', 1)


@pytest.mark.parametrize(
    "position, kept, dropped",
    [
        ("first", ("r2", "r3"), "r1"),
        ("middle", ("r1", "r3"), "r2"),
        ("last", ("r1", "r2"), "r3"),
    ],
)
def test_deleting_a_reaction_from_any_position_is_refused(
    position: str, kept: tuple, dropped: str
) -> None:
    result = _repair(THREE_REACTIONS_MALFORMED, _reactions(*kept))

    assert result.outcome == REPAIR_SEMANTIC_GUARD_FAILED, position
    assert result.guard["divergence"] == "removed"
    assert dropped in result.guard["removed_sample"]


def test_keeping_all_three_reactions_is_accepted() -> None:
    result = _repair(THREE_REACTIONS_MALFORMED, _reactions("r1", "r2", "r3"))

    assert result.ok
    assert [row["name"] for row in result.payload["processes"]["reactions"]] == ["r1", "r2", "r3"]


# ---------------------------------------------------------------------------
# Which syntax faults authorize a deletion. (Answer: only truncation.)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label, malformed",
    [
        ("trailing comma", '{"a": "x", "b": "y",}'),
        ("missing comma", '{"a": "x" "b": "y"}'),
        ("code fence", '```json\n{"a": "x", "b": "y"}\n```'),
        ("missing closing brace", '{"a": "x", "b": "y"'),
        ("missing closing bracket", '{"a": ["x", "y"'),
    ],
)
def test_no_ordinary_syntax_fault_authorizes_dropping_a_field(
    label: str, malformed: str
) -> None:
    """Every literal in these is complete. There is nothing to legally delete."""

    ok, finding = json_repair_preserves_content(malformed, {"a": "x"})

    assert ok is False, label
    assert finding["divergence"] == "removed"
    assert finding["droppable_tail"] == 0


@pytest.mark.parametrize(
    "label, malformed, repaired",
    [
        ("trailing comma", '{"a": "x", "b": "y",}', {"a": "x", "b": "y"}),
        ("missing comma", '{"a": "x" "b": "y"}', {"a": "x", "b": "y"}),
        ("code fence", '```json\n{"a": "x", "b": "y"}\n```', {"a": "x", "b": "y"}),
        ("missing closing brace", '{"a": "x", "b": "y"', {"a": "x", "b": "y"}),
        ("missing closing bracket", '{"a": ["x", "y"', {"a": ["x", "y"]}),
    ],
)
def test_every_ordinary_syntax_fault_is_still_repairable(
    label: str, malformed: str, repaired: Dict[str, Any]
) -> None:
    """The other half: closing the delimiter and keeping everything is correct."""

    ok, _finding = json_repair_preserves_content(malformed, repaired)

    assert ok is True, label


def test_only_a_truncated_value_lets_its_own_key_go() -> None:
    """The one authorized deletion, and it is authorized for one token."""

    truncated = '{"pathway_name": "Glutathione biosynthesis", "likely_organism": "Homo sap'

    ok, finding = json_repair_preserves_content(
        truncated, {"pathway_name": "Glutathione biosynthesis"}
    )

    assert ok is True
    assert finding["input_truncated"] is True
    assert finding["droppable_tail"] == 1


def test_the_authorized_deletion_does_not_extend_to_a_second_field() -> None:
    """One token, not "the tail". Dropping the previous pair too is a deletion."""

    truncated = '{"pathway_name": "Glutathione biosynthesis", "likely_organism": "Homo sap'

    ok, finding = json_repair_preserves_content(truncated, {})

    assert ok is False
    assert finding["divergence"] == "removed"
    assert "Glutathione biosynthesis" in finding["removed_sample"]


def test_a_truncated_value_in_the_middle_of_an_array_authorizes_nothing() -> None:
    """The last complete literal is a VALUE, not the key of the truncated thing.

    ``{"a": ["x", "y`` was cut off inside the array, so ``"x"`` is complete and
    must survive; there is no key to drop.
    """

    ok, finding = json_repair_preserves_content('{"a": ["x", "y', {"a": []})

    assert ok is False
    assert finding["droppable_tail"] == 0
    assert "x" in finding["removed_sample"]

    ok_kept, _ = json_repair_preserves_content('{"a": ["x", "y', {"a": ["x"]})
    assert ok_kept is True


def test_a_truncated_key_authorizes_nothing() -> None:
    """When the KEY is what was cut off, it produced no literal to begin with."""

    ok, finding = json_repair_preserves_content('{"a": "x", "b', {"a": "x"})

    assert ok is True
    assert finding["droppable_tail"] == 0


# ---------------------------------------------------------------------------
# The stored fixture must keep recovering.
# ---------------------------------------------------------------------------
def test_the_stored_invalid_json_fixture_still_recovers() -> None:
    """Pinned here as well as in the replay harness.

    The equality rule is the strictest thing in this module, and the stored
    fixture is the one case that must pass it. Checking it directly means a
    tightening that breaks it fails in the file that did the tightening.
    """

    cases = json.loads(
        (ROOT / "tests" / "fixtures" / "early_failures" / "cases.json").read_text(
            encoding="utf-8"
        )
    )["cases"]
    case = next(item for item in cases if item["id"] == "stage1_invalid_json_localized_repair")
    malformed = case["replies"][1]["content"]
    repaired_text = case["replies"][2]["content"]

    ok, finding = json_repair_preserves_content(
        malformed, json.loads(repaired_text), repaired_text=repaired_text
    )

    assert ok is True, finding
    assert finding["input_literal_count"] == finding["repaired_literal_count"]
