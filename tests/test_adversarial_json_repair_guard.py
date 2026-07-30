"""ADVERSARIAL GROUP 1 — a JSON repairer that returns valid JSON and lies.

The system prompt in ``localized_repair`` tells the repair model, in absolute
terms, not to add, remove, rename, reorder or reword anything. **A model
instruction is not enforcement.** A model that ignores it returns perfectly
well-formed JSON containing a reaction nobody extracted, and from that point on
the invention is indistinguishable from extracted biology: it has a name, it has
participants, it flows through cleaning, mapping and export exactly like a real
one. The repair pass would have become the most dangerous stage in the pipeline
— the only one that can manufacture a pathway from a syntax error.

:func:`json_repair_preserves_content` is the mechanical check that makes the
instruction binding. Every complete key and scalar the repaired document
contains must appear in the malformed input, **in the same relative order**. This
file attacks that check from every direction a repair can go wrong: invention,
substitution, reordering, duplication, and the several *legitimate* repairs it
must not break.

The tests are named for the attack, not the mechanism, because that is how a
reviewer decides whether the guard still covers the thing they care about.
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

from t2pw.pipeline.extraction_diagnostics import (  # noqa: E402
    BOUNDARY_REPORT_NAME,
    OUTCOME_SEMANTIC_GUARD_FAILED,
    activate,
    deactivate,
)
from t2pw.pipeline.localized_repair import (  # noqa: E402
    REPAIR_OK,
    REPAIR_SEMANTIC_GUARD_FAILED,
    json_repair_preserves_content,
    repair_json_text,
)


@pytest.fixture(autouse=True)
def _clean_recorder():
    deactivate()
    yield
    deactivate()


class _Reply:
    def __init__(self, text: str, **diagnostics: Any) -> None:
        self.text = text
        self.diagnostics = {
            "model": "test-model",
            "finish_reason": "stop",
            "attempts": 1,
            "response_status": "ok",
            "terminal_reason": "",
            "attempt_log": [],
            **diagnostics,
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


#: The real shape: one missing comma between ``"r1"`` and ``"inputs"``. Valid
#: content, invalid syntax, and the prefix salvage drops the whole processes
#: block — which is why a repair is attempted at all.
MALFORMED = (
    '{"entities": {"compounds": [{"name": "ATP"}, {"name": "ADP"}]}, '
    '"processes": {"reactions": [{"name": "r1" "inputs": ["ATP"], "outputs": ["ADP"]}]}}'
)

#: The only honest repair of MALFORMED: the same document, one comma added.
HONEST = (
    '{"entities": {"compounds": [{"name": "ATP"}, {"name": "ADP"}]}, '
    '"processes": {"reactions": [{"name": "r1", "inputs": ["ATP"], "outputs": ["ADP"]}]}}'
)


def _repair(reply: str) -> Any:
    provider = _Provider(_Reply(reply))
    return repair_json_text(MALFORMED, "JSONDecodeError: Expecting ',' delimiter", chat_fn=provider)


# ---------------------------------------------------------------------------
# The control: an honest repair must still work.
# ---------------------------------------------------------------------------
def test_the_honest_repair_is_accepted() -> None:
    """If this ever fails, the guard has stopped being a guard and started being
    a wall, and the stored invalid-JSON fixture stops recovering with it."""

    result = _repair(HONEST)

    assert result.ok
    assert result.outcome == REPAIR_OK
    assert result.payload["processes"]["reactions"][0]["inputs"] == ["ATP"]


# ---------------------------------------------------------------------------
# Attack 1: invention.
# ---------------------------------------------------------------------------
def test_an_invented_reaction_is_refused() -> None:
    """The worst case. A whole second reaction, syntactically flawless.

    Nothing downstream can tell an invented reaction from an extracted one, so
    this has to die here or not at all.
    """

    invented = json.dumps(
        {
            "entities": {"compounds": [{"name": "ATP"}, {"name": "ADP"}]},
            "processes": {
                "reactions": [
                    {"name": "r1", "inputs": ["ATP"], "outputs": ["ADP"]},
                    {"name": "r2", "inputs": ["ADP"], "outputs": ["AMP"]},
                ]
            },
        }
    )

    result = _repair(invented)

    assert not result.ok
    assert result.outcome == REPAIR_SEMANTIC_GUARD_FAILED
    assert result.payload is None
    # The token that runs out is ``name`` -- the second reaction needs a second
    # occurrence of that key and the input supplies one. Worth pinning rather
    # than accepting any refusal: it shows the guard catches the invention at the
    # first literal the input cannot account for, not by noticing "AMP" several
    # tokens later, so a reaction invented entirely out of REAL names is caught
    # just the same.
    assert result.guard["first_unmatched"] == "name"
    assert result.guard["repaired_literal_count"] > result.guard["input_literal_count"]


def test_an_invented_participant_inside_a_real_reaction_is_refused() -> None:
    """Subtler and likelier: the reaction is real, one cofactor is not."""

    padded = HONEST.replace('"inputs": ["ATP"]', '"inputs": ["ATP", "NADPH"]')

    result = _repair(padded)

    assert result.outcome == REPAIR_SEMANTIC_GUARD_FAILED
    assert "NADPH" in result.reason


def test_an_invented_key_is_refused() -> None:
    """A repairer that can add a field can add a reaction. Same defect."""

    annotated = HONEST.replace(
        '"outputs": ["ADP"]', '"outputs": ["ADP"], "evidence": "ATP is hydrolysed."'
    )

    result = _repair(annotated)

    assert result.outcome == REPAIR_SEMANTIC_GUARD_FAILED
    assert "evidence" in result.reason


def test_an_invented_numeric_value_is_refused() -> None:
    """Scalars are guarded too, or a stoichiometry could be conjured."""

    with_number = HONEST.replace('"outputs": ["ADP"]', '"outputs": ["ADP"], "stoichiometry": 2')

    result = _repair(with_number)

    assert result.outcome == REPAIR_SEMANTIC_GUARD_FAILED


# ---------------------------------------------------------------------------
# Attack 2: substitution.
# ---------------------------------------------------------------------------
def test_a_changed_participant_is_refused() -> None:
    """ADP silently becomes AMP. Every count stays identical, so nothing that
    merely compares sizes would notice."""

    swapped = HONEST.replace('"outputs": ["ADP"]}]', '"outputs": ["AMP"]}]')

    result = _repair(swapped)

    assert result.outcome == REPAIR_SEMANTIC_GUARD_FAILED
    assert "AMP" in result.reason


def test_a_reworded_value_is_refused() -> None:
    """"r1" -> "ATP hydrolysis" reads like a helpful tidy-up. It is a rewrite."""

    renamed = HONEST.replace('"name": "r1"', '"name": "ATP hydrolysis"')

    result = _repair(renamed)

    assert result.outcome == REPAIR_SEMANTIC_GUARD_FAILED


def test_a_renamed_key_is_refused() -> None:
    substituted = HONEST.replace('"outputs"', '"products"')

    result = _repair(substituted)

    assert result.outcome == REPAIR_SEMANTIC_GUARD_FAILED


# ---------------------------------------------------------------------------
# Attack 3: reordering.
# ---------------------------------------------------------------------------
def test_a_reordered_array_is_refused() -> None:
    """Direction matters in this schema: inputs are not outputs.

    The two compounds are both present and both legitimate, so a set-based check
    would wave this through and the reaction would ship backwards.
    """

    reversed_sides = (
        '{"entities": {"compounds": [{"name": "ATP"}, {"name": "ADP"}]}, '
        '"processes": {"reactions": [{"name": "r1", "inputs": ["ADP"], "outputs": ["ATP"]}]}}'
    )

    result = _repair(reversed_sides)

    assert result.outcome == REPAIR_SEMANTIC_GUARD_FAILED


def test_reordered_entity_rows_are_refused() -> None:
    swapped_rows = (
        '{"entities": {"compounds": [{"name": "ADP"}, {"name": "ATP"}]}, '
        '"processes": {"reactions": [{"name": "r1", "inputs": ["ATP"], "outputs": ["ADP"]}]}}'
    )

    result = _repair(swapped_rows)

    assert result.outcome == REPAIR_SEMANTIC_GUARD_FAILED


def test_duplicated_content_is_refused() -> None:
    """Two copies of a real reaction need two copies of its literals, and the
    input only supplies one."""

    doubled = json.dumps(
        {
            "entities": {"compounds": [{"name": "ATP"}, {"name": "ADP"}]},
            "processes": {
                "reactions": [
                    {"name": "r1", "inputs": ["ATP"], "outputs": ["ADP"]},
                    {"name": "r1", "inputs": ["ATP"], "outputs": ["ADP"]},
                ]
            },
        }
    )

    result = _repair(doubled)

    assert result.outcome == REPAIR_SEMANTIC_GUARD_FAILED


# ---------------------------------------------------------------------------
# The guard must not break legitimate repairs.
# ---------------------------------------------------------------------------
def test_deleting_a_truncated_pair_stays_legal() -> None:
    """What the system prompt actually asks for.

    A reply cut off mid-value cannot be completed without guessing, so the model
    is told to drop the incomplete pair. The scanner ignores incomplete literals
    for exactly this reason.
    """

    truncated = '{"pathway_name": "Glutathione biosynthesis", "likely_organism": "Homo sap'
    ok, _finding = json_repair_preserves_content(
        truncated, {"pathway_name": "Glutathione biosynthesis"}
    )

    assert ok is True


def test_guessing_the_truncated_value_is_refused() -> None:
    """The other half of the same case: completing it IS inventing it."""

    truncated = '{"pathway_name": "Glutathione biosynthesis", "likely_organism": "Homo sap'
    ok, finding = json_repair_preserves_content(
        truncated,
        {"pathway_name": "Glutathione biosynthesis", "likely_organism": "Homo sapiens"},
    )

    assert ok is False
    assert finding["first_unmatched"] == "Homo sapiens"


def test_a_trailing_comma_repair_stays_legal() -> None:
    ok, _ = json_repair_preserves_content('{"a": 1, "b": [2,],}', {"a": 1, "b": [2]})
    assert ok is True


def test_a_code_fence_around_a_valid_body_stays_legal() -> None:
    ok, _ = json_repair_preserves_content('```json\n{"a": "x"}\n```', {"a": "x"})
    assert ok is True


def test_renumbering_a_scalar_format_stays_legal() -> None:
    """``1e3`` re-serialized as ``1000.0`` is the same number.

    Rejecting a correct repair over a float format would make the guard fire on
    honest work, which is how guards get switched off.
    """

    ok, _ = json_repair_preserves_content('{"confidence": 1e3, "flag": true}', {"confidence": 1000.0, "flag": True})
    assert ok is True


def test_a_number_inside_a_string_is_not_read_as_a_scalar() -> None:
    """``"2 ATP"`` is one string, not a number and a word."""

    ok, _ = json_repair_preserves_content('{"note": "2 ATP"}', {"note": "2 ATP"})
    assert ok is True


# ---------------------------------------------------------------------------
# What a refusal costs, and what it leaves behind.
# ---------------------------------------------------------------------------
def test_a_refused_repair_is_not_retried() -> None:
    """The reply was valid JSON: the model did not fail at the task it was set,
    it did something else. Asking the identical question again is not what
    changes that, and the deterministic salvage is right there."""

    provider = _Provider(_Reply(HONEST.replace('"r1"', '"invented"')), _Reply(HONEST))

    result = repair_json_text(
        MALFORMED, "JSONDecodeError", max_attempts=3, chat_fn=provider
    )

    assert result.outcome == REPAIR_SEMANTIC_GUARD_FAILED
    assert len(provider.prompts) == 1


def test_the_refusal_is_recorded_as_its_own_outcome(tmp_path: Path) -> None:
    """``semantic_guard_failed``, not ``invalid_json``.

    An operator reading the artifact needs to know a guard fired, not that a
    model failed to close a brace — the two have opposite fixes.
    """

    activate(artifact_dir=tmp_path)
    _repair(HONEST.replace('"r1"', '"invented"'))

    report = json.loads((tmp_path / BOUNDARY_REPORT_NAME).read_text(encoding="utf-8"))
    record = report["boundaries"][-1]
    assert record["outcome"] == OUTCOME_SEMANTIC_GUARD_FAILED
    assert record["boundary"] == "stage1_json_repair"
    assert record["semantic_guard"]["first_unmatched"] == "invented"
    assert record["semantic_guard"]["first_unmatched_kind"] == "string"


def test_a_refused_repair_falls_back_to_deterministic_salvage() -> None:
    """End to end through ``_run_json_stage``: the run keeps going.

    The salvage is the pre-repair behaviour and it cannot invent anything,
    because it never asks a model. Losing the repair must not cost the run the
    part of the payload the prefix scan can still recover.
    """

    import t2pw.pipeline.pipeline as pipeline_mod
    from t2pw.llm.client import CompletionDiagnostics, CompletionResult
    from t2pw.pipeline.pipeline import _run_json_stage

    def _extractor(_prev: Any, _err: Any) -> str:
        return "prompt"

    def _stage_draw(*_a: Any, **_k: Any) -> CompletionResult:
        return CompletionResult(
            MALFORMED,
            CompletionDiagnostics(model="test-model", stage="extraction", attempts=1),
        )

    original = pipeline_mod.chat_detailed
    pipeline_mod.chat_detailed = _stage_draw  # type: ignore[assignment]
    try:
        payload, attempts = _run_json_stage(
            stage_name="extraction",
            system_prompt="sys",
            build_user_prompt=_extractor,
            max_attempts=1,
            temperature=0.0,
            max_tokens=100,
            repair_chat_fn=_Provider(_Reply(HONEST.replace('"r1"', '"invented"'))),
        )
    finally:
        pipeline_mod.chat_detailed = original  # type: ignore[assignment]

    # The prefix salvage recovered the entities that precede the syntax error.
    assert payload["entities"]["compounds"][0]["name"] == "ATP"
    assert "processes" not in payload
    assert any("semantic_guard_failed" in str(row.get("note", "")) for row in attempts)
