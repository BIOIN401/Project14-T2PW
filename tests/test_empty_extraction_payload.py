"""An empty extraction result: when it is a bug, and when it is the answer.

Run ``2026-08-02_2130`` failed four legs on the same missing distinction. Three
of them (``PMC12452463/strict``, ``PMC12312563/strict``, ``PMC12782028/research``)
recorded ``raw_chars: 2``, ``finish_reason: stop``, ``attempts: 1`` — the model
returned ``{}``. Because ``{}`` parses to a dict, the only retry trigger
(``json.JSONDecodeError``) never fired and the degenerate draw was final;
``PMC12452463`` passed in the other mode, so the paper was fine and the draw was
not. The fourth, ``PMC13231680``, is the gold set's NEGATIVE CONTROL: its research
leg returned 4 entities, 0 reactions and discarded nothing, which its own
``export_rationale`` calls the correct outcome — "an empty pathway plus a
rejection reason" — and it was killed as a contract violation.

Both are "zero reactions reached the contract", and the fixes point in opposite
directions: one must be re-drawn, the other must be allowed through. So the two
halves are pinned together in one file, because a change that fixes either one
alone is a regression:

* allowing empty containers without the re-draw lets ``{}`` PASS the contract as
  a legitimate empty pathway — silently shipping nothing, which is worse than
  today's loud failure;
* re-drawing without allowing empty containers leaves the negative control dead.

The third guard here is the one that is easy to get wrong in the other direction:
Stage 2 nests its output under ``additions``, so *every* ``stage2_inference``
boundary in that run recorded ``ents 0 | procs 0`` — including replies of 13,000
characters. A structurally-empty test applied to Stage 2 would re-draw every
successful inference in the pipeline and lean on a model that correctly proposed
nothing to invent something.
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

import t2pw.pipeline.pipeline as pipeline_mod  # noqa: E402
from t2pw.llm.client import CompletionDiagnostics, CompletionResult  # noqa: E402
from t2pw.pipeline.pipeline import (  # noqa: E402
    EMPTY_PAYLOAD_RETRY_REASON,
    NOTE_EMPTY_PAYLOAD_EXHAUSTED,
    NOTE_EMPTY_PAYLOAD_RETRY,
    PipelineFailure,
    _build_extraction_prompt,
    _run_json_stage,
    clean_stage_one,
    clean_stage_one_with_report,
)
from t2pw.pipeline.stage_contracts import validate_post_extraction  # noqa: E402

#: The negative control's shape: the model read the paper and found no chemistry.
ENTITIES_ONLY = json.dumps(
    {
        "entities": {
            "proteins": [{"name": "LpxC", "evidence": "PSA may interact with LpxC"}],
            "compounds": [{"name": "meropenem", "evidence": "meropenem bactericidal activity"}],
        },
        "processes": {},
    }
)

WITH_REACTIONS = json.dumps(
    {
        "entities": {"compounds": [{"name": "ATP", "evidence": "ATP is consumed"}]},
        "processes": {
            "reactions": [
                {
                    "name": "r1",
                    "inputs": ["ATP"],
                    "outputs": ["ADP"],
                    "evidence": "ATP is consumed to give ADP",
                }
            ]
        },
    }
)


class _Draws:
    """Canned ``chat_detailed`` returning one reply per attempt."""

    def __init__(self, *texts: str) -> None:
        self._texts = list(texts)
        self.calls: List[str] = []

    def __call__(self, messages: Any, **_kwargs: Any) -> CompletionResult:
        self.calls.append(messages[-1]["content"])
        text = self._texts[min(len(self.calls) - 1, len(self._texts) - 1)]
        return CompletionResult(
            text,
            CompletionDiagnostics(model="test-model", stage="extraction", attempts=1),
        )


@pytest.fixture
def draws(monkeypatch: pytest.MonkeyPatch):
    def _install(*texts: str) -> _Draws:
        provider = _Draws(*texts)
        monkeypatch.setattr(pipeline_mod, "chat_detailed", provider)
        return provider

    return _install


def _stage(
    max_attempts: int = 2,
    *,
    stage_name: str = "extraction",
    retry_on_empty_payload: bool = True,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    return _run_json_stage(
        stage_name=stage_name,
        system_prompt="sys",
        build_user_prompt=lambda prev, err: f"prompt|{prev}|{err}",
        max_attempts=max_attempts,
        temperature=0.0,
        max_tokens=100,
        repair_json=False,
        retry_on_empty_payload=retry_on_empty_payload,
    )


# ---------------------------------------------------------------------------
# Half one: the degenerate draw is re-drawn, and never becomes a pathway.
# ---------------------------------------------------------------------------


def test_a_bare_empty_object_is_retried_instead_of_being_returned(draws) -> None:
    """The exact shape of the three dead legs: ``{}``, ``stop``, one attempt.

    Attempt 1 being terminal is the whole defect, so the assertion that matters
    is the second draw happening at all.
    """

    provider = draws("{}", WITH_REACTIONS)
    payload, attempts = _stage(max_attempts=2)

    assert len(provider.calls) == 2
    assert payload["processes"]["reactions"][0]["name"] == "r1"
    assert any(row.get("note") == NOTE_EMPTY_PAYLOAD_RETRY for row in attempts)


def test_a_bare_empty_object_never_becomes_an_empty_pathway(draws) -> None:
    """Exhausting the retries must raise, not return ``{}``.

    This is the interlock with the cleaning change below. Once an empty
    ``processes`` container is a legal payload, returning ``{}`` from the
    extraction boundary would sail through the contract and export an empty
    pathway with no error anywhere — strictly worse than the loud failure this
    replaces. The leg still fails; it now fails naming the cause.
    """

    provider = draws("{}", "{}")
    with pytest.raises(PipelineFailure) as excinfo:
        _stage(max_attempts=2)

    assert len(provider.calls) == 2
    assert "structurally empty" in str(excinfo.value)
    notes = [row.get("note") for row in excinfo.value.attempts]
    assert NOTE_EMPTY_PAYLOAD_RETRY in notes
    assert NOTE_EMPTY_PAYLOAD_EXHAUSTED in notes


def test_an_all_empty_bucket_reply_counts_as_empty_too(draws) -> None:
    """``{"entities": {"proteins": []}}`` is degenerate in the same way as ``{}``.

    Testing row counts rather than key presence is what catches it; a key test
    would call this payload populated and return it.
    """

    provider = draws(json.dumps({"entities": {"proteins": []}, "processes": {}}), WITH_REACTIONS)
    payload, _attempts = _stage(max_attempts=2)

    assert len(provider.calls) == 2
    assert payload["processes"]["reactions"][0]["name"] == "r1"


def test_the_retry_prompt_does_not_claim_the_json_was_invalid() -> None:
    """``{}`` has no syntax error to fix, and saying otherwise wastes the re-draw.

    The instruction to leave a genuinely empty result empty is load-bearing: this
    prompt must not become pressure to invent chemistry, which is precisely what
    the gold set's ``forbidden_identifiers`` controls exist to catch.
    """

    prompt = _build_extraction_prompt("source text", "{}", EMPTY_PAYLOAD_RETRY_REASON)

    assert "returned invalid JSON" not in prompt
    assert "empty JSON object" in prompt
    assert "Do not invent" in prompt


def test_a_parse_error_still_reports_itself_as_a_parse_error() -> None:
    """The pre-existing repair branch keeps its own wording."""

    prompt = _build_extraction_prompt("source text", "{oops", "JSONDecodeError: boom")

    assert "returned invalid JSON" in prompt
    assert "JSONDecodeError: boom" in prompt


# ---------------------------------------------------------------------------
# Half two: a real empty result survives.
# ---------------------------------------------------------------------------


def test_an_entities_only_reply_is_not_retried(draws) -> None:
    """The model read the paper and found no reactions. That is an answer.

    Distinguishing this from ``{}`` is the entire point of the split; a retry
    trigger of "zero processes" would re-draw the negative control and invite the
    model to produce the reactions the paper does not contain.
    """

    provider = draws(ENTITIES_ONLY, WITH_REACTIONS)
    payload, _attempts = _stage(max_attempts=2)

    assert len(provider.calls) == 1
    assert payload["entities"]["proteins"][0]["name"] == "LpxC"


def test_an_entities_only_reply_survives_cleaning_and_passes_the_contract() -> None:
    """The negative control's leg, end to end from reply to contract.

    ``clean_stage_one`` used to drop the falsy ``processes`` container, and
    ``validate_post_extraction`` requires the key — so this payload died as
    ``processes_required`` with nothing wrong with it.
    """

    cleaned = clean_stage_one(json.loads(ENTITIES_ONLY))

    assert cleaned["processes"] == {}
    assert cleaned["entities"]["proteins"][0]["name"] == "LpxC"

    report = validate_post_extraction(cleaned)
    assert report["ok"] is True
    assert [issue["code"] for issue in report["errors"]] == []


def test_both_containers_are_always_present_even_when_everything_is_empty() -> None:
    """Emitting the keys is what makes an empty result representable at all."""

    cleaned = clean_stage_one({})

    assert cleaned["entities"] == {}
    assert cleaned["processes"] == {}
    assert validate_post_extraction(cleaned)["ok"] is True


def test_zero_in_zero_out_is_reported_as_its_own_finding() -> None:
    """``all_processes_discarded`` cannot fire here — nothing was discarded.

    Without a separate flag the report gives a reader nothing to distinguish "the
    model declared none" from "cleaning removed them all", which are the two
    causes of an empty pathway and have opposite fixes.
    """

    _cleaned, report = clean_stage_one_with_report(json.loads(ENTITIES_ONLY))

    assert report["all_processes_discarded"] is False
    assert report["no_processes_declared"] is True
    assert report["discarded_total"] == 0


def test_processes_removed_by_cleaning_stays_distinguishable() -> None:
    """The other cause keeps its own flag, and does not claim none were declared."""

    payload = {
        "entities": {"compounds": [{"name": "ATP", "evidence": "ATP is consumed"}]},
        "processes": {"reactions": [{"name": "", "inputs": [], "outputs": []}]},
    }
    _cleaned, report = clean_stage_one_with_report(payload)

    assert report["raw_process_counts"]["total"] == 1
    assert report["all_processes_discarded"] is True
    assert report["no_processes_declared"] is False


# ---------------------------------------------------------------------------
# The guard on the guard: Stage 2's empty is legitimate.
# ---------------------------------------------------------------------------


def test_stage_two_is_not_subject_to_the_empty_payload_retry(draws) -> None:
    """Its output nests under ``additions``, so the top level is always empty.

    Every ``stage2_inference`` boundary in run ``2026-08-02_2130`` recorded
    ``ents 0 | procs 0``, including replies of 5,000-13,000 characters that were
    entirely successful. A stage-agnostic emptiness test would re-draw all of
    them. The inference system prompt also names an empty result as correct:
    "propose no additions. Return an empty additions object".
    """

    provider = draws(json.dumps({"additions": {"processes": {"reactions": [{"name": "r9"}]}}}))
    payload, _attempts = _stage(
        max_attempts=2, stage_name="inference", retry_on_empty_payload=False
    )

    assert len(provider.calls) == 1
    assert payload["additions"]["processes"]["reactions"][0]["name"] == "r9"


def test_a_deliberately_empty_inference_is_returned_not_redrawn(draws) -> None:
    """``{"additions": {}}`` is what the Stage-2 prompt asks for when nothing applies."""

    provider = draws(json.dumps({"additions": {}}))
    payload, _attempts = _stage(
        max_attempts=2, stage_name="inference", retry_on_empty_payload=False
    )

    assert len(provider.calls) == 1
    assert payload == {"additions": {}}


def test_the_retry_is_opt_in_so_a_new_stage_cannot_inherit_it_silently(draws) -> None:
    """Default off. A stage that wants the re-draw has to say so."""

    provider = draws("{}", WITH_REACTIONS)
    payload, _attempts = _run_json_stage(
        stage_name="extraction",
        system_prompt="sys",
        build_user_prompt=lambda prev, err: "prompt",
        max_attempts=2,
        temperature=0.0,
        max_tokens=100,
        repair_json=False,
    )

    assert len(provider.calls) == 1
    assert payload == {}
