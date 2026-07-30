"""The Stage-1 -> Stage-2 boundary contract: settle it, or say why you cannot.

The boundary used to be ``validate_post_extraction(stage_one)`` followed by
``st.error(...); st.stop()``. Two things were wrong with that and these tests pin
both fixes:

  * a *recoverable* failure was terminal. One reaction row in the wrong shape
    stopped the whole run, and nothing tried to fix the one row;
  * a *silent* loss was never addressed. A participant named by a reaction but
    absent from the entity registry passes this contract and then costs that
    reaction later, quietly, inside ``filter_unresolvable_reactions``.

And the property that must survive both: when recovery is spent, the boundary
reports an explicit incomplete outcome. It never pads the payload to satisfy
``processes_required``.

Offline: repairs go through an injected ``chat_fn``, reconstruction uses no model
at all.
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

from t2pw.pipeline.export_mode import PATHWHIZ, RESEARCH  # noqa: E402
from t2pw.pipeline.extraction_diagnostics import (  # noqa: E402
    BOUNDARY_REPORT_NAME,
    OUTCOME_CONTRACT_FAILED,
    OUTCOME_DISCARDED_BY_CLEANING,
    OUTCOME_OK,
    OUTCOME_ZERO_PROCESSES,
    activate,
    deactivate,
)
from t2pw.pipeline.stage_contracts import StageContractError  # noqa: E402
from t2pw.pipeline.stage_one_boundary import settle_stage_one  # noqa: E402


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
            raise AssertionError("the boundary made more repair draws than were scripted")
        return self._replies.pop(0)


def _healthy() -> Dict[str, Any]:
    return {
        "entities": {"compounds": [{"name": "ATP"}, {"name": "ADP"}]},
        "processes": {
            "reactions": [
                {
                    "name": "hydrolysis",
                    "inputs": ["ATP"],
                    "outputs": ["ADP"],
                    "evidence": "ATP is hydrolysed to ADP.",
                }
            ]
        },
    }


# ---------------------------------------------------------------------------
# The passing path.
# ---------------------------------------------------------------------------
def test_a_healthy_payload_settles_with_no_model_call() -> None:
    provider = _Provider()

    outcome = settle_stage_one(_healthy(), chat_fn=provider)

    assert outcome.ok
    assert outcome.outcome == OUTCOME_OK
    assert provider.prompts == []
    assert outcome.repair is None


def test_the_boundary_does_not_mutate_the_payload_it_was_given() -> None:
    """Reconstruction adds rows, so the caller's object must be left alone."""

    payload = _healthy()
    before = json.dumps(payload, sort_keys=True)

    settle_stage_one(payload, chat_fn=_Provider())

    assert json.dumps(payload, sort_keys=True) == before


# ---------------------------------------------------------------------------
# Reconstruction runs first, and it runs unconditionally.
# ---------------------------------------------------------------------------
def test_reconstruction_runs_before_validation_and_without_a_model() -> None:
    """It is free and it cannot fail, so a shell it adds saves a repair draw.

    This is also the *silent* loss being closed: the contract accepts this
    payload either way, so nothing here would ever have raised -- the reaction
    would simply have disappeared three stages later.
    """

    payload = {
        "entities": {"compounds": [{"name": "ATP"}]},
        "processes": {
            "reactions": [
                {"name": "hydrolysis", "inputs": ["ATP"], "outputs": ["ADP"], "evidence": "e"}
            ]
        },
    }
    provider = _Provider()

    outcome = settle_stage_one(payload, chat_fn=provider)

    assert outcome.ok
    assert provider.prompts == []
    assert outcome.reconstruction["shell_count"] == 1
    shell = next(
        row for row in outcome.payload["entities"]["compounds"] if row["name"] == "ADP"
    )
    assert shell["resolution_status"] == "unresolved"
    assert outcome.to_summary()["registry_reconstruction"]["shell_count"] == 1


# ---------------------------------------------------------------------------
# Localized repair on the failing path.
# ---------------------------------------------------------------------------
_NAMELESS = {
    "entities": {"proteins": [{"name": "", "evidence": "LpxA acetylates UDP-GlcNAc."}]},
    "processes": {
        "reactions": [
            {"name": "r", "inputs": ["UDP-GlcNAc"], "outputs": ["product"], "evidence": "e"}
        ]
    },
}


def test_a_repairable_row_is_repaired_instead_of_ending_the_run() -> None:
    """One nameless entity row used to stop the whole extraction."""

    provider = _Provider(
        _Reply(
            json.dumps(
                {
                    "repaired_rows": [
                        {
                            "pointer": "/entities/proteins/0",
                            "row": {
                                "name": "LpxA",
                                "evidence": "LpxA acetylates UDP-GlcNAc.",
                            },
                        }
                    ]
                }
            )
        )
    )

    outcome = settle_stage_one(_NAMELESS, chat_fn=provider)

    assert outcome.ok
    assert outcome.outcome == OUTCOME_OK
    assert outcome.repair.repaired_pointers == ["/entities/proteins/0"]
    assert outcome.payload["entities"]["proteins"][0]["name"] == "LpxA"
    # The reaction was never in the repair prompt: it was not the broken row.
    assert "UDP-GlcNAc" in provider.prompts[0]  # via the row's own evidence
    assert '"pointer": "/processes/reactions/0"' not in provider.prompts[0]


def test_a_repair_that_invents_a_name_is_refused_and_the_run_reports_incomplete() -> None:
    """The contract asked for a ``name``. The repair also wrote an ``organism``.

    Two independent reasons to refuse it, and the STRICTER one fires first: the
    contract error is ``entity_missing_name``, whose pointer names exactly one
    repairable field, so ``organism`` is a field the repair had no licence to
    introduce at all -- regardless of whether the evidence would have supported
    it. (It would not have: the passage never mentions the organism. That path is
    covered by ``test_a_repaired_row_that_invents_biology_is_rejected`` in
    ``tests/test_localized_repair.py``, where the invented value lands in a field
    the contract DID name.)

    Taking any value the model offers is what would make repair a laundering step
    for invented identity.
    """

    provider = _Provider(
        _Reply(
            json.dumps(
                {
                    "repaired_rows": [
                        {
                            "pointer": "/entities/proteins/0",
                            "row": {
                                "name": "LpxA",
                                "organism": "Escherichia coli",
                                "evidence": "LpxA acetylates UDP-GlcNAc.",
                            },
                        }
                    ]
                }
            )
        ),
        _Reply(json.dumps({"repaired_rows": []})),
    )

    outcome = settle_stage_one(_NAMELESS, chat_fn=provider)

    assert outcome.ok is False
    assert outcome.payload["entities"]["proteins"][0]["name"] == ""
    rejection = outcome.repair.rejected[0]
    assert rejection["reason"] == "unrelated_field_added"
    assert rejection["lost"][0]["field"] == "organism"
    assert "Escherichia coli" in rejection["lost"][0]["added"]


# ---------------------------------------------------------------------------
# The incomplete outcome: explicit, artifact-backed, never padded.
# ---------------------------------------------------------------------------
def test_an_unrecoverable_payload_reports_incomplete_and_invents_nothing() -> None:
    """Requirement 6. A payload with no processes must stay one.

    ``processes_required`` is a container guard: no row edit can satisfy it, and
    the only way to make it pass would be to write a reaction nobody extracted.
    """

    provider = _Provider()

    outcome = settle_stage_one({"entities": {"compounds": [{"name": "ATP"}]}}, chat_fn=provider)

    assert outcome.ok is False
    assert outcome.outcome == OUTCOME_CONTRACT_FAILED
    assert provider.prompts == []  # no draw was spent on an unrepairable guard
    assert "processes_required" in outcome.incomplete_reason
    assert "not attempted" in outcome.incomplete_reason
    assert outcome.payload.get("processes") in (None, {})
    assert isinstance(outcome.failure, StageContractError)


def test_the_incomplete_reason_names_cleaning_when_cleaning_is_the_cause() -> None:
    """The distinction no message in this pipeline used to draw.

    "The model gave us nothing" and "we deleted what it gave us" produce the same
    contract failure. The cleaning report is what separates them, and the reason
    string is where a person reading RESULT.txt at 09:00 sees it.
    """

    cleaning = {
        "all_processes_discarded": True,
        "raw_process_counts": {"reactions": 3, "total": 3},
        "cleaned_process_counts": {"total": 0},
        "discarded_by_reason": {"reaction_no_usable_outputs": 3},
    }

    outcome = settle_stage_one({"entities": {}}, cleaning_report=cleaning, repair_rows=False)

    assert outcome.ok is False
    assert "cleaning discarded every one of the 3 process row(s)" in outcome.incomplete_reason
    assert "reaction_no_usable_outputs" in outcome.incomplete_reason


def test_the_boundary_writes_its_verdict_to_disk_on_the_failing_path(
    tmp_path: Path,
) -> None:
    activate(artifact_dir=tmp_path)

    settle_stage_one({"entities": {}}, repair_rows=False)

    report = json.loads((tmp_path / BOUNDARY_REPORT_NAME).read_text(encoding="utf-8"))
    record = report["boundaries"][-1]
    assert record["outcome"] == OUTCOME_CONTRACT_FAILED
    assert record["contract_ok"] is False
    assert "processes_required" in record["failing_codes"]
    assert record["note"] == "stage_one_boundary_incomplete"


# ---------------------------------------------------------------------------
# A settled boundary is not the same as a complete pathway.
# ---------------------------------------------------------------------------
def test_an_empty_but_structurally_valid_payload_is_reported_as_zero_processes() -> None:
    """``processes: {}`` satisfies the structural guard and is still no pathway.

    The guard asks whether ``processes`` is an object; an empty one is. So the
    contract is content, ``ok`` is True -- and the outcome is
    ``valid_json_zero_processes``, not ``ok``. Both facts have to be reported:
    calling a settled boundary a success is how a pipeline starts calling nothing
    a pathway, and calling it a contract failure would mislabel the cause.

    (In a real run this shape does not survive to here: ``clean_stage_one`` drops
    an empty ``processes`` key entirely, so the boundary sees a payload with no
    key at all and ``processes_required`` fires -- the case the replay harness
    covers. Both routes end with no reactions and neither invents one.)
    """

    outcome = settle_stage_one(
        {"entities": {"compounds": [{"name": "ATP"}]}, "processes": {}}, repair_rows=False
    )

    assert outcome.ok is True
    assert outcome.outcome == OUTCOME_ZERO_PROCESSES
    assert outcome.payload["processes"] == {}


def test_a_payload_whose_processes_key_is_gone_fails_the_guard() -> None:
    """What cleaning actually hands the boundary when nothing survived."""

    outcome = settle_stage_one({"entities": {"compounds": [{"name": "ATP"}]}}, repair_rows=False)

    assert outcome.ok is False
    assert outcome.outcome == OUTCOME_CONTRACT_FAILED
    assert "processes_required" in outcome.incomplete_reason


def test_a_settled_boundary_over_no_reactions_is_labelled_by_its_cause() -> None:
    """The two "no pathway" outcomes are told apart by the cleaning report."""

    payload = {"entities": {"compounds": [{"name": "ATP"}]}, "processes": {"interactions": []}}

    nothing_extracted = settle_stage_one(payload, repair_rows=False)
    all_discarded = settle_stage_one(
        payload,
        cleaning_report={
            "all_processes_discarded": True,
            "raw_process_counts": {"total": 4},
            "discarded_by_reason": {"reaction_no_usable_inputs": 4},
        },
        repair_rows=False,
    )

    assert nothing_extracted.outcome == OUTCOME_ZERO_PROCESSES
    assert all_discarded.outcome == OUTCOME_DISCARDED_BY_CLEANING
    # Both settled the contract; neither is a pathway, and both say so.
    assert nothing_extracted.ok and all_discarded.ok


@pytest.mark.parametrize("mode", [PATHWHIZ, RESEARCH])
def test_the_boundary_honours_the_export_mode_it_is_given(mode: str) -> None:
    """Structural guards abort in both modes -- there is no candidate pathway to
    review when the payload is not a payload -- so the verdict must not depend on
    the mode for this shape."""

    outcome = settle_stage_one({"entities": {}}, mode=mode, repair_rows=False)

    assert outcome.ok is False
    assert "processes_required" in outcome.incomplete_reason


def test_the_summary_is_json_safe_and_bounded() -> None:
    """It goes into session state and into an artifact, so it must serialize."""

    outcome = settle_stage_one(_healthy(), repair_rows=False)

    text = json.dumps(outcome.to_summary())
    assert "payload_hash" in text
    # Counts and hashes, never the payload itself.
    assert "ATP is hydrolysed" not in text
