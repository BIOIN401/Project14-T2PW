"""Offline replay of the stored strict-failure shapes.

Each case in ``fixtures/strict_failures/cases.json`` is a compact payload that
reproduces one class of pre-export strict failure. The harness runs the same
question over every one of them:

    given this payload, does pre-export quarantine now produce a SMALLER VALID
    strict graph, or does it correctly refuse?

Both answers are correct results, and the fixture says which is expected. What is
never a correct result is a graph that passes because the pathway was emptied out
or replaced by unrelated survivors -- ``every_reaction_unresolvable`` and
``only_unrelated_reactions_survive`` exist to keep that from ever reading as a win.

Pure and offline: no network, no database, no batch run.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.process_normalizer import (  # noqa: E402
    GateValidationError,
    run_strict_post_normalization_gates,
)
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402
from t2pw.pwml.ir import validate_required_pwml_contract  # noqa: E402


FIXTURES = Path(__file__).parent / "fixtures" / "strict_failures"


def _cases() -> List[Dict[str, Any]]:
    return json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))["cases"]


def _case_ids() -> List[str]:
    return [case["id"] for case in _cases()]


def _case(case_id: str) -> Dict[str, Any]:
    return next(case for case in _cases() if case["id"] == case_id)


def _strict_verdict(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Exactly the two gate stacks ``run_pwml_export`` runs, in that order.

    Both, not one: the Stage 3 revalidation owns registry references and
    connectivity, the required-field gate owns identity and PWML shape, and the
    fixture classes split across the two. Reporting only one would let a case
    look recovered while still failing the other.
    """

    reasons: List[str] = []
    try:
        run_strict_post_normalization_gates(deepcopy(payload), enforce_all_proteins_connected=True)
    except GateValidationError as exc:
        reasons.extend(str(row.get("reason", "")) for row in (exc.details.get("errors") or []))
    contract = validate_required_pwml_contract(deepcopy(payload), strict_db=True)
    if not contract.get("ok"):
        reasons.extend(str(issue.get("code", "")) for issue in contract.get("errors", []))
    return (not reasons), reasons


def _reaction_count(payload: Dict[str, Any]) -> int:
    return len(payload.get("processes", {}).get("reactions") or [])


def _process_count(payload: Dict[str, Any]) -> int:
    """Every process bucket, not just reactions.

    A coupled transport leaving is a smaller graph even when the reaction count
    is untouched; counting reactions alone reported that case as "no change".
    """

    return sum(
        len(rows)
        for rows in (payload.get("processes") or {}).values()
        if isinstance(rows, list)
    )


def _entity_count(payload: Dict[str, Any]) -> int:
    return sum(
        len(rows)
        for rows in (payload.get("entities") or {}).values()
        if isinstance(rows, list)
    )


@pytest.mark.parametrize("case_id", _case_ids())
def test_every_stored_strict_failure_replays_to_its_recorded_verdict(case_id: str) -> None:
    case = _case(case_id)
    expect = case["expect"]
    payload = case["payload"]

    ok_before, _reasons_before = _strict_verdict(payload)
    assert ok_before is expect["strict_before"], (
        f"{case_id}: the stored payload's pre-quarantine strict verdict changed; "
        "the fixture records what the gates said when it was written."
    )

    result = quarantine_and_close(deepcopy(payload), strict_db=True)

    assert result.ok is expect["recovers"]
    if expect["recovers"]:
        ok_after, reasons_after = _strict_verdict(result.payload)
        assert ok_after is True, f"{case_id} still fails strict export: {reasons_after}"
    else:
        assert expect.get("coverage_reason")
        assert any(
            reason.startswith(expect["coverage_reason"]) for reason in result.coverage["reasons"]
        )


@pytest.mark.parametrize("case_id", _case_ids())
def test_recovered_cases_are_smaller_and_refused_cases_are_not_claimed(case_id: str) -> None:
    case = _case(case_id)
    expect = case["expect"]
    result = quarantine_and_close(deepcopy(case["payload"]), strict_db=True)

    # "smaller" is short for "a smaller VALID strict graph". A refused case
    # shrank plenty -- that is what refusal looks like from the inside -- but it
    # produced no graph at all, so recording it as smaller would read as a win.
    shrank = (
        _process_count(result.payload) < _process_count(case["payload"])
        or _entity_count(result.payload) < _entity_count(case["payload"])
    )
    assert (result.ok and shrank) is expect["smaller"]

    if "surviving_reactions" in expect:
        assert _reaction_count(result.payload) == expect["surviving_reactions"]


@pytest.mark.parametrize("case_id", _case_ids())
def test_quarantine_decisions_match_the_fixture(case_id: str) -> None:
    case = _case(case_id)
    expect = case["expect"]
    if "quarantined" not in expect:
        pytest.skip("case records a coverage refusal rather than per-process decisions")

    result = quarantine_and_close(deepcopy(case["payload"]), strict_db=True)
    decided = {
        name: state for name, state in result.states().items() if not state.endswith("_accepted")
    }

    assert decided == expect["quarantined"]


@pytest.mark.parametrize("case_id", _case_ids())
def test_removed_entities_match_the_fixture(case_id: str) -> None:
    case = _case(case_id)
    expected = case["expect"].get("removed_entities")
    if expected is None:
        pytest.skip("case records no entity removals")

    result = quarantine_and_close(deepcopy(case["payload"]), strict_db=True)
    removed = [row["name"] for row in result.removed_entity_report["removed_entities"]]

    assert sorted(removed) == sorted(expected)


@pytest.mark.parametrize("case_id", _case_ids())
def test_closure_converges_on_every_stored_shape(case_id: str) -> None:
    result = quarantine_and_close(deepcopy(_case(case_id)["payload"]), strict_db=True)

    assert result.closure_report["converged"] is True
    assert result.closure_report["iterations"][-1]["changed"] is False
    # Replaying the reduced payload must be a no-op: a closure that keeps finding
    # work on its own output has not reached a fixpoint, it has just stopped.
    again = quarantine_and_close(deepcopy(result.payload), strict_db=True)
    assert again.payload == result.payload


def test_the_base_pathway_shared_by_every_case_passes_strict_on_its_own() -> None:
    """Otherwise a failure below could belong to the base rather than the defect."""

    unknown_case = _case("unknown_backed_functional_complex")
    ok, reasons = _strict_verdict(unknown_case["payload"])

    assert ok is True, reasons


def test_the_fixture_covers_both_verdicts() -> None:
    verdicts = {case["expect"]["recovers"] for case in _cases()}

    assert verdicts == {True, False}, (
        "a replay corpus of only-recovering cases cannot show that quarantine ever "
        "refuses, which is half of what it is for"
    )
