"""Offline replay of the stored strict-failure shapes.

Each case in ``fixtures/strict_failures/cases.json`` is a compact payload that
reproduces one class of pre-export strict failure. The harness runs the same
question over every one of them:

    given this payload, does pre-export quarantine now produce a SMALLER VALID
    strict graph, or does it correctly refuse -- and WHAT IS THE RUN CLASSIFIED
    AS, either way?

The third clause is not decoration, and it is the seam this module now measures.
``quarantine_and_close(...).ok`` answers "may this graph be frozen", and since
**C-041a** (``4177fe5``, under **D-002**) that stopped being the same question as
"was the requested pathway recovered". A ``minimum_core`` shortfall over a
defensible surviving core is routed to ``quarantine_report["review_reasons"]``
instead of ``refusal_reasons``, so the run keeps its graph and ``ok`` is True
while the verdict is carried by ``quarantine_report["release"]`` --
``status`` and ``strict_acceptance_eligible``. An EMPTY graph is not a shortfall
and still hard-refuses to ``diagnostic_only``; that surviving distinction is what
makes the split a relabelling rather than a hole, and
``test_an_emptied_graph_and_an_off_topic_survivor_reach_different_verdicts`` pins
it head-on.

Both answers are correct results, and the fixture says which is expected. What is
never a correct result is a graph that READS AS A WIN because the pathway was
emptied out or replaced by unrelated survivors -- ``every_reaction_unresolvable``
and ``only_unrelated_reactions_survive`` exist to keep that from ever happening.
Since D-002 that "win" is measured at the release seam rather than at ``ok``: for
``only_unrelated_reactions_survive`` these tests must go RED if ``release.status``
ever becomes ``release_ready``, or ``strict_acceptance_eligible`` ever becomes
True. The expectation, not the protection, is what F-142 found stale.

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
from t2pw.pipeline.release_status import (  # noqa: E402
    COVERAGE_REASON_EMPTY,
    DIAGNOSTIC_ONLY,
    RELEASE_READY,
    REVIEW_REQUIRED,
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


def _release(result: Any) -> Dict[str, Any]:
    """``quarantine_report["release"]``, asserted present before it is read.

    This is where C-041a routed the verdict ``ok`` used to carry, so a test that
    silently found nothing here would be measuring the absence of the seam rather
    than its content. Named explicitly so the failure says which seam went missing
    instead of raising a bare ``KeyError``.
    """

    release = result.quarantine_report.get("release")
    assert isinstance(release, dict) and release, (
        "quarantine_report carries no 'release' classification: the D-002 / C-041a "
        "seam this module measures is gone"
    )
    return release


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

    # 1. MAY THE GRAPH BE FROZEN? That, and since C-041a only that, is what ``ok``
    #    answers.
    assert result.ok is expect["recovers"]

    # 2. WHAT IS THE RUN? Read for every case and never from inside a branch --
    #    which is precisely how the pre-D-002 shape lost an assertion: it hung the
    #    refusal check on ``recovers``, so the one case whose ``ok`` moved stopped
    #    being checked at all instead of being checked differently (F-142).
    release = _release(result)
    assert release["status"] == expect["release_status"], (
        f"{case_id}: expected release status {expect['release_status']!r}, "
        f"got {release['status']!r}"
    )
    assert release["strict_acceptance_eligible"] is expect["strict_acceptance_eligible"]

    # 3. WHICH CHANNEL CARRIED THE VERDICT. A reason moving between these two lists
    #    is the D-002 product change itself, not a detail, so both are pinned
    #    exactly. An empty list here is an assertion, not a skip: it fails if a
    #    reason ever appears where the fixture records none.
    assert result.quarantine_report["review_reasons"] == expect["review_reasons"]
    assert result.refusal_reasons == expect["refusal_reasons"]

    # 4. Anything frozen must still survive the real strict export gates.
    if expect["recovers"]:
        ok_after, reasons_after = _strict_verdict(result.payload)
        assert ok_after is True, f"{case_id} still fails strict export: {reasons_after}"

    # 5. The coverage gate itself -- keyed on the fixture DECLARING a shortfall,
    #    not on ``recovers``. The two coincided before C-041a and do not now, and
    #    ``only_unrelated_reactions_survive`` is exactly where the difference bites.
    if "coverage_reason" in expect:
        coverage_reasons = result.coverage["reasons"]
        assert coverage_reasons, (
            f"{case_id}: the fixture records a coverage shortfall but the coverage "
            "gate emitted no reason at all, so the prefix test below would pass "
            "vacuously over an empty list"
        )
        assert any(
            reason.startswith(expect["coverage_reason"]) for reason in coverage_reasons
        ), (
            f"{case_id}: no coverage reason starts with "
            f"{expect['coverage_reason']!r}: {coverage_reasons}"
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

    # ...and since D-002, smaller-and-ok is no longer sufficient to mean a win.
    # ``only_unrelated_reactions_survive`` now satisfies both and is still not
    # acceptable, so the thing this test must actually hold down for that case is
    # acceptance eligibility, which the fixture records for every case.
    assert _release(result)["strict_acceptance_eligible"] is expect[
        "strict_acceptance_eligible"
    ]

    if "surviving_reactions" in expect:
        assert _reaction_count(result.payload) == expect["surviving_reactions"]


@pytest.mark.parametrize("case_id", _case_ids())
def test_quarantine_decisions_match_the_fixture(case_id: str) -> None:
    case = _case(case_id)
    expect = case["expect"]
    if "quarantined" not in expect:
        pytest.skip("case records a coverage verdict rather than per-process decisions")

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


def test_an_emptied_graph_and_an_off_topic_survivor_reach_different_verdicts() -> None:
    """The two never-a-win cases must stay DISTINGUISHED, not merged.

    D-002 draws its line between "no defensible connected core" and "a core that
    is merely smaller than the request"; C-041a implements it at one seam, and
    ``CoverageVerdict.has_surviving_core`` states it. If that line were erased in
    either direction -- an emptied graph promoted to ``review_required``, or a
    shortfall demoted back to a refusal -- the per-case rows in the fixture could
    be updated to match and stay internally consistent while the product rule was
    gone. So this test is unparametrized and names both cases: neither half can be
    skipped by a branch, and the closing assertion fails the moment they agree.
    """

    empty = quarantine_and_close(
        deepcopy(_case("every_reaction_unresolvable")["payload"]), strict_db=True
    )
    off_topic = quarantine_and_close(
        deepcopy(_case("only_unrelated_reactions_survive")["payload"]), strict_db=True
    )

    # The emptied graph HARD-REFUSES, exactly as it did before D-002.
    assert empty.ok is False
    assert _release(empty)["status"] == DIAGNOSTIC_ONLY
    assert empty.refusal_reasons, "an emptied graph refused while recording no reason"
    assert any(
        reason.startswith("minimum_core:" + COVERAGE_REASON_EMPTY)
        for reason in empty.refusal_reasons
    ), empty.refusal_reasons
    assert empty.quarantine_report["review_reasons"] == []

    # The off-topic survivor is CLASSIFIED rather than refused -- and is still
    # never a win: not release_ready, not acceptance-eligible, zero completeness,
    # every requested anchor recorded as missing.
    off_release = _release(off_topic)
    assert off_topic.ok is True
    assert off_release["status"] == REVIEW_REQUIRED
    assert off_release["strict_acceptance_eligible"] is False
    assert off_release["completeness"] == 0.0
    assert off_release["missing_anchors"], "a zero-coverage run named no missing anchor"
    assert off_topic.refusal_reasons == []
    shortfall = [
        reason
        for reason in off_topic.quarantine_report["review_reasons"]
        if reason.startswith("minimum_core:")
    ]
    assert shortfall, "the coverage shortfall reached neither review nor refusal reasons"

    assert _release(empty)["status"] != off_release["status"]


def test_the_base_pathway_shared_by_every_case_passes_strict_on_its_own() -> None:
    """Otherwise a failure below could belong to the base rather than the defect."""

    unknown_case = _case("unknown_backed_functional_complex")
    ok, reasons = _strict_verdict(unknown_case["payload"])

    assert ok is True, reasons


def test_the_fixture_covers_both_verdicts() -> None:
    """Neither half of the corpus may quietly empty out.

    ``recovers`` alone stopped being enough at C-041a: after it,
    ``only_unrelated_reactions_survive`` recovers a graph, so a corpus audited on
    ``recovers`` would still look balanced having lost its only below-threshold
    case. The three D-002 classifications and the coverage-shortfall count are
    audited beside it, because the branch in the replay test above is reached only
    by cases that declare a ``coverage_reason``.
    """

    cases = _cases()
    verdicts = {case["expect"]["recovers"] for case in cases}

    assert verdicts == {True, False}, (
        "a replay corpus of only-recovering cases cannot show that quarantine ever "
        "refuses, which is half of what it is for"
    )

    statuses = {case["expect"]["release_status"] for case in cases}
    assert statuses == {RELEASE_READY, REVIEW_REQUIRED, DIAGNOSTIC_ONLY}, (
        "the corpus must exercise all three D-002 classifications; it carries "
        f"{sorted(statuses)}"
    )

    shortfalls = [case["id"] for case in cases if case["expect"].get("coverage_reason")]
    assert len(shortfalls) >= 2, (
        "the coverage-reason assertion is reached only by cases that declare one; "
        f"{shortfalls} would leave it all but vacuous"
    )
