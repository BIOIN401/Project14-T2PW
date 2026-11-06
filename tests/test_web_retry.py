"""NEW ACCEPTANCE TEST (W4, new capability): the auto-retry policy, ``t2pw.web.retry``.

Pure: ``decide`` reads a persisted result dict. No process, no LLM, no network.
"""

from __future__ import annotations

import pytest

from t2pw.pipeline.release_status import RELEASE_READY, ReleaseStatus
from t2pw.service.pathway import (
    KIND_AMBIGUOUS,
    KIND_CONTRACT,
    KIND_CRASH,
    KIND_LLM,
    KIND_NETWORK,
    KIND_NO_REACTIONS,
    KIND_TIMEOUT,
    KIND_UNKNOWN,
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SCOPE_CONFLICT,
    STATUS_TIMEOUT,
    PathwayResult,
)
from t2pw.web import retry
from t2pw.web.jobs import result_to_dict
from t2pw.web.retry import decide


def _res(status=STATUS_FAIL, kind="", pwml=None, release=None) -> dict:
    return result_to_dict(PathwayResult(status=status, failure_kind=kind, pwml_path=pwml,
                                        release_status=release))


def test_constants_are_the_decided_defaults():
    """NEW ACCEPTANCE TEST (W4, new capability): D-W07 defaults and the cap arithmetic."""
    assert retry.MAX_RETRIES == 2
    assert retry.RUN_TIMEOUT_S == 90 * 60          # Q-E2E-1
    assert retry.PER_ITEM_WALL_CAP_S == retry.RUN_TIMEOUT_S
    assert KIND_TIMEOUT not in retry.RETRY_KINDS


def test_a_timeout_is_final():
    """Q-E2E-1: one 90 min try; a timeout is never re-queued, even with budget left."""
    d = decide(_res(STATUS_TIMEOUT, KIND_TIMEOUT), attempt=1, wall_used_s=60)
    assert not d.retry and "not retryable" in d.reason


@pytest.mark.parametrize("status,kind", [
    (STATUS_ERROR, KIND_CRASH),
    (STATUS_FAIL, KIND_LLM), (STATUS_ERROR, KIND_LLM), (STATUS_FAIL, KIND_NETWORK)])
def test_each_infrastructure_kind_retries(status, kind):
    """NEW ACCEPTANCE TEST (W4, new capability): crash, llm and network retry (not timeout)."""
    d = decide(_res(status, kind), attempt=1, wall_used_s=60)
    assert d.retry and kind in d.reason


@pytest.mark.parametrize("status,kind", [
    (STATUS_FAIL, KIND_CONTRACT), (STATUS_ERROR, KIND_CONTRACT),
    (STATUS_FAIL, KIND_AMBIGUOUS), (STATUS_FAIL, KIND_NO_REACTIONS),
    (STATUS_SCOPE_CONFLICT, ""), (STATUS_SCOPE_CONFLICT, KIND_LLM),
    (STATUS_FAIL, "some_new_kind")])
def test_content_and_scope_verdicts_never_retry(status, kind):
    """NEW ACCEPTANCE TEST (W4, new capability): contract, ambiguous_review_scope,
    no_reactions and scope_conflict are final on attempt 1 (a retry must not be a way
    around a biological gate), and so is a kind the policy does not know (fail closed)."""
    assert not decide(_res(status, kind), attempt=1, wall_used_s=0).retry


@pytest.mark.parametrize("kind", ["", KIND_UNKNOWN])
def test_error_with_no_pwml_and_no_verdict_retries(kind):
    """NEW ACCEPTANCE TEST (W4, new capability, review F1): the only retryable "no PWML"
    outcome is a run that broke (status error) with no kind, or unknown, no pwml_path
    and no release classification."""
    res = _res(STATUS_ERROR, kind)
    assert res["pwml_path"] is None and res["release_status"] is None
    d = decide(res, attempt=1, wall_used_s=0)
    assert d.retry and "no PWML" in d.reason


@pytest.mark.parametrize("res", [
    _res(STATUS_FAIL, KIND_UNKNOWN),          # strict outcome that may be an unnamed gate refusal
    _res(STATUS_FAIL, ""),
    _res(STATUS_PASS, ""),                    # research-mode success without a PWML
    _res(STATUS_PASS, KIND_UNKNOWN)])
def test_fail_or_pass_without_pwml_is_final(res):
    """NEW ACCEPTANCE TEST (W4, new capability, review F1, merge gate 6): a fail/unknown
    (possibly a lexically unnamed gate refusal) and a pass without a PWML are never
    retried, so a retry cannot re-roll the LLM until a gate passes."""
    assert res["pwml_path"] is None
    assert not decide(res, attempt=1, wall_used_s=0).retry


@pytest.mark.parametrize("status,kind", [
    (STATUS_FAIL, KIND_LLM), (STATUS_ERROR, KIND_CRASH), (STATUS_ERROR, KIND_UNKNOWN),
    (STATUS_TIMEOUT, KIND_TIMEOUT), (STATUS_FAIL, KIND_NETWORK), (STATUS_ERROR, "")])
def test_any_result_with_a_release_status_is_final(status, kind):
    """NEW ACCEPTANCE TEST (W4, new capability, review F1): a result that carries a
    release_status reached a verdict, whatever its failure_kind."""
    ready = ReleaseStatus(status=RELEASE_READY, pipeline_executed=True, strict_gates_passed=True)
    assert not decide(_res(status, kind, release=ready), 1, 0).retry


def test_a_pass_with_a_pwml_is_final(tmp_path):
    """NEW ACCEPTANCE TEST (W4, new capability): success is final."""
    assert not decide(_res(STATUS_PASS, "", pwml=tmp_path / "pathway.pwml"), 1, 0).retry


def test_stops_after_three_attempts():
    """NEW ACCEPTANCE TEST (W4, new capability): MAX_RETRIES = 2, so attempts 1 and 2
    retry and attempt 3 is final, whatever the failure."""
    crash = _res(STATUS_ERROR, KIND_CRASH)
    assert [decide(crash, n, 0).retry for n in (1, 2, 3, 4)] == [True, True, False, False]
    assert decide(crash, 1, 0, max_retries=0).retry is False


def test_wall_cap_stops_further_attempts():
    """NEW ACCEPTANCE TEST (W4, new capability): the summed run time ends retries once
    less than MIN_RETRY_BUDGET_S of the cap is left, even on attempt 1."""
    crash = _res(STATUS_ERROR, KIND_CRASH)
    cap, floor = retry.PER_ITEM_WALL_CAP_S, retry.MIN_RETRY_BUDGET_S
    assert decide(crash, 1, cap - floor).retry
    d = decide(crash, 1, cap - floor + 1)
    assert not d.retry and "wall-clock" in d.reason
    assert not decide(crash, 1, 99, wall_cap_s=100).retry
    # one attempt's worth of run time is the whole cap: nothing is left to retry with
    assert not decide(crash, 1, retry.RUN_TIMEOUT_S).retry
