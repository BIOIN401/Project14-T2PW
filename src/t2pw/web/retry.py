"""Auto-retry policy for the web job worker (W4, D-W07).

One pure function, :func:`decide`, answers "retry or final?" for a finished attempt.
It reads the persisted result dict (``jobs.result_to_dict``) and never looks at
biology: a retry is only for runs that were *interrupted*, never for a run that
reached a verdict. Retrying a content or scope verdict would be a way around a
biological gate, so those are final on the first attempt.

``failure_kind`` vocabulary is ``t2pw.service.pathway.KIND_*`` (frozen), which mirrors
``t2pw.batch.driver``; the driver's own operational / content split is reused, not
re-invented. Evidence per kind is in ``docs/webserver/results/W4.md``.

No delay between attempts: see the RESULT doc (the in-run LLM client already backs
off, and an attempt takes minutes before it can fail).
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple

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
    STATUS_PASS,
    STATUS_SCOPE_CONFLICT,
)

#: Retries after the first attempt, so at most ``MAX_RETRIES + 1`` = 3 attempts.
MAX_RETRIES = 2

#: Wall-clock cap for ONE attempt, from dequeue: 90 min (PO ruling Q-E2E-1,
#: docs/webserver/results/PO-RULINGS-2026-11-04.md). History: median 18 min, p90 43 min,
#: 15 of 280 legs over 60 min; E2E-1 timed out at 60 min and E2E-2 passed in 35 min on
#: the same input, so a timeout is LLM-latency variance, not a defect.
RUN_TIMEOUT_S = 90 * 60

#: Hard cap per pathway, summed over the run time of all its attempts (queue waits are
#: not counted). Equal to one attempt: the single worker is held at most 90 min per
#: pathway, retries (crash / network / LLM) included. A timeout is not retried, because a
#: restart doubles the cost of a slow run without making it less likely to be slow.
PER_ITEM_WALL_CAP_S = RUN_TIMEOUT_S

#: A retry is not started with less than this left of the wall cap. The history median run is
#: ~18 min (fastest runs are shorter), so this 15 min budget sits below the median: a retry
#: started with only this much left will often time out. It is a floor, not a promise of success.
MIN_RETRY_BUDGET_S = 15 * 60

#: Interrupted, not judged. KIND_TIMEOUT is deliberately absent (Q-E2E-1: one 90 min
#: try, a timeout is final). driver.py: KIND_CRASH is the
#: driver's own wrapper for an exception, not a finding (2365-2368, 3112-3120);
#: KIND_NETWORK / KIND_LLM are the transport and unparseable / empty / truncated /
#: rate-limited reply markers (268-318), and ``_classify`` (1637-1672) tests the
#: content markers first, so a contract problem is never reported as llm / network.
RETRY_KINDS = frozenset({KIND_CRASH, KIND_LLM, KIND_NETWORK})

#: Verdicts. driver.py ``_classify`` 1653-1663: structured evidence of a scope or
#: contract finding, or no reactions in the source. Never retried.
CONTENT_KINDS = frozenset({KIND_CONTRACT, KIND_AMBIGUOUS, KIND_NO_REACTIONS})

#: "Nothing matched": the driver returns UNKNOWN when it has no evidence either way
#: (1672); an empty kind means the run recorded none. Only an ``error`` status with
#: one of these is retried (the no-PWML rule); ``fail`` / ``pass`` are final.
NO_EVIDENCE_KINDS = frozenset({"", KIND_UNKNOWN})


class Decision(NamedTuple):
    retry: bool
    reason: str


def decide(result: Dict[str, Any], attempt: int, wall_used_s: float, *,
           max_retries: int = MAX_RETRIES,
           wall_cap_s: float = PER_ITEM_WALL_CAP_S) -> Decision:
    """Retry or final, for attempt number ``attempt`` (1-based) that just ended.

    ``wall_used_s`` is the run time spent on this item so far, this attempt included
    (``JobStore.wall_used``). ``result`` is the attempt's persisted result dict.
    """

    status, kind = result.get("status"), result.get("failure_kind") or ""
    if status == STATUS_SCOPE_CONFLICT:
        return Decision(False, "final: scope_conflict is a verdict")
    if result.get("release_status") is not None:
        return Decision(False, "final: the run reached a release classification (a verdict)")
    if kind in CONTENT_KINDS:
        return Decision(False, f"final: {kind} is a content verdict")
    if status == STATUS_PASS and result.get("pwml_path"):
        return Decision(False, "final: pass with a PWML")

    if kind in RETRY_KINDS:
        why = f"retry: infrastructure failure ({kind})"
    elif status == STATUS_ERROR and kind in NO_EVIDENCE_KINDS and not result.get("pwml_path"):
        # "No PWML" (narrowed by review F1): the run BROKE (status error) without
        # reaching any verdict (release_status is None, checked above). A strict
        # ``fail`` / unknown can be a gate refusal the lexical ``_classify`` could not
        # name, and a ``pass`` without a PWML is a research-mode success; retrying
        # either would re-roll the LLM until a gate passes (merge gate 6).
        why = "retry: errored with no PWML and no verdict"
    else:
        return Decision(False, f"final: {kind or 'no kind'} is not retryable")

    if attempt > max_retries:
        return Decision(False, f"final: {attempt} attempts is the limit")
    if wall_cap_s - wall_used_s < MIN_RETRY_BUDGET_S:
        return Decision(False, "final: per-pathway wall-clock cap")
    return Decision(True, why)
