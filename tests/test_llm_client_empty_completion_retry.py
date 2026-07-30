"""Retry behaviour for empty-but-successful LLM completions in ``t2pw.llm.client``.

WHY THIS FILE EXISTS. Until 2026-07-28 both retry loops in ``t2pw/llm/client.py``
fired only on a RAISED exception (RateLimitError, APITimeoutError, APIError,
json.JSONDecodeError). An HTTP 200 whose ``message.content`` was ``""`` or
``None`` was returned to the caller as a success, and ``finish_reason`` was never
inspected anywhere in the module. Two legs of the overnight batch in
``runs/2026-07-28_0919`` died on that hole:

    PMC13278307 / research   FAIL after  137s   "Payload must include a processes object"
    PMC13231680 / strict     FAIL after   55s   "Payload must include a processes object"

Both are downstream of an empty Stage 0 reply - Stage 0 is the only stage that
calls ``chat()`` once with no parse-retry wrapper - and PMC13231680 had extracted
3 reactions successfully from the identical PDF the previous day, proving the
empty reply was a transient rather than a property of the paper.

These tests pin the three behaviours that fix depends on:

  1. an empty completion is retried inside the EXISTING loop/backoff/budget and a
     later non-empty reply is returned normally;
  2. once ``LLM_MAX_RETRIES`` is spent the empty string is still RETURNED, not
     raised, because ``pipeline/preprocessor.py`` already turns "" into status
     ``empty_reply`` and a new exception type would change every call site's
     contract at once;
  3. a reply with NO content but WITH ``tool_calls`` is a legitimate
     function-calling turn and must NOT be retried - re-issuing a round the model
     already answered would be a worse regression than the bug being fixed.

Everything here is offline and deterministic: ``_client`` is replaced with a
canned-response stub and ``time`` with a recorder, so no network call and no real
sleep happens.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ``t2pw.llm.client`` does ``from openai import OpenAI`` and builds a client at
# import time. Per the repo convention, any test that pulls in the llm client
# stubs ``openai`` itself so it runs in isolation (not only when an earlier test
# in the session happened to have stubbed it first).
if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *_: object, **__: object) -> None:
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **__: None)
            )

    openai_stub.OpenAI = _OpenAI
    openai_stub.RateLimitError = RuntimeError
    openai_stub.APIError = RuntimeError
    openai_stub.APITimeoutError = RuntimeError
    openai_stub.AuthenticationError = RuntimeError
    openai_stub.BadRequestError = RuntimeError
    sys.modules["openai"] = openai_stub

from t2pw.llm import client as client_mod  # noqa: E402
from t2pw.llm.client import (  # noqa: E402
    _completion_is_empty,
    _finish_reason,
    chat,
    chat_detailed,
    chat_with_tools,
)


# ---------------------------------------------------------------------------
# Canned response / client doubles
# ---------------------------------------------------------------------------
def _tool_call(name: str = "lookup", arguments: str = '{"q": "x"}', call_id: str = "call_1") -> Any:
    return types.SimpleNamespace(
        id=call_id,
        function=types.SimpleNamespace(name=name, arguments=arguments),
    )


def _resp(content: Any, *, finish_reason: str = "stop", tool_calls: Any = None) -> Any:
    """A minimal stand-in for an openai ChatCompletion.

    ``usage=None`` so ``_record_usage`` returns early and the module's global
    token counters stay untouched by these tests.
    """
    message = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    return types.SimpleNamespace(choices=[choice], usage=None)


class _FakeCompletions:
    def __init__(self, responses: List[Any]) -> None:
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError(
                f"client made {len(self.calls)} calls but only "
                f"{len(self.calls) - 1} canned responses were provided"
            )
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, responses: List[Any]) -> None:
        self.completions = _FakeCompletions(responses)
        self.chat = types.SimpleNamespace(completions=self.completions)


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch):
    """Install the canned client + a sleep recorder and shrink the retry budget.

    The budget is set to 3 (rather than the shipped default of 8) purely so the
    exhaustion tests stay short; the code under test must read it from the env on
    every call, which is itself part of "reuse the budget already there".
    """
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("LLM_RETRY_BASE_SLEEP", "0.5")
    monkeypatch.setenv("LLM_RETRY_MAX_SLEEP", "2.0")
    monkeypatch.setenv("LLM_CALL_SPACING", "0.01")

    sleeps: List[float] = []
    # Replace the module-level `time` reference rather than patching stdlib's
    # time.sleep globally, so nothing outside this module is affected.
    monkeypatch.setattr(client_mod, "time", types.SimpleNamespace(sleep=sleeps.append))

    def _install(responses: List[Any]) -> _FakeClient:
        fake = _FakeClient(responses)
        monkeypatch.setattr(client_mod, "_client", fake)
        return fake

    return types.SimpleNamespace(install=_install, sleeps=sleeps)


MESSAGES = [{"role": "user", "content": "extract the pathway"}]


# ---------------------------------------------------------------------------
# chat()
# ---------------------------------------------------------------------------
def test_chat_returns_first_non_empty_reply_without_retrying(llm) -> None:
    """Regression guard: the fix must not add retries to the happy path."""
    fake = llm.install([_resp('{"pathway_name": "lipid A biosynthesis"}')])

    out = chat(MESSAGES, model_override="test-model")

    assert out == '{"pathway_name": "lipid A biosynthesis"}'
    assert len(fake.completions.calls) == 1


def test_chat_retries_empty_completion_then_succeeds(llm) -> None:
    """The PMC13231680 case: two dead 200s, then the reply that was always there.

    The whitespace-only second reply is the same shape the preprocessor's own
    test feeds in ("   \\n\\t ") and must count as empty, since every caller
    ``.strip()``s the result anyway.
    """
    fake = llm.install(
        [
            _resp(""),
            _resp("   \n\t "),
            _resp('{"pathway_name": "lipid A biosynthesis"}'),
        ]
    )

    out = chat(MESSAGES, model_override="test-model")

    assert out == '{"pathway_name": "lipid A biosynthesis"}'
    assert len(fake.completions.calls) == 3
    # Backoff came from the existing loop, not from a second retry mechanism:
    # base_sleep * 2**attempt for attempts 0 and 1 == 0.5 and 1.0.
    assert 0.5 in llm.sleeps and 1.0 in llm.sleeps


def test_chat_retries_content_none_not_just_empty_string(llm) -> None:
    """``content=None`` is what an OpenAI-compatible 200 actually carries; the
    pre-fix code laundered it into "" via ``(content or "")`` and called it a
    success."""
    fake = llm.install([_resp(None), _resp("recovered")])

    assert chat(MESSAGES, model_override="test-model") == "recovered"
    assert len(fake.completions.calls) == 2


def test_chat_returns_empty_string_after_budget_exhausted(llm) -> None:
    """Budget spent -> return "", never raise.

    ``pipeline/preprocessor.py`` inspects the returned text and records status
    ``empty_reply`` with the empty context; raising a new exception type here
    would change the contract of every call site in the pipeline at once.
    """
    fake = llm.install([_resp(""), _resp(""), _resp("")])

    out = chat(MESSAGES, model_override="test-model")

    assert out == ""
    assert len(fake.completions.calls) == 3  # exactly LLM_MAX_RETRIES, no more


def test_chat_logs_finish_reason_so_a_content_filter_is_distinguishable(
    llm, caplog: pytest.LogCaptureFixture
) -> None:
    """A moderation stop and a provider hiccup both arrive as an empty 200.

    The 2026-07-28 postmortem could not tell them apart because nothing logged
    ``finish_reason``. Retrying a content_filter stop cannot help, so the
    operator has to be able to see which one they got.

    As of 2026-07-29 the loop also STOPS on it rather than merely logging it --
    see the durable-exit section below -- so only the first canned reply is
    consumed here. The log-line requirement is unchanged and is what this test
    still pins.
    """
    llm.install([_resp("", finish_reason="content_filter"), _resp("", finish_reason="content_filter"),
                 _resp("", finish_reason="content_filter")])

    with caplog.at_level(logging.WARNING, logger="t2pw.llm.client"):
        assert chat(MESSAGES, model_override="test-model", stage_name="preprocessor") == ""

    assert "content_filter" in caplog.text
    assert "preprocessor" in caplog.text


# ---------------------------------------------------------------------------
# content_filter: a DURABLE stop, not a transient (added 2026-07-29)
#
# The 2026-07-28 fix folded every empty 200 into one retry loop. That is right
# for a provider hiccup and wrong for a moderation stop: moderation is a
# deterministic function of the prompt, so re-sending the IDENTICAL prompt gets
# the identical verdict. LLM_MAX_RETRIES draws then buy nothing but latency, and
# the caller ends up with the same empty string it could have had immediately --
# with no recorded reason, so nobody could tell that editing the prompt rather
# than re-running the leg was the fix.
#
# Retrying a DIFFERENT prompt is still allowed, and is what localized repair
# does. What is forbidden is the blind identical retry.
# ---------------------------------------------------------------------------
def test_content_filter_exits_immediately_instead_of_retrying(llm) -> None:
    """One call, not LLM_MAX_RETRIES calls, on a moderation stop."""

    fake = llm.install(
        [
            _resp("", finish_reason="content_filter"),
            _resp("this must never be reached"),
            _resp("nor this"),
        ]
    )

    out = chat(MESSAGES, model_override="test-model", stage_name="preprocessor")

    assert out == ""
    assert len(fake.completions.calls) == 1
    # No backoff was spent either: the only pause is the LLM_CALL_SPACING one
    # every call makes, not the exponential retry sleep.
    assert llm.sleeps == [0.01]


def test_content_filter_carries_a_durable_reason_the_caller_can_act_on(llm) -> None:
    """The reason has to survive the call, not just the log line.

    ``terminal_reason`` is what lets Stage 0, ``_run_json_stage`` and the repair
    passes each decline to retry without re-deriving "was that moderation?" from
    a finish_reason string of their own.
    """

    llm.install([_resp("", finish_reason="content_filter")])

    result = chat_detailed(MESSAGES, model_override="test-model", stage_name="preprocessor")

    assert result.text == ""
    assert result.diagnostics.terminal_reason == "content_filter"
    assert result.diagnostics.response_status == "content_filter"
    assert result.diagnostics.finish_reason == "content_filter"
    assert result.diagnostics.attempts == 1
    assert result.diagnostics.model == "test-model"


def test_a_content_filter_stop_that_carries_text_is_still_a_result(llm) -> None:
    """A partial answer is an answer.

    Only an EMPTY content_filter reply is a dead end. One with text is a
    truncated result and must be returned like any other, or this change would
    start discarding real content.
    """

    fake = llm.install([_resp('{"pathway_name": "partial"}', finish_reason="content_filter")])

    result = chat_detailed(MESSAGES, model_override="test-model")

    assert result.text == '{"pathway_name": "partial"}'
    assert result.diagnostics.terminal_reason == ""
    assert len(fake.completions.calls) == 1


@pytest.mark.parametrize("reason", ["content_filter", "content-filter", "SAFETY"])
def test_moderation_spellings_are_all_recognised(llm, reason: str) -> None:
    """OpenAI-compatible gateways are not consistent about this string, and a
    spelling we fail to recognise silently restores the burn-the-budget
    behaviour this section exists to end."""

    fake = llm.install([_resp("", finish_reason=reason), _resp("unreachable")])

    assert chat(MESSAGES, model_override="test-model") == ""
    assert len(fake.completions.calls) == 1


def test_an_ordinary_empty_completion_still_retries(llm) -> None:
    """The regression guard for the tests above: only moderation is durable.

    A ``stop`` or ``length`` finish_reason with no text is still the transient
    the 2026-07-28 fix was written for and must still be retried.
    """

    fake = llm.install([_resp("", finish_reason="stop"), _resp("recovered")])

    assert chat(MESSAGES, model_override="test-model") == "recovered"
    assert len(fake.completions.calls) == 2


def test_chat_with_tools_also_stops_on_a_moderation_refusal(llm) -> None:
    """The same rule at the other call site, which has its own retry loop."""

    fake = llm.install(
        [_resp("", finish_reason="content_filter"), _resp("unreachable")]
    )

    out = chat_with_tools(MESSAGES, TOOLS, _noop_executor, model_override="test-model")

    assert out == ""
    assert len(fake.completions.calls) == 1


# ---------------------------------------------------------------------------
# Diagnostics on the ordinary paths
#
# The text ``chat`` returns is unchanged; what is new is that the run can say
# which model answered, on what finish_reason, and what each attempt cost.
# ---------------------------------------------------------------------------
def test_empty_completion_then_successful_retry_is_fully_described(llm) -> None:
    """The PMC13231680 shape, now with the record the postmortem needed."""

    fake = llm.install([_resp(""), _resp("   \n\t "), _resp('{"pathway_name": "lipid A"}')])

    result = chat_detailed(MESSAGES, model_override="test-model", stage_name="preprocessor")

    assert result.text == '{"pathway_name": "lipid A"}'
    assert len(fake.completions.calls) == 3
    diagnostics = result.diagnostics
    assert diagnostics.attempts == 3
    assert diagnostics.model == "test-model"
    assert diagnostics.stage == "preprocessor"
    assert diagnostics.response_status == "ok"
    assert diagnostics.terminal_reason == ""
    assert [row["status"] for row in diagnostics.attempt_log] == ["empty", "empty", "ok"]
    assert diagnostics.raw_chars == len('{"pathway_name": "lipid A"}')
    assert diagnostics.request_hash.startswith("sha256:")
    assert diagnostics.response_hash.startswith("sha256:")


def test_the_exhausted_empty_budget_is_reported_as_such(llm) -> None:
    llm.install([_resp(""), _resp(""), _resp("")])

    result = chat_detailed(MESSAGES, model_override="test-model")

    assert result.text == ""
    assert result.diagnostics.terminal_reason == "empty_after_retries"
    assert result.diagnostics.attempts == 3


def test_transient_provider_errors_appear_in_the_attempt_log(llm) -> None:
    """A transient that was retried away is still a fact about the run.

    A truncated HTTP body -- what the SDK raises ``json.JSONDecodeError`` for --
    is used rather than ``RateLimitError`` because it is constructible whether
    the real ``openai`` package or this file's stub is installed, and the client
    treats both the same way.
    """

    llm.install([json.JSONDecodeError("Expecting value", "", 0), _resp("recovered")])

    result = chat_detailed(MESSAGES, model_override="test-model")

    assert result.text == "recovered"
    assert [row["status"] for row in result.diagnostics.attempt_log] == ["error", "ok"]
    assert "Expecting value" in result.diagnostics.attempt_log[0]["error"]


def test_chat_is_still_exactly_the_text(llm) -> None:
    """``chat`` keeps its historical contract: a string, nothing wrapped.

    Dozens of call sites depend on it, and a diagnostics envelope leaking into
    them would be a far bigger change than the one being made.
    """

    llm.install([_resp("plain text")])

    assert chat(MESSAGES, model_override="test-model") == "plain text"


# ---------------------------------------------------------------------------
# chat_with_tools(): the tool round
# ---------------------------------------------------------------------------
def _noop_executor(name: str, args: Dict[str, Any]) -> Any:
    return {"ok": True, "tool": name, "args": args}


TOOLS = [
    {
        "type": "function",
        "function": {"name": "lookup", "description": "look something up", "parameters": {}},
    }
]


def test_chat_with_tools_does_not_retry_a_tool_call_only_reply(llm) -> None:
    """THE regression this fix must not cause.

    An assistant message with ``tool_calls`` and ``content=None`` is the normal,
    correct shape of a function-calling turn. Retrying it would re-issue a round
    the model already answered and could double-execute the tool.
    """
    executed: List[str] = []

    def executor(name: str, args: Dict[str, Any]) -> Any:
        executed.append(name)
        return {"ok": True}

    fake = llm.install(
        [
            _resp(None, finish_reason="tool_calls", tool_calls=[_tool_call()]),
            _resp("final answer"),
        ]
    )

    out = chat_with_tools(MESSAGES, TOOLS, executor, model_override="test-model")

    assert out == "final answer"
    assert executed == ["lookup"]  # executed exactly once, not once per retry
    assert len(fake.completions.calls) == 2  # no retry was spent on the tool turn


def test_chat_with_tools_retries_an_empty_reply_with_no_tool_calls(llm) -> None:
    """No content and no tool_calls is a dead 200 even on the tool round."""
    fake = llm.install(
        [
            _resp("", tool_calls=None),
            _resp("final answer"),
        ]
    )

    out = chat_with_tools(MESSAGES, TOOLS, _noop_executor, model_override="test-model")

    assert out == "final answer"
    assert len(fake.completions.calls) == 2


def test_chat_with_tools_returns_empty_after_budget_exhausted(llm) -> None:
    fake = llm.install([_resp(""), _resp(""), _resp("")])

    out = chat_with_tools(MESSAGES, TOOLS, _noop_executor, model_override="test-model")

    assert out == ""
    assert len(fake.completions.calls) == 3


# ---------------------------------------------------------------------------
# chat_with_tools(): the forced final round (include_tools=False)
# ---------------------------------------------------------------------------
def test_chat_with_tools_retries_empty_content_on_the_forced_final_round(llm) -> None:
    """Third site: after ``max_tool_rounds`` the loop makes one more call with no
    tools at all and reduces it to ``(content or "").strip()``. Text is the only
    payload that can exist there, so an empty one is retryable."""
    fake = llm.install(
        [
            _resp(None, finish_reason="tool_calls", tool_calls=[_tool_call()]),
            _resp(""),
            _resp("forced final answer"),
        ]
    )

    out = chat_with_tools(
        MESSAGES, TOOLS, _noop_executor, model_override="test-model", max_tool_rounds=1
    )

    assert out == "forced final answer"
    assert len(fake.completions.calls) == 3
    # The tool round sends tools; the forced final round deliberately does not.
    assert "tools" in fake.completions.calls[0]
    assert "tools" not in fake.completions.calls[1]
    assert "tools" not in fake.completions.calls[2]


# ---------------------------------------------------------------------------
# The emptiness predicate itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "content, tool_calls, tools_were_sent, expected",
    [
        ("text", None, False, False),
        ("text", None, True, False),
        ("", None, False, True),
        (None, None, True, True),
        ("   \n\t ", None, False, True),
        # tool_calls are payload only when we actually asked for tools.
        (None, [_tool_call()], True, False),
        (None, [_tool_call()], False, True),
    ],
)
def test_completion_is_empty_predicate(content, tool_calls, tools_were_sent, expected) -> None:
    resp = _resp(content, tool_calls=tool_calls)
    assert _completion_is_empty(resp, tools_were_sent=tools_were_sent) is expected


def test_emptiness_helpers_never_raise_on_a_malformed_response() -> None:
    """Diagnostics must not be able to kill a call. A response shaped unlike
    anything we recognise is reported as empty (so it takes the retry path) and
    its finish_reason degrades to a marker string rather than an exception."""
    junk = types.SimpleNamespace(choices=[])

    assert _completion_is_empty(junk, tools_were_sent=False) is True
    assert _finish_reason(junk) == "unavailable"
    assert _finish_reason(_resp("x", finish_reason=None)) == "unknown"
