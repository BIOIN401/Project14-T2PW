"""Every LLM request is bounded, and a timeout is its own outcome.

Before this file, ``grep -c "timeout=" src/t2pw/llm/client.py`` was 0: neither
``OpenAI(...)`` construction set one, so a request's only ceiling was the SDK's
600 s READ default times its own 3 internal HTTP attempts, and one
``chat_detailed`` could burn far past the 3480 s child deadline. Offline
throughout -- the client is a canned stub, nothing reaches a provider.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List

import dotenv
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Stubbed like every other client test, but with DISTINCT exception classes in
# the real SDK's hierarchy (APITimeoutError < APIError), so "a timeout is not
# merely another APIError" is exercised rather than assumed.
if "openai" not in sys.modules:
    _stub = types.ModuleType("openai")
    _stub.OpenAI = type("OpenAI", (), {"__init__": lambda self, **kw: None})
    _stub.APIError = type("APIError", (Exception,), {})
    _stub.APITimeoutError = type("APITimeoutError", (_stub.APIError,), {})
    for _name in ("RateLimitError", "AuthenticationError", "BadRequestError"):
        setattr(_stub, _name, type(_name, (Exception,), {}))
    sys.modules["openai"] = _stub

import openai  # noqa: E402
from t2pw.llm import client as client_mod  # noqa: E402

MESSAGES = [{"role": "user", "content": "extract the pathway"}]
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
HANG = object()  #: script item meaning "wait out the timeout, then time out"
UNBOUNDED_SECONDS = 3.0  #: what one attempt costs when nothing bounds it


def _resp(text: str) -> Any:
    message = types.SimpleNamespace(content=text, tool_calls=None)
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], usage=None)


class _Completions:
    """Transport double. ``HANG`` waits the timeout it was handed, then raises."""

    def __init__(self, script: List[Any]) -> None:
        self._script: List[Any] = list(script)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self._script.pop(0) if self._script else HANG
        if item is HANG:
            time.sleep(min(kwargs["timeout"].read, UNBOUNDED_SECONDS))
            raise openai.APITimeoutError("request timed out")
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch):
    """Canned client on a short but REAL budget -- test (a) measures elapsed time."""
    for name, value in (
        ("LLM_MAX_RETRIES", "2"),
        ("LLM_RETRY_BASE_SLEEP", "0.01"),
        ("LLM_RETRY_MAX_SLEEP", "0.01"),
        ("LLM_CALL_SPACING", "0"),
    ):
        monkeypatch.setenv(name, value)

    def _install(script: List[Any]) -> _Completions:
        fake = _Completions(script)
        chat_ns = types.SimpleNamespace(completions=fake)
        monkeypatch.setattr(client_mod, "_client", types.SimpleNamespace(chat=chat_ns))
        return fake

    return _install


def test_a_timeout_raises_the_distinct_outcome_inside_the_computed_bound(llm) -> None:
    """(a) The bound is asserted as elapsed time, not merely as an exception type."""
    fake = llm([HANG, HANG])
    bound = client_mod.worst_case_call_seconds(timeout=0.05)  # 2*0.05 + 2*0.01 + 0
    started = time.monotonic()
    with pytest.raises(client_mod.LLMOperationTimeout) as caught:
        client_mod.chat_detailed(MESSAGES, model_override="m", timeout=0.05)
    elapsed = time.monotonic() - started
    assert caught.value.terminal_reason == "operation_timeout"
    assert [c["timeout"].read for c in fake.calls] == [0.05, 0.05]
    assert elapsed < UNBOUNDED_SECONDS, "the request reached the transport unbounded"
    assert elapsed <= bound + 1.0, f"{elapsed:.3f}s exceeded the bound of {bound}s"


def test_b_worst_case_duration_is_the_hand_computed_value(monkeypatch) -> None:
    """(b) The exact number a deadline owner budgets against before calling."""
    # 4 attempts x 30 s = 120.0 | backoff min(1.0 * 2**i, 20.0), i=0..3 = 15.0
    # | spacing 4 x 0.35 = 1.4 | total = 136.4
    cfg = dict(timeout=30.0, max_retries=4, base_sleep=1.0, max_sleep=20.0, spacing=0.35)
    assert client_mod.worst_case_call_seconds(**cfg) == pytest.approx(136.4, abs=1e-9)
    # chat_with_tools: max_tool_rounds tool rounds plus one forced final round.
    assert client_mod.worst_case_call_seconds(tool_rounds=1, **cfg) == pytest.approx(272.8, abs=1e-9)
    for name in ("LLM_MAX_RETRIES", "LLM_RETRY_BASE_SLEEP", "LLM_RETRY_MAX_SLEEP", "LLM_CALL_SPACING"):
        monkeypatch.delenv(name, raising=False)
    # Shipped defaults, SDK retries pinned off: 8 x 300 + (1+2+4+8+16+20+20+20)
    # + 8 x 0.35 = 2493.8 s, what makes D-005's 3480 s deadline enforceable.
    assert client_mod.REQUEST_TIMEOUT_SECONDS == 300.0
    assert client_mod.worst_case_call_seconds() == pytest.approx(2493.8, abs=1e-9)
    assert client_mod.worst_case_call_seconds() < 3480.0
    # The SDK multiplier must be PRICED, not assumed to be 1: at the SDK's own
    # default of 2 this config really costs 3 x 30 s + 2 x 8 s backoff per attempt.
    assert client_mod.SDK_MAX_RETRIES == 0
    monkeypatch.setattr(client_mod, "SDK_MAX_RETRIES", 2)
    assert client_mod.worst_case_call_seconds(**cfg) == pytest.approx(440.4, abs=1e-9)


def test_c_a_per_call_override_is_honoured_and_does_not_leak(llm) -> None:
    """(c) An override bounds its own call only; the module default is untouched."""
    default = client_mod.REQUEST_TIMEOUT_SECONDS
    fake = llm([_resp("one"), _resp("two"), _resp("three")])
    client_mod.chat_detailed(MESSAGES, model_override="m", timeout=7.5)
    client_mod.chat_detailed(MESSAGES, model_override="m")
    client_mod.chat_with_tools(MESSAGES, TOOLS, lambda *_: {}, model_override="m", timeout=9.0)
    assert [c["timeout"].read for c in fake.calls] == [7.5, default, 9.0]
    assert [c["timeout"].connect for c in fake.calls] == [5.0, 5.0, 5.0]
    assert client_mod.REQUEST_TIMEOUT_SECONDS == default


def test_d_a_timeout_is_never_filed_as_an_empty_completion(llm, monkeypatch) -> None:
    """(d) Distinct status, and the empty-completion retry budget is left intact."""
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    fake = llm([openai.APITimeoutError("t"), _resp(""), _resp("recovered")])
    result = client_mod.chat_detailed(MESSAGES, model_override="m")
    assert result.text == "recovered"
    assert [r["status"] for r in result.diagnostics.attempt_log] == ["timeout", "empty", "ok"]
    assert result.diagnostics.terminal_reason == ""
    assert len(fake.calls) == 3


def test_d_a_budget_spent_on_timeouts_is_not_the_empty_string_exit(llm) -> None:
    """(d) Both loops raise operation_timeout instead of returning "" as empty."""
    llm([openai.APITimeoutError("t"), openai.APITimeoutError("t")])
    with pytest.raises(client_mod.LLMOperationTimeout) as caught:
        client_mod.chat_detailed(MESSAGES, model_override="m")
    assert caught.value.diagnostics.terminal_reason == client_mod.TERMINAL_OPERATION_TIMEOUT
    assert [r["status"] for r in caught.value.diagnostics.attempt_log] == ["timeout", "timeout"]
    llm([openai.APITimeoutError("t"), openai.APITimeoutError("t")])
    with pytest.raises(client_mod.LLMOperationTimeout):
        client_mod.chat_with_tools(MESSAGES, TOOLS, lambda *_: {}, model_override="m")


def _second_copy(name: str) -> None:
    """Run another copy of client.py so its import-time construction reruns.

    ``sys.modules["t2pw.llm.client"]`` is deliberately left untouched.
    """
    spec = importlib.util.spec_from_file_location(name, SRC / "t2pw" / "llm" / "client.py")
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_e_both_openai_constructions_receive_the_timeout(monkeypatch) -> None:
    """(e) Covering only the openrouter path would leave the local path unbounded."""
    built: List[Dict[str, Any]] = []
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: built.append(kw))
    monkeypatch.setattr(dotenv, "load_dotenv", lambda **_: False)  # a real .env must not win
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "42")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-" + "probe")
    monkeypatch.setenv("OPENROUTER_MODEL", "probe-model")
    for provider in ("openrouter", "local"):
        monkeypatch.setenv("LLM_PROVIDER", provider)
        _second_copy(f"c014_probe_{provider}")
    assert [kw["timeout"].read for kw in built] == [42.0, 42.0]
    # A bare float expands to connect=42.0 as well, relaxing the SDK's own 5 s
    # connect: MORE permissive than base, which MUST NOT CHANGE forbids.
    assert [kw["timeout"].connect for kw in built] == [5.0, 5.0]
    # Left unset, the SDK's default of 2 makes one create() three HTTP attempts,
    # so worst_case_call_seconds under-reports every call by ~3x.
    assert [kw["max_retries"] for kw in built] == [0, 0]
