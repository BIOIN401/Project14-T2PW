"""OBS-RAW: ``T2PW_KEEP_RAW_LLM`` keeps the raw reply of every provider attempt beside the run's leg trace.

Offline: ``_client`` is a canned fake, no network. Off by default; on, one JSON line per attempt in
``llm_raw_responses.jsonl`` in the folder of the active leg trace.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, List

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "openai" not in sys.modules:  # same stub the other llm-client tests use
    _stub = types.ModuleType("openai")
    _stub.OpenAI = lambda *a, **k: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **__: None)))
    for _name in ("RateLimitError", "APIError", "APITimeoutError", "AuthenticationError", "BadRequestError"):
        setattr(_stub, _name, RuntimeError)
    sys.modules["openai"] = _stub

from t2pw.batch import leg_trace  # noqa: E402
from t2pw.llm import client as client_mod  # noqa: E402

FLAG = "T2PW_KEEP_RAW_LLM"
MESSAGES = [{"role": "user", "content": "audit this"}]
EXPECTED_KEYS = {"ts", "stage", "model", "attempt", "status", "finish_reason", "error", "request_hash",
                 "response_chars", "response"}


_QUEUE: List[Any] = []  # the canned replies the fake client hands out, in order


def _resp(content: Any, finish_reason: str = "stop") -> Any:
    message = types.SimpleNamespace(content=content, tool_calls=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason=finish_reason)],
                                 usage=None)


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A fake client plus an active leg trace in ``tmp_path`` (what a job or a batch leg has)."""

    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("LLM_CALL_SPACING", "0")
    monkeypatch.setenv("LLM_RETRY_BASE_SLEEP", "0")
    monkeypatch.delenv(FLAG, raising=False)
    queue = _QUEUE
    queue.clear()

    def create(**_kw: Any) -> Any:
        return queue.pop(0)

    monkeypatch.setattr(client_mod, "_client", types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))))
    leg_trace.activate(tmp_path / leg_trace.LEG_TRACE_NAME)
    yield tmp_path
    leg_trace.deactivate()


def _lines(folder: Path) -> List[dict]:
    path = folder / client_mod.RAW_LLM_FILE
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_flag_off_writes_nothing(run_dir) -> None:
    _QUEUE.append(_resp("hello"))
    assert client_mod.chat(MESSAGES, model_override="m", stage_name="audit loop") == "hello"
    assert not (run_dir / client_mod.RAW_LLM_FILE).exists()


def test_flag_on_one_line_per_call_with_the_raw_reply(run_dir, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "1")
    _QUEUE.extend([_resp('  {"a": 1}\n'), _resp("second", "length")])
    assert client_mod.chat(MESSAGES, model_override="m", stage_name="audit loop") == '{"a": 1}'
    client_mod.chat(MESSAGES, model_override="m", stage_name="gap resolver")
    first, second = _lines(run_dir)
    assert set(first) == EXPECTED_KEYS
    assert first["response"] == '  {"a": 1}\n'  # exactly as returned, not stripped
    assert (first["stage"], first["model"], first["status"], first["finish_reason"]) == ("audit loop", "m", "ok", "stop")
    assert first["request_hash"].startswith("sha256:") and first["ts"].endswith("+00:00")
    assert (second["stage"], second["finish_reason"], second["response"]) == ("gap resolver", "length", "second")


def test_tool_loop_is_kept_too(run_dir, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "1")
    _QUEUE.append(_resp("final text"))
    assert client_mod.chat_with_tools(MESSAGES, [], lambda *_: None, model_override="m", stage_name="curator") == "final text"
    (row,) = _lines(run_dir)
    assert (row["stage"], row["response"]) == ("curator", "final text")
    assert [e for e in leg_trace.read_events(run_dir) if e.get("kind") == "model_attempt"] == []  # leg trace unchanged


def test_empty_attempt_is_a_line_with_no_response(run_dir, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "1")
    _QUEUE.extend([_resp(""), _resp("ok")])
    client_mod.chat(MESSAGES, model_override="m")
    empty, ok = _lines(run_dir)
    assert (empty["status"], "response" in empty, ok["response"]) == ("empty", False, "ok")


def test_no_secret_or_provider_field(run_dir, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-0123456789abcdef0123456789abcdef")
    _QUEUE.append(_resp("reply"))
    client_mod.chat(MESSAGES, model_override="m")
    text = (run_dir / client_mod.RAW_LLM_FILE).read_text(encoding="utf-8")
    assert "sk-or-v1" not in text
    assert not {"headers", "api_key", "base_url", "provider", "authorization"} & set(_lines(run_dir)[0])


def test_long_reply_is_truncated_with_a_marker(run_dir, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(client_mod, "RAW_LLM_MAX_CHARS", 10)
    _QUEUE.append(_resp("x" * 25))
    client_mod.chat(MESSAGES, model_override="m")
    (row,) = _lines(run_dir)
    assert row["truncated"] is True and row["response_chars"] == 25
    assert row["response"].startswith("x" * 10 + "\n[TRUNCATED: 25 chars")


def test_each_activation_starts_a_fresh_file(run_dir, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "1")
    _QUEUE.extend([_resp("job one"), _resp("job two")])
    client_mod.chat(MESSAGES, model_override="m")
    leg_trace.activate(run_dir / leg_trace.LEG_TRACE_NAME)  # the next job in the same folder
    client_mod.chat(MESSAGES, model_override="m")
    assert [r["response"] for r in _lines(run_dir)] == ["job two"]


def test_a_failing_write_never_fails_the_call(run_dir, monkeypatch, caplog) -> None:
    monkeypatch.setenv(FLAG, "1")
    (run_dir / client_mod.RAW_LLM_FILE).mkdir()  # opening it for write now raises
    _QUEUE.append(_resp("still fine"))
    assert client_mod.chat(MESSAGES, model_override="m") == "still fine"
    assert "could not write" in caplog.text
