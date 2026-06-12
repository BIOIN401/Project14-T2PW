from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.llm import client as llm_client  # noqa: E402


class _FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return self.responses.pop(0)


def _fake_client(responses: list[object]) -> tuple[object, _FakeCompletions]:
    completions = _FakeCompletions(responses)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return fake, completions


def _response(content: str) -> object:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=None)


def test_chat_retries_json_mode_without_response_format_when_provider_returns_no_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    bad_response = SimpleNamespace(choices=None, usage=None, id="bad-response")
    fake, completions = _fake_client([bad_response, _response('{"ok": true}')])
    monkeypatch.setattr(llm_client, "_client", fake)
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("LLM_RETRY_BASE_SLEEP", "0")
    monkeypatch.setenv("LLM_RETRY_MAX_SLEEP", "0")
    monkeypatch.setenv("LLM_CALL_SPACING", "0")

    result = llm_client.chat(
        [{"role": "user", "content": "Return JSON."}],
        response_json=True,
    )

    assert result == '{"ok": true}'
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in completions.calls[1]


def test_chat_reports_malformed_provider_response_instead_of_type_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, _completions = _fake_client([SimpleNamespace(choices=None, usage=None, id="bad-response")])
    monkeypatch.setattr(llm_client, "_client", fake)
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    monkeypatch.setenv("LLM_RETRY_BASE_SLEEP", "0")
    monkeypatch.setenv("LLM_RETRY_MAX_SLEEP", "0")
    monkeypatch.setenv("LLM_CALL_SPACING", "0")

    with pytest.raises(RuntimeError, match="no chat choices"):
        llm_client.chat([{"role": "user", "content": "Hello"}])
