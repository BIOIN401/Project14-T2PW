"""NEW TESTS (RT1, part 1): per-call logging on the website path.

Offline: ``t2pw.llm.client._client`` is replaced with a canned fake, retry sleeps are set to 0 by
env. The rule these pin: a website job (``run_pathway``) writes a per-call trace with start time,
duration, tokens and provider, curator calls included, in the job's folder and not among its
downloads; everything else (batch, the app) records exactly what it did before; and no value
returned by a call, no ``CompletionDiagnostics.to_dict()`` and no attempt row changes.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, List, Optional

import pytest

try:  # the real SDK when installed, so its error classes keep their real hierarchy
    import openai  # noqa: F401
except ImportError:  # same stub convention as test_llm_client_empty_completion_retry.py
    _stub = types.ModuleType("openai")
    _stub.OpenAI = lambda *a, **k: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **__: None)))
    for _name in ("RateLimitError", "APIError", "APITimeoutError", "AuthenticationError", "BadRequestError"):
        setattr(_stub, _name, RuntimeError)
    sys.modules["openai"] = _stub

from t2pw.batch import leg_trace  # noqa: E402
from t2pw.llm import client as client_mod  # noqa: E402
from t2pw.service import pathway as pw  # noqa: E402
from t2pw.service.pathway import PathwayRequest  # noqa: E402


def _resp(text: Optional[str] = "answer", *, finish: str = "stop", tool_calls: Any = None,
          prompt: int = 11, completion: int = 22, reasoning: int = 7,
          provider: str = "DeepInfra", gid: str = "gen-abc") -> Any:
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text, tool_calls=tool_calls),
                                       finish_reason=finish)],
        usage=types.SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                                    completion_tokens_details=types.SimpleNamespace(reasoning_tokens=reasoning)),
        provider=provider, id=gid, model="served-model", model_extra={})


def _api_error() -> Exception:
    return client_mod.APIError("boom")


def _fake_client(replies: List[Any]) -> Any:
    queue = list(replies)

    def create(**kwargs: Any) -> Any:
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    # Distinct error classes whatever an earlier test file did to the ``openai`` import (some stub every
    # error as RuntimeError, which would send an APIError down the authentication branch).
    for name in ("RateLimitError", "APIError", "APITimeoutError", "AuthenticationError", "BadRequestError"):
        monkeypatch.setattr(client_mod, name, type(name, (Exception,), {}))
    monkeypatch.setenv("LLM_CALL_SPACING", "0")
    monkeypatch.setenv("LLM_RETRY_BASE_SLEEP", "0")
    monkeypatch.setenv("LLM_RETRY_MAX_SLEEP", "0")
    yield
    leg_trace.deactivate()


def _trace(tmp_path: Path) -> Path:
    leg_trace.activate(tmp_path / leg_trace.LEG_TRACE_NAME)
    return tmp_path


def _attempts(tmp_path: Path) -> List[dict]:
    return [e for e in leg_trace.read_events(tmp_path) if e["kind"] == "model_attempt"]


def _chat(**kw: Any):
    return client_mod.chat_detailed([{"role": "user", "content": "hi"}], model_override="m",
                                    stage_name=kw.pop("stage_name", "Stage 1 extraction"), **kw)


# -- the new fields ----------------------------------------------------------------------
def test_a_web_call_records_start_duration_tokens_provider_and_generation_id(monkeypatch, tmp_path):
    monkeypatch.setattr(client_mod, "_client", _fake_client([_resp()]))
    _trace(tmp_path)
    with client_mod.web_run_scope():
        result = _chat()
    (row,) = _attempts(tmp_path)
    assert row["stage"] == "Stage 1 extraction" and row["status"] == "ok" and row["finish_reason"] == "stop"
    assert (row["prompt_tokens"], row["completion_tokens"], row["reasoning_tokens"]) == (11, 22, 7)
    assert row["provider"] == "DeepInfra" and row["generation_id"] == "gen-abc"
    assert row["response_model"] == "served-model" and row["model"] == "m"
    assert row["started_at"].endswith("+00:00") and isinstance(row["duration_s"], float) and row["duration_s"] >= 0
    assert result.text == "answer"


def test_each_attempt_of_a_retried_call_has_its_own_timing(monkeypatch, tmp_path):
    monkeypatch.setattr(client_mod, "_client", _fake_client([_resp(""), _resp(""), _resp("third")]))
    _trace(tmp_path)
    with client_mod.web_run_scope():
        result = _chat()
    rows = _attempts(tmp_path)
    assert [r["status"] for r in rows] == ["empty", "empty", "ok"] and result.text == "third"
    assert all("duration_s" in r and "started_at" in r for r in rows)
    assert rows[0]["started_at"] <= rows[1]["started_at"] <= rows[2]["started_at"]


def test_a_failed_request_still_gets_a_duration(monkeypatch, tmp_path):
    monkeypatch.setattr(client_mod, "_client", _fake_client([_api_error(), _resp()]))
    _trace(tmp_path)
    with client_mod.web_run_scope():
        _chat()
    first, second = _attempts(tmp_path)
    assert first["status"] == "error" and "duration_s" in first and "prompt_tokens" not in first
    assert second["status"] == "ok" and second["prompt_tokens"] == 11


def test_odd_response_objects_add_nothing_wrong(monkeypatch, tmp_path):
    odd = _resp()
    odd.usage = types.SimpleNamespace(prompt_tokens=1.5, completion_tokens=None)
    odd.provider, odd.id = 5, ""
    monkeypatch.setattr(client_mod, "_client", _fake_client([odd]))
    _trace(tmp_path)
    with client_mod.web_run_scope():
        _chat()
    (row,) = _attempts(tmp_path)
    assert not {"prompt_tokens", "completion_tokens", "reasoning_tokens", "provider", "generation_id"} & set(row)


# -- nothing else changes ----------------------------------------------------------------
def test_a_batch_leg_trace_gets_exactly_the_old_row(monkeypatch, tmp_path):
    monkeypatch.setattr(client_mod, "_client", _fake_client([_resp()]))
    _trace(tmp_path)
    _chat()  # no web_run_scope: the batch / app path
    (row,) = _attempts(tmp_path)
    assert set(row) == {"seq", "kind", "elapsed_seconds", "stage", "attempt", "status", "model", "reason",
                        "finish_reason", "content_chars", "request_hash", "response_hash"}


def test_result_and_diagnostics_are_identical_with_and_without_the_scope(monkeypatch, tmp_path):
    replies = lambda: [_api_error(), _resp(""), _resp("final")]  # noqa: E731
    monkeypatch.setattr(client_mod, "_client", _fake_client(replies()))
    plain = _chat()
    monkeypatch.setattr(client_mod, "_client", _fake_client(replies()))
    _trace(tmp_path)
    with client_mod.web_run_scope():
        scoped = _chat()
    assert (scoped.text, scoped.diagnostics.to_dict()) == (plain.text, plain.diagnostics.to_dict())
    flat = str(scoped.diagnostics.to_dict())
    assert "duration_s" not in flat and "started_at" not in flat and "prompt_tokens" not in flat


def test_the_scope_is_off_by_default_and_restored(monkeypatch):
    assert client_mod._WEB_RUN is False
    with client_mod.web_run_scope():
        with client_mod.web_run_scope():
            assert client_mod._WEB_RUN is True
        assert client_mod._WEB_RUN is True
    assert client_mod._WEB_RUN is False
    with pytest.raises(ZeroDivisionError):
        with client_mod.web_run_scope():
            1 / 0
    assert client_mod._WEB_RUN is False


# -- the curator -------------------------------------------------------------------------
def _tool_turn() -> Any:
    call = types.SimpleNamespace(id="c1", function=types.SimpleNamespace(name="lookup", arguments="{}"))
    return _resp(None, finish="tool_calls", tool_calls=[call])


def _with_tools(**kw: Any) -> str:
    return client_mod.chat_with_tools(
        [{"role": "user", "content": "go"}], [{"type": "function", "function": {"name": "lookup"}}],
        lambda name, args: {"ok": True}, model_override="m", stage_name="curator", **kw)


def test_curator_rounds_are_visible_in_a_web_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(client_mod, "_client", _fake_client([_tool_turn(), _resp("curated")]))
    _trace(tmp_path)
    with client_mod.web_run_scope():
        assert _with_tools() == "curated"
    rows = _attempts(tmp_path)
    assert [r["stage"] for r in rows] == ["curator", "curator"]
    assert [r["finish_reason"] for r in rows] == ["tool_calls", "stop"]
    assert rows[0]["request_hash"] != rows[1]["request_hash"]  # the second round carries the tool result
    assert all({"started_at", "duration_s", "prompt_tokens", "provider"} <= set(r) for r in rows)


def test_curator_retries_and_empty_replies_are_visible(monkeypatch, tmp_path):
    monkeypatch.setattr(client_mod, "_client", _fake_client([_api_error(), _resp(""), _resp("done")]))
    _trace(tmp_path)
    with client_mod.web_run_scope():
        assert _with_tools() == "done"
    assert [r["status"] for r in _attempts(tmp_path)] == ["error", "empty", "ok"]


def test_the_curator_is_silent_outside_a_web_job(monkeypatch, tmp_path):
    monkeypatch.setattr(client_mod, "_client", _fake_client([_tool_turn(), _resp("curated")]))
    _trace(tmp_path)
    assert _with_tools() == "curated"
    assert _attempts(tmp_path) == []  # batch traces keep today's counts


# -- run_pathway -------------------------------------------------------------------------
def _job(tmp_path: Path, monkeypatch, replies: List[Any]) -> Any:
    monkeypatch.setattr(client_mod, "_client", _fake_client(replies))

    def drive(request, paper, outcome, state, reporter, deadline):
        _chat()  # a real client call, inside whatever scope run_pathway set
        outcome.status, outcome.stage = "fail", "stage1"

    monkeypatch.setattr(pw, "_drive", drive)
    return pw.run_pathway(PathwayRequest(texts=("The paper.",), out_dir=tmp_path / "out"))


def test_run_pathway_writes_a_trace_that_is_not_a_download(monkeypatch, tmp_path):
    result = _job(tmp_path, monkeypatch, [_resp()])
    out = tmp_path / "out"
    assert (out / leg_trace.LEG_TRACE_NAME).is_file()
    kinds = [e["kind"] for e in leg_trace.read_events(out)]
    assert kinds[0] == "job_begin" and kinds[-1] == "job_end" and "model_attempt" in kinds
    (row,) = _attempts(out)
    assert row["stage"] == "Stage 1 extraction" and "duration_s" in row and row["prompt_tokens"] == 11
    # not offered: downloads are looked up by key in report_paths / pwml_path / image_path
    delivered = {str(p) for p in [result.pwml_path, result.image_path, *result.report_paths.values()] if p}
    assert not any(leg_trace.LEG_TRACE_NAME in p for p in delivered)
    assert leg_trace.LEG_TRACE_NAME not in result.report_paths
    assert leg_trace.active() is None and client_mod._WEB_RUN is False  # nothing left switched on
    text = (out / leg_trace.LEG_TRACE_NAME).read_text(encoding="utf-8")
    assert "answer" not in text and "The paper." not in text  # no prompt or reply text


def test_run_pathway_leaves_a_batch_trace_that_is_already_active_alone(monkeypatch, tmp_path):
    batch = leg_trace.activate(tmp_path / "batch" / leg_trace.LEG_TRACE_NAME)
    _job(tmp_path, monkeypatch, [_resp()])
    assert leg_trace.active() is batch  # not replaced, not deactivated
    assert not (tmp_path / "out" / leg_trace.LEG_TRACE_NAME).exists()
    assert any(e["kind"] == "model_attempt" for e in leg_trace.read_events(tmp_path / "batch"))


def test_run_paper_alone_is_not_a_web_run(monkeypatch, tmp_path):
    """Batch calls ``run_paper`` directly: no trace of its own, no web scope."""
    seen = {}
    monkeypatch.setattr(pw, "_drive", lambda *a, **k: seen.update(web=client_mod._WEB_RUN, trace=leg_trace.active()))
    pw.run_paper({"paper_id": "p"}, PathwayRequest(texts=("x",)), None)
    assert seen == {"web": False, "trace": None}
