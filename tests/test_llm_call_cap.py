"""NEW TESTS (RT1, part 2): the 12-minute wall-clock cap on one LLM request in a website job.

Offline and fast: ``client._client`` is a fake whose calls can be held on an Event, the cap is
monkeypatched to a fraction of a second, retry sleeps are 0. What these pin: a request past the cap
is a failed attempt in the EXISTING loop (no new retries); its late answer changes nothing,
token totals included; only the measured stages are capped; and the cap exists only inside
``run_pathway``'s web scope, never for batch (``run_paper``) or the app.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable, List

import pytest

try:  # the real SDK when installed (see test_llm_call_logging.py)
    import openai  # noqa: F401
except ImportError:
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

CAP = 0.15
CAPPED_STAGES = ["preprocessor", "Stage 1 extraction", "Stage 2 inference", "rag_prose_extraction"]
UNCAPPED_STAGES = ["audit loop", "final_completeness_audit", "gap resolver", "curator", "chat", "overwatch",
                   "focused_repair", "Stage 1 extraction json repair", "Stage 2 inference row repair"]


def _resp(text: str = "answer", prompt: int = 11, completion: int = 22) -> Any:
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text, tool_calls=None),
                                       finish_reason="stop")],
        usage=types.SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                                    completion_tokens_details=None),
        provider="P", id="gen", model="m", model_extra={})


class FakeProvider:
    """``create`` calls answered by a list of callables; each sees the thread it ran on."""

    def __init__(self, *steps: Callable[[], Any]) -> None:
        self.steps, self.threads, self.lock = list(steps), [], threading.Lock()

    @property
    def calls(self) -> int:
        return len(self.threads)

    def create(self, **kwargs: Any) -> Any:
        with self.lock:
            self.threads.append(threading.current_thread())
            step = self.steps.pop(0)
        return step()

    @property
    def client(self) -> Any:
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create)))


def _ok(text: str = "answer", **kw: Any) -> Callable[[], Any]:
    return lambda: _resp(text, **kw)


def _held(gate: threading.Event, then: Any = None) -> Callable[[], Any]:
    """A request that hangs until ``gate`` is set, then answers (its late reply)."""
    def step() -> Any:
        gate.wait(20)
        return then if then is not None else _resp("LATE", prompt=1000, completion=2000)
    return step


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    for name in ("RateLimitError", "APIError", "APITimeoutError", "AuthenticationError", "BadRequestError"):
        monkeypatch.setattr(client_mod, name, type(name, (Exception,), {}))
    monkeypatch.setenv("LLM_CALL_SPACING", "0")
    monkeypatch.setenv("LLM_RETRY_BASE_SLEEP", "0")
    monkeypatch.setenv("LLM_RETRY_MAX_SLEEP", "0")
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    monkeypatch.setattr(client_mod, "LLM_CALL_CAP_SECONDS", CAP)
    client_mod.reset_token_stats()
    yield
    leg_trace.deactivate()
    client_mod.reset_token_stats()


def _use(monkeypatch, provider: FakeProvider) -> FakeProvider:
    monkeypatch.setattr(client_mod, "_client", provider.client)
    return provider


def _chat(stage: str = "Stage 1 extraction"):
    return client_mod.chat_detailed([{"role": "user", "content": "hi"}], model_override="m", stage_name=stage)


def _workers() -> List[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "t2pw-llm-call"]


def _join_workers() -> None:
    for t in _workers():
        t.join(10)
    assert not _workers()


def _rows(path: Path) -> List[dict]:
    return [e for e in leg_trace.read_events(path) if e["kind"] == "model_attempt"]


# -- the cap is a failed attempt in the existing loop ---------------------------------
def test_a_request_past_the_cap_is_a_failed_attempt_and_the_existing_retry_follows(monkeypatch, tmp_path):
    gate = threading.Event()
    provider = _use(monkeypatch, FakeProvider(_held(gate), _ok("second try")))
    leg_trace.activate(tmp_path / leg_trace.LEG_TRACE_NAME)
    started = time.monotonic()
    with client_mod.web_run_scope():
        result = _chat()
    assert time.monotonic() - started < 5  # waited the cap, not the hold
    gate.set(); _join_workers()
    assert result.text == "second try" and provider.calls == 2  # one retry, no more
    assert [r["status"] for r in result.diagnostics.attempt_log] == ["call_cap", "ok"]
    assert result.diagnostics.attempts == 2 and result.diagnostics.terminal_reason == ""
    first, second = _rows(tmp_path)
    assert (first["status"], first["attempt"], second["status"], second["attempt"]) == ("call_cap", 1, "ok", 2)
    assert "wall-clock cap" in first["reason"] and first["duration_s"] >= CAP * 0.9


def test_every_attempt_capped_ends_as_the_distinct_operation_timeout(monkeypatch):
    gate = threading.Event()
    provider = _use(monkeypatch, FakeProvider(*(_held(gate) for _ in range(3))))
    with client_mod.web_run_scope(), pytest.raises(client_mod.LLMOperationTimeout) as raised:
        _chat()
    gate.set(); _join_workers()
    assert provider.calls == 3  # LLM_MAX_RETRIES, as before
    assert raised.value.terminal_reason == "operation_timeout"
    assert raised.value.diagnostics.terminal_reason == "operation_timeout"
    assert f"{CAP}s" in str(raised.value)


def test_an_error_inside_the_worker_reaches_the_normal_retry_path(monkeypatch):
    provider = _use(monkeypatch, FakeProvider(lambda: (_ for _ in ()).throw(client_mod.APIError("boom")), _ok()))
    with client_mod.web_run_scope():
        result = _chat()
    assert [r["status"] for r in result.diagnostics.attempt_log] == ["error", "ok"] and provider.calls == 2
    assert provider.threads[0] is not threading.current_thread()


# -- the late result changes nothing --------------------------------------------------
def test_a_late_result_is_discarded_and_touches_no_totals_or_records(monkeypatch, tmp_path):
    gate = threading.Event()
    provider = _use(monkeypatch, FakeProvider(_held(gate), _ok("kept", prompt=11, completion=22)))
    leg_trace.activate(tmp_path / leg_trace.LEG_TRACE_NAME)
    with client_mod.web_run_scope():
        result = _chat()
    before = (client_mod.get_token_stats(), result.diagnostics.to_dict(), len(leg_trace.read_events(tmp_path)),
              dict(client_mod._token_stats))
    gate.set(); _join_workers()  # the abandoned request now answers, with 1000 / 2000 tokens
    after = (client_mod.get_token_stats(), result.diagnostics.to_dict(), len(leg_trace.read_events(tmp_path)),
             dict(client_mod._token_stats))
    assert before == after
    stats = client_mod.get_token_stats()
    assert (stats["prompt_tokens"], stats["completion_tokens"], stats["api_calls"]) == (11, 22, 1)
    assert result.text == "kept" and "LATE" not in str(result.diagnostics.to_dict())
    assert provider.calls == 2


def test_the_worker_is_a_daemon_so_a_stuck_request_cannot_hold_the_process(monkeypatch):
    gate = threading.Event()
    _use(monkeypatch, FakeProvider(_held(gate), _ok()))
    with client_mod.web_run_scope():
        _chat()
    (worker,) = _workers()
    assert worker.daemon is True and worker.is_alive()
    gate.set(); _join_workers()


# -- only the measured stages ---------------------------------------------------------
def test_the_allowlist_is_exactly_the_four_measured_stages():
    assert client_mod.LLM_CALL_CAP_STAGES == frozenset(CAPPED_STAGES)


@pytest.mark.parametrize("stage", CAPPED_STAGES)
def test_an_allowlisted_stage_is_capped(monkeypatch, stage):
    gate = threading.Event()
    provider = _use(monkeypatch, FakeProvider(_held(gate), _ok()))
    with client_mod.web_run_scope():
        result = _chat(stage)
    gate.set(); _join_workers()
    assert [r["status"] for r in result.diagnostics.attempt_log] == ["call_cap", "ok"] and provider.calls == 2


@pytest.mark.parametrize("stage", UNCAPPED_STAGES)
def test_every_other_stage_runs_on_the_callers_thread_with_no_cap(monkeypatch, stage):
    slow_then_ok = lambda: (time.sleep(CAP * 3), _resp("slow but fine"))[1]  # noqa: E731 - well past the cap
    provider = _use(monkeypatch, FakeProvider(slow_then_ok))
    with client_mod.web_run_scope():
        result = _chat(stage)
    assert result.text == "slow but fine" and provider.calls == 1
    assert provider.threads == [threading.current_thread()] and not _workers()
    assert [r["status"] for r in result.diagnostics.attempt_log] == ["ok"]


def test_the_curator_loop_is_never_capped_even_when_slow(monkeypatch):
    slow = lambda: (time.sleep(CAP * 3), _resp("curated"))[1]  # noqa: E731
    provider = _use(monkeypatch, FakeProvider(slow))
    with client_mod.web_run_scope():
        text = client_mod.chat_with_tools([{"role": "user", "content": "go"}], [], lambda n, a: {},
                                          model_override="m", stage_name="Stage 1 extraction")
    assert text == "curated" and provider.threads == [threading.current_thread()]


# -- the cap exists only in a website job ---------------------------------------------
def test_without_the_web_scope_a_slow_allowlisted_call_is_never_capped(monkeypatch):
    slow = lambda: (time.sleep(CAP * 3), _resp("app or batch"))[1]  # noqa: E731
    provider = _use(monkeypatch, FakeProvider(slow))
    result = _chat("Stage 1 extraction")  # the Streamlit app and batch legs call it exactly like this
    assert result.text == "app or batch" and provider.calls == 1
    assert provider.threads == [threading.current_thread()] and not _workers()


def _slow_stage1_drive(provider: FakeProvider, seen: dict) -> Callable[..., None]:
    def drive(request, paper, outcome, state, reporter, deadline):
        seen["web"] = client_mod._WEB_RUN
        seen["result"] = _chat("Stage 1 extraction")
        outcome.status, outcome.stage = "fail", "stage1"
    return drive


def test_run_pathway_caps_but_run_paper_does_not(monkeypatch, tmp_path):
    # run_pathway: the web entry
    gate = threading.Event()
    web = _use(monkeypatch, FakeProvider(_held(gate), _ok("web")))
    seen: dict = {}
    monkeypatch.setattr(pw, "_drive", _slow_stage1_drive(web, seen))
    pw.run_pathway(PathwayRequest(texts=("The paper.",), out_dir=tmp_path / "out"))
    gate.set(); _join_workers()
    assert seen["web"] is True and web.calls == 2
    assert [r["status"] for r in seen["result"].diagnostics.attempt_log] == ["call_cap", "ok"]
    # run_paper: what batch calls
    slow = lambda: (time.sleep(CAP * 3), _resp("batch"))[1]  # noqa: E731
    batch = _use(monkeypatch, FakeProvider(slow))
    seen = {}
    monkeypatch.setattr(pw, "_drive", _slow_stage1_drive(batch, seen))
    pw.run_paper({"paper_id": "p"}, PathwayRequest(texts=("x",)), None)
    assert seen["web"] is False and batch.calls == 1 and seen["result"].text == "batch"
    assert batch.threads == [threading.current_thread()]


def test_the_scope_is_off_after_run_pathway_even_when_the_run_crashes(monkeypatch, tmp_path):
    monkeypatch.setattr(pw, "_drive", lambda *a, **k: 1 / 0)
    result = pw.run_pathway(PathwayRequest(texts=("The paper.",), out_dir=tmp_path / "out"))
    assert result.status == "error" and client_mod._WEB_RUN is False and leg_trace.active() is None


# -- a normal call is untouched -------------------------------------------------------
@pytest.mark.parametrize("stage", CAPPED_STAGES)
def test_a_call_that_finishes_normally_is_identical_with_the_cap_on_or_off(monkeypatch, stage):
    _use(monkeypatch, FakeProvider(_ok("same"), _ok("same")))
    plain = _chat(stage)
    with client_mod.web_run_scope():
        capped = _chat(stage)
    assert (capped.text, capped.diagnostics.to_dict()) == (plain.text, plain.diagnostics.to_dict())
    stats = client_mod.get_token_stats()
    assert (stats["prompt_tokens"], stats["completion_tokens"], stats["api_calls"]) == (22, 44, 2)
    _join_workers()


def test_the_cap_is_one_named_constant_above_the_longest_good_call_measured():
    lines = Path(client_mod.__file__).read_text(encoding="utf-8").splitlines()
    assert lines.count("LLM_CALL_CAP_SECONDS = 720.0") == 1  # the PO's 12 minutes, written once
    assert 720.0 > 11.97 * 60  # the longest good Stage-1 call measured
