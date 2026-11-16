"""L9: state that concurrent Stage-1 draws share is safe, and 429 retries do not run in lockstep.

Offline. Each test hammers one shared object from several threads (with a tiny thread switch
interval) and checks nothing was lost or torn; the jitter test checks the 429 backoff stays
inside its old ceiling.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from t2pw.llm import client as client_mod  # noqa: E402
from t2pw.pipeline.deadline import LegDeadline  # noqa: E402
from t2pw.pipeline.extraction_diagnostics import BOUNDARY_REPORT_NAME, ExtractionDiagnostics  # noqa: E402

THREADS = 8


@pytest.fixture(autouse=True)
def tight_switching() -> Any:
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(previous)


def _in_threads(work: Callable[[], None]) -> None:
    errors: List[BaseException] = []

    def run() -> None:
        try:
            work()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def test_token_counters_lose_no_updates() -> None:
    before = dict(client_mod._token_stats)
    resp = types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=5))
    _in_threads(lambda: [client_mod._record_usage(resp) for _ in range(5000)])
    calls = THREADS * 5000
    assert client_mod._token_stats["calls"] - before["calls"] == calls
    assert client_mod._token_stats["prompt_tokens"] - before["prompt_tokens"] == 3 * calls
    assert client_mod._token_stats["completion_tokens"] - before["completion_tokens"] == 5 * calls


def test_checkpoint_sequence_numbers_are_unique() -> None:
    deadline = LegDeadline(1000.0, reserve_seconds=0.0)
    seen: List[int] = []
    _in_threads(lambda: seen.extend(deadline.checkpoint("x").sequence for _ in range(500)))
    assert sorted(seen) == list(range(1, THREADS * 500 + 1))


def test_concurrent_boundary_writes_leave_a_whole_report(tmp_path: Path) -> None:
    recorder = ExtractionDiagnostics(run_id="l9", artifact_dir=tmp_path)
    _in_threads(lambda: [recorder.record_boundary(stage="extraction", boundary="b", attempts=i) for i in range(20)])
    report = json.loads((tmp_path / BOUNDARY_REPORT_NAME).read_text(encoding="utf-8"))
    assert len(report["boundaries"]) == THREADS * 20
    assert recorder.write_errors == []


def test_429_backoff_is_jittered_and_never_above_the_old_value(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RateLimitError", "APIError", "APITimeoutError", "AuthenticationError", "BadRequestError"):
        monkeypatch.setattr(client_mod, name, type(name, (Exception,), {}))
    monkeypatch.setenv("LLM_CALL_SPACING", "0")
    monkeypatch.setenv("LLM_RETRY_BASE_SLEEP", "8")
    monkeypatch.setenv("LLM_RETRY_MAX_SLEEP", "20")
    slept: List[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    reply = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok", tool_calls=None),
                                       finish_reason="stop")],
        usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1), id="g", model="m", model_extra={})

    for _ in range(25):
        queue: List[Any] = [client_mod.RateLimitError("429"), reply]

        def create(**_: Any) -> Any:
            item = queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        monkeypatch.setattr(client_mod, "_client", types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))))
        client_mod.chat_detailed([{"role": "user", "content": "hi"}], model_override="m", stage_name="t")

    backoffs = [s for s in slept if s > 0]
    assert len(backoffs) == 25
    assert all(4.0 <= s <= 8.0 for s in backoffs)  # first retry: 8 s ceiling, jittered down to 4 s
    assert len(set(backoffs)) > 1
