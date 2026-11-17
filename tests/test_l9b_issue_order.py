"""L9b: request i of draw k is issued after request i of draw k-1, whatever the thread timing.

``tests/test_equivalence_run_pathway.py`` replays three byte-identical Stage-1 requests (one cassette key,
three recorded answers served in the order the requests ARRIVE). With sequential draws the k-th request was
draw k's by construction; concurrent threads that get going in another order swapped two answers, the
first-occurrence-wins merge changed, and Stage 2 asked for a request the cassette never held (about 1 run
in 3). Offline, no model: the draws here go through the same gate the real client uses, and "arrival" is
recorded inside the request, as the cassette lookup is.
"""

from __future__ import annotations

import itertools
import random
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import t2pw.pipeline.stage_one_ensemble as ens  # noqa: E402
from t2pw.llm import client as client_mod  # noqa: E402
from t2pw.pipeline import extraction_diagnostics as diag  # noqa: E402
from t2pw.pipeline.pipeline import PipelineFailure  # noqa: E402

TIMEOUT = 5.0


@pytest.fixture(autouse=True)
def recorder(tmp_path: Path) -> Any:
    yield diag.activate(run_id="l9b", artifact_dir=tmp_path)
    diag.deactivate()


def _payload(k: int) -> Dict[str, Any]:
    return {"entities": {"compounds": [{"name": f"c{k}", "class": "compound"}]},
            "processes": {"reactions": [{"name": "shared", "inputs": ["a"], "outputs": ["b"], "evidence": f"draw {k}"}]}}


class Draws:
    """Each draw waits ``start[k]`` seconds (a late thread start), then issues ``requests`` requests.

    A request takes a random few milliseconds to reach its lookup (the cassette hashes the whole prompt
    first), records its arrival, then lasts ``length`` seconds.
    """

    def __init__(self, start: Dict[int, float], requests: int = 1, length: float = 0.0, fail_before_request: Any = ()) -> None:
        self.start, self.requests, self.length = start, requests, length
        self.fail_before_request = set(fail_before_request)
        self.arrivals: List[tuple] = []
        self.lock = threading.Lock()

    def __call__(self) -> Any:
        k = ens.current_draw()
        time.sleep(self.start[k])
        if k in self.fail_before_request:
            raise PipelineFailure("extraction", f"draw {k} failed before any request", [])
        for i in range(1, self.requests + 1):
            gate = client_mod.REQUEST_GATE.get()
            with gate():
                time.sleep(random.random() * 0.01)
                with self.lock:
                    self.arrivals.append((i, k))
                time.sleep(self.length)
        return _payload(k), []


@pytest.mark.parametrize("order", list(itertools.permutations([1, 2, 3])))
def test_first_requests_arrive_in_draw_order_and_the_merge_is_the_same(order: Any) -> None:
    # ``order`` is the order the threads get going in: the first of them starts at once, the rest later.
    draws = Draws({k: 0.1 * order.index(k) for k in (1, 2, 3)})
    merged, _ = ens.run_ensemble(draws, 3, failure=PipelineFailure, paper_text="a b")
    assert draws.arrivals == [(1, 1), (1, 2), (1, 3)]
    reference, _ = ens.merge_draws([_payload(1), _payload(2), _payload(3)], paper_text="a b")
    assert merged == reference
    assert merged["processes"]["reactions"][0]["evidence"] == "draw 1"


def test_draws_released_together_still_arrive_in_order() -> None:
    """The first design let a draw in as soon as the one before it ENTERED; draws waiting together were
    then released in a burst and raced to the lookup. 40 bursts, each draw's lookup taking random time."""

    for _ in range(40):
        draws = Draws({1: 0.05, 2: 0.0, 3: 0.0})  # 2 and 3 are waiting at the gate when 1 gets there
        ens.run_ensemble(draws, 3, failure=PipelineFailure, paper_text="a b")
        assert draws.arrivals == [(1, 1), (1, 2), (1, 3)]


def test_later_requests_are_ordered_the_same_way() -> None:
    draws = Draws({1: 0.3, 2: 0.0, 3: 0.1}, requests=3)
    ens.run_ensemble(draws, 3, failure=PipelineFailure, paper_text="a b")
    for i in (1, 2, 3):
        assert [k for (n, k) in draws.arrivals if n == i] == [1, 2, 3], draws.arrivals


def test_the_draws_still_overlap_while_their_requests_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ens, "ISSUE_INFLIGHT_WAIT_SECONDS", 0.05)
    started = time.monotonic()
    ens.run_ensemble(Draws({1: 0.0, 2: 0.0, 3: 0.0}, length=0.5), 3, failure=PipelineFailure, paper_text="a b")
    assert time.monotonic() - started < 0.9  # serial draws would take 1.5 s


@pytest.mark.parametrize("failing", [(1,), (2,), (1, 2), (1, 2, 3)])
def test_a_draw_that_fails_before_its_first_request_frees_the_next(failing: Any) -> None:
    done: Dict[str, Any] = {}

    def run() -> None:
        try:
            done["result"] = ens.run_ensemble(Draws({1: 0.2, 2: 0.0, 3: 0.0}, fail_before_request=failing), 3,
                                              failure=PipelineFailure, paper_text="a b")
        except PipelineFailure as exc:
            done["failure"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(TIMEOUT)
    assert not thread.is_alive(), "a failed draw left the next one waiting"
    assert ("failure" in done) == (len(failing) == 3)


def test_a_draw_that_returns_without_a_request_frees_the_next() -> None:
    def draw() -> Any:
        k = ens.current_draw()
        if k == 1:
            return _payload(1), []
        with client_mod.REQUEST_GATE.get()():
            pass
        return _payload(k), []

    started = time.monotonic()
    ens.run_ensemble(draw, 3, failure=PipelineFailure, paper_text="a b")
    assert time.monotonic() - started < TIMEOUT


def test_one_draw_has_no_gate_at_all() -> None:
    seen: List[Any] = []

    def draw() -> Any:
        seen.append(client_mod.REQUEST_GATE.get())
        return _payload(1), []

    ens.run_ensemble(draw, 1, failure=PipelineFailure, paper_text="a b")
    assert seen == [None]


def test_the_client_wraps_each_request_in_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    events: List[str] = []
    reply = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="{}", tool_calls=None), finish_reason="stop")],
        usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1), id="g", model="m", model_extra={})

    def create(**_: Any) -> Any:
        events.append("request")
        return reply

    class Gate:
        def __enter__(self) -> None:
            events.append("enter")

        def __exit__(self, *_: Any) -> None:
            events.append("exit")

    monkeypatch.setattr(client_mod, "_client", types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))))
    monkeypatch.setenv("LLM_CALL_SPACING", "0")
    token = client_mod.REQUEST_GATE.set(Gate)
    try:
        client_mod.chat_detailed([{"role": "user", "content": "hi"}], model_override="m", stage_name="t")
    finally:
        client_mod.REQUEST_GATE.reset(token)
    assert events == ["enter", "request", "exit"]
