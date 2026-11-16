"""L9 (D-W24 item 8): the Stage-1 draws run concurrently, the merge does not change.

Offline: ``run_ensemble`` is driven with fake draw functions, no model, no network.
Finish order is forced with events (draw in finish position p waits for position p-1), so a
sequential implementation cannot complete: it times out instead of passing.
"""

from __future__ import annotations

import itertools
import json
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import t2pw.pipeline.stage_one_ensemble as ens  # noqa: E402
from t2pw.pipeline import extraction_diagnostics as diag  # noqa: E402
from t2pw.pipeline.extraction_ladder import current_leg_deadline  # noqa: E402
from t2pw.pipeline.pipeline import PipelineFailure  # noqa: E402

WAIT = 3.0  # a draw that waits longer than this for its turn means the draws are not concurrent
PAPER = "Glucose is phosphorylated by hexokinase to glucose 6-phosphate using ATP."

# On the base commit ``current_draw`` does not exist; draws then ran in call order, which IS
# the draw number, so a call counter stands in and the tests can be run against the base.
_calls = itertools.count(1)


def _slot() -> int:
    current = getattr(ens, "current_draw", None)
    return current() if current else next(_calls)


def payload(k: int) -> Dict[str, Any]:
    """Draw k: one reaction every draw states (different evidence each: first draw wins), one its own."""

    return {
        "entities": {"compounds": [{"name": f"compound {k}", "class": "compound"}]},
        "processes": {"reactions": [
            {"name": "shared step", "inputs": ["glucose", "ATP"], "outputs": ["G6P", "ADP"],
             "evidence": f"evidence from draw {k}"},
            {"name": f"step only in draw {k}", "inputs": [f"s{k}"], "outputs": [f"p{k}"],
             "evidence": f"only {k}"},
        ]},
    }


class Draws:
    """A draw function whose k-th draw finishes ``order.index(k)``-th and returns ``results[k]``."""

    def __init__(self, order: Sequence[int], results: Dict[int, Any]) -> None:
        self.order, self.results = list(order), results
        self.events = [threading.Event() for _ in order]
        self.finished: List[int] = []
        self.lock = threading.Lock()
        self.threads: List[int] = []

    def __call__(self) -> Any:
        k = _slot()
        pos = self.order.index(k)
        self.threads.append(threading.get_ident())
        if pos and not self.events[pos - 1].wait(WAIT):
            raise AssertionError(f"draw {k} never got its turn: the draws are not concurrent")
        with self.lock:
            self.finished.append(k)
        self.events[pos].set()
        out = self.results[k]
        if isinstance(out, BaseException):
            raise out
        return deepcopy(out), [{"attempt": 1, "raw": f"raw of draw {k}"}]


@pytest.fixture(autouse=True)
def fresh(tmp_path: Path) -> Any:
    global _calls
    _calls = itertools.count(1)
    recorder = diag.activate(run_id="l9", artifact_dir=tmp_path)
    yield recorder
    diag.deactivate()


def _report(tmp_path: Path) -> List[Dict[str, Any]]:
    return json.loads((tmp_path / ens.REPORT_NAME).read_text(encoding="utf-8"))


PERMS = list(itertools.permutations([1, 2, 3]))


def test_merge_and_report_do_not_depend_on_finish_order(tmp_path: Path) -> None:
    reference, _ = ens.merge_draws([payload(1), payload(2), payload(3)], paper_text=PAPER)
    seen = []
    for order in PERMS:
        _reset(tmp_path)
        draws = Draws(order, {k: payload(k) for k in (1, 2, 3)})
        merged, attempts = ens.run_ensemble(draws, 3, failure=PipelineFailure, paper_text=PAPER)
        assert draws.finished == list(order)  # the forced order really happened
        assert merged == reference
        reactions = merged["processes"]["reactions"]
        assert reactions[0]["evidence"] == "evidence from draw 1"  # draw order, not completion order
        assert [a["draw"] for a in attempts] == [1, 2, 3]
        assert [a["raw"] for a in attempts] == [f"raw of draw {k}" for k in (1, 2, 3)]
        seen.append(_report(tmp_path))
    assert all(r == seen[0] for r in seen)
    assert seen[0][0]["draws_succeeded"] == 3


def _reset(tmp_path: Path) -> None:
    global _calls
    _calls = itertools.count(1)
    for name in (ens.REPORT_NAME, Path(ens.REPORT_NAME).with_suffix(".txt").name):
        (tmp_path / name).unlink(missing_ok=True)
    diag.activate(run_id="l9", artifact_dir=tmp_path)


def test_draws_overlap_in_wall_time() -> None:
    def draw() -> Any:
        k = _slot()
        time.sleep(0.5)
        return payload(k), []

    started = time.monotonic()
    ens.run_ensemble(draw, 3, failure=PipelineFailure, paper_text=PAPER)
    assert time.monotonic() - started < 1.2  # ~max (0.5 s), not the sum (1.5 s)


def test_draw_threads_see_the_callers_context(fresh: Any) -> None:
    seen: List[Any] = []

    def draw() -> Any:
        seen.append((diag.current(), current_leg_deadline(), threading.current_thread().name))
        return payload(_slot()), []

    ens.run_ensemble(draw, 3, failure=PipelineFailure, paper_text=PAPER)
    assert [r for r, _, _ in seen] == [fresh] * 3
    assert all(name != threading.current_thread().name for _, _, name in seen)
    [boundary] = [b for b in fresh.boundaries if b["boundary"] == ens.BOUNDARY_STAGE1_ENSEMBLE]
    assert boundary["ensemble"]["draws_succeeded"] == 3


# ------------------------------------------------------------------ failures: as today


@pytest.mark.parametrize("order", PERMS)
def test_one_failed_draw_is_survived_whatever_finishes_first(order: Sequence[int], tmp_path: Path) -> None:
    failure = PipelineFailure("extraction", "draw 2 produced nothing", [{"attempt": 1, "raw": "", "error": "empty"}])
    draws = Draws(order, {1: payload(1), 2: failure, 3: payload(3)})
    merged, attempts = ens.run_ensemble(draws, 3, failure=PipelineFailure, paper_text=PAPER)
    reference, _ = ens.merge_draws([payload(1), None, payload(3)], paper_text=PAPER)
    assert merged == reference
    assert [a["draw"] for a in attempts] == [1, 2, 3]  # the failed draw's attempt log is kept, in place
    [report] = _report(tmp_path)
    assert report["draws_succeeded"] == 2 and report["draws_failed"] == 1
    assert report["failed_draws"] == [{"draw": 2, "stage": "extraction", "error": "draw 2 produced nothing"}]


@pytest.mark.parametrize("order", PERMS)
def test_every_draw_failing_raises_the_last_draws_failure(order: Sequence[int]) -> None:
    failures = {k: PipelineFailure("extraction", f"draw {k} failed", []) for k in (1, 2, 3)}
    with pytest.raises(PipelineFailure) as excinfo:
        ens.run_ensemble(Draws(order, failures), 3, failure=PipelineFailure, paper_text=PAPER)
    assert excinfo.value is failures[3]  # draw 3, not whichever finished last
    assert [(i, e) for i, e in excinfo.value.ensemble_failures] == [(1, failures[1]), (2, failures[2]), (3, failures[3])]


def test_an_unrelated_exception_propagates_untouched_lowest_draw_first() -> None:
    boom2, boom3 = RuntimeError("leg deadline"), ValueError("other")
    draws = Draws([3, 2, 1], {1: payload(1), 2: boom2, 3: boom3})
    with pytest.raises(RuntimeError) as excinfo:
        ens.run_ensemble(draws, 3, failure=PipelineFailure, paper_text=PAPER)
    assert excinfo.value is boom2
    assert sorted(draws.finished) == [1, 2, 3]  # no draw is left running behind the exception


# ------------------------------------------------------------------ N = 1


def test_one_draw_runs_inline_with_no_pool() -> None:
    caller = threading.get_ident()
    ran: List[Any] = []

    def draw() -> Any:
        ran.append((threading.get_ident(), getattr(ens, "current_draw", lambda: 0)()))
        return payload(1), [{"attempt": 1, "raw": "r"}]

    merged, attempts = ens.run_ensemble(draw, 1, failure=PipelineFailure, paper_text=PAPER)
    assert ran == [(caller, 0)]
    assert merged == ens.merge_draws([payload(1)], paper_text=PAPER)[0]
    assert attempts == [{"attempt": 1, "raw": "r", "draw": 1}]


def test_one_draw_failing_raises_it_unchanged() -> None:
    failure = PipelineFailure("extraction", "nothing", [])

    def draw() -> Any:
        raise failure

    with pytest.raises(PipelineFailure) as excinfo:
        ens.run_ensemble(draw, 1, failure=PipelineFailure)
    assert excinfo.value is failure and excinfo.value.ensemble_failures == [(1, failure)]
