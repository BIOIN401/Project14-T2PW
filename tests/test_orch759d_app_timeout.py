"""ORCH-759d -- **G9 BASE-FAILURE PROOF. A REGRESSION FIX, NOT NEW WORK.**

THE DEFECT. ``driver.run_one`` defaulted ``app_timeout`` to ``DEFAULT_APP_TIMEOUT``
(3600 s) and ``_Budget.slice()`` caps every app interaction at
``min(app_timeout, remaining)``. No production caller passes ``app_timeout``
(``runner.run_single`` calls ``run_one(paper, mode, timeout=...)`` and nothing
else), so a leg approved for 10,800 s still lost any single interaction -- e.g. the
Stage 0-2 submit -- at 3,600 s. Measured: ORCH-759 Wnt/Ca2+ strict,
``runs_papers/ORCH-759/legs/wnt_ca-strict/2026-10-22_1528/papers/PMC10425814/strict``,
``extraction did not finish inside the time budget`` at 5,296 s of a 10,800 s leg.

HOW. A fixture app speaks the real app's contract (``tests/test_batch_driver.py``
harness). Wall time is faked: ``driver.time`` is replaced by a fake clock, and
``AppTest.run`` is wrapped so each interaction runs the (instant) fixture script and
then *costs* ``fake_cost`` fake seconds, which the script sets in session_state. An
interaction whose cost exceeds the timeout the driver asked for behaves the way
AppTest does: the clock advances by the timeout and ``RuntimeError('... timed out
...')`` is raised. This models a COOPERATIVE interaction only; a blocked syscall is
bounded by the parent's process-tree kill, not by this budget (see
``docs/batch_runner.md`` section 7).

Every name used here exists at base ``5d1ea64c`` with the same signature, so this
module imports cleanly there and fails on VALUES: ``'timeout' != 'pass'``.

No pipeline, no LLM, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.batch import driver, leg_trace, runner  # noqa: E402
from t2pw.batch.driver import MODE_STRICT, STRICT, run_one  # noqa: E402
from t2pw.pipeline.deadline import child_deadline_seconds  # noqa: E402
from test_batch_driver import (  # noqa: E402
    PAPER,
    _RELEASE_READY,
    _artifacts,
    _post_pipeline_body,
    _write_app,
    real_streamlit,  # noqa: F401 -- autouse fixture, re-exported on purpose
)
from test_batch_run import _make_run_dir  # noqa: E402

#: The ORCH-756 / ORCH-759 operating ruling: 10,800 s per paper leg.
LEG_TIMEOUT = 10_800.0
#: What the parent hands the child (``runner.child_command``): leg minus the grace.
CHILD_DEADLINE = child_deadline_seconds(LEG_TIMEOUT, grace=runner._CHILD_GRACE)

_PWML = {
    "ok": True,
    "_xml": "<pathway><name>menaquinone</name></pathway>",
    "pwml_ir": {"pathway": {"name": "menaquinone"}},
    "pwml_ir_report": {"counts": {"reactions": 1}},
    "validation_report": {"issues": 0},
    "qa": {"ok": True},
    "counts": {"reactions": 1, "compounds": 2},
    "output_path": "outputs/pathway.pwml",
    "quarantine_report": {"release": _RELEASE_READY},
}


def _app(tmp_path: Path, submit_cost: float) -> Path:
    """The strict-pass fixture, whose submit interaction costs ``submit_cost`` s."""

    body = (
        f'\nst.session_state["fake_cost"] = {submit_cost!r} if submitted else 1.0\n'
        + _post_pipeline_body(_artifacts("pathwhiz"), pwml=_PWML)
    )
    return _write_app(tmp_path, "slow_submit", body)


def _fake_wall_time(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, float]]:
    """Fake the driver's clock and AppTest's interaction cost. Returns the call log."""

    from streamlit.testing.v1 import AppTest

    clock = SimpleNamespace(now=0.0)
    calls: List[Dict[str, float]] = []
    real_run = AppTest.run

    def fake_run(self: Any, *, timeout: Any = None) -> Any:
        asked = float(self.default_timeout if timeout is None else timeout)
        start = clock.now
        real_run(self, timeout=60.0)  # the fixture script itself is instant
        cost = float(driver._ss(self, "fake_cost", 1.0))
        calls.append({"start": start, "timeout": asked, "cost": cost})
        if cost > asked:
            clock.now += asked
            raise RuntimeError(f"AppTest script run timed out after {asked}(s)")
        clock.now += cost
        return self

    monkeypatch.setattr(AppTest, "run", fake_run)
    monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return calls


def _leg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cost: float, deadline: float):
    """One strict leg through the PRODUCTION call path: ``runner.run_single``."""

    calls = _fake_wall_time(monkeypatch)
    monkeypatch.setattr(driver, "DEFAULT_APP_PATH", _app(tmp_path, cost))
    run_dir = _make_run_dir(tmp_path, slugs=("PMC1__alpha",), modes=(MODE_STRICT,))
    row = runner.run_single(run_dir, "PMC1__alpha", MODE_STRICT, timeout=deadline)
    return row, calls


def test_g9_a_4000s_submit_finishes_inside_a_10800s_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**BASE: ``timeout`` at 3,600 s, "extraction did not finish". TIP: ``pass``.**"""

    row, calls = _leg(tmp_path, monkeypatch, cost=4000.0, deadline=CHILD_DEADLINE)

    assert row["status"] == "pass", (row["status"], row["message"], calls)
    assert row["stage"] == "pwml_export"
    submit = next(call for call in calls if call["cost"] == 4000.0)
    # The submit was offered everything the leg had left -- not a fixed hour.
    assert submit["timeout"] == pytest.approx(CHILD_DEADLINE - submit["start"])
    # ...and no interaction was ever offered time past the child's own deadline.
    assert all(c["start"] + c["timeout"] <= CHILD_DEADLINE + 1e-6 for c in calls), calls


@pytest.mark.parametrize(
    ("deadline", "cost"),
    [
        (3_000.0, 4_000.0),  # leg shorter than the old cap: identical at base and tip
        (CHILD_DEADLINE, 20_000.0),  # the 10,800 s leg: the uncapped slice still stops
    ],
)
def test_preservation_the_leg_deadline_still_caps_an_interaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deadline: float, cost: float
) -> None:
    """An interaction that outruns the leg times out AT OR BEFORE the child deadline,
    so the parent's finalization reserve (the 120 s grace) is never touched."""

    row, calls = _leg(tmp_path, monkeypatch, cost=cost, deadline=deadline)

    assert row["status"] == "timeout"
    assert row["message"] == "extraction did not finish inside the time budget"
    assert row["termination_reason"] == "operation_timeout"
    assert all(c["start"] + c["timeout"] <= deadline + 1e-6 for c in calls), calls
    assert row["seconds"] <= deadline
    reserve = leg_trace.finalization_reserve_record(
        elapsed_seconds=row["seconds"],
        leg_timeout_seconds=deadline + runner._CHILD_GRACE,
        grace_seconds=runner._CHILD_GRACE,
    )
    assert reserve["used_seconds"] == 0.0 and not reserve["exhausted"], reserve
    if deadline < 3600.0:
        assert row["seconds"] == pytest.approx(deadline)


def test_preservation_an_explicit_app_timeout_still_caps_each_interaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller that passes ``app_timeout`` keeps exactly the cap it asked for."""

    calls = _fake_wall_time(monkeypatch)
    app = _app(tmp_path, 4000.0)

    outcome = run_one(PAPER, STRICT, app_path=app, timeout=CHILD_DEADLINE, app_timeout=3600.0)

    assert outcome.status == "timeout"
    assert outcome.message == "extraction did not finish inside the time budget"
    assert [c["timeout"] for c in calls] == [3600.0, 3600.0, 3600.0]
