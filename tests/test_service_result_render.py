"""VIS-6: the structure-fetch budget the service passes to the PathBank render (no network)."""
from __future__ import annotations

import time

import pytest

from t2pw.render import pathbank, structures
from t2pw.service import pathway


@pytest.fixture
def seen(monkeypatch, tmp_path):
    monkeypatch.setenv("T2PW_RESULT_RENDER", "pathbank")
    monkeypatch.setattr(structures, "default_budget", lambda: 6.0)
    calls = []
    monkeypatch.setattr(pathbank, "render_pwml", lambda p, *a, **k: calls.append(k) or b"PNG")
    return calls, tmp_path / "x.pwml"


def test_budget_is_default_without_deadline_then_shrinks_to_zero(seen):
    calls, p = seen
    pathway._pathbank_png(p)
    pathway._pathbank_png(p, time.time() + 600)
    pathway._pathbank_png(p, time.time() + 8)
    pathway._pathbank_png(p, time.time() + 2)
    b = [c["structure_budget_s"] for c in calls]
    assert b[0] is None and b[1] == 6.0 and 2.0 < b[2] <= 3.0 and b[3] == 0.0


def test_flag_off_renders_nothing(monkeypatch, seen):
    calls, p = seen
    monkeypatch.delenv("T2PW_RESULT_RENDER")
    assert pathway._pathbank_png(p) is None and not calls
