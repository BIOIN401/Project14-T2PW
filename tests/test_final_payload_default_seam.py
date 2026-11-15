"""Fix round 3 (review 3, R3-1 / R3-2 / R3-3 / R3-8): the DEFAULT (flag on) post-pipeline hook with a dead network, a job deadline and
the budget switch. The suite pins T2PW_FINAL_PAYLOAD=0; every test here sets it to 1 itself. The network is never touched: the live
fetcher's inner ``UrllibFetcher`` is replaced, so the real ``_live_fetcher`` / ``_Budget`` / deadline cap run, exactly as ``postpipeline`` calls
``prefreeze_with_final_fill(payload, resolve, pathway_context, state)`` (no ``fetcher=``, no ``cache=``)."""

from __future__ import annotations

import copy
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import t2pw.service.final_payload as fp  # noqa: E402
from t2pw.clean_model.fill.http import FetchError  # noqa: E402
from t2pw.pwml.compound_resolution import NO_DB_RESOLVER  # noqa: E402
from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution  # noqa: E402
from t2pw.service.final_payload import FINAL_PAYLOAD_FILL_FIELD, FINAL_PAYLOAD_PATHWAY_FIELD, prefreeze_with_final_fill  # noqa: E402


class _Inner:
    """A stand-in for ``UrllibFetcher``: counts calls, then raises FetchError (``mode="dead"``) or answers nothing (``mode="empty"``)."""

    timeout, retries = 30.0, 3

    def __init__(self, mode: str) -> None:
        self.mode, self.calls, self.urls = mode, 0, []

    def _go(self, url, params=None):
        self.calls += 1
        self.urls.append(url)
        if self.mode == "dead":
            raise FetchError("network unreachable (test)")
        return None

    get_json = get_text = _go


@pytest.fixture
def net(monkeypatch):
    """Flag on, memory-only cache, and a swappable inner fetcher behind the real ``_live_fetcher``."""
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD_BUDGET_S", raising=False)
    monkeypatch.setattr("t2pw.clean_model.fill.cache.Cache._path", lambda self, key: None)  # no read of clean_model_cache/
    box = {}

    def install(mode):
        box["inner"] = inner = _Inner(mode)
        monkeypatch.setattr("t2pw.clean_model.fill.http.UrllibFetcher", lambda throttle=None: inner)
        return inner

    return install


def _payload():
    names = ["chorismate", "ATP", "zinc", "theobromine", "caffeine", "glucose", "pyruvate", "NADH"]
    return {
        "metadata": {"name": "x", "pathway_name": "x", "subject": "Metabolic", "pathway_subject": "Metabolic"},
        "entities": {"species": [{"name": "Escherichia coli", "pathwhiz_id": 1}],
                     "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
                     "compounds": [{"name": n, "pathbank_compound_id": 100 + i} for i, n in enumerate(names)],
                     "proteins": [], "protein_complexes": []},
        "biological_states": [], "processes": {"reactions": [], "transports": [], "interactions": []},
    }


def _resolve(p):
    return run_prefreeze_resolution(p, strict_db=False, db_resolver=NO_DB_RESOLVER)


def _hook(p, state=None):
    """What ``postpipeline`` does at the seam: the default fetcher, the default cache."""
    return prefreeze_with_final_fill(p, _resolve, {"likely_organism": "Escherichia coli"}, {} if state is None else state)


def _flag_off_payload(monkeypatch):
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "0")
    p = _payload()
    res, rep = _hook(p)
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    assert rep == {}
    return p, res


# ---- R3-2 (a): dead network, default hook -------------------------------------------------------------------------------------
def test_dead_network_fails_open_and_the_circuit_opens(net, monkeypatch, tmp_path):
    from t2pw.service.postpipeline import freeze_canonical_payload
    off, off_res = _flag_off_payload(monkeypatch)
    inner = net("dead")
    p = _payload()
    res, rep = _hook(p)
    assert res == off_res and p == off  # the payload is exactly what the flag-off run produced
    assert FINAL_PAYLOAD_FILL_FIELD not in p and FINAL_PAYLOAD_PATHWAY_FIELD not in p
    assert rep["enabled"] and rep["written"] == 0 and rep["error"] is None  # the fill failed open: every request was a FetchError
    assert rep["circuit_open"] is True and rep["budget_exhausted"] is True
    assert rep["nucleic_types"]["circuit_open"] is True and rep["pathway_meta"]["circuit_open"] is False  # the title has its own
    per_host = Counter(urlsplit(u).netloc for u in inner.urls)  # D-W23 c: the circuit is per host, so a dead network costs 5 per host
    assert max(per_host.values()) == fp.CIRCUIT_BREAKER_N  # not one request per entity
    assert len(per_host) <= 3  # the fill reaches 2 hosts here (6 requests): the cost stays capped at N x hosts

    def freeze(payload):
        return freeze_canonical_payload(payload, strict_db=False, outputs_dir=tmp_path, pathway_context={}, export_mode="pathwhiz",
                                        research_mode=False, tmp=tmp_path, state={})
    assert freeze(p)["quarantine_ok"] == freeze(off)["quarantine_ok"]  # and the run goes on through the real freeze


def test_title_step_with_a_dead_network_falls_back_to_the_given_title(net):
    inner = net("dead")
    p = _payload()
    rep = fp.maybe_apply_final_pathway_meta(p, {"likely_organism": "Escherichia coli"}, {"final_payload_paper": {"pmid": "555", "title": "T"}})
    assert rep["written"] is True and rep["title_source"] == "paper metadata"  # falls back to the given title, nothing invented
    # the title needs only a couple of requests, so its own breaker (never the id fill's) stays closed: it just fell back
    assert 1 <= inner.calls < fp.CIRCUIT_BREAKER_N and rep["circuit_open"] is False and rep["budget_exhausted"] is False


# ---- R3-2 (b): a fetcher that finds nothing is fast ---------------------------------------------------------------------------
def test_empty_answers_return_quickly_and_leave_the_payload(net, monkeypatch):
    off, off_res = _flag_off_payload(monkeypatch)
    inner = net("empty")
    p = _payload()
    t0 = time.perf_counter()
    res, rep = _hook(p)
    assert time.perf_counter() - t0 < 30.0
    assert res == off_res and p == off and rep["written"] == 0 and rep["error"] is None
    assert rep["circuit_open"] is False and rep["budget_exhausted"] is False and inner.calls > 0  # no answer is not an error


# ---- R3-2 (c) / R3-1: the job deadline --------------------------------------------------------------------------------------
def test_deadline_nearly_spent_means_no_network_call(net, monkeypatch):
    off, _ = _flag_off_payload(monkeypatch)
    inner = net("dead")
    p = _payload()
    state = {fp.DEADLINE_STATE_KEY: time.time() + fp.DEADLINE_MARGIN_S - 5, "final_payload_paper": {"pmid": "555", "title": "T"}}
    res, rep = _hook(p, state)
    assert inner.calls == 0 and rep["error"] is None and rep["written"] == 0
    assert rep["budget_exhausted"] is True and rep["circuit_open"] is False
    assert rep["pathway_meta"]["budget_exhausted"] is True and rep["pathway_meta"]["title_source"] == "paper metadata"
    off.pop(FINAL_PAYLOAD_PATHWAY_FIELD, None)
    p.pop(FINAL_PAYLOAD_PATHWAY_FIELD)  # the title is the only write: it came from the given metadata, not the network
    assert p == off


def test_a_deadline_already_gone_means_no_network_call_either(net):
    inner = net("dead")
    rep = fp.maybe_apply_final_id_fill(_payload(), {}, {fp.DEADLINE_STATE_KEY: time.time() - 100})
    assert inner.calls == 0 and rep["error"] is None


def test_the_cap_only_shortens_and_bounds_one_request(net):
    inner = net("dead")
    f = fp._live_fetcher()
    assert 299 < f.deadline - time.monotonic() <= fp.BUDGET_SECONDS and (inner.timeout, inner.retries) == (30.0, 3)
    fp._cap_to_job_deadline(f, {fp.DEADLINE_STATE_KEY: time.time() + 10_000})  # far deadline: the 300 s budget stays
    assert 299 < f.deadline - time.monotonic() <= fp.BUDGET_SECONDS
    fp._cap_to_job_deadline(f, {fp.DEADLINE_STATE_KEY: time.time() + fp.DEADLINE_MARGIN_S + 100})
    assert 95 < f.deadline - time.monotonic() <= 100
    assert (inner.timeout, inner.retries) == (10.0, 2)  # a hard deadline also bounds one request (4 x 30 s + backoff before)
    f2 = fp._live_fetcher()
    fp._cap_to_job_deadline(f2, {})  # no deadline in state (Streamlit, tests): behave as before
    fp._cap_to_job_deadline(f2, {fp.DEADLINE_STATE_KEY: None})
    fp._cap_to_job_deadline(f2, {fp.DEADLINE_STATE_KEY: float("nan")})
    assert 299 < f2.deadline - time.monotonic() <= fp.BUDGET_SECONDS


def test_the_deadline_key_is_the_one_run_pathway_publishes():
    from t2pw.service import extraction
    assert fp.DEADLINE_STATE_KEY == extraction.DEADLINE_STATE_KEY


# ---- R3-1: the circuit -----------------------------------------------------------------------------------------------------
def test_circuit_opens_after_n_in_a_row_and_a_success_resets_the_count():
    class Flaky:
        def __init__(self, script):
            self.script, self.calls = list(script), 0

        def get_json(self, url, params=None):
            self.calls += 1
            if self.script.pop(0):
                raise FetchError("x")
            return {"ok": 1}

        get_text = get_json

    inner = Flaky([True] * 4 + [False] + [True] * 4 + [False])
    b = fp._Budget(inner, 1000.0, breaker=5)
    for _ in range(4):
        with pytest.raises(FetchError):
            b.get_json("u")
    assert b.get_json("u") == {"ok": 1} and b.circuit_open is False  # the success reset the count
    for _ in range(4):
        with pytest.raises(FetchError):
            b.get_json("u")
    assert b.circuit_open is False and b.refused == 0
    assert b.get_json("u") == {"ok": 1}
    inner2 = Flaky([True] * 5 + [False])
    b2 = fp._Budget(inner2, 1000.0, breaker=5)
    for _ in range(5):
        with pytest.raises(FetchError):
            b2.get_text("u")
    assert b2.circuit_open is True and b2.refused == 0
    with pytest.raises(FetchError, match="circuit open"):
        b2.get_json("u")
    assert inner2.calls == 5 and b2.refused == 1 and b2.exhausted is True  # refused without touching the network


def test_the_circuit_is_per_host_five_failures_from_one_host_do_not_stop_another():
    class Hosts:
        def __init__(self):
            self.calls = []

        def get_json(self, url, params=None):
            self.calls.append(url)
            if "dead.example" in url:
                raise FetchError("x")
            return {"ok": 1}

        get_text = get_json

    inner = Hosts()
    b = fp._Budget(inner, 1000.0, breaker=5)
    for _ in range(5):
        with pytest.raises(FetchError):
            b.get_json("https://dead.example/a")
    assert b.circuit_open is True and b.refused == 0
    assert b.get_json("https://live.example/a") == {"ok": 1}  # another host is still served
    with pytest.raises(FetchError, match="circuit open"):
        b.get_json("https://dead.example/b?q=1")  # the dead host is refused without touching the network
    assert len(inner.calls) == 6 and b.refused == 1
    # a success resets only its own host's count
    inner2 = Hosts()
    b2 = fp._Budget(inner2, 1000.0, breaker=5)
    for _ in range(4):
        with pytest.raises(FetchError):
            b2.get_json("https://dead.example/a")
    assert b2.get_json("https://live.example/a") == {"ok": 1}  # a success on another host does not reset dead.example's count
    with pytest.raises(FetchError):
        b2.get_json("https://dead.example/a")
    assert b2.circuit_open is True


# ---- R3-3: budget_exhausted is per step -------------------------------------------------------------------------------------
def test_budget_exhausted_reports_a_refusal_during_that_step_only():
    b = fp._Budget(_Inner("empty"), 1000.0)
    b.refused = 3  # an earlier step was refused on the shared fetcher
    rep = fp.maybe_apply_paper_nucleic_types(_payload(), {}, fetcher=b)
    assert rep["budget_exhausted"] is False and rep["circuit_open"] is False
    spent = fp._Budget(_Inner("empty"), 0.0)
    rep = fp.apply_final_id_fill(_payload(), {"organism": "Escherichia coli"}, fetcher=spent)
    assert rep["budget_exhausted"] is True


# ---- R3-8: T2PW_FINAL_PAYLOAD_BUDGET_S ---------------------------------------------------------------------------------------
def test_budget_zero_is_the_offline_switch_for_every_step(net, monkeypatch):
    inner = net("dead")
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD_BUDGET_S", "0")
    p = _payload()
    rep = fp.maybe_apply_final_id_fill(p, {"likely_organism": "Escherichia coli"}, {"final_payload_paper": {"pmid": "555", "title": "T"}})
    assert inner.calls == 0 and rep["error"] is None and rep["written"] == 0
    assert rep["budget_exhausted"] and rep["pathway_meta"]["budget_exhausted"]


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "-1", "-0.5", "abc", "", "  "])
def test_bad_budget_values_fall_back_to_the_defaults(net, monkeypatch, raw):
    net("empty")
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD_BUDGET_S", raw)
    assert fp._env_budget() is None and fp._title_budget() == fp.TITLE_BUDGET_SECONDS
    assert 299 < fp._live_fetcher().deadline - time.monotonic() <= fp.BUDGET_SECONDS


def test_the_env_budget_caps_the_title_budget_too(net, monkeypatch):
    net("empty")
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD_BUDGET_S", "20")
    assert fp._title_budget() == 20.0 and 19 < fp._live_fetcher().deadline - time.monotonic() <= 20
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD_BUDGET_S", "1000")
    assert fp._title_budget() == fp.TITLE_BUDGET_SECONDS  # the env never lengthens the title budget past its own default
    assert 999 < fp._live_fetcher().deadline - time.monotonic() <= 1000  # but it does set the id-fill budget
