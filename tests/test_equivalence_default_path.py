"""OFFLINE DEFAULT-PATH TEST (D-W23 a, FINAL-PAYLOAD-P5 sec. 0.2 / review R3-2): the REAL defaults, end to end.

``tests/equivalence/llm_replay.py`` pins ``T2PW_FINAL_PAYLOAD=0`` and ``T2PW_PWML_LAYOUT=v1`` for the two equivalence files
(they compare against a recorded run made before those defaults flipped). This file replays the same recorded case with BOTH
variables left UNSET, i.e. final payload on and layout v2, through the whole default path (the real app and the real
``run_pathway``), with the final payload's network fetcher faked as a dead network (every request a ``FetchError``), so no
request leaves the machine. It asserts that the run completes, writes a PWML, the final payload ran and failed open, the app
and ``run_pathway`` arms agree on the PWML bytes and the canonical payload, and the PWML is v2 layout.

RUN IT IN ITS OWN PROCESS (one pytest invocation naming only this file), like the two equivalence files:
``harness.replay_env`` refuses a process that already imported the LLM client.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equivalence import harness, llm_replay  # noqa: E402


class _DeadNetwork:
    """Stands in for ``UrllibFetcher``: records the request, then fails like an unreachable network."""

    timeout, retries = 30.0, 3
    urls: list = []

    def __init__(self, throttle=None) -> None:
        pass

    def _go(self, url, params=None):
        from t2pw.clean_model.fill.http import FetchError

        type(self).urls.append(url)
        raise FetchError("network unreachable (offline default-path test)")

    get_json = get_text = _go


@pytest.fixture(scope="module")
def default_path(tmp_path_factory):
    fixture = harness.load_fixture()
    mp = pytest.MonkeyPatch()
    try:
        harness.replay_env(mp, fixture)
        # The two defaults under test: unset, exactly as a deployment without the variables runs.
        mp.delenv("T2PW_FINAL_PAYLOAD", raising=False)
        mp.delenv("T2PW_PWML_LAYOUT", raising=False)
        mp.setattr("t2pw.clean_model.fill.http.UrllibFetcher", _DeadNetwork)
        mp.setattr("t2pw.clean_model.fill.cache.Cache._path", lambda self, key: None)  # memory-only cache
        import os

        from t2pw.pwml.layout import layout_version
        from t2pw.service.final_payload import final_payload_enabled

        # read here: tests/conftest.py re-pins T2PW_FINAL_PAYLOAD=0 for every test function, after this module fixture ran
        defaults = {"unset": "T2PW_FINAL_PAYLOAD" not in os.environ and "T2PW_PWML_LAYOUT" not in os.environ,
                    "final_payload": final_payload_enabled(), "layout": layout_version()}
        base = tmp_path_factory.mktemp("equivalence_default_path")
        app = harness.run_app_arm(base / "app", llm_replay.Cassette("replay", fixture["entries"]), 240.0, fixture["case"])
        app_requests = len(_DeadNetwork.urls)
        pathway = harness.run_pathway_arm(app, base / "pathway", llm_replay.Cassette("replay", fixture["entries"]),
                                          fixture["case"])
        yield {"defaults": defaults, "app": app, "pathway": pathway, "app_requests": app_requests, "all_requests": len(_DeadNetwork.urls)}
    finally:
        mp.undo()


def test_the_defaults_are_final_payload_on_and_layout_v2(default_path):
    assert default_path["defaults"] == {"unset": True, "final_payload": True, "layout": "v2"}


def test_the_run_completes_and_writes_a_pwml(default_path):
    app, pathway = default_path["app"], default_path["pathway"]
    assert app["outcome"]["status"] == "pass" and app["xml"], app["outcome"]
    assert pathway["result"]["status"] == "pass" and pathway["xml"], pathway["result"]


def test_the_final_payload_ran_on_both_arms_and_failed_open(default_path):
    # each arm reached the (faked, dead) network, so the default hook ran; neither run failed because of it
    assert default_path["app_requests"] > 0 and default_path["all_requests"] > default_path["app_requests"]
    for arm in (default_path["app"], default_path["pathway"]):
        assert arm["misses"] == [] and arm["refused"] == [], (arm["misses"][:3], arm["refused"][:5])  # nothing left the machine
        assert [row for row in arm["unused"] if row[0] == "llm"] == []
    from t2pw.service.final_payload import FINAL_PAYLOAD_FILL_FIELD

    assert FINAL_PAYLOAD_FILL_FIELD not in (default_path["app"]["canonical_payload"] or {})  # a dead network fills nothing


def test_the_app_and_run_pathway_agree(default_path):
    app, pathway = default_path["app"], default_path["pathway"]
    assert pathway["xml"] == app["xml"]
    assert app["canonical_payload"] and pathway["canonical_payload"] == app["canonical_payload"]
    assert pathway["pwml_files"] == app["pwml_files"]


def test_the_pwml_is_layout_v2(default_path):
    # The writer always emits the (possibly empty) <membrane-visualizations/> section, so a substring test would also pass for v1.
    # v2 adds one membrane-visualization and one label-location per compartment; v1 leaves both sections empty.
    from lxml import etree

    root = etree.fromstring(default_path["app"]["xml"])
    assert len(root.findall(".//membrane-visualizations/membrane-visualization")) > 0
    assert len(root.findall(".//label-locations/label-location")) > 0


def test_the_app_payload_is_the_recorded_live_payload_plus_the_final_payload_title(default_path):
    """Against the recorded live run (made with the final payload off): the same leaves, except the ``retrieved_at`` stamps (live
    wall-clock times; the replay freezes the clock) and the ``final_payload_pathway`` record the default-ON title step adds
    (a dead network falls back to the paper's own title). Nothing else may differ."""

    live = harness.load_fixture()["recorded"]["canonical_payload"]
    a, b = harness.leaves(live), harness.leaves(default_path["app"]["canonical_payload"])
    assert set(a) <= set(b) and all(k.startswith("/final_payload_pathway/") for k in set(b) - set(a)), sorted(set(b) ^ set(a))[:10]
    assert sorted(k for k in a if a[k] != b[k] and not k.endswith("/retrieved_at")) == []
