"""NEW ACCEPTANCE TEST (R26, new capability): the app and the service agree on a SECOND recorded case.

The first fixture (``test_equivalence_app_vs_service.py``) has Stage 2 additions 0, no out-of-scope reaction
and the default export mode, so three mutations of the extraction pass it (R3d review F1). This case replays
one live RESEARCH-mode run of an open-access paper (``harness.SECOND_CASE``) in which the model added
reactions at Stage 2 and labelled a reaction out of scope, through the same harness: app, app, service, all
offline, the service arm starting from the case's text.

Research mode writes no PWML, so the comparison is the canonical payload and what the extraction published.
PWML bytes stay covered by the first fixture. RAG is off, as in the first fixture (R26.md sec. 8).

RUN IT IN ITS OWN PROCESS, like the first test: ``harness.replay_env`` refuses a process that already
imported ``t2pw.llm.client``.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equivalence import harness, llm_replay  # noqa: E402
from test_equivalence_app_vs_service import APP_TIMEOUT, SECRET_PATTERNS, _is_stamp, _outcome, _secret_hits  # noqa: E402

FIXTURE = harness.SECOND_CASE["fixture"]


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    fixture = harness.load_fixture(FIXTURE)
    mp = pytest.MonkeyPatch()
    try:
        harness.replay_env(mp, fixture)
        before = harness.checkout_state()
        base = tmp_path_factory.mktemp("equivalence2")
        case = fixture["case"]

        def cassette() -> llm_replay.Cassette:
            return llm_replay.Cassette("replay", fixture["entries"])

        app1 = harness.run_app_arm(base / "app1", cassette(), APP_TIMEOUT, case)
        app2 = harness.run_app_arm(base / "app2", cassette(), APP_TIMEOUT, case)
        service = harness.run_service_arm(app1, base / "service", cassette(), case=case)
        yield {"fixture": fixture, "app1": app1, "app2": app2, "service": service,
               "before": before, "after": harness.checkout_state()}
    finally:
        mp.undo()


def test_fixture_holds_no_secrets():
    text = gzip.decompress(FIXTURE.read_bytes()).decode("utf-8")
    paper = harness.load_fixture(FIXTURE)["case"]["full_text"]
    public = frozenset(m.group(0) for m in SECRET_PATTERNS["email address"].finditer(paper))
    assert _secret_hits(text, public) == {}


def test_fixture_is_the_recorded_research_run():
    fixture = harness.load_fixture(FIXTURE)
    recorded = fixture["recorded"]
    assert fixture["case"]["mode"] == "research" and fixture["env"]["RAG_ENABLED"] == "0"
    assert recorded["tree_dirty"] is False and recorded["tree_head"] and recorded["live_llm_errors"] == []
    assert recorded["outcome"]["status"] == "pass" and recorded["outcome"]["stage"] == "research_report"
    assert recorded["canonical_payload_sha256"]


def test_a_app_runs_offline_from_the_fixture_alone(arms):
    recorded = arms["fixture"]["recorded"]
    for name in ("app1", "app2"):
        arm = arms[name]
        assert arm["misses"] == [], (name, arm["misses"][:3])
        assert arm["refused"] == [], (name, arm["refused"][:5])
        assert arm["unused"] == [], (name, arm["unused"][:10])
        assert _outcome(arm["outcome"]) == _outcome(recorded["outcome"]), name


def _names(payload) -> set:
    return {r.get("name") for r in payload["processes"]["reactions"]}


def test_the_case_reaches_the_paths_the_first_fixture_cannot_see(arms):
    """What makes this fixture worth having (R3d review F1): additions, an out-of-scope hit, research mode."""

    published = arms["app1"]["published"]
    assert published["export_mode"] == "research"
    removed = published["out_of_scope_removed_reactions"]
    assert removed, "the recording has no out-of-scope reaction"
    assert set(removed) <= _names(published["stage_one"]) and not set(removed) & _names(published["final_payload"])
    additions = published["stage_two"].get("additions")
    assert additions, "Stage 2 added nothing"
    assert any(v for v in additions.values() if isinstance(v, (dict, list)))


def test_b_two_app_runs_are_identical(arms):
    app1, app2 = arms["app1"], arms["app2"]
    assert app1["canonical_payload_sha256"] and app1["canonical_payload_sha256"] == app2["canonical_payload_sha256"]
    assert app1["canonical_payload"] == app2["canonical_payload"]
    assert app1["published"] == app2["published"]


def test_b_replay_reproduces_the_recorded_live_run(arms):
    from t2pw.pipeline.gate_reports import payload_sha256

    recorded = arms["fixture"]["recorded"]
    live_payload = recorded["canonical_payload"]
    assert payload_sha256(live_payload) == recorded["canonical_payload_sha256"]
    live, replay = harness.leaves(live_payload), harness.leaves(arms["app1"]["canonical_payload"])
    assert set(live) == set(replay)
    differing = sorted(k for k in live if live[k] != replay[k])
    assert all(_is_stamp(k) for k in differing), differing[:10]


def test_c_the_service_equals_the_app(arms):
    app1, service = arms["app1"], arms["service"]
    assert app1["post_call"] and app1["post_call"][1]["export_mode"] == "research"
    assert app1["export_call"] is None and app1["xml"] == b"", "research mode writes no PWML"
    assert service["misses"] == [] and service["refused"] == [], (service["misses"][:3], service["refused"][:5])
    assert [row for row in service["unused"] if row[0] == "llm"] == [], service["unused"]
    assert service["canonical_payload_sha256"] and service["canonical_payload_sha256"] == app1["canonical_payload_sha256"]
    assert service["canonical_payload"] == app1["canonical_payload"]
    assert service["published"] == app1["published"]


def test_isolation_left_the_checkout_untouched(arms):
    assert arms["after"] == arms["before"]
