"""NEW ACCEPTANCE TEST (R2.5, new capability): the app and the service give the same canonical payload
hash and the same PWML bytes for one recorded case.

Under the recorded fixture ``tests/fixtures/equivalence/heme_ferrochelatase_strict.json.gz`` (LLM, HTTP and
PathBank DB answers, replayed by ``tests/equivalence/llm_replay.py``; every non-loopback socket refused)
and the offline flags (``T2PW_OFFLINE_CURATOR=1``, ``T2PW_SPECIES_LLM=0``, ``T2PW_SPECIES_NCBI=0``):

(a) the case runs through the real app (``AppTest``, via the batch driver's widget path) to a strict PWML;
(b) a second app run gives the identical canonical payload sha256 and identical PWML bytes;
(c) the SERVICE post-pipeline (``t2pw.service.postpipeline.run_post_pipeline_sbml_artifacts``, which
    freezes via ``freeze_canonical_payload``, then ``run_pwml_export``), called directly on the Stage-2
    payload the app reached, gives the same hash and the same bytes as the app. Nothing is normalised.

Each arm has its own project root, output dir and empty caches (``tests/equivalence/harness.py``).
From R3, arm (c) starts from the text instead (see ``run_service_arm``'s TODO).

RUN IT IN ITS OWN PROCESS: one pytest invocation naming only this file. ``harness.replay_env`` refuses to
pin the environment when another test has already imported ``t2pw.llm.client`` (its provider and model are
read at import), so a shared process fails loudly in the fixture instead of replaying under the wrong
configuration. Runtime: about one minute (three arms; see docs/webserver/results/R2.5.md).
"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equivalence import harness, llm_replay  # noqa: E402

APP_TIMEOUT = 240.0

#: Secret SHAPES the committed fixture must not contain (record mode separately refuses the actual values).
SECRET_PATTERNS = {
    "openrouter key": re.compile(r"sk-or-[A-Za-z0-9-]{8,}"),
    "bearer token": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{8,}"),
    "authorization header": re.compile(r"(?i)\"(proxy-)?authorization\""),
    "cookie header": re.compile(r"(?i)\"(set-)?cookie\""),
    "api key param": re.compile(r"(?i)\b(api_?key|access_token|token)=(?!<redacted>)[^&\s\"]+"),
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
    "db credential env": re.compile(r"PATHBANK_DB_(HOST|USER|PASSWORD)"),
    "password field": re.compile(r"(?i)\"(db_)?password\"\s*:"),
    "openrouter headers": re.compile(r"HTTP-Referer|X-Title"),
}


def _secret_hits(text: str) -> dict:
    return {name: pattern.findall(text)[:3] for name, pattern in SECRET_PATTERNS.items() if pattern.search(text)}


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    fixture = harness.load_fixture()
    mp = pytest.MonkeyPatch()
    try:
        harness.replay_env(mp, fixture)
        before = harness.checkout_state()
        base = tmp_path_factory.mktemp("equivalence")
        app1 = harness.run_app_arm(base / "app1", llm_replay.Cassette("replay", fixture["entries"]), APP_TIMEOUT)
        app2 = harness.run_app_arm(base / "app2", llm_replay.Cassette("replay", fixture["entries"]), APP_TIMEOUT)
        service = harness.run_service_arm(app1, base / "service", llm_replay.Cassette("replay", fixture["entries"]))
        after = harness.checkout_state()
        yield {"fixture": fixture, "app1": app1, "app2": app2, "service": service, "before": before, "after": after}
    finally:
        mp.undo()


def test_fixture_holds_no_secrets():
    text = gzip.decompress(harness.FIXTURE.read_bytes()).decode("utf-8")
    assert _secret_hits(text) == {}
    # The scan bites: each shape is caught when present.
    planted = ('"sk-or-' + 'v1-0123456789abcdef" "Authorization" Bearer abcdefghijkl a.b@example.org '
               'PATHBANK_DB_PASSWORD "password": api_key=XYZ12345 "set-cookie" X-Title')
    assert set(_secret_hits(planted)) == set(SECRET_PATTERNS)


def test_fixture_was_recorded_from_a_clean_tree_and_reached_strict_pwml():
    recorded = harness.load_fixture()["recorded"]
    assert recorded["tree_dirty"] is False
    assert recorded["outcome"]["status"] == "pass"
    assert recorded["pwml_bytes"] > 0 and recorded["canonical_payload_sha256"]


def test_a_app_run_reaches_strict_pwml_offline_from_the_fixture_alone(arms):
    for name in ("app1", "app2"):
        arm = arms[name]
        assert arm["outcome"]["status"] == "pass", (name, arm["outcome"])
        assert arm["misses"] == [], (name, arm["misses"][:3])
        assert arm["refused"] == [], (name, arm["refused"][:5])
        assert arm["unused"] == [], (name, arm["unused"][:10])
        assert arm["xml"] and arm["canonical_payload_sha256"]


def test_b_two_app_runs_are_byte_identical(arms):
    app1, app2 = arms["app1"], arms["app2"]
    assert app1["canonical_payload_sha256"] == app2["canonical_payload_sha256"]
    assert app1["xml"] == app2["xml"]
    assert app1["pwml_files"] == app2["pwml_files"] and app1["pwml_files"]


def test_b_replay_reproduces_the_recorded_live_run(arms):
    recorded = arms["fixture"]["recorded"]
    assert arms["app1"]["canonical_payload_sha256"] == recorded["canonical_payload_sha256"]
    assert harness.sha256(arms["app1"]["xml"]) == recorded["pwml_sha256"]
    assert arms["app1"]["pwml_files"] == recorded["pwml_files"]


def test_c_service_post_pipeline_equals_the_app(arms):
    app1, service = arms["app1"], arms["service"]
    # The Stage-2 payload the service started from is the one the app's session held.
    from t2pw.pipeline.pipeline import propagate_context_organism

    (post_args, _kwargs) = app1["post_call"]
    assert post_args[0] == propagate_context_organism(app1["stage2_payload"], app1["pathway_context"])
    assert service["misses"] == [] and service["refused"] == []
    assert service["ok"] is True, service["error"]
    assert service["canonical_payload_sha256"] == app1["canonical_payload_sha256"]
    assert service["canonical_payload"] == app1["canonical_payload"]
    assert service["xml"] == app1["xml"]
    assert service["pwml_files"] == app1["pwml_files"]


def test_isolation_left_the_checkout_untouched(arms):
    assert arms["after"] == arms["before"]


def test_an_unrecorded_request_fails_loudly(arms):
    import t2pw.llm.client as client

    cassette = llm_replay.Cassette("replay", {})
    with pytest.MonkeyPatch.context() as mp:
        cassette.install(mp)
        with pytest.raises(llm_replay.ReplayMiss):
            client._client.chat.completions.create(model="m", messages=[{"role": "user", "content": "unrecorded"}])
        with pytest.raises(OSError):
            __import__("socket").create_connection(("example.org", 443), timeout=1)
    assert [m["kind"] for m in cassette.misses] == ["llm"]
    assert cassette.refused


def test_keying_ignores_the_model_and_the_timeout_only():
    base = {"messages": [{"role": "user", "content": "x"}], "temperature": 0.0, "max_tokens": 10}
    key, _ = llm_replay.llm_key(base)
    assert llm_replay.llm_key({**base, "model": "other/model", "timeout": 5})[0] == key
    assert llm_replay.llm_key({**base, "temperature": 0.1})[0] != key
    assert llm_replay.llm_key({**base, "max_tokens": 11})[0] != key
    assert llm_replay.redact_url("https://h/p?email=a%40b.org&id=1&monkey=2") == "https://h/p?email=<redacted>&id=1&monkey=2"
