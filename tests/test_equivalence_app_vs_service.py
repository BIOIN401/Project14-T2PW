"""NEW ACCEPTANCE TEST (R2.5, new capability): the app and the service agree, byte for byte, on one
recorded case.

Under the recorded fixture ``tests/fixtures/equivalence/sour_taste_v2_strict.json.gz`` (LLM, HTTP and
PathBank DB answers, replayed by ``tests/equivalence/llm_replay.py``; every non-loopback socket refused)
and the offline flags (``T2PW_OFFLINE_CURATOR=1``, ``T2PW_SPECIES_LLM=0``, ``T2PW_SPECIES_NCBI=0``):

(a) the case runs through the real app (``AppTest``, via the batch driver's widget path) from the pasted
    text through Stage 0-2, audit and DB mapping, to the strict PWML export step, offline;
(b) a second app run gives the identical canonical payload sha256, canonical payload, export verdict and
    PWML bytes, and both reproduce the recorded live run;
(c) the SERVICE post-pipeline (``t2pw.service.postpipeline.run_post_pipeline_sbml_artifacts``, which
    freezes via ``freeze_canonical_payload``, then ``run_pwml_export``), called directly on the Stage-2
    payload the app reached, gives the same canonical hash and payload, the same export verdict and the
    same PWML bytes as the app. Nothing is normalised.

The case is a full open-access paper, pinned with scope (``harness.TOPIC_LINE``, from
docs/evaluation_v2/paper_runs_orch755/topics/sour_taste_v2.txt), with committed evidence that strict mode
exports a PWML from it (ORCH-767-RESULT.md; tests/fixtures/orch768). No short text has such evidence.

NON-DEFAULT SETTING: the app runs with the sidebar's "Audit repair timeout" at 1800 s (its maximum), not 240 s,
in recording and replay alike (``harness.AUDIT_TIMEOUT_SECONDS``). The post-pipeline audit loop branches on
wall-clock time against that budget, so at 240 s the round it stops at depends on live LLM latency and no
replay can follow it (docs/webserver/results/R2.5.md sec. 8).

CLOCK FREEZE (PO ruling 2026-11-05): during replay ``enrich_entities._utc_now_iso`` reads
``harness.FROZEN_UTC``, because enrichment stamps wall-clock ``retrieved_at`` times into the canonical payload.
The frozen arms (app1, app2, service) must then agree on the FULL canonical hash. Against the live run, which
carries real times, the comparison is PWML bytes plus every canonical field except ``retrieved_at``; an
unfrozen control arm proves the freeze is effective and touches only those stamps.

Each arm has its own project root, output dir and empty caches (``tests/equivalence/harness.py``).
From R3, arm (c) starts from the text instead (see ``run_service_arm``'s TODO).

RUN IT IN ITS OWN PROCESS: one pytest invocation naming only this file. ``harness.replay_env`` refuses to
pin the environment when another test has already imported ``t2pw.llm.client`` (its provider and model are
read at import), so a shared process fails loudly in the fixture instead of replaying under the wrong
configuration. Runtime: see docs/webserver/results/R2.5.md.
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


def _secret_hits(text: str, public: frozenset = frozenset()) -> dict:
    """Pattern hits, minus strings in ``public`` (e-mail addresses printed in the source paper itself)."""

    hits = {}
    for name, pattern in SECRET_PATTERNS.items():
        found = [m.group(0) for m in pattern.finditer(text) if m.group(0) not in public]
        if found:
            hits[name] = found[:3]
    return hits


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    fixture = harness.load_fixture()
    mp = pytest.MonkeyPatch()
    try:
        harness.replay_env(mp, fixture)
        before = harness.checkout_state()
        base = tmp_path_factory.mktemp("equivalence")
        case = fixture["case"]
        app1 = harness.run_app_arm(base / "app1", llm_replay.Cassette("replay", fixture["entries"]), APP_TIMEOUT, case)
        app2 = harness.run_app_arm(base / "app2", llm_replay.Cassette("replay", fixture["entries"]), APP_TIMEOUT, case)
        service = harness.run_service_arm(app1, base / "service", llm_replay.Cassette("replay", fixture["entries"]))
        # The control: the same replay with the real clock, to prove the freeze is what makes the hashes equal.
        unfrozen = harness.run_app_arm(base / "unfrozen", llm_replay.Cassette("replay", fixture["entries"]),
                                       APP_TIMEOUT, case, freeze=False)
        after = harness.checkout_state()
        yield {"fixture": fixture, "app1": app1, "app2": app2, "service": service, "unfrozen": unfrozen,
               "before": before, "after": after}
    finally:
        mp.undo()


def test_fixture_holds_no_secrets():
    text = gzip.decompress(harness.FIXTURE.read_bytes()).decode("utf-8")
    # The case is a published open-access paper: an address printed in it (a corresponding author's) is public
    # paper text, carried into every prompt. Only those exact strings are exempt; any other address fails.
    paper = harness.load_fixture()["case"]["full_text"]
    public = frozenset(m.group(0) for m in SECRET_PATTERNS["email address"].finditer(paper))
    assert _secret_hits(text, public) == {}
    # The scan bites: each shape is caught when present.
    planted = ('"sk-or-' + 'v1-0123456789abcdef" "Authorization" Bearer abcdefghijkl a.b@example.org '
               'PATHBANK_DB_PASSWORD "password": api_key=XYZ12345 "set-cookie" X-Title')
    assert set(_secret_hits(planted)) == set(SECRET_PATTERNS)


OUTCOME_FIELDS = ("status", "failure_kind", "stage", "message")


def _outcome(outcome: dict) -> dict:
    return {k: outcome.get(k) for k in OUTCOME_FIELDS}


def test_fixture_was_recorded_from_a_clean_tree_through_to_the_export_step():
    recorded = harness.load_fixture()["recorded"]
    assert recorded["tree_dirty"] is False and recorded["tree_head"]
    assert recorded["outcome"]["stage"] == "pwml_export"  # the export step ran; its verdict is pinned below
    assert recorded["canonical_payload_sha256"] and recorded["live_llm_errors"] == []


def test_a_app_runs_offline_from_the_fixture_alone_to_the_recorded_verdict(arms):
    recorded = arms["fixture"]["recorded"]
    for name in ("app1", "app2"):
        arm = arms[name]
        assert arm["misses"] == [], (name, arm["misses"][:3])
        assert arm["refused"] == [], (name, arm["refused"][:5])
        assert arm["unused"] == [], (name, arm["unused"][:10])
        assert _outcome(arm["outcome"]) == _outcome(recorded["outcome"]), name


def test_b_two_app_runs_are_identical(arms):
    app1, app2 = arms["app1"], arms["app2"]
    assert app1["canonical_payload_sha256"] and app1["canonical_payload_sha256"] == app2["canonical_payload_sha256"]
    assert app1["canonical_payload"] == app2["canonical_payload"]
    assert app1["export_verdict"] == app2["export_verdict"]
    assert app1["xml"] == app2["xml"]
    assert app1["pwml_files"] == app2["pwml_files"]


def _is_stamp(path: str) -> bool:
    return path.endswith("/retrieved_at")


def test_b_replay_reproduces_the_recorded_live_run(arms):
    """Against the LIVE run: PWML bytes exactly, and the canonical payload field by field EXCEPT the
    ``retrieved_at`` stamps, which in the live run are real wall-clock times (PO ruling 2026-11-05). The live
    canonical hash is therefore not compared; the frozen arms compare full hashes in test_b/test_c."""

    from t2pw.pipeline.gate_reports import payload_sha256

    recorded = arms["fixture"]["recorded"]
    assert harness.sha256(arms["app1"]["xml"]) == recorded["pwml_sha256"]
    assert arms["app1"]["pwml_files"] == recorded["pwml_files"]
    live_payload = recorded["canonical_payload"]
    assert payload_sha256(live_payload) == recorded["canonical_payload_sha256"]  # it IS the live run's payload
    live, replay = harness.leaves(live_payload), harness.leaves(arms["app1"]["canonical_payload"])
    assert set(live) == set(replay)
    differing = sorted(k for k in live if live[k] != replay[k])
    assert differing and all(_is_stamp(k) for k in differing), differing[:10]
    assert {replay[k] for k in differing} == {harness.FROZEN_UTC}


def test_the_clock_freeze_is_effective_and_touches_only_retrieved_at(arms):
    frozen, unfrozen = arms["app1"], arms["unfrozen"]
    assert frozen["frozen_bindings"] >= 1 and unfrozen["frozen_bindings"] == 0
    assert unfrozen["canonical_payload_sha256"] != frozen["canonical_payload_sha256"]
    a, b = harness.leaves(frozen["canonical_payload"]), harness.leaves(unfrozen["canonical_payload"])
    assert set(a) == set(b)
    differing = sorted(k for k in a if a[k] != b[k])
    assert differing and all(_is_stamp(k) for k in differing), differing[:10]
    assert unfrozen["xml"] == frozen["xml"]


def test_c_service_post_pipeline_equals_the_app(arms):
    app1, service = arms["app1"], arms["service"]
    # The Stage-2 payload the service started from is the one the app's session held.
    from t2pw.pipeline.pipeline import propagate_context_organism

    assert app1["post_call"] and app1["export_call"], "the app arm never reached the post-pipeline and export"
    assert app1["canonical_payload_sha256"], "the app arm produced no canonical payload"
    (post_args, _kwargs) = app1["post_call"]
    assert post_args[0] == propagate_context_organism(app1["stage2_payload"], app1["pathway_context"])
    assert service["misses"] == [] and service["refused"] == []
    assert service["canonical_payload_sha256"] == app1["canonical_payload_sha256"]
    assert service["canonical_payload"] == app1["canonical_payload"]
    assert service["export_verdict"] == app1["export_verdict"]
    assert service["xml"] == app1["xml"]
    assert service["pwml_files"] == app1["pwml_files"]


def test_pwml_bytes_are_produced_and_equal_in_all_three_arms(arms):
    # Not skippable: a fixture whose case writes no PWML cannot prove byte equality, so it fails here.
    assert arms["fixture"]["recorded"]["pwml_bytes"] > 0, "the fixture's case wrote no PWML"
    xml = arms["app1"]["xml"]
    assert xml and arms["app2"]["xml"] == xml and arms["service"]["xml"] == xml
    assert harness.sha256(xml) == arms["fixture"]["recorded"]["pwml_sha256"]


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
