"""NEW ACCEPTANCE TEST (R5, new capability): ``run_pathway`` matches the app on both recorded cases.

PLAN section 3: the app and ``run_pathway(PathwayRequest)`` give the same canonical payload and PWML bytes. Each
case replays offline through the real app (AppTest, the driver's widget path) and through ``run_pathway``, in
its own project root, under the existing harness. Nothing is normalised. The one setting overridden is the
audit timeout both arms run under (``harness.AUDIT_TIMEOUT_SECONDS``).

RUN THIS FILE IN ITS OWN PROCESS: ``harness.replay_env`` refuses a process that already imported the LLM client.
The second case runs in a child for its fixed ``PYTHONHASHSEED`` (see ``test_equivalence_second_fixture.py``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equivalence import harness, llm_replay  # noqa: E402

#: The ``RunOutcome`` fields the job and result pages read, against the same names on the result.
RESULT_FIELDS = ("counts", "issue_codes", "warnings")


@pytest.fixture(scope="module")
def sour(tmp_path_factory):
    fixture = harness.load_fixture()
    mp = pytest.MonkeyPatch()
    try:
        harness.replay_env(mp, fixture)
        before = harness.checkout_state()
        base = tmp_path_factory.mktemp("equivalence_r5")
        app = harness.run_app_arm(base / "app", llm_replay.Cassette("replay", fixture["entries"]), 240.0, fixture["case"])
        pathway = harness.run_pathway_arm(app, base / "pathway", llm_replay.Cassette("replay", fixture["entries"]),
                                          fixture["case"])
        yield {"app": app, "pathway": pathway, "before": before, "after": harness.checkout_state()}
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def farnesol(tmp_path_factory):
    base = tmp_path_factory.mktemp("equivalence_r5_2")
    out = base / "arms.json"
    before = harness.checkout_state()
    done = subprocess.run(
        [sys.executable, "-u", str(Path(harness.__file__)), "replay-pathway", "--case", "farnesol_research",
         "--base", str(base / "arms"), "--out", str(out)],
        env={**os.environ, "PYTHONHASHSEED": harness.SECOND_CASE["hash_seed"]},
        capture_output=True, text=True, timeout=500)
    assert done.returncode == 0, (done.stdout[-3000:], done.stderr[-3000:])
    yield {**json.loads(out.read_text(encoding="utf-8")), "before": before, "after": harness.checkout_state()}


def _replayed_cleanly(arm: dict) -> None:
    assert arm["misses"] == [] and arm["refused"] == [], (arm["misses"][:3], arm["refused"][:5])
    assert [row for row in arm["unused"] if row[0] == "llm"] == [], arm["unused"]


def _same_verdict(app: dict, pathway: dict) -> None:
    result = pathway["result"]
    for name in ("status", "failure_kind", "stage", "message"):
        assert result[name] == app["outcome"][name], name
    for name in RESULT_FIELDS:
        assert result[name] == app["outcome_row"][name], name
    release, frozen = result["release_status"], app["outcome_row"]["release_status"]
    # ReleaseStatus has no field for the boundary's extra keys; the bridge drops them the same way.
    assert (release is None) == (frozen is None)
    if frozen is not None:
        assert release and {key: frozen[key] for key in release} == release


def test_strict_case_matches_the_app_payload_and_pwml_bytes(sour):
    app, pathway = sour["app"], sour["pathway"]
    _replayed_cleanly(app)
    _replayed_cleanly(pathway)
    assert pathway["frozen_bindings"] >= 1
    assert app["outcome"]["status"] == "pass" and app["xml"], app["outcome"]
    assert app["canonical_payload"] and pathway["canonical_payload"] == app["canonical_payload"]
    assert pathway["xml"] == app["xml"]
    assert pathway["pwml_files"] == app["pwml_files"]


def test_strict_case_renders_the_same_result(sour):
    app, pathway = sour["app"], sour["pathway"]
    _same_verdict(app, pathway)
    row = app["outcome_row"]
    assert pathway["result"]["pwml_name"] == row["pwml_artifact"]
    assert pathway["result"]["reports"] == sorted(set(row["artifacts"]) - {row["pwml_artifact"]})


def test_strict_case_hands_the_pipeline_the_settings_the_form_does(sour):
    """The shared defaults, end to end: the audit and the export get the arguments the real form gave them."""

    pathway = sour["pathway"]
    assert pathway["calls"] == {"run_post_pipeline_sbml_artifacts": 1, "run_pwml_export": 1}
    assert pathway["same_post_settings"] is True
    assert pathway["same_export_settings"] is True


def test_strict_case_saves_the_rendered_diagram(sour):
    assert sour["pathway"]["result"]["has_image"] is True


def test_strict_case_left_the_checkout_untouched(sour):
    assert sour["after"] == sour["before"]


def test_research_case_matches_the_app_payload(farnesol):
    app, pathway = farnesol["app"], farnesol["pathway"]
    _replayed_cleanly(app)
    _replayed_cleanly(pathway)
    assert app["outcome"]["status"] == "pass" and app["canonical_payload"], app["outcome"]
    assert pathway["canonical_payload"] == app["canonical_payload"]
    assert app["xml_bytes"] == 0 and pathway["xml_bytes"] == 0, "research mode writes no PWML"


def test_research_case_renders_the_same_result(farnesol):
    app, pathway = farnesol["app"], farnesol["pathway"]
    _same_verdict(app, pathway)
    assert pathway["result"]["reports"] == app["outcome_row"]["artifacts"]
    assert pathway["same_post_settings"] is True


def test_research_case_left_the_checkout_untouched(farnesol):
    assert farnesol["after"] == farnesol["before"]
