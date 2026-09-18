"""Run artifacts, schema versioning, checkpoints, resume and cancellation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import helpers_mining as H  # noqa: E402
from t2pw.mining.artifacts import (  # noqa: E402
    INCOMPLETE_STATES,
    STATE_CANCELLED,
    STATE_PREFLIGHT_FAILED,
    STATE_STOPPED,
    RunDirectory,
    list_runs,
    read_jsonl,
)
from t2pw.mining.config import ARTIFACT_SCHEMA_VERSION, PROMPT_VERSION  # noqa: E402

TOP_LEVEL_FILES = (
    "run_manifest.json",
    "config.json",
    "preflight.json",
    "papers.jsonl",
    "queries.jsonl",
    "frontier.jsonl",
    "evidence_spans.jsonl",
    "claims.jsonl",
    "decisions.jsonl",
    "model_calls.jsonl",
    "costs.json",
    "primary_graph.json",
    "candidate_subgraphs.json",
    "conflicts.json",
    "summary.json",
    "report.md",
    "checkpoint.json",
    "state.json",
)


@pytest.fixture(scope="module")
def mined(tmp_path_factory):
    return H.run_fixture(tmp_path_factory.mktemp("mining_artifacts"))


def test_every_required_artifact_is_written(mined):
    root = mined.outcome.run_dir.root
    for name in TOP_LEVEL_FILES:
        assert (root / name).exists(), f"missing {name}"
    for name in (
        "reaction_network.json", "provenance_network.json",
        "reaction_network.dot", "provenance_network.dot", "evidence_table.json",
    ):
        assert (root / "graph_views" / name).exists(), f"missing graph_views/{name}"


def test_the_manifest_carries_everything_an_audit_needs(mined):
    manifest = json.loads(
        (mined.outcome.run_dir.root / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_id"]
    assert manifest["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert manifest["prompt_version"] == PROMPT_VERSION
    assert manifest["started_at"] and manifest["updated_at"] and manifest["finished_at"]
    assert "commit" in manifest["git"] and "dirty" in manifest["git"]
    assert "branch" in manifest["git"]
    assert manifest["environment"]["python"]
    assert "T2PW_PATHWAY_MINING_ENABLED" in manifest["feature_flags"]
    assert manifest["config"]["target_species"] == H.TARGET_SPECIES
    assert manifest["config"]["budgets"]["max_rounds"]
    assert manifest["config"]["model"]["model"]
    assert manifest["completion_state"]
    assert manifest["stop_reason"]
    assert manifest["produces_pwml"] is False


def test_no_secret_is_written_anywhere_in_the_run(mined):
    root = mined.outcome.run_dir.root
    forbidden = ("api_key", "sk-", "OPENROUTER_API_KEY", "password")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for token in forbidden:
            assert token.lower() not in text, f"{path} contains {token!r}"


def test_jsonl_artifacts_are_one_object_per_line(mined):
    root = mined.outcome.run_dir.root
    for name in ("papers.jsonl", "claims.jsonl", "decisions.jsonl", "evidence_spans.jsonl"):
        rows = read_jsonl(root / name)
        assert rows, f"{name} is empty"
        assert all(isinstance(r, dict) for r in rows)


def test_every_claim_records_its_schema_and_prompt_version(mined):
    for row in read_jsonl(mined.outcome.run_dir.root / "claims.jsonl"):
        assert row["schema_version"] == ARTIFACT_SCHEMA_VERSION
        assert row["prompt_version"] == PROMPT_VERSION
        assert row["model"]
        assert row["model_call_id"]


def test_each_round_directory_is_self_contained(mined):
    rounds = sorted((mined.outcome.run_dir.root / "rounds").iterdir())
    assert rounds
    for directory in rounds:
        payload = json.loads((directory / "round.json").read_text(encoding="utf-8"))
        checkpoint = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
        assert checkpoint["completed"] is True
        assert checkpoint["round_index"] == payload["round"]
        assert checkpoint["schema_version"] == ARTIFACT_SCHEMA_VERSION
        assert "usage" in checkpoint
        assert "frontier" in checkpoint
        metrics = payload["metrics"]
        assert "graph_hash_before" in metrics and "graph_hash_after" in metrics
        assert "usage" in metrics


def test_the_report_names_the_stop_reason_and_the_policies(mined):
    report = (mined.outcome.run_dir.root / "report.md").read_text(encoding="utf-8")
    assert "research graph, not PWML" in report
    assert "DEFERRED in version 1" in report
    assert "Stop reason" in report
    assert "Growth by round" in report
    assert "estimate" in report.lower()


def test_list_runs_reports_the_run(mined, tmp_path_factory):
    runs_dir = mined.outcome.run_dir.root.parent
    rows = list_runs(runs_dir)
    assert rows
    assert rows[0]["run_id"] == mined.outcome.run_dir.run_id
    assert rows[0]["species"] == H.TARGET_SPECIES


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------
def test_a_cancelled_run_keeps_its_rounds_and_is_never_reported_as_completed(tmp_path):
    wired = H.build_controller(tmp_path, budgets={"max_rounds": 6})
    controller = wired.controller
    source = wired.source
    original = controller._progress

    def progress(event, payload):
        if original is not None:
            original(event, payload)
        if event == "round_end" and int(payload.get("round") or 0) >= 2:
            controller.cancel()

    controller._progress = progress
    outcome = controller.run(probe_model=False, probe_embeddings=False)

    assert outcome.manifest.completion_state == STATE_CANCELLED
    assert outcome.completed_normally is False
    assert outcome.stop_reason == "cancelled_by_user"
    assert outcome.manifest.completion_state in INCOMPLETE_STATES
    assert any("did NOT complete normally" in n for n in outcome.manifest.notes)

    rounds = sorted((outcome.run_dir.root / "rounds").iterdir())
    assert len(rounds) == 2, "every completed round's artifacts survive a cancellation"
    for directory in rounds:
        assert (directory / "checkpoint.json").exists()
    report = (outcome.run_dir.root / "report.md").read_text(encoding="utf-8")
    assert "did not complete normally" in report
    assert source.search_calls > 0


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
def test_an_interrupted_run_resumes_from_its_checkpoint(tmp_path):
    first = H.build_controller(tmp_path, budgets={"max_rounds": 2})
    outcome = first.controller.run(probe_model=False, probe_embeddings=False)
    assert outcome.manifest.completion_state == STATE_STOPPED
    assert outcome.manifest.rounds_completed == 2
    reactions_before = len(outcome.state.graph.reactions)
    papers_before = len(outcome.state.papers)
    calls_before = outcome.ledger.usage.llm_calls
    assert reactions_before > 0

    from t2pw.mining.controller import MiningController

    source = H.ScriptedPaperSource()
    model = H.ScriptedModel()

    def progress(event, payload):
        if event == "round_start":
            source.round = int(payload.get("round") or 1)

    resumed = MiningController.resume(
        outcome.run_dir.root,
        source=source,
        chat_fn=model,
        progress=progress,
        enabled_override=True,
        overrides={"max_rounds": 5},
    )
    resumed._index = H.offline_index(resumed.config, tmp_path / "resume_index")
    for pid, record in resumed.state.papers.items():
        text = resumed.state.paper_texts.get(pid, "")
        if text and not record.exclusion_reasons:
            from t2pw.mining.acquisition import LoadedPaper

            resumed.index.add_paper(LoadedPaper(record=record, text=text))

    # The restored state is the state that was saved.
    assert len(resumed.state.graph.reactions) == reactions_before
    assert len(resumed.state.papers) == papers_before
    assert resumed.ledger.usage.llm_calls == calls_before, "budgets stay cumulative"
    assert resumed.manifest.resumed_from_round == 2
    assert any("resumed from round 2" in n for n in resumed.manifest.notes)

    second = resumed.continue_run()
    assert second.manifest.rounds_completed > 2
    assert len(second.state.graph.reactions) >= reactions_before
    assert second.run_dir.root == outcome.run_dir.root, "the same run directory is extended"
    assert len(sorted((second.run_dir.root / "rounds").iterdir())) > 2


def test_state_round_trips_without_losing_the_graph(mined):
    from t2pw.mining.controller import MiningState

    payload = mined.outcome.state.to_dict()
    restored = MiningState.from_dict(payload)
    assert restored.graph.graph_hash() == mined.outcome.state.graph.graph_hash()
    assert len(restored.store.claims) == len(mined.outcome.state.store.claims)
    assert len(restored.evidence) == len(mined.outcome.state.evidence)
    assert len(restored.papers) == len(mined.outcome.state.papers)
    assert len(restored.store.conflicts) == len(mined.outcome.state.store.conflicts)
    # Decision history survives the round trip.
    assert len(restored.store.decisions()) == len(mined.outcome.state.store.decisions())


# ---------------------------------------------------------------------------
# Preflight failure
# ---------------------------------------------------------------------------
def test_a_failed_preflight_stops_before_any_model_call(tmp_path, monkeypatch):
    wired = H.build_controller(tmp_path)
    controller = wired.controller
    # Force a failure the run must respect: an unwritable outputs directory is
    # simulated by pointing the check at a path that cannot be created.
    monkeypatch.setattr(
        controller.config, "runs_dir", "\x00invalid", raising=False
    )
    outcome = controller.run(probe_model=False, probe_embeddings=False)
    assert outcome.manifest.completion_state == STATE_PREFLIGHT_FAILED
    assert outcome.stop_reason == "preflight_failed"
    assert wired.model.calls == [], "no model call may happen after a failed preflight"


def test_preflight_reports_the_policies_and_never_echoes_a_key(tmp_path):
    from t2pw.mining.preflight import run_preflight

    config = H.make_config(tmp_path)
    report = run_preflight(
        config, probe_model=False, probe_embeddings=False,
        enabled_override=True, model_client_injected=True,
    )
    names = {c.name for c in report.checks}
    assert "feature.flag" in names
    assert "policy.database_validation" in names
    assert "policy.pwml" in names
    assert "budgets.selected" in names
    rendered = report.render()
    assert "DEFERRED" in rendered
    assert "PWML exporter is not invoked" in rendered
    assert "sk-" not in rendered


# ---------------------------------------------------------------------------
# Regeneration
# ---------------------------------------------------------------------------
def test_report_and_views_regenerate_from_artifacts_alone(mined):
    from t2pw.mining.regenerate import regenerate

    run_dir = RunDirectory.open(mined.outcome.run_dir.root)
    before = (run_dir.path("report.md")).read_text(encoding="utf-8")
    written = regenerate(run_dir)
    assert len(written) == 7
    after = (run_dir.path("report.md")).read_text(encoding="utf-8")
    assert "Pathway mining run" in after
    assert after.splitlines()[0] == before.splitlines()[0]
    view = json.loads(
        run_dir.path("graph_views", "reaction_network.json").read_text(encoding="utf-8")
    )
    assert view["view"] == "reaction_network"
    assert view["nodes"]
