"""Two stability corrections: the preflight probe budget and the per-round
reaction cap.

Both are G9 corrections of pre-existing observable behaviour, not new features.
Each test below fails behaviourally at base ``9c085967`` and passes at the tip.
Nothing here imports a symbol the base lacks, so the base failure is an
assertion, never an ImportError. No live call: the model is scripted.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import helpers_mining as H  # noqa: E402
from t2pw.mining.artifacts import STATE_STOPPED  # noqa: E402
from t2pw.mining.preflight import FAIL, OK, run_preflight  # noqa: E402

#: Output tokens the scripted reasoning model spends thinking before it writes
#: any answer text -- the DeepSeek behaviour that made the old 8-token probe
#: fail a working model.
REASONING_TOKENS = 40


def _probe_with_reasoning_model(tmp_path, monkeypatch):
    """Run the model preflight against a scripted reasoning model; no network."""
    calls = []

    def chat_detailed(messages, **kwargs):  # noqa: ARG001
        calls.append(kwargs)
        if int(kwargs.get("max_tokens") or 0) <= REASONING_TOKENS:
            return SimpleNamespace(text="", diagnostics=SimpleNamespace(finish_reason="length"))
        return SimpleNamespace(text="ok", diagnostics=SimpleNamespace(finish_reason="stop"))

    # A stand-in module, so the real client (which loads .env and builds a
    # provider client at import) is never imported by this test.
    fake = types.ModuleType("t2pw.llm.client")
    fake.chat_detailed = chat_detailed
    monkeypatch.setitem(sys.modules, "t2pw.llm.client", fake)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder-not-a-key")

    config = H.make_config(tmp_path, offline=False)
    report = run_preflight(config, probe_model=True, probe_embeddings=False,
                           enabled_override=True)
    check = next(c for c in report.checks if c.name == "model.reachable")
    return report, check, calls


def test_the_preflight_probe_budget_lets_a_reasoning_model_answer(tmp_path, monkeypatch):
    """At base the probe asked for 8 tokens, the reasoning model spent them all
    thinking, and preflight FAILED a model that works."""
    monkeypatch.delenv("PATHWAY_MINING_PROBE_MAX_TOKENS", raising=False)
    report, check, calls = _probe_with_reasoning_model(tmp_path, monkeypatch)
    assert check.status == OK, check.detail
    assert "model.reachable" not in [c.name for c in report.failures]
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 64


def test_an_exhausted_probe_budget_is_named_and_the_knob_is_honoured(tmp_path, monkeypatch):
    """At base the override was ignored (the probe always sent 8) and the failure
    read like an outage, naming neither the budget nor the knob."""
    monkeypatch.setenv("PATHWAY_MINING_PROBE_MAX_TOKENS", "16")
    report, check, calls = _probe_with_reasoning_model(tmp_path, monkeypatch)
    assert check.status == FAIL
    assert report.ok is False
    assert "PATHWAY_MINING_PROBE_MAX_TOKENS" in check.detail, check.detail
    assert "finish_reason=length" in check.detail
    assert calls[0]["max_tokens"] == 16


def test_max_new_reactions_per_round_stops_the_run_and_keeps_the_round(tmp_path):
    """At base the cap was declared and serialized but never checked: the run
    went on for more rounds and stopped for some other reason."""
    mined = H.run_fixture(tmp_path, budgets={"max_new_reactions_per_round": 1})
    outcome = mined.outcome

    first = outcome.state.deltas[0]
    new_ids = list(first.added_reaction_ids) + list(first.added_candidate_reaction_ids)
    assert new_ids, "fixture round 1 must add a reaction for this test to mean anything"

    assert outcome.stop_reason == "max_new_reactions_per_round_reached"
    assert outcome.manifest.completion_state == STATE_STOPPED
    assert outcome.manifest.rounds_completed == 1
    assert len(outcome.state.deltas) == 1
    # The capped round is complete: nothing it found is dropped or deferred.
    assert set(new_ids) <= set(outcome.state.graph.reactions)
    assert outcome.run_dir.path("rounds", "round_001", "checkpoint.json").is_file()

    on_disk = json.loads(outcome.run_dir.path("run_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["stop_reason"] == "max_new_reactions_per_round_reached"
    assert f"{len(new_ids)} >= limit 1" in on_disk["stop_detail"]
