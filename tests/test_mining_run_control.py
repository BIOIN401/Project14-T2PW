"""ORCH-745 change set C: run control.

The pilot-003 audit's run-control defects: the per-round cap stopped the run on
quarantined junk (B1 / F-7), a round started after the discovery budget was spent
(A5), the frontier cap cut every target-species search (F-3), thin reasoning-free
answers passed as ``ok`` (B2), and the CLI never loaded ``.env`` (F9).

Every ``test_g9_745c_*`` node fails behaviourally at base ``f9556288`` (a wrong
value or a failed assertion, never an ImportError) and imports only what the base
has. A ``test_new_acceptance_745c_*`` node pins a new helper and may import it
inside the test. No network and no model: every source and model is scripted.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import helpers_mining as H  # noqa: E402
from t2pw.mining.controller import MiningController  # noqa: E402

_EVIDENCE = re.compile(r"\[evidence_id:\s*([A-Za-z0-9_]+)\]")


# ---------------------------------------------------------------------------
# B1 / F-7: max_new_reactions_per_round counts newly admitted reactions only
# ---------------------------------------------------------------------------
def _off_species_review(count):
    """A model reading a dense review of another organism: ``count`` distinct,
    fully cited Zea mays conversions per batch. Every one is a quarantined lead."""

    def chat(messages, **kwargs):  # noqa: ARG001
        user = "".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
        ev = _EVIDENCE.findall(user)[0]
        claims = [
            {
                "substrates": [{"name": f"maize lipid {k}", "evidence_ids": [ev]}],
                "products": [{"name": f"maize oxylipin {k}", "evidence_ids": [ev]}],
                "species": "Zea mays",
                "species_evidence_ids": [ev],
                "evidence_ids": [ev],
            }
            for k in range(count)
        ]
        return SimpleNamespace(text=json.dumps({"claims": claims}), diagnostics=None)

    return chat


def test_g9_745c_b1_quarantined_candidates_do_not_stop_the_run(tmp_path):
    """Pilot-003: round 1 added 0 admitted and 35 quarantined reactions, and
    35 >= 25 ended the run. At f9556288 this run stops after round 1 with
    max_new_reactions_per_round_reached."""
    config = H.make_config(tmp_path, budgets={"max_rounds": 3})
    assert config.budgets.max_new_reactions_per_round == 25
    controller = MiningController(
        config,
        source=H.ScriptedPaperSource({1: ["P5"]}),
        chat_fn=_off_species_review(26),
        index=H.offline_index(config, tmp_path),
        enabled_override=True,
    )
    outcome = controller.run(probe_model=False, probe_embeddings=False)

    first = outcome.state.deltas[0]
    assert first.added_reaction_ids == []
    assert len(first.added_candidate_reaction_ids) >= 25
    assert outcome.stop_reason != "max_new_reactions_per_round_reached"
    assert outcome.manifest.rounds_completed >= 2


def test_g9_745c_b1_the_fixture_round_counts_its_admitted_reactions_only(tmp_path):
    """Fixture round 1 admits 2 reactions and quarantines 1 (Zea mays). With the cap
    at 3, f9556288 counted 3 and stopped after round 1; only 2 were admitted."""
    mined = H.run_fixture(tmp_path, budgets={"max_new_reactions_per_round": 3})
    first = mined.outcome.state.deltas[0]
    assert (len(first.added_reaction_ids), len(first.added_candidate_reaction_ids)) == (2, 1)
    assert mined.outcome.stop_reason != "max_new_reactions_per_round_reached"
    assert mined.outcome.manifest.rounds_completed > 1


def test_745c_b1_a_round_admitting_the_cap_still_stops_the_run(tmp_path):
    """The cap stays a stop condition: 2 admitted reactions meet a cap of 2."""
    mined = H.run_fixture(tmp_path, budgets={"max_new_reactions_per_round": 2})
    outcome = mined.outcome
    assert outcome.stop_reason == "max_new_reactions_per_round_reached"
    assert outcome.manifest.rounds_completed == 1
    assert "newly admitted reactions in round 1: 2 >= limit 2" in outcome.manifest.stop_detail


# ---------------------------------------------------------------------------
# A5: no phantom round after the discovery budget is spent
# ---------------------------------------------------------------------------
def test_g9_745c_a5_a_spent_discovery_budget_starts_no_further_round(tmp_path):
    """The seed's query takes the whole 4-paper discovery budget in round 1, and the
    topic item records max_papers_discovered_reached mid-round. At f9556288 a
    round 2 still started; it could not search, and a round 3 followed."""
    mined = H.run_fixture(tmp_path, budgets={"max_papers_discovered": 4})
    outcome = mined.outcome
    assert outcome.ledger.usage.papers_discovered == 4
    assert outcome.stop_reason == "max_papers_discovered_reached"
    assert outcome.ledger.usage.rounds == 1
    assert outcome.manifest.rounds_completed == 1
    assert outcome.run_dir.completed_rounds() == [1]
    assert not outcome.run_dir.path("rounds", "round_002").exists()


def test_g9_745c_a5_check_before_round_stops_on_the_discovery_budget():
    """The budget can be spent exactly by a round's last query, so no mid-round check
    records it. At f9556288 check_before_round did not look at it (returned None)."""
    from t2pw.mining.budget import BudgetLedger
    from t2pw.mining.config import MiningBudgets

    ledger = BudgetLedger(MiningBudgets(max_papers_discovered=3))
    ledger.record_papers_discovered(3)
    assert ledger.check_before_round() == "max_papers_discovered_reached"
    assert ledger.stop_detail == "papers discovered: 3 >= limit 3"


def test_g9_745c_a5_a_stop_recorded_mid_round_forbids_the_next_round():
    """A stop recorded mid-round is final. At f9556288 check_before_round ignored it
    (returned None) whenever the counters it re-checks were under their caps."""
    from t2pw.mining.budget import BudgetLedger
    from t2pw.mining.config import MiningBudgets

    ledger = BudgetLedger(MiningBudgets())
    ledger.stop("max_papers_discovered_reached", "recorded mid-round")
    assert ledger.check_before_round() == "max_papers_discovered_reached"
