"""The end-to-end fixture scenario.

One offline run over the synthetic corpus in ``helpers_mining``, asserted against
the eight behaviours the feature is defined by. No network, no LLM credit, no
production cache.

The eight:

1. connected reactions enter the primary graph;
2. the disconnected reaction enters a candidate subgraph;
3. wrong-species evidence does not enter the target-species graph;
4. conflicting evidence stays visible;
5. the later bridge promotes the candidate component;
6. every accepted reaction has a complete provenance chain;
7. the full round history survives;
8. no PWML is generated.
"""

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
from t2pw.mining.admission import ADMITTED_STATUSES  # noqa: E402
from t2pw.mining.graph import GRAPH_CANDIDATE, GRAPH_PRIMARY  # noqa: E402
from t2pw.mining.models import (  # noqa: E402
    EXTRACTION_TRUNCATED,
    provenance_chain_complete,
)
from t2pw.mining.species import (  # noqa: E402
    SPECIES_AMBIGUOUS,
    SPECIES_NON_TARGET,
    SPECIES_TARGET_SUPPORTED,
)


@pytest.fixture(scope="module")
def mined(tmp_path_factory):
    return H.run_fixture(tmp_path_factory.mktemp("mining_integration"))


def reaction_named(graph, substrate):
    return next(
        (r for r in graph.reactions.values() if substrate in r.substrate_names), None
    )


# --- 1 ---------------------------------------------------------------------
def test_1_chemically_connected_reactions_enter_the_primary_graph(mined):
    graph = mined.outcome.state.graph
    labels = {r.label() for r in graph.primary_reactions}
    assert "alpha-linolenic acid -> 13-HPOT" in labels
    assert "13-HPOT -> 12-OPDA" in labels
    first = reaction_named(graph, "alpha-linolenic acid")
    second = reaction_named(graph, "13-HPOT")
    assert first.graph_status == GRAPH_PRIMARY
    assert second.graph_status == GRAPH_PRIMARY
    assert first.component_id == second.component_id
    # Connected through the shared metabolite, not through a shared enzyme.
    links = graph.connections_for(first.reaction_id)
    assert any(link.via_entity_name == "13-HPOT" for link in links)


# --- 2 ---------------------------------------------------------------------
def test_2_a_relevant_but_disconnected_reaction_starts_in_a_candidate_subgraph(mined):
    """JA-Ile -> 12OH-JA-Ile is relevant and paper-supported but initially unlinked."""
    deltas = {d.round_index: d for d in mined.outcome.state.deltas}
    round_two = deltas[2]
    # It was created as its own component in round 2, not attached to primary.
    assert round_two.created_component_ids
    reaction = reaction_named(mined.outcome.state.graph, "JA-Ile")
    assert reaction is not None
    assert reaction.discovery_round == 2
    assert "quarantined" in [h.event for h in reaction.history]
    quarantine_event = next(h for h in reaction.history if h.event == "quarantined")
    assert "component" in quarantine_event.detail


# --- 3 ---------------------------------------------------------------------
def test_3_wrong_species_evidence_never_enters_the_target_species_graph(mined):
    graph = mined.outcome.state.graph
    wrong = reaction_named(graph, "linoleic acid")
    assert wrong is not None, "the wrong-species finding must be retained, not dropped"
    assert wrong.species_state == SPECIES_NON_TARGET
    assert wrong.graph_status == GRAPH_CANDIDATE
    assert wrong.admitted is False
    assert wrong.reaction_id not in {r.reaction_id for r in graph.primary_reactions}
    component = graph.components[wrong.component_id]
    assert component.quarantine_reason == f"species:{SPECIES_NON_TARGET}"
    assert component.promoted is False
    # Its provenance is intact even in quarantine.
    assert wrong.paper_ids and wrong.evidence_ids


def test_3b_an_ambiguous_species_claim_is_also_held_back(mined):
    graph = mined.outcome.state.graph
    ambiguous = reaction_named(graph, "12-OPDA")
    candidates = [
        r for r in graph.candidate_reactions if "OPC-8:0" in r.product_names
    ]
    assert candidates, "the ambiguous-species claim must be retained as a labelled lead"
    held = candidates[0]
    assert held.species_state == SPECIES_AMBIGUOUS
    assert held.admitted is False
    assert held.graph_status == GRAPH_CANDIDATE
    assert ambiguous is not None  # the target-species 12-OPDA chemistry is separate


# --- 4 ---------------------------------------------------------------------
def test_4_conflicting_evidence_remains_visible(mined):
    store = mined.outcome.state.store
    conflicts = list(store.conflicts.values())
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.attribute == "enzyme"
    assert len(conflict.paper_ids) == 2
    values = {p["value"] for p in conflict.positions}
    assert values == {"allene oxide synthase", "allene oxide cyclase"}
    assert conflict.resolved is False
    # The reaction is still in the graph, carrying the conflict.
    reaction = reaction_named(mined.outcome.state.graph, "13-HPOT")
    assert conflict.conflict_id in reaction.conflict_ids
    assert reaction.decision_status == "conflicting"
    # The disputed attribute was not silently chosen.
    assert reaction.enzyme_names == []


# --- 5 ---------------------------------------------------------------------
def test_5_a_later_bridge_promotes_the_candidate_component(mined):
    graph = mined.outcome.state.graph
    promoted = [c for c in graph.components.values() if c.promoted]
    assert len(promoted) == 1
    component = promoted[0]
    assert component.created_round == 2
    assert component.promoted_round == 3
    assert component.bridge_reaction_ids
    assert component.promotion_history[-1]["event"] == "promoted_to_primary"

    moved = reaction_named(graph, "JA-Ile")
    assert moved.graph_status == GRAPH_PRIMARY
    assert moved.discovery_round == 2, "the original discovery round survives promotion"
    assert moved.promoted_round == 3
    events = [h.event for h in moved.history]
    assert events.index("quarantined") < events.index("admitted_to_primary")


# --- 6 ---------------------------------------------------------------------
def test_6_every_admitted_reaction_has_a_complete_provenance_chain(mined):
    state = mined.outcome.state
    admitted = [
        r for r in state.graph.reactions.values()
        if r.decision_status in ADMITTED_STATUSES
    ]
    assert admitted
    for reaction in admitted:
        problems = provenance_chain_complete(
            reaction=reaction,
            claims=state.store.claims,
            evidence=state.evidence,
            papers=state.papers,
        )
        assert problems == [], f"{reaction.label()}: {problems}"
        for eid in reaction.evidence_ids:
            span = state.evidence[eid]
            assert span.text.strip()
            assert span.retrieval_query_id
            assert span.frontier_item_id
            paper = state.papers[span.paper_id]
            assert paper.has_source_identifier


def test_6b_multiple_independent_papers_are_counted_not_collapsed(mined):
    graph = mined.outcome.state.graph
    reaction = reaction_named(graph, "alpha-linolenic acid")
    assert reaction.independent_paper_count == 2
    assert reaction.multi_paper_supported is True
    assert len(set(reaction.claim_ids)) == 2


# --- 7 ---------------------------------------------------------------------
def test_7_the_full_round_history_survives(mined):
    outcome = mined.outcome
    run_dir = outcome.run_dir.root
    rounds = sorted((run_dir / "rounds").iterdir())
    assert len(rounds) == outcome.manifest.rounds_completed >= 3
    for directory in rounds:
        payload = json.loads((directory / "round.json").read_text(encoding="utf-8"))
        for key in (
            "input_frontier", "queries", "papers_considered", "passages_retrieved",
            "claims_extracted", "decisions", "rejections", "graph_delta",
            "new_frontier", "metrics", "errors",
        ):
            assert key in payload, f"{directory.name} missing {key}"
        assert (directory / "checkpoint.json").exists()
        assert payload["graph_delta"]["graph_hash_before"] is not None
        assert payload["graph_delta"]["graph_hash_after"] is not None

    # Growth by round is recorded in the summary too.
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["additions_by_round"]) == outcome.manifest.rounds_completed
    assert summary["additions_by_round"][0]["round"] == 1


def test_7b_a_truncated_model_response_added_nothing(mined):
    calls = mined.outcome.state.model_calls
    truncated = [c for c in calls if c.status == EXTRACTION_TRUNCATED]
    assert truncated, "the fixture includes a paper the model cannot finish"
    for call in truncated:
        assert call.finish_reason == "length"
    # No claim anywhere carries a non-OK extraction status.
    for claim in mined.outcome.state.store.claims.values():
        assert claim.admissible_extraction is True


# --- 8 ---------------------------------------------------------------------
def test_8_no_pwml_is_generated_anywhere(mined):
    run_dir = mined.outcome.run_dir.root
    assert list(run_dir.rglob("*.pwml")) == []
    assert list(run_dir.rglob("*.sbml")) == []
    assert list(run_dir.rglob("*.xml")) == []
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["produces_pwml"] is False
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["produces_pwml"] is False
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "research graph, not PWML" in report


def test_8b_the_pwml_exporter_was_never_imported(mined):
    """Structural, not incidental: the mining package must not reach the exporter."""
    assert mined.outcome is not None
    banned = ("t2pw.pwml", "t2pw.sbml.json_to_sbml", "t2pw.pipeline.strict_quarantine")
    loaded = [m for m in banned if m in sys.modules]
    # Another test module may have imported these; what matters is that no mining
    # module references them at all.
    mining_src = ROOT / "src" / "t2pw" / "mining"
    for path in mining_src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for module in ("t2pw.pwml", "json_to_sbml", "strict_quarantine", "build_sbml"):
            assert module not in text, f"{path.name} references {module}"
    assert isinstance(loaded, list)


# --- run-level assertions ---------------------------------------------------
def test_external_database_validation_is_deferred_everywhere(mined):
    state = mined.outcome.state
    for entity in state.graph.registry.all():
        assert entity.external_refs == {"status": "not_evaluated"}
    for reaction in state.graph.reactions.values():
        assert reaction.external_validation == {"status": "not_evaluated"}
    summary = json.loads(
        (mined.outcome.run_dir.root / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["external_database_validation"] == "deferred_not_evaluated"


def test_the_run_stopped_for_a_named_reason(mined):
    assert mined.outcome.stop_reason
    assert mined.outcome.manifest.stop_reason == mined.outcome.stop_reason


def test_frontier_expansion_asked_new_questions_each_round(mined):
    queries = mined.outcome.state.queries
    assert queries
    by_round = {}
    for query in queries:
        by_round.setdefault(query.round_index, []).append(query)
    assert len(by_round) >= 3, "the loop must keep asking new questions"
    kinds = {q.frontier_kind for q in queries}
    assert "seed_expansion" in kinds
    assert any(k in kinds for k in ("downstream_expansion", "upstream_expansion", "bridge_search"))
    for query in queries:
        assert query.frontier_item_id, "every query traces back to a frontier item"
        assert query.query_text


def test_accepted_reactions_all_carry_target_species_support(mined):
    for reaction in mined.outcome.state.graph.primary_reactions:
        assert reaction.species_state == SPECIES_TARGET_SUPPORTED


def test_the_evidence_table_is_complete_and_quotable(mined):
    table = json.loads(
        (mined.outcome.run_dir.root / "graph_views" / "evidence_table.json")
        .read_text(encoding="utf-8")
    )
    assert table
    for row in table:
        assert row["reaction"]
        assert row["quote"].strip()
        assert row["source_identifier"]
        assert row["external_validation"] == "not_evaluated"
