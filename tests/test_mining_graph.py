"""Graph semantics: what may and may not connect two reactions.

The negative tests here are the important ones. Anything that lets a shared
enzyme, gene, pathway name, paper or vocabulary create a biochemical edge turns
the research graph into a co-occurrence map, which is the failure this design
exists to prevent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.mining.graph import (  # noqa: E402
    CONNECTION_BRANCH,
    CONNECTION_CONVERGENT,
    CONNECTION_SEQUENTIAL,
    GRAPH_CANDIDATE,
    GRAPH_PRIMARY,
    ResearchGraph,
    ResearchReaction,
    connection_between,
    load_starting_graph,
)
from t2pw.mining.ids import reaction_id as make_reaction_id  # noqa: E402
from t2pw.mining.models import reaction_identity  # noqa: E402
from t2pw.mining.normalize import EntityRegistry  # noqa: E402


def make_reaction(
    registry: EntityRegistry,
    substrates,
    products,
    *,
    enzymes=(),
    pathway=(),
    papers=("PMC1",),
    reversible=None,
    admitted=True,
    species="arabidopsis thaliana",
):
    subs = [registry.upsert(s, declared_type="metabolite") for s in substrates]
    prods = [registry.upsert(p, declared_type="metabolite") for p in products]
    enzs = [registry.upsert(e, declared_type="enzyme") for e in enzymes]
    identity = reaction_identity(substrates, products, species=species)
    return ResearchReaction(
        reaction_id=make_reaction_id(identity),
        claim_identity=identity,
        substrate_ids=[e.entity_id for e in subs],
        product_ids=[e.entity_id for e in prods],
        substrate_names=list(substrates),
        product_names=list(products),
        enzyme_names=list(enzymes),
        enzyme_entity_ids=[e.entity_id for e in enzs],
        pathway_names=list(pathway),
        paper_ids=list(papers),
        claim_ids=[f"cl_{identity[-8:]}"],
        evidence_ids=[f"ev_{identity[-8:]}"],
        reversible=reversible,
        admitted=admitted,
        decision_status="accepted" if admitted else "candidate",
        quarantine_reason="" if admitted else "species:non_target_species",
    )


# ---------------------------------------------------------------------------
# Positive connectivity
# ---------------------------------------------------------------------------
def test_product_to_substrate_is_a_sequential_connection():
    reg = EntityRegistry()
    a = make_reaction(reg, ["A"], ["B"])
    b = make_reaction(reg, ["B"], ["C"])
    link = connection_between(a, b, registry=reg)
    assert link is not None
    assert link.kind == CONNECTION_SEQUENTIAL
    assert link.via_entity_name == "B"


def test_shared_substrate_is_a_branch_point():
    reg = EntityRegistry()
    a = make_reaction(reg, ["B"], ["C"])
    b = make_reaction(reg, ["B"], ["D"])
    link = connection_between(a, b, registry=reg)
    assert link is not None and link.kind == CONNECTION_BRANCH


def test_shared_product_is_convergent():
    reg = EntityRegistry()
    a = make_reaction(reg, ["A"], ["X"])
    b = make_reaction(reg, ["B"], ["X"])
    link = connection_between(a, b, registry=reg)
    assert link is not None and link.kind == CONNECTION_CONVERGENT


def test_a_reversible_reaction_can_supply_either_side():
    reg = EntityRegistry()
    a = make_reaction(reg, ["A"], ["B"], reversible=True)
    b = make_reaction(reg, ["A"], ["Z"])
    link = connection_between(a, b, registry=reg)
    assert link is not None


# ---------------------------------------------------------------------------
# Negative connectivity -- the rules that matter
# ---------------------------------------------------------------------------
def test_a_shared_enzyme_does_not_connect_two_reactions():
    reg = EntityRegistry()
    a = make_reaction(reg, ["A"], ["B"], enzymes=["LOX2"])
    b = make_reaction(reg, ["P"], ["Q"], enzymes=["LOX2"])
    assert connection_between(a, b, registry=reg) is None


def test_a_shared_pathway_name_does_not_connect_two_reactions():
    reg = EntityRegistry()
    a = make_reaction(reg, ["A"], ["B"], pathway=["jasmonate biosynthesis"])
    b = make_reaction(reg, ["P"], ["Q"], pathway=["jasmonate biosynthesis"])
    assert connection_between(a, b, registry=reg) is None


def test_a_shared_paper_does_not_connect_two_reactions():
    reg = EntityRegistry()
    a = make_reaction(reg, ["A"], ["B"], papers=["PMC7"])
    b = make_reaction(reg, ["P"], ["Q"], papers=["PMC7"])
    assert connection_between(a, b, registry=reg) is None


def test_similar_vocabulary_does_not_connect_two_reactions():
    """Lexically close but chemically distinct names stay distinct."""
    reg = EntityRegistry()
    a = make_reaction(reg, ["linoleic acid"], ["13-HPOD"])
    b = make_reaction(reg, ["alpha-linolenic acid"], ["13-HPOT"])
    assert connection_between(a, b, registry=reg) is None


def test_an_enzyme_on_a_participant_list_still_cannot_bridge():
    """Defence in depth: even a mis-typed participant cannot be a chemical bridge."""
    reg = EntityRegistry()
    enzyme = reg.upsert("LOX2", declared_type="enzyme")
    a = make_reaction(reg, ["A"], ["B"])
    b = make_reaction(reg, ["P"], ["Q"])
    a.product_ids.append(enzyme.entity_id)
    b.substrate_ids.append(enzyme.entity_id)
    assert connection_between(a, b, registry=reg) is None


def test_a_shared_cofactor_does_not_connect_by_default():
    reg = EntityRegistry()
    a = make_reaction(reg, ["A", "ATP"], ["B", "ADP"])
    b = make_reaction(reg, ["P", "ATP"], ["Q", "ADP"])
    assert connection_between(a, b, registry=reg) is None
    assert connection_between(a, b, registry=reg, allow_cofactor_bridges=True) is not None


# ---------------------------------------------------------------------------
# Reaction representation
# ---------------------------------------------------------------------------
def test_a_multi_substrate_reaction_is_one_node_with_all_participants():
    reg = EntityRegistry()
    reaction = make_reaction(reg, ["jasmonic acid", "L-isoleucine", "ATP"], ["JA-Ile", "AMP"])
    assert len(reaction.substrate_ids) == 3
    assert len(reaction.product_ids) == 2
    assert reaction.label() == "jasmonic acid + L-isoleucine + ATP -> JA-Ile + AMP"
    # No pairwise decomposition anywhere: every participant hangs off ONE node.
    assert reaction.substrate_names == ["jasmonic acid", "L-isoleucine", "ATP"]
    assert reaction.product_names == ["JA-Ile", "AMP"]


# ---------------------------------------------------------------------------
# Placement, quarantine and promotion
# ---------------------------------------------------------------------------
def test_connected_admitted_reactions_land_in_one_primary_component():
    graph = ResearchGraph(topic="jasmonate")
    reg = graph.registry
    graph.add_reaction(make_reaction(reg, ["A"], ["B"]), round_index=1)
    graph.add_reaction(make_reaction(reg, ["B"], ["C"]), round_index=1)
    graph.select_primary_component(round_index=1, topic="jasmonate")
    assert len(graph.primary_reactions) == 2
    assert len(graph.candidate_components) == 0


def test_a_disconnected_relevant_reaction_is_quarantined_not_dropped():
    graph = ResearchGraph(topic="jasmonate")
    reg = graph.registry
    graph.add_reaction(make_reaction(reg, ["A"], ["B"]), round_index=1)
    graph.add_reaction(make_reaction(reg, ["B"], ["C"]), round_index=1)
    graph.select_primary_component(round_index=1)
    graph.add_reaction(make_reaction(reg, ["X"], ["Y"]), round_index=2)

    assert len(graph.candidate_components) == 1
    assert len(graph.candidate_reactions) == 1
    assert graph.candidate_reactions[0].graph_status == GRAPH_CANDIDATE
    # Nothing was discarded and its provenance survives.
    assert graph.candidate_reactions[0].paper_ids == ["PMC1"]


def test_a_chemical_bridge_promotes_a_candidate_component():
    graph = ResearchGraph(topic="jasmonate")
    reg = graph.registry
    graph.add_reaction(make_reaction(reg, ["A"], ["B"]), round_index=1)
    graph.select_primary_component(round_index=1)
    graph.add_reaction(make_reaction(reg, ["X"], ["Y"]), round_index=2)
    assert len(graph.candidate_components) == 1

    bridge = make_reaction(reg, ["B"], ["X"], papers=["PMC_BRIDGE"])
    graph.add_reaction(bridge, round_index=3)

    assert len(graph.candidate_components) == 0
    promoted = [c for c in graph.components.values() if c.promoted]
    assert len(promoted) == 1
    component = promoted[0]
    assert component.created_round == 2
    assert component.promoted_round == 3
    assert "PMC_BRIDGE" in component.bridge_paper_ids
    # The original discovery round survives promotion.
    moved = next(r for r in graph.primary_reactions if r.substrate_names == ["X"])
    assert moved.discovery_round == 2
    assert moved.promoted_round == 3
    events = [h.event for h in moved.history]
    assert "quarantined" in events and "admitted_to_primary" in events


def test_a_non_admitted_reaction_cannot_reach_primary_by_chemistry():
    """The species guardrail outranks connectivity.

    A wrong-species reaction that shares a metabolite with the primary graph is
    still wrong-species. Chemistry decides WHERE an admitted reaction sits, never
    WHETHER it is admitted.
    """
    graph = ResearchGraph(topic="jasmonate")
    reg = graph.registry
    graph.add_reaction(make_reaction(reg, ["A"], ["B"]), round_index=1)
    graph.select_primary_component(round_index=1)

    intruder = make_reaction(reg, ["B"], ["C"], admitted=False, species="zea mays")
    graph.add_reaction(intruder, round_index=2)

    assert intruder.graph_status == GRAPH_CANDIDATE
    assert intruder.reaction_id not in {r.reaction_id for r in graph.primary_reactions}
    component = graph.components[intruder.component_id]
    assert component.quarantine_reason == "species:non_target_species"
    assert component.promoted is False


def test_a_quarantine_component_is_never_promoted_by_a_bridge():
    graph = ResearchGraph(topic="jasmonate")
    reg = graph.registry
    graph.add_reaction(make_reaction(reg, ["A"], ["B"]), round_index=1)
    graph.select_primary_component(round_index=1)
    graph.add_reaction(
        make_reaction(reg, ["X"], ["Y"], admitted=False, species="zea mays"), round_index=2
    )
    graph.add_reaction(make_reaction(reg, ["B"], ["X"]), round_index=3)

    quarantined = [c for c in graph.components.values() if c.quarantine_reason]
    assert quarantined and all(not c.promoted for c in quarantined)


def test_reconsideration_lifts_a_reaction_out_of_quarantine():
    graph = ResearchGraph(topic="jasmonate")
    reg = graph.registry
    graph.add_reaction(make_reaction(reg, ["A"], ["B"]), round_index=1)
    graph.select_primary_component(round_index=1)

    held = make_reaction(reg, ["B"], ["C"], admitted=False)
    graph.add_reaction(held, round_index=2)
    assert held.graph_status == GRAPH_CANDIDATE

    better = make_reaction(reg, ["B"], ["C"], admitted=True, papers=["PMC_NEW"])
    graph.add_reaction(better, round_index=4)

    assert held.graph_status == GRAPH_PRIMARY
    assert held.admitted is True
    assert "PMC_NEW" in held.paper_ids
    assert [h.event for h in held.history].count("quarantined") == 1
    assert "quarantine_lifted" in [h.event for h in held.history]


# ---------------------------------------------------------------------------
# Primary-component selection
# ---------------------------------------------------------------------------
def test_primary_component_selection_is_deterministic_and_explained():
    graph = ResearchGraph(topic="jasmonate biosynthesis")
    reg = graph.registry
    # Component 1: two reactions, two papers, topic-relevant naming.
    graph.add_reaction(
        make_reaction(reg, ["12-OPDA"], ["jasmonic acid"], papers=["P1", "P2"],
                      pathway=["jasmonate biosynthesis"]),
        round_index=1,
    )
    graph.add_reaction(
        make_reaction(reg, ["jasmonic acid"], ["JA-Ile"], papers=["P2"],
                      pathway=["jasmonate biosynthesis"]),
        round_index=1,
    )
    # Component 2: one reaction, one paper, unrelated naming.
    graph.add_reaction(make_reaction(reg, ["glucose"], ["pyruvate"], papers=["P9"]), round_index=1)

    chosen = graph.select_primary_component(round_index=1, topic="jasmonate biosynthesis")
    assert chosen
    assert len(graph.primary_reactions) == 2
    assert len(graph.candidate_components) == 1
    assert "evidence-weighted topic relevance" in graph.primary_selection_rationale
    # No connection was fabricated to merge them.
    assert graph.candidate_reactions[0].substrate_names == ["glucose"]


def test_selection_is_stable_across_repeated_builds():
    def build():
        graph = ResearchGraph(topic="jasmonate biosynthesis")
        reg = graph.registry
        graph.add_reaction(make_reaction(reg, ["A"], ["B"], papers=["P1"]), round_index=1)
        graph.add_reaction(make_reaction(reg, ["X"], ["Y"], papers=["P2"]), round_index=1)
        graph.select_primary_component(round_index=1, topic="jasmonate biosynthesis")
        return graph.primary_component_id, graph.graph_hash()

    assert build() == build()


# ---------------------------------------------------------------------------
# Starting graph
# ---------------------------------------------------------------------------
def test_an_existing_pathway_is_adopted_read_only_without_pwml():
    payload = {
        "topic": "jasmonate biosynthesis",
        "species": "Arabidopsis thaliana",
        "reactions": [
            {
                "name": "step 1",
                "inputs": ["alpha-linolenic acid"],
                "outputs": ["13-HPOT"],
                "enzymes": [{"protein": "LOX2"}],
                "biological_state": "chloroplast",
            },
            {"name": "step 2", "inputs": ["13-HPOT"], "outputs": ["12-OPDA"]},
        ],
    }
    graph = load_starting_graph(payload)
    assert len(graph.reactions) == 2
    assert len(graph.primary_reactions) == 2
    reaction = graph.primary_reactions[0]
    assert reaction.decision_status == "starting_graph"
    assert reaction.external_validation == {"status": "not_evaluated"}
    assert "starting_graph_input" in [h.event for h in reaction.history]
    # Enzymes are annotations, not chemistry.
    assert all(
        eid not in reaction.substrate_ids + reaction.product_ids
        for eid in reaction.enzyme_entity_ids
    )


def test_graph_round_trips_through_json():
    graph = ResearchGraph(topic="t")
    reg = graph.registry
    graph.add_reaction(make_reaction(reg, ["A"], ["B"]), round_index=1)
    graph.add_reaction(make_reaction(reg, ["B"], ["C"]), round_index=1)
    graph.select_primary_component(round_index=1)
    restored = ResearchGraph.from_dict(graph.to_dict())
    assert restored.graph_hash() == graph.graph_hash()
    assert len(restored.primary_reactions) == len(graph.primary_reactions)
    assert restored.primary_component_id == graph.primary_component_id
