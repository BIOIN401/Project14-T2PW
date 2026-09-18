"""Streamlit gating and visualization serialization.

Streamlit is exercised through a small stub rather than ``AppTest``: the
project's AppTest-driven files are slow enough to dominate a focused run, and
what matters here is the gate and the payloads, both of which are pure functions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import helpers_mining as H  # noqa: E402
from t2pw.mining.flags import FEATURE_ENV_VAR  # noqa: E402
from t2pw.mining.ui.entry import MODE_KEY, mining_available, render_entry  # noqa: E402
from t2pw.mining.ui.network import render_network_html  # noqa: E402
from t2pw.mining.viz import (  # noqa: E402
    EDGE_CATALYZES,
    EDGE_PRODUCT,
    EDGE_SUBSTRATE,
    NODE_ENZYME,
    NODE_METABOLITE,
    NODE_REACTION,
    evidence_table,
    filter_reaction_network,
    provenance_network,
    provenance_network_dot,
    reaction_network,
    reaction_network_dot,
)


class StubSidebar:
    def __init__(self, choice_index: int = 0) -> None:
        self.choice_index = choice_index
        self.calls: List[str] = []

    def radio(self, label, options, index=0, key=None, help=None):  # noqa: A002
        self.calls.append(label)
        return options[self.choice_index]


class StubStreamlit:
    """The minimum surface ``render_entry`` touches."""

    def __init__(self, choice_index: int = 0) -> None:
        self.session_state: Dict[str, Any] = {}
        self.sidebar = StubSidebar(choice_index)
        self.rendered: List[str] = []

    def title(self, text):
        self.rendered.append(f"title:{text}")

    def caption(self, text):
        self.rendered.append(f"caption:{text}")


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------
def test_the_entry_hook_is_a_noop_when_the_feature_is_off(monkeypatch):
    monkeypatch.delenv(FEATURE_ENV_VAR, raising=False)
    monkeypatch.setattr("t2pw.config._ENV_LOADED", True, raising=False)
    st = StubStreamlit()
    assert mining_available() is False
    assert render_entry(st) is False
    assert st.sidebar.calls == [], "no control is added to the sidebar when disabled"
    assert st.session_state == {}, "no session state is touched when disabled"
    assert st.rendered == []


def test_the_entry_hook_offers_the_mode_but_does_not_take_over_by_default(monkeypatch):
    monkeypatch.setenv(FEATURE_ENV_VAR, "true")
    st = StubStreamlit(choice_index=0)  # the user stays on the PWML pipeline
    assert render_entry(st) is False
    assert st.sidebar.calls == ["Mode"]
    assert st.session_state[MODE_KEY] is False
    assert st.rendered == [], "the mining page did not render"


def test_selecting_the_mining_mode_takes_over_the_render(monkeypatch):
    monkeypatch.setenv(FEATURE_ENV_VAR, "true")
    st = StubStreamlit(choice_index=1)
    calls = {}

    def fake_render_page(target):
        calls["rendered"] = target

    monkeypatch.setattr("t2pw.mining.ui.page.render_page", fake_render_page)
    assert render_entry(st) is True
    assert st.session_state[MODE_KEY] is True
    assert calls["rendered"] is st


def test_the_main_app_hook_is_guarded_and_additive():
    """The production Streamlit app's integration must be a no-op by default."""
    source = (ROOT / "src" / "t2pw" / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "from t2pw.mining.ui.entry import render_entry" in source
    assert "if _render_mining_entry(st):" in source
    assert "except ImportError" in source
    # The hook must not sit before set_page_config, and must not import the
    # controller, the model client or the vector store at module scope.
    assert source.index("st.set_page_config") < source.index("_render_mining_entry")
    assert "MiningController" not in source
    assert "t2pw.mining.controller" not in source


# ---------------------------------------------------------------------------
# View serialization
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mined(tmp_path_factory):
    return H.run_fixture(tmp_path_factory.mktemp("mining_ui"))


@pytest.fixture(scope="module")
def views(mined):
    state = mined.outcome.state
    conflicts = dict(state.store.conflicts)
    return {
        "reaction": reaction_network(state.graph, papers=state.papers, conflicts=conflicts),
        "provenance": provenance_network(
            graph=state.graph, papers=state.papers, claims=state.store.claims,
            evidence=state.evidence, conflicts=conflicts,
        ),
        "table": evidence_table(
            graph=state.graph, papers=state.papers, claims=state.store.claims,
            evidence=state.evidence, decisions=state.store.decisions(),
        ),
    }


def test_the_reaction_view_keeps_reactions_as_nodes(views):
    view = views["reaction"]
    reactions = [n for n in view["nodes"] if n["type"] == NODE_REACTION]
    assert reactions
    multi = next(
        (n for n in reactions if len(n["label"].split("->")[0].split("+")) > 1), None
    )
    assert multi is not None, "the fixture contains a multi-substrate reaction"
    incident = [
        e for e in view["edges"]
        if e["source"] == multi["id"] or e["target"] == multi["id"]
    ]
    assert len(incident) >= 3
    # No metabolite-to-metabolite edge exists anywhere.
    metabolites = {n["id"] for n in view["nodes"] if n["type"] == NODE_METABOLITE}
    for edge in view["edges"]:
        assert not (edge["source"] in metabolites and edge["target"] in metabolites)


def test_enzyme_edges_are_marked_as_annotations_not_chemistry(views):
    view = views["reaction"]
    enzymes = {n["id"] for n in view["nodes"] if n["type"] == NODE_ENZYME}
    assert enzymes
    for edge in view["edges"]:
        if edge["type"] == EDGE_CATALYZES:
            assert edge["chemical"] is False
            assert edge["annotation"] is True
        if edge["type"] in (EDGE_SUBSTRATE, EDGE_PRODUCT):
            assert edge["chemical"] is True
            assert edge["source"] not in enzymes or edge["type"] == EDGE_SUBSTRATE
    # An enzyme is never on both sides of two reactions as a chemical participant.
    chemical_endpoints = {
        e["source"] for e in view["edges"] if e.get("chemical")
    } | {e["target"] for e in view["edges"] if e.get("chemical")}
    assert not (enzymes & chemical_endpoints)


def test_the_reaction_view_exposes_everything_a_selection_should_reveal(views):
    reaction = next(n for n in views["reaction"]["nodes"] if n["type"] == NODE_REACTION)
    for key in (
        "label", "species", "enzymes", "compartment", "discovery_round", "graph_status",
        "supporting_papers", "evidence_ids", "conflicts", "history", "direction",
        "reversible", "promoted_round", "decision_status", "decision_reason",
        "external_validation", "multi_paper_supported",
    ):
        assert key in reaction, f"missing {key}"


def test_primary_and_candidate_are_distinguishable_in_the_payload(views):
    view = views["reaction"]
    statuses = {
        n["graph_status"] for n in view["nodes"] if n["type"] == NODE_REACTION
    }
    assert "primary" in statuses and "candidate" in statuses
    assert view["components"]
    assert any(c["is_primary"] for c in view["components"].values())


def test_the_round_slider_reconstructs_the_network_as_it_stood(views):
    view = views["reaction"]
    at_one = filter_reaction_network(view, max_round=1)
    at_three = filter_reaction_network(view, max_round=3)
    count = lambda v: sum(1 for n in v["nodes"] if n["type"] == NODE_REACTION)  # noqa: E731
    assert count(at_one) < count(at_three)
    assert all(
        n["discovery_round"] <= 1
        for n in at_one["nodes"] if n["type"] == NODE_REACTION
    )


def test_a_promoted_reaction_reads_as_candidate_before_its_promotion_round(views):
    view = views["reaction"]
    # A reaction that was DISCOVERED before it was promoted -- the quarantine
    # story. A reaction discovered and promoted in the same round has no earlier
    # candidate state to show.
    promoted = [
        n for n in view["nodes"]
        if n["type"] == NODE_REACTION
        and n.get("promoted_round", -1) > n.get("discovery_round", 0)
    ]
    assert promoted
    target = promoted[0]
    earlier = filter_reaction_network(
        view, max_round=target["promoted_round"] - 1, graph_status="candidate"
    )
    assert target["id"] in {n["id"] for n in earlier["nodes"]}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"graph_status": "primary"},
        {"only_multi_paper": True},
        {"species_state": "non_target_species"},
    ],
)
def test_reaction_filters_narrow_the_payload(views, kwargs):
    view = views["reaction"]
    filtered = filter_reaction_network(view, **kwargs)
    assert filtered["filtered"] is True
    assert len(filtered["nodes"]) <= len(view["nodes"])


def test_the_provenance_view_states_that_its_edges_are_not_chemistry(views):
    view = views["provenance"]
    assert view["provenance_edges_are_not_chemical_connectivity"] is True
    assert "NOT chemically connected" in view["note"]
    for edge in view["edges"]:
        assert edge["chemical"] is False


def test_the_provenance_view_answers_the_questions_it_is_for(views):
    view = views["provenance"]
    papers = [n for n in view["nodes"] if n["type"] == "paper"]
    assert papers
    for paper in papers:
        for key in (
            "evidence_type", "discovered_round", "discovered_via_query_id",
            "inclusion_reasons", "exclusion_reasons", "excluded",
            "is_bridge_paper", "supports_candidate_subgraph",
        ):
            assert key in paper
    assert view["first_support"], "which paper introduced a reaction first"
    assert any(p["is_bridge_paper"] for p in papers), "papers involved in a bridge"


def test_excluded_papers_are_filterable_rather_than_mixed_in(views):
    view = views["provenance"]
    excluded = [n for n in view["nodes"] if n["type"] == "paper" and n["excluded"]]
    for paper in excluded:
        assert paper["exclusion_reasons"]


def test_the_evidence_table_is_auditable_row_by_row(views):
    rows = views["table"]
    assert rows
    for row in rows:
        for key in (
            "reaction_id", "reaction", "graph_status", "discovery_round", "species_state",
            "decision_status", "decision_reason", "claim_id", "paper_id",
            "source_identifier", "evidence_id", "quote", "retrieval_query_id",
            "frontier_item_id", "external_validation",
        ):
            assert key in row
        assert row["quote"].strip()


def test_dot_exports_render_and_carry_the_caveat(views):
    reaction_dot = reaction_network_dot(views["reaction"])
    assert reaction_dot.startswith("digraph reaction_network {")
    assert reaction_dot.rstrip().endswith("}")
    assert "enzyme annotation" in reaction_dot
    assert 'style=dashed' in reaction_dot

    provenance_dot = provenance_network_dot(views["provenance"])
    assert provenance_dot.startswith("digraph provenance_network {")
    assert "NOT chemical connectivity" in provenance_dot


def test_the_html_renderer_is_self_contained_and_offline(views):
    html = render_network_html(views["reaction"], uid="t")
    assert "<svg" in html and "<script>" in html
    # No CDN, no remote asset, no network call: the page must work offline. The
    # one http:// in the file is the SVG XML namespace, which is an identifier
    # and is never dereferenced.
    for token in ("<script src", "<link ", "<img ", "cdn.", "fetch(", "XMLHttpRequest",
                  "import(", "WebSocket"):
        assert token not in html, f"renderer reaches out via {token!r}"
    assert html.count("http://www.w3.org/2000/svg") >= 1
    payload = json.loads(html.split("const DATA = ")[1].split(";\n")[0])
    assert payload["nodes"]
    assert payload["view"] == "reaction_network"


def test_the_provenance_renderer_prints_its_caveat_on_the_canvas(views):
    html = render_network_html(views["provenance"], uid="p")
    assert "bibliographic" in html
    assert "not</b> chemically" in html or "<b>not</b> chemically" in html


def test_a_large_graph_is_trimmed_rather_than_rendered_unreadable(views):
    html = render_network_html(views["provenance"], uid="p", max_nodes=3)
    payload = json.loads(html.split("const DATA = ")[1].split(";\n")[0])
    assert len(payload["nodes"]) == 3
    assert payload["trimmed"] is True
    assert payload["trimmed_from"] > 3
    # Reactions are kept first: they are the subject of the view.
    assert all(n["type"] == NODE_REACTION for n in payload["nodes"])
