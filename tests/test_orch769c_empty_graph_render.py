"""ORCH-769c (D-112 § 3c): a graph with nothing to draw renders to ``b""``, it does not raise.

CORRECTION of pre-existing observable behaviour, so G9 applies: the two ``test_g9_*`` tests
fail on the base commit ``e3d5b65f``, where ``render_draft_graph_to_png_bytes`` raised
``RuntimeError("DraftGraph has no reaction nodes ...")``.

The defect, measured on ORCH-767 WNT PCP (``wnt_pcp-strict/2026-10-24_0131``): quarantine
admitted 18 interactions and no reaction or transport, the review diagram (which lays out only
reactions and transports) raised inside ``initialize_refinement_review_state``, the review step
never opened and the "Generate PWML" button was never rendered. The review page already skips an
empty image and the curator already treats a failed drawing as non-fatal, so empty bytes are the
"no diagram" value both callers understand.
"""

from __future__ import annotations

from t2pw.pipeline.draft_graph import build_draft_graph
from t2pw.pipeline.draft_graph_render import render_draft_graph_to_png_bytes


def _interactions_only() -> dict:
    """The WNT PCP shape: proteins and interactions, no reaction, no transport."""

    return {
        "entities": {"proteins": [{"name": "Dvl"}, {"name": "Inversin"}, {"name": "Prickle"}]},
        "processes": {
            "reactions": [],
            "transports": [],
            "interactions": [
                {"name": "Dvl recruits Inversin", "entity_1": "Dvl", "entity_2": "Inversin", "relationship": "binds"},
                {"name": "Inversin degrades Prickle", "entity_1": "Inversin", "entity_2": "Prickle",
                 "relationship": "inhibits"},
            ],
        },
    }


def test_g9_an_interactions_only_graph_renders_to_empty_bytes() -> None:
    graph = build_draft_graph(_interactions_only()).to_dict()
    assert any(node["kind"] == "interaction" for node in graph["nodes"])  # not vacuous: there are nodes
    assert render_draft_graph_to_png_bytes(graph, dpi=100) == b""


def test_g9_a_graph_with_no_nodes_renders_to_empty_bytes() -> None:
    assert render_draft_graph_to_png_bytes({"nodes": [], "edges": []}, dpi=100) == b""


def test_preservation_a_graph_with_a_reaction_still_renders_a_png() -> None:
    """Not a G9 test (it passes at base too): the guard must not swallow a drawable graph."""

    payload = _interactions_only()
    payload["entities"]["compounds"] = [{"name": "L-glutamate"}, {"name": "glutathione"}]
    payload["processes"]["reactions"] = [
        {"name": "glutathione synthesis", "inputs": ["L-glutamate"], "outputs": ["glutathione"], "enzymes": []}
    ]
    png = render_draft_graph_to_png_bytes(build_draft_graph(payload).to_dict(), dpi=100)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
