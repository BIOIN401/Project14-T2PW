"""VIS-8: interaction edges and chain folding (src/t2pw/pwml/layout.py)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from t2pw.pwml.layout import CROSS_W, _layout_cost, _path_points, _plen, _relayout, relayout  # noqa: E402
from test_pwml_vis5_layout import _chain, _ids  # noqa: E402


def _interacting_chain(n, a, b, second=None, fold=1.8, public=False):
    """Chain of n reactions with an interaction between the enzymes of reactions a and b (edges 900 Left, 901 Right)."""
    comps, prots, pcvs, edges, rvs = _chain(n)
    pairs = [(a, b)] + ([second] if second else [])
    ivs = []
    for k, (x, y) in enumerate(pairs):
        edges += [{"id": 900 + 2 * k, "path": "M0 0", "hidden": True, "options": "{}", "visualization-template-id": 195},
                  {"id": 901 + 2 * k, "path": "M0 0", "hidden": False, "options": "{}", "visualization-template-id": 195}]
        ivs.append({"id": 700 + k, "interaction_protein_complex_visualizations": [
            {"protein-complex-visualization-id": 300 + x, "edge-id": 900 + 2 * k, "side": "Left"},
            {"protein-complex-visualization-id": 300 + y, "edge-id": 901 + 2 * k, "side": "Right"}]})
    items = {"biological-states": [{"id": 1, "name": "cytosol state", "subcellular-location-id": 11}],
             "subcellular-locations": [{"id": 11, "name": "cytosol"}],
             "proteins": [{"id": 100 + i, "name": f"E{i}"} for i in range(n)],
             "compound-locations": comps, "protein-locations": prots, "protein-complex-visualizations": pcvs,
             "edges": edges, "reaction-visualizations": rvs, "interaction-visualizations": ivs}
    (relayout(items, _ids()) if public else _relayout(items, _ids(), None, None, fold))
    return items


def _by_id(items):
    return {e["id"]: e for e in items["edges"]}


def test_relayout_turns_a_blocked_enzyme_to_enzyme_interaction_into_one_arch():
    # VIS-7 review F5: enzymes 0 and 2 of a row stand with compounds between them, so the straight curve is blocked
    items = _interacting_chain(3, 0, 2)
    e = _by_id(items)[901]
    pts = _path_points(e["path"])
    assert e["path"].count("C") == 1 and len(pts) == 4  # one cubic, not a routed polyline
    locs = {p["id"]: p for p in items["protein-locations"]}
    assert pts[0][1] == int(locs[102]["y"]) and pts[3][1] == min(int(locs[100]["y"]), int(locs[199]["y"]))  # both on top sides
    assert pts[1][1] < pts[0][1] and pts[2][1] < pts[3][1]  # the control points stand off above the boxes: an arch


def test_the_two_edges_of_a_pair_end_on_the_same_points_and_the_visible_one_follows_the_arch():
    # VIS-7 review F2
    items = _interacting_chain(3, 0, 2)
    edges = _by_id(items)
    hidden, visible = _path_points(edges[900]["path"]), _path_points(edges[901]["path"])
    assert hidden[0] == visible[3] and hidden[3] == visible[0]
    assert hidden[1] == visible[2] and hidden[2] == visible[1]  # same arch, reversed


def test_arch_ends_face_along_the_pairs_longer_axis():
    # VIS-7 review F3: a pair side by side leaves from top or bottom sides; the arch never runs back through its own box
    items = _interacting_chain(3, 0, 2)
    pts = _path_points(_by_id(items)[901]["path"])
    locs = {p["id"]: p for p in items["protein-locations"]}
    for (x, y), ids in ((pts[0], (102,)), (pts[3], (100, 199))):  # enzyme 100 stands in a stack with the isozyme 199
        ls = [locs[k] for k in ids]
        x0, x1 = min(int(l["x"]) for l in ls), max(int(l["x"]) + 150 for l in ls)
        y0, y1 = min(int(l["y"]) for l in ls), max(int(l["y"]) + 70 for l in ls)
        assert x0 <= x <= x1 and y in (y0, y1)  # on the top or bottom side of the box


def test_arches_of_one_hub_nest_the_longer_span_stands_higher():
    items = _interacting_chain(3, 0, 1, second=(0, 2))
    edges = _by_id(items)
    near, far = _path_points(edges[901]["path"]), _path_points(edges[903]["path"])  # (0,1) and (0,2)
    assert far[1][1] != near[1][1] or abs(_plen(far) - _plen(near)) > 0  # distinct arches
    assert abs(far[1][1] - far[0][1]) >= abs(near[1][1] - near[0][1])  # the longer one is at least as high


def test_a_long_flat_chain_folds_into_rows_so_the_picture_is_not_a_strip():
    items = _interacting_chain(12, 5, 6, public=True)
    cy = {int(c["y"]) for c in items["compound-locations"]}
    assert max(cy) - min(cy) > 400  # more than one row
    assert max(int(l["x"]) + 150 for l in items["protein-locations"]) < 12 * 670  # narrower than the straight chain


def test_relayout_keeps_the_layout_with_the_lowest_cost():
    strip = _interacting_chain(12, 5, 6, fold=100)
    folded = _interacting_chain(12, 5, 6, fold=1.2)
    public = _interacting_chain(12, 5, 6, public=True)
    assert max(int(l["x"]) for l in strip["protein-locations"]) > max(int(l["x"]) for l in folded["protein-locations"])
    wh = lambda it: (max(int(l["x"]) + 150 for l in it["protein-locations"]) + 40, max(int(c["y"]) + 190 for c in it["compound-locations"]) + 40)  # noqa: E731
    costs = [_layout_cost(it, *wh(it)) for it in (strip, folded, public)]
    assert costs[2] <= min(costs[0], costs[1]) + 1e-6


def test_the_cost_charges_crossings_through_nodes_and_interaction_length():
    base = {"compound-locations": [], "protein-locations": [], "edges": [
        {"id": 1, "path": "M0 0 C50 0 100 0 150 0", "hidden": False}, {"id": 2, "path": "M80 -45 C80 -20 80 20 80 50", "hidden": False}]}
    far = {"compound-locations": [], "protein-locations": [], "edges": [
        {"id": 1, "path": "M0 0 C50 0 100 0 150 0", "hidden": False}, {"id": 2, "path": "M75 150 C75 160 75 190 75 200", "hidden": False}]}
    assert _layout_cost(base, 1000, 700) - _layout_cost(far, 1000, 700) >= CROSS_W - 0.02  # the crossing adds CROSS_W (lengths differ a little)
    assert _layout_cost(far, 2000, 700) > _layout_cost(far, 1000, 700)  # a wider picture is harder to read


def test_chooser_off_gives_the_first_candidates_layout(monkeypatch):
    from t2pw.pwml.layout import LAYOUT_CANDIDATES
    monkeypatch.setenv("T2PW_LAYOUT_CHOOSER", "0")
    off = _interacting_chain(12, 5, 6, public=True)
    first = _interacting_chain(12, 5, 6, fold=LAYOUT_CANDIDATES[0][0])
    assert [(l["x"], l["y"]) for l in off["protein-locations"]] == [(l["x"], l["y"]) for l in first["protein-locations"]]


def test_an_enzymeless_reaction_point_sits_on_its_substrate_even_across_compartments():
    # the writer hides the substrate edge of such a reaction, so the visible product edge must start at the substrate
    from test_pwml_vis5_layout import _cl
    comps = [_cl(1, 2), _cl(2, 1)]
    edges = [{"id": 10, "path": "M0 0", "hidden": True, "options": '{"start_arrow": false}'},
             {"id": 11, "path": "M0 0", "hidden": False, "options": '{"start_arrow": true}'}]
    rv = {"id": 20, "reaction-id": 1, "biological-state-id": 1, "reaction_compound_visualizations": [
        {"compound-location-id": 1, "edge-id": 10, "side": "Left"}, {"compound-location-id": 2, "edge-id": 11, "side": "Right"}]}
    items = {"biological-states": [{"id": 1, "name": "cytosol state", "subcellular-location-id": 11},
                                   {"id": 2, "name": "ER state", "subcellular-location-id": 12}],
             "subcellular-locations": [{"id": 11, "name": "cytosol"}, {"id": 12, "name": "endoplasmic reticulum"}],
             "compound-locations": comps, "edges": edges, "reaction-visualizations": [rv]}
    _relayout(items, _ids(), None, None, 1.8)
    pts = _path_points(edges[1]["path"])
    sub = next(c for c in comps if c["id"] == 1)
    x0, y0, x1, y1 = int(sub["x"]), int(sub["y"]), int(sub["x"]) + 200, int(sub["y"]) + 190
    end = pts[-1]
    assert x0 - 14 <= end[0] <= x1 + 14 and y0 - 14 <= end[1] <= y1 + 14  # the edge starts on the substrate's border


def test_crossings_counts_proper_crossings_once_per_polyline():
    from t2pw.pwml.layout import _crossings
    line = [(0.0, 0.0), (100.0, 0.0)]
    cross = [(50.0, -10.0), (50.0, 10.0)]
    twice = [(20.0, -10.0), (20.0, 10.0), (80.0, 10.0), (80.0, -10.0)]  # crosses the line twice: one polyline, one count
    touch = [(100.0, 0.0), (150.0, 40.0)]  # shares an end: not a crossing
    far = [(0.0, 50.0), (100.0, 50.0)]
    assert _crossings(line, [cross, twice, touch, far, []]) == 2


def test_a_picture_with_more_edges_than_the_limit_is_laid_out_once(monkeypatch):
    import t2pw.pwml.layout as lay
    monkeypatch.setattr(lay, "CHOOSER_MAX_EDGES", 0)
    limited = _interacting_chain(12, 5, 6, public=True)
    first = _interacting_chain(12, 5, 6, fold=lay.LAYOUT_CANDIDATES[0][0])
    assert [(l["x"], l["y"]) for l in limited["protein-locations"]] == [(l["x"], l["y"]) for l in first["protein-locations"]]
