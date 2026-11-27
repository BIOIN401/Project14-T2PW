"""VIS-8: interaction edges and chain folding (src/t2pw/pwml/layout.py)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from t2pw.pwml.layout import _path_points, _plen, relayout  # noqa: E402
from test_pwml_vis5_layout import _chain, _ids  # noqa: E402


def _interacting_chain(n, a, b, second=None):
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
    relayout(items, _ids())
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
    items = _interacting_chain(12, 5, 6)
    w = max(int(l["x"]) + 150 for l in items["protein-locations"])
    h = max(int(l["y"]) + 70 for l in items["protein-locations"]) - min(int(l["y"]) for l in items["protein-locations"])
    cy = {int(c["y"]) for c in items["compound-locations"]}
    assert max(cy) - min(cy) > 400  # more than one row
    assert w < 12 * 670  # narrower than the straight chain
