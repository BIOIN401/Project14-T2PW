"""VIS-1: PathBank edges and flow in PWML layout v2 (src/t2pw/pwml/layout.py). Geometry only."""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from t2pw.pwml.layout import relayout  # noqa: E402
from test_pwml_layout import _ids, _loc  # noqa: E402  (tests/ is on sys.path, as for the other cross-test imports)

ARROW = json.dumps({"start_arrow": True, "start_flat_arrow": False, "end_arrow": False, "end_flat_arrow": False,
                    "start_arrow_path": "M 0 0 L 0 0 L 0 0"})
NO_ARROW = json.dumps({"end_arrow": False, "end_flat_arrow": False, "start_arrow": False, "start_flat_arrow": False})


def _nums(path):
    return [float(t) for t in re.findall(r"-?\d+(?:\.\d+)?", path)]


def _rect(loc):
    return loc["x"], loc["y"], loc["x"] + int(loc["width"]), loc["y"] + int(loc["height"])


def _reaction_items():
    """S + ATP -> P + ADP, catalysed by E; one extra connector edge from E to its own reaction (no member uses it)."""
    return {
        "biological-states": [{"id": 1, "name": "cytosol state"}],
        "compounds": [{"id": 1, "name": "citrate"}, {"id": 2, "name": "isocitrate"}, {"id": 3, "name": "ATP"},
                      {"id": 4, "name": "ADP"}],
        "compound-locations": [_loc(1, 1), _loc(2, 1), _loc(3, 1, "50", "30", tid=42), _loc(4, 1, "50", "30", tid=43)],
        "protein-locations": [dict(_loc(5, 1, "150", "70", tid=2), **{"protein-id": 5}),
                              dict(_loc(6, 2, "150", "70", tid=2), **{"protein-id": 5})],  # E again, in a state no process uses
        "protein-complex-visualizations": [{"id": 50, "protein_complex_protein_visualizations": [{"id": 51, "protein-location-id": 5}]}],
        "edges": [{"id": 70, "path": "M0 0", "hidden": False, "options": NO_ARROW},   # S
                  {"id": 71, "path": "M0 0", "hidden": False, "options": NO_ARROW},   # ATP
                  {"id": 72, "path": "M0 0", "hidden": False, "options": ARROW},      # P
                  {"id": 73, "path": "M0 0", "hidden": False, "options": ARROW},      # ADP
                  {"id": 74, "path": "M9 9 C9 9 9 9 9 9", "hidden": False}],          # E -> its own reaction (IR connector)
        "reaction-visualizations": [{"id": 80, "reaction-id": 1, "biological-state-id": 1,
                                     "reaction_compound_visualizations": [
                                         {"compound-location-id": 1, "edge-id": 70, "side": "Left"},
                                         {"compound-location-id": 3, "edge-id": 71, "side": "Left"},
                                         {"compound-location-id": 2, "edge-id": 72, "side": "Right"},
                                         {"compound-location-id": 4, "edge-id": 73, "side": "Right"}],
                                     "reaction_enzyme_visualizations": [{"protein-complex-visualization-id": 50}]}],
    }


def _ir_connector():
    ir = {"process_visualizations": [{"type": "reaction_visualization", "process_key": "R1",
                                      "members": [{"edge_key": "E74", "location_key": "L5"}]}]}
    return ir, {"processes": {"R1": 1}, "edges": {"E74": 74}, "locations": {"L5": 5}}


def test_reaction_is_a_fork_on_one_shaft_with_pathbank_curves():
    items = _reaction_items()
    ir, keys = _ir_connector()
    a, b = copy.deepcopy(items), copy.deepcopy(items)
    assert relayout(a, _ids(), ir, keys) == relayout(b, _ids(), ir, keys) and a == b  # deterministic
    loc = {it["id"]: it for sec in ("compound-locations", "protein-locations") for it in a[sec]}
    e = {it["id"]: it for it in a["edges"]}
    ex0, ey0, ex1, ey1 = _rect(loc[5])
    cy = (ey0 + ey1) // 2
    # main substrate and product on the enzyme's shaft, left and right of it
    for i in (1, 2):
        r = _rect(loc[i])
        assert (r[1] + r[3]) // 2 == cy
    assert _rect(loc[1])[2] < ex0 and ex1 < _rect(loc[2])[0]
    # every edge is one cubic; the substrate edge ends at the middle of the enzyme's left side, the product edge
    # starts at the product (where its arrow is) and ends at the middle of the right side
    for eid in (70, 71, 72, 73):
        assert re.fullmatch(r"M-?\d+ -?\d+ C(-?\d+ -?\d+ ){2}-?\d+ -?\d+", e[eid]["path"]), e[eid]["path"]
    s, p = _nums(e[70]["path"]), _nums(e[72]["path"])
    assert s[-2:] == [ex0, cy] and s[:2] == [_rect(loc[1])[2], cy] and s[1] == s[3] == s[5] == cy  # straight on the shaft
    assert p[:2] == [_rect(loc[2])[0], cy] and p[-2:] == [ex1, cy]
    # side compounds: small nodes just above or below the shaft, beside the reaction point, on curved branches that
    # leave the node at right angles and meet the shaft along it
    for i, eid, point in ((3, 71, ex0), (4, 73, ex1)):
        r = _rect(loc[i])
        assert r[3] <= cy - 40 or r[1] >= cy + 40, r
        assert abs((r[0] + r[2]) / 2 - point) <= 100
        n = _nums(e[eid]["path"])
        assert n[-2:] == [point, cy]
        assert n[0] == n[2] and n[1] != n[3]  # leaves the pill vertically
        assert n[5] == cy and n[4] != n[6]    # meets the shaft horizontally
    # arrow paths follow the new geometry: the product's head sits at the path's start
    tip = json.loads(e[72]["options"])["start_arrow_path"].split()[4:6]
    assert [float(t) for t in tip] == p[:2]
    # no zero-length path: the connector from the enzyme to its own reaction is the shaft through the enzyme
    c = _nums(e[74]["path"])
    assert c[:2] == [ex0, cy] and c[-2:] == [ex1, cy]
    # E's second location (no process refers to it; PathWhiz drops it) sits on the drawn one, not as a stray box
    assert (loc[6]["x"], loc[6]["y"]) == (loc[5]["x"], loc[5]["y"])


def test_side_compound_needs_a_main_partner_and_a_single_use():
    items = _reaction_items()
    # ATP -> ADP alone (no main member on either side): both stay on the main chain
    rv = items["reaction-visualizations"][0]
    rv["reaction_compound_visualizations"] = [m for m in rv["reaction_compound_visualizations"] if m["compound-location-id"] in (3, 4)]
    relayout(items, _ids())
    loc = {it["id"]: it for it in items["compound-locations"]}
    enz = _rect(items["protein-locations"][0])
    cy = (enz[1] + enz[3]) // 2
    for i in (3, 4):
        r = _rect(loc[i])
        assert (r[1] + r[3]) // 2 == cy


def test_drug_box_is_drawn_as_a_compound_box():
    items = _reaction_items()
    items["compound-locations"][0]["visualization-template-id"] = 157  # a metabolite with a DrugBank id
    relayout(items, _ids())
    loc = items["compound-locations"][0]
    assert (loc["visualization-template-id"], loc["width"], loc["height"]) == (81, "200", "190")


def test_transport_edges_take_pathbanks_transport_template():
    items = _reaction_items()
    items["edges"] += [{"id": 76, "path": "M0 0", "hidden": False, "visualization-template-id": 5, "options": NO_ARROW},
                       {"id": 77, "path": "M0 0", "hidden": False, "visualization-template-id": 5, "options": ARROW}]
    items["transport-visualizations"] = [{"id": 90, "transport-id": 2, "transport_compound_visualizations": [
        {"compound-location-id": 2, "edge-id": 76, "side": "Left"}, {"compound-location-id": 1, "edge-id": 77, "side": "Right"}],
        "transport_transporter_visualizations": []}]
    for e in items["edges"][:4]:
        e["visualization-template-id"] = 5
    relayout(items, _ids())
    tpl = {e["id"]: e.get("visualization-template-id") for e in items["edges"]}
    assert (tpl[76], tpl[77]) == (83, 83) and {tpl[i] for i in (70, 71, 72, 73)} == {5}


def test_every_edge_has_length_and_a_real_arrow_tangent():
    items = _reaction_items()
    ir, keys = _ir_connector()
    items["edges"].append({"id": 75, "path": "M575 660 C575 660 575 660 575 660", "hidden": False, "options": ARROW})  # tied to nothing
    relayout(items, _ids(), ir, keys)
    for e in items["edges"]:
        n = _nums(e["path"])
        pts = list(zip(n[0::2], n[1::2]))
        assert len(set(pts)) > 1, e
        if json.loads(e.get("options") or "{}").get("start_arrow"):
            assert pts[0] != pts[1], e  # PathWhiz draws the head from the start point toward the first control point
