"""VIS-2: PathBank cell and organelle art in PWML layout v2 (src/t2pw/pwml/cell_art.py, layout.py). Visualization only."""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from t2pw.pwml import cell_art  # noqa: E402
from t2pw.pwml.layout import relayout  # noqa: E402
from test_pwml_layout import _ids, _loc  # noqa: E402

ARROW = json.dumps({"start_arrow": True, "start_flat_arrow": False, "end_arrow": False, "end_flat_arrow": False,
                    "start_arrow_path": "M 0 0 L 0 0 L 0 0"})
NO_ARROW = json.dumps({"end_arrow": False, "start_arrow": False})


def _reaction(rid, state, s, p, eid):
    return {"id": 80 + rid, "reaction-id": rid, "biological-state-id": state,
            "reaction_compound_visualizations": [{"compound-location-id": s, "edge-id": eid, "side": "Left"},
                                                 {"compound-location-id": p, "edge-id": eid + 1, "side": "Right"}]}


def _items(species, locations, states, reactions):
    """``states``: [(state id, location id)]; ``reactions``: [(state, substrate loc id, product loc id)]."""
    locs, edges, rvs = [], [], []
    for n, (st, s, p) in enumerate(reactions):
        locs += [_loc(s, st), _loc(p, st)]
        edges += [{"id": 100 + 2 * n, "path": "M0 0", "hidden": False, "options": NO_ARROW},
                  {"id": 101 + 2 * n, "path": "M0 0", "hidden": False, "options": ARROW}]
        rvs.append(_reaction(n + 1, st, s, p, 100 + 2 * n))
    return {"species": [{"id": 3, "name": species[0], "classification": species[1]}],
            "subcellular-locations": [{"id": i, "name": n} for i, n in locations],
            "biological-states": [{"id": s, "name": f"state {s}", "subcellular-location-id": l, "species-id": 3} for s, l in states],
            "compounds": [{"id": l["id"], "name": f"c{l['id']}"} for l in locs],
            "compound-locations": locs, "edges": edges, "reaction-visualizations": rvs}


def _bacterium(classification="Prokaryote", name="Escherichia coli"):
    return _items((name, classification), [(1, "cytoplasm"), (2, "Outer Membrane")], [(1, 1), (2, 2)],
                  [(1, 1, 2), (1, 3, 4)])


def _eukaryote():
    return _items(("Homo sapiens", "Eukaryote"), [(1, "cytosol"), (2, "mitochondrion")], [(1, 1), (2, 2)],
                  [(1, 1, 2), (2, 3, 4), (2, 5, 6)])


def _box(d):
    s = float(d["scale-x"])
    return d["x"], d["y"], d["x"] + int(d["width"]) * s, d["y"] + int(d["height"]) * s


def _inside(p, r):
    return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]


def _path_box(path):
    n = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", path)]
    return min(n[0::2]), min(n[1::2]), max(n[0::2]), max(n[1::2])


def test_bacterium_gets_icon_left_zoom_chromosome_and_gram_negative_envelope():
    items = _bacterium()
    w, h = relayout(items, _ids())
    art = {d["drawable-image-id"]: d for d in items["drawable-element-locations"]}
    assert set(art) == {cell_art.BACTERIUM, cell_art.CHROMOSOME}
    (z,) = items["zoom-visualizations"]
    assert z["visualization-template-id"] == cell_art.ZOOM_LEFT and z["color"] is None and z["zindex"] == 4
    bug = _box(art[cell_art.BACTERIUM])
    assert _inside((z["p1x"], z["p1y"]), bug) and bug[2] <= z["p2x"] + 20  # the small box on the bacterium, left of the cell
    mem = items["membrane-visualizations"]
    pg = [m for m in mem if m["visualization-template-id"] == cell_art.PEPTIDOGLYCAN_TEMPLATE]
    lines = sorted((_path_box(m["path"]) for m in mem if m["visualization-template-id"] == 1), key=lambda b: b[0])
    assert len(pg) == 1 and len(lines) == 2  # outer and inner membrane lines with the peptidoglycan between them
    outer, inner, mid = lines[0], lines[1], _path_box(pg[0]["path"])
    assert inner[0] - outer[0] == cell_art.ENVELOPE_GAP and outer[0] < mid[0] < inner[0]
    assert (z["p2x"], z["p2y"], z["p3x"], z["p3y"]) == (outer[0] - 27, outer[1] - 27, outer[2] + 27, outer[3] + 27)
    assert _inside(_box(art[cell_art.CHROMOSOME])[:2], inner) and _inside(_box(art[cell_art.CHROMOSOME])[2:], inner)
    for loc in items["compound-locations"]:  # nodes inside the inner membrane, not under the chromosome
        assert _inside((loc["x"], loc["y"]), inner)
        c = _box(art[cell_art.CHROMOSOME])
        assert loc["x"] + int(loc["width"]) <= c[0] or c[2] <= loc["x"] or loc["y"] + int(loc["height"]) <= c[1] or c[3] <= loc["y"]
    assert z["p3x"] <= w and z["p3y"] <= h and all(_box(d)[2] <= w and _box(d)[3] <= h for d in art.values())


def test_bacterium_without_outer_membrane_or_periplasm_keeps_one_membrane():
    items = _items(("Bacillus subtilis", "Prokaryote"), [(1, "cytoplasm")], [(1, 1)], [(1, 1, 2)])
    relayout(items, _ids())
    assert {d["drawable-image-id"] for d in items["drawable-element-locations"]} == {cell_art.BACTERIUM, cell_art.CHROMOSOME}
    assert [m["visualization-template-id"] for m in items["membrane-visualizations"]] == [1]


def test_eukaryote_gets_cluster_and_top_zoom_into_its_organelle():
    items = _eukaryote()
    w, h = relayout(items, _ids())
    (d,) = items["drawable-element-locations"]
    (z,) = items["zoom-visualizations"]
    assert d["drawable-image-id"] == cell_art.CLUSTER_ANIMAL and d["zindex"] == 2 and d["degree"] == 0
    assert z["visualization-template-id"] in (cell_art.ZOOM_TOP, cell_art.ZOOM_LEFT) and z["color"] == "#FFEBEB"
    mito = [_path_box(m["path"]) for m in items["membrane-visualizations"] if m["visualization-template-id"] == 84]
    outer = min(mito, key=lambda b: b[0])  # the mitochondrion's outer line (double membrane)
    assert (z["p2x"], z["p2y"], z["p3x"], z["p3y"]) == (outer[0] - 27, outer[1] - 27, outer[2] + 27, outer[3] + 27)
    assert _inside((z["p1x"], z["p1y"]), _box(d))
    if z["visualization-template-id"] == cell_art.ZOOM_TOP:  # from the cluster down into the box
        assert z["p1y"] + 20 < z["p2y"] and _box(d)[3] < outer[1]
    else:  # from the cluster, left of the box, across into it
        assert z["p1x"] + 25 < z["p2x"] and _box(d)[2] < outer[0]
    assert _box(d)[2] <= w and _box(d)[3] <= h and z["p3x"] <= w and z["p3y"] <= h


def test_plant_gets_the_plant_cluster_zooming_from_its_chloroplast():
    items = _items(("Arabidopsis thaliana", "Eukaryote"), [(1, "cytosol"), (2, "chloroplast")], [(1, 1), (2, 2)],
                   [(1, 1, 2), (2, 3, 4)])
    relayout(items, _ids())
    (d,) = items["drawable-element-locations"]
    (z,) = items["zoom-visualizations"]
    assert d["drawable-image-id"] == cell_art.CLUSTER_PLANT and z["color"] == "#EBFFEB"


def test_unknown_cell_type_gets_no_art():
    for items in (_bacterium(classification=None, name="Unknownia obscura"),  # genus in no table: undetermined
                  _bacterium(name="Methanococcus maripaludis"),  # an archaeon: no PathBank bacterium image for it
                  _items(("Homo sapiens", "Eukaryote"), [(1, "cytosol")], [(1, 1)], [(1, 1, 2)])):  # no organelle box
        relayout(items, _ids())
        assert items.get("drawable-element-locations", []) == [] and items.get("zoom-visualizations", []) == []
        assert all(m["visualization-template-id"] != cell_art.PEPTIDOGLYCAN_TEMPLATE for m in items.get("membrane-visualizations", []))


def test_art_is_deterministic_and_moves_whole_picture():
    for make in (_bacterium, _eukaryote):
        a, b = make(), make()
        assert relayout(a, _ids()) == relayout(b, _ids()) and a == b
        # every node, label and edge stays inside the canvas after the picture moved to make room
        w, h = relayout(copy.deepcopy(make()), _ids())
        for loc in a["compound-locations"]:
            assert 0 <= loc["x"] and loc["x"] + int(loc["width"]) <= w and 0 <= loc["y"] and loc["y"] + int(loc["height"]) <= h
        for e in a["edges"]:
            x0, y0, x1, y1 = _path_box(e["path"])
            assert 0 <= x0 and x1 <= w and 0 <= y0 and y1 <= h
            assert json.loads(e["options"]).get("start_arrow_path", "M 1") != "M 0 0 L 0 0 L 0 0"
