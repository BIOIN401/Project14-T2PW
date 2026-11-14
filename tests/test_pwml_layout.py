"""PWML layout v2 (``T2PW_PWML_LAYOUT=v2``, src/t2pw/pwml/layout.py): geometry changes, content does not."""
from __future__ import annotations

import copy
import json
import re
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.layout import _label_lines, _path, _pava, label_text, layout_version, relayout  # noqa: E402
from t2pw.pwml.validate import discover_structure_signature  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402

IR = json.loads((ROOT / "tests" / "fixtures" / "pwml_layout" / "ir_run4.json").read_text(encoding="utf-8"))
LOCS = ("compound-location", "protein-location", "nucleic-acid-location", "element-collection-location")


def _build(monkeypatch, layout):
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
    if layout is None:
        monkeypatch.delenv("T2PW_PWML_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("T2PW_PWML_LAYOUT", layout)
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000",
                           height=1400, width=3200, background_color="#FFFFFF",
                           ref=str(ROOT / "reference" / "PW000001.pwml"))
    sig = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    root = DeterministicPwmlBuilder(extraction=copy.deepcopy(IR), signature=sig, args=args).build().root
    return etree.fromstring(etree.tostring(root))


def _strip(root):
    """Same rules as docs/pwml_output/evidence/layout/scripts/content_check.py."""
    root = copy.deepcopy(root)
    for el in list(root.iter()):
        if not isinstance(el.tag, str):
            continue
        tag, ptag = etree.QName(el).localname, etree.QName(el.getparent()).localname if el.getparent() is not None else ""
        if tag in ("membrane-visualizations", "label-locations"):
            for c in list(el):
                el.remove(c)
            el.text = None
        elif tag in ("x", "y", "width", "height") and len(el) == 0:
            el.text = "#"
        elif tag == "path" and ptag == "edge":
            el.text = "#"
        elif tag == "options" and ptag == "edge" and el.text:
            o = json.loads(el.text)
            el.text = json.dumps({k: ("#" if k.endswith("_path") and v is not None else v) for k, v in o.items()}, sort_keys=True)
    return etree.tostring(root, method="c14n")


def _boxes(root):
    out = []
    for tag in LOCS:
        for el in root.iter(tag):
            if el.findtext("hidden") != "true":
                x, y = int(el.findtext("x")), int(el.findtext("y"))
                out.append((el.findtext("biological-state-id"), (x, y, x + int(el.findtext("width")), y + int(el.findtext("height")))))
    return out


def test_flag_values(monkeypatch):
    monkeypatch.delenv("T2PW_PWML_LAYOUT", raising=False)
    assert layout_version() == "v1"
    for value, expected in (("v2", "v2"), (" V2 ", "v2"), ("v1", "v1"), ("yes", "v1"), ("", "v1")):
        monkeypatch.setenv("T2PW_PWML_LAYOUT", value)
        assert layout_version() == expected


def test_unset_and_v1_are_the_same_bytes(monkeypatch):
    assert etree.tostring(_build(monkeypatch, None)) == etree.tostring(_build(monkeypatch, "v1"))


def test_v2_changes_geometry_only(monkeypatch):
    v1, v2 = _build(monkeypatch, None), _build(monkeypatch, "v2")
    assert etree.tostring(v1) != etree.tostring(v2)
    assert _strip(v1) == _strip(v2)


def test_v2_quality(monkeypatch):
    root = _build(monkeypatch, "v2")
    pv = root.find(".//pathway-visualization")
    W, H = int(pv.findtext("width")), int(pv.findtext("height"))
    boxes = _boxes(root)
    assert boxes
    # no two visible nodes overlap, every node inside the canvas
    for (_, a), (_, b) in combinations(boxes, 2):
        assert min(a[2], b[2]) <= max(a[0], b[0]) or min(a[3], b[3]) <= max(a[1], b[1]), (a, b)
    assert all(r[0] >= 0 and r[1] >= 0 and r[2] <= W and r[3] <= H for _, r in boxes)
    # one closed membrane per compartment that holds a node PathWhiz draws (one a process refers to, directly or
    # through a used complex visualization); each such node sits in a membrane holding only its own state
    pv = root.find(".//pathway-visualization")
    procs = [v for t in ("reaction-visualization", "transport-visualization", "interaction-visualization") for v in pv.iter(t)]
    ref = {e.text for v in procs for e in v.iter() if isinstance(e.tag, str) and e.tag.endswith("location-id")}
    pcvs = {e.text for v in procs for e in v.iter("protein-complex-visualization-id")}
    for c in pv.iter("protein-complex-visualization"):
        if c.findtext("id") in pcvs:
            ref |= {e.text for e in c.iter("protein-location-id")}
    drawn = []
    for tag in LOCS:
        for el in root.iter(tag):
            if el.findtext("hidden") != "true" and el.findtext("id") in ref:
                x, y = int(el.findtext("x")), int(el.findtext("y"))
                drawn.append((el.findtext("biological-state-id"), (x, y, x + int(el.findtext("width")), y + int(el.findtext("height")))))
    rects = []
    for m in root.iter("membrane-visualization"):
        nums = [int(t) for t in re.findall(r"-?\d+", m.findtext("path"))]
        xs, ys = nums[0::2], nums[1::2]
        assert (xs[0], ys[0]) == (xs[-1], ys[-1])
        rects.append((min(xs), min(ys), max(xs), max(ys)))
    assert len(rects) == len({st for st, _ in drawn})
    for r in rects:
        inside = {st for st, bx in drawn if r[0] <= bx[0] and r[1] <= bx[1] and bx[2] <= r[2] and bx[3] <= r[3]}
        assert len(inside) == 1  # no empty box, no mixed box
    texts = [lb.findtext("label-text") for lb in root.iter("label-location")]
    assert texts and not any(t.startswith("__") for t in texts)
    # every edge stays inside the canvas
    for e in root.iter("edge"):
        nums = [int(float(t)) for t in re.findall(r"-?\d+(?:\.\d+)?", e.findtext("path"))]
        assert all(0 <= v <= W for v in nums[0::2]) and all(0 <= v <= H for v in nums[1::2])


def test_path_is_straight_cubic_segments():
    path, cp1, cp2 = _path([(0, 0), (30, 0), (30, 60)])
    assert path == "M0 0 C10 0 20 0 30 0 C30 20 30 40 30 60"
    assert cp1 == (10, 0) and cp2 == (30, 40)
    assert _path([(5, 5)])[0] == "M5 5 C5 5 5 5 5 5"


def test_pava_keeps_separation_and_fits_targets():
    assert _pava([0, 0, 0], [0, 10, 10]) == [-10.0, 0.0, 10.0]
    assert _pava([0, 100], [0, 10]) == [0.0, 100.0]


def _ids():
    n = [1000]

    def nxt():
        n[0] += 1
        return n[0]
    return nxt


def _loc(i, state, w="160", h="60", hidden=False, tid=81):
    return {"id": i, "compound-id": i, "biological-state-id": state, "visualization-template-id": tid,
            "hidden": hidden, "x": 0, "y": 0, "zindex": 10, "font-size": "regular", "width": w, "height": h}


def test_relayout_empty_items_gives_a_small_valid_canvas():
    w, h = relayout({}, _ids())
    assert w > 0 and h > 0


def test_relayout_edge_cases_and_determinism():
    arrow = json.dumps({"start_arrow": True, "start_flat_arrow": False, "end_arrow": False, "end_flat_arrow": False,
                        "start_arrow_path": "M 0 0 L 0 0 L 0 0"})
    items = {
        "biological-states": [{"id": 1, "name": "cytosol state"}, {"id": 2, "name": "nucleus state"}],
        "compound-locations": [_loc(1, 1), _loc(2, 1, w="160.0"), _loc(3, 1, hidden=True), _loc(4, 2)],
        "protein-locations": [dict(_loc(5, 1, "150", "70", tid=2), **{"protein-id": 5})],
        "protein-complex-visualizations": [
            {"id": 50, "protein_complex_protein_visualizations": [{"id": 51, "protein-location-id": 5}]},
            {"id": 60, "protein_complex_protein_visualizations": []},  # a complex with nothing drawable
        ],
        "edges": [{"id": 70, "path": "M0 0", "hidden": False, "options": None},
                  {"id": 71, "path": "M0 0", "hidden": False, "options": arrow},
                  {"id": 72, "path": "M0 0", "hidden": True, "options": "{}"},
                  {"id": 73, "path": "M0 0", "hidden": False, "options": arrow},
                  {"id": 74, "path": "M9 9 L8 8", "hidden": False, "options": None}],  # tied to nothing
        "reaction-visualizations": [{"id": 80, "reaction-id": 1, "biological-state-id": 1,
                                     "reaction_compound_visualizations": [
                                         {"compound-location-id": 1, "edge-id": 70, "side": "Right"},
                                         {"compound-location-id": 2, "edge-id": 71, "side": "Left"}],
                                     "reaction_enzyme_visualizations": [{"protein-complex-visualization-id": 50}]}],
        "interaction-visualizations": [{"id": 90, "interaction_compound_visualizations": [
            {"compound-location-id": 4, "edge-id": 72, "side": "Left"}],
            "interaction_protein_complex_visualizations": [
            {"protein-complex-visualization-id": 60, "edge-id": 73, "side": "Right"}]}],
        "bound-visualizations": [{"id": 95, "biological-state-id": 1}, {"id": 96, "biological-state-id": 3}],
    }
    a, b = copy.deepcopy(items), copy.deepcopy(items)
    wa, ha = relayout(a, _ids())
    assert (wa, ha) == relayout(b, _ids()) and a == b  # deterministic
    # side is the reaction side PathWhiz reads, never geometry
    assert [m["side"] for m in a["reaction-visualizations"][0]["reaction_compound_visualizations"]] == ["Right", "Left"]
    # substrate (no arrow) left of the enzyme, product (arrow) right of it
    loc = {it["id"]: it for sec in ("compound-locations", "protein-locations") for it in a[sec]}
    assert loc[1]["x"] < loc[5]["x"] < loc[2]["x"]
    tip = json.loads(a["edges"][1]["options"])["start_arrow_path"].split()[4:6]
    assert tip == [a["edges"][1]["path"].split()[0][1:], a["edges"][1]["path"].split()[1]]  # arrow at the product end
    # interaction with an undrawable side: a stub point, an arrow that is still a real arrow
    assert json.loads(a["edges"][3]["options"])["start_arrow_path"] is not None
    moved = [int(t) for t in re.findall(r"-?[0-9]+", a["edges"][4]["path"])]  # tied to nothing: same shape, in the canvas
    assert (moved[0] - moved[2], moved[1] - moved[3]) == (1, 1) and max(moved[0::2]) <= wa and max(moved[1::2]) <= ha
    assert loc[3]["x"] >= 0 and loc[3]["y"] >= 0 and loc[3]["x"] + 200 <= wa
    assert loc[2]["width"] == "200"  # compound template 81 is drawn at its template size
    bv = {b["id"]: b for b in a["bound-visualizations"]}
    assert bv[95]["x"] <= loc[1]["x"] and bv[96]["width"] == "200"  # empty state: a small box of its own
    assert [l["label-text"] for l in a["label-locations"]] == ["cytosol", "nucleus"]
    assert len(a["membrane-visualizations"]) == 2


def test_labels_never_show_internal_names_and_wrap_like_pathwhiz():
    assert label_text({"name": "cytosol state"}) == "cytosol"
    assert label_text({"name": "__auto_state__"}) == "" and label_text({"name": "  "}) == ""
    assert _label_lines("Bacillus subtilis plasma membrane") == 2  # 28-character lines, as divide_label_text
    assert _label_lines("x" * 60) == 3


def test_no_box_for_a_state_whose_nodes_pathwhiz_drops_and_orphan_edges_move_into_the_canvas():
    items = {
        "biological-states": [{"id": 1, "name": "cytosol state"}, {"id": 2, "name": "__auto_state__"},
                              {"id": 3, "name": "plasma membrane state"}],
        "compound-locations": [_loc(1, 1), _loc(2, 1)],
        "nucleic-acid-locations": [dict(_loc(9, 1, tid=2), **{"nucleic-acid-id": 9})],
        "protein-locations": [dict(_loc(5, 2, "150", "70", tid=2), **{"protein-id": 5}),   # in an unused complex
                              dict(_loc(6, 3, "150", "70", tid=2), **{"protein-id": 6})],  # in no complex
        "protein-complex-visualizations": [{"id": 50, "protein_complex_protein_visualizations": [{"id": 51, "protein-location-id": 5}]}],
        "edges": [{"id": 70, "path": "M0 0", "hidden": False, "options": None},
                  {"id": 71, "path": "M0 0", "hidden": False, "options": None},
                  {"id": 74, "path": "M5000 300 L5100 320", "hidden": False, "options": None}],  # tied to nothing
        "reaction-visualizations": [{"id": 80, "reaction-id": 1, "biological-state-id": 1,
                                     "reaction_compound_visualizations": [
                                         {"compound-location-id": 1, "edge-id": 70, "side": "Left"},
                                         {"compound-location-id": 2, "edge-id": 71, "side": "Right"}],
                                     "reaction_enzyme_visualizations": []}],
    }
    w, h = relayout(items, _ids())
    assert len(items["membrane-visualizations"]) == 1  # only cytosol holds a node PathWhiz draws
    assert [lb["label-text"] for lb in items["label-locations"]] == ["cytosol"]
    lb = items["label-locations"][0]
    assert (lb["visualization-template-id"], lb["scale-x"]) == (235, "2.0")
    top = min(it["y"] for it in items["compound-locations"])
    assert lb["y"] + (40 + 16) * 2 < top  # drawn label box ends above the first node
    assert items["nucleic-acid-locations"][0]["width"] == "150" and items["nucleic-acid-locations"][0]["height"] == "70"
    nums = [int(t) for t in re.findall(r"-?\d+", items["edges"][2]["path"])]  # same shape, moved under the content
    assert items["edges"][2]["path"].startswith("M") and " L" in items["edges"][2]["path"]
    assert (nums[2] - nums[0], nums[3] - nums[1]) == (100, 20) and nums[2] + 40 <= w and nums[3] + 40 <= h
    assert nums[1] >= max(it["y"] + int(it["height"]) for it in items["compound-locations"])
    every = items["compound-locations"] + items["protein-locations"] + items["nucleic-acid-locations"]
    assert all(0 <= it["x"] and it["x"] + int(it["width"]) <= w and it["y"] + int(it["height"]) <= h for it in every)
