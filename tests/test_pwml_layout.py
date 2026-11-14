"""PWML layout v2 (``T2PW_PWML_LAYOUT=v2``, src/t2pw/pwml/layout.py): geometry changes, content does not."""
from __future__ import annotations

import copy
import json
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.layout import _path, _pava, layout_version  # noqa: E402
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
        elif (tag == "path" and ptag == "edge") or (tag == "side" and ptag.startswith("reaction-")):
            el.text = "#"
        elif tag == "options" and ptag == "edge" and el.text:
            o = json.loads(el.text)
            el.text = json.dumps({k: ("#" if k.endswith("_path") else v) for k, v in o.items()}, sort_keys=True)
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
    # one closed membrane per compartment; each node sits in a membrane holding only its own state
    rects = []
    for m in root.iter("membrane-visualization"):
        nums = [int(t) for t in m.findtext("path").replace("M", " ").replace("C", " ").split()]
        xs, ys = nums[0::2], nums[1::2]
        assert (xs[0], ys[0]) == (xs[-1], ys[-1])
        rects.append((min(xs), min(ys), max(xs), max(ys)))
    states = {s for s, _ in boxes}
    assert len(rects) == len(states)
    for r in rects:
        inside = {s for s, b in boxes if r[0] <= b[0] and r[1] <= b[1] and b[2] <= r[2] and b[3] <= r[3]}
        assert len(inside) == 1
    assert len(list(root.iter("label-location"))) == len(rects)
    # every visible edge stays inside the canvas
    for e in root.iter("edge"):
        nums = [int(t) for t in e.findtext("path").replace("M", " ").replace("C", " ").split()]
        assert all(0 <= v <= W for v in nums[0::2]) and all(0 <= v <= H for v in nums[1::2])


def test_path_is_straight_cubic_segments():
    path, cp1, cp2 = _path([(0, 0), (30, 0), (30, 60)])
    assert path == "M0 0 C10 0 20 0 30 0 C30 20 30 40 30 60"
    assert cp1 == (10, 0) and cp2 == (30, 40)
    assert _path([(5, 5)])[0] == "M5 5 C5 5 5 5 5 5"


def test_pava_keeps_separation_and_fits_targets():
    assert _pava([0, 0, 0], [0, 10, 10]) == [-10.0, 0.0, 10.0]
    assert _pava([0, 100], [0, 10]) == [0.0, 100.0]
