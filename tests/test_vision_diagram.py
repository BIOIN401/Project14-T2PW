import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import t2pw.vision.diagram as dg

DATA = Path(__file__).parent / "data" / "vision" / "diagram"
FIXTURES = sorted(DATA.glob("*.json"))


def _ir(p):
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_svg_valid_complete_and_checked(path):
    ir = _ir(path)
    svg = dg.render_svg(ir)
    root = ET.fromstring(svg)  # parses as XML
    assert root.tag.endswith("svg") and "<style>" in svg
    assert svg.count("data-ent=") >= 1  # one element per drawn molecule
    P = ir["processes"]
    for kind in ("reactions", "transports", "interactions"):
        for p in P.get(kind, []):  # every process key is carried by a drawn element or the hidden list
            assert p["key"] in svg
    assert dg.selfcheck(ir, svg)["failures"] == []


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_deterministic_and_metrics(path):
    ir = _ir(path)
    assert dg.render_svg(ir) == dg.render_svg(ir)
    m = dg.diagram_metrics(ir)
    assert m["style"] in ("bands", "cell", "spine") and m["drawings"] >= 1 and m["W"] > 0


def test_reaction_transport_fixture_counts_and_pin():
    ir = _ir(DATA / "ir_reaction_transport.json")
    P = ir["processes"]
    assert P["reactions"] and P["transports"] and any(r.get("enzymes") for r in P["reactions"])
    svg = dg.render_svg(ir)
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    jn = [g for g in root.iter(ns + "g") if "jn" in (g.get("class") or "").split() and g.get("data-k")]
    assert sum("reaction" in g.get("class") for g in jn) == len(P["reactions"])
    assert sum("transport" in g.get("class") for g in jn) == len(P["transports"])
    drawn = [g for g in root.iter(ns + "g") if g.get("data-ent")]
    n_ent = sum(len(ir["entities"][k]) for k in ("compounds", "proteins", "protein_complexes"))
    assert len(drawn) >= n_ent
    assert {g.get("data-ent").split(":")[0] for g in drawn} >= {"compound", "protein_complex"}
    assert dg.selfcheck(ir, svg)["failures"] == []
    if "segoeui" in Path(str(getattr(dg._R._FONT, "path", ""))).name.lower():  # layout depends on the measuring font
        assert hashlib.sha256(svg.encode()).hexdigest() == "f4a4de0dbcbc9d8b14b1a77e1b09918a05318f9b0a265d113b0e55daa135805e"


def test_bad_font_env_falls_through(tmp_path):
    bad = tmp_path / "x.ttf"
    bad.write_text("not a font")
    import importlib
    import os
    os.environ["T2PW_DIAGRAM_FONT"] = str(bad)
    try:
        importlib.reload(dg._R)
    finally:
        del os.environ["T2PW_DIAGRAM_FONT"]
        importlib.reload(dg._R)


def test_html_is_self_contained():
    ir = _ir(FIXTURES[0])
    html = dg.render_html(ir, title="T <x>")
    assert html.startswith("<!doctype html>") and "<title>T &lt;x&gt;</title>" in html
    assert re.search(r"<svg [^>]*class=\"dg\"", html) and "src=" not in html and "http://" not in html.replace("http://www.w3.org/2000/svg", "")
