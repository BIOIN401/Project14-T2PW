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


def test_html_is_self_contained():
    ir = _ir(FIXTURES[0])
    html = dg.render_html(ir, title="T <x>")
    assert html.startswith("<!doctype html>") and "<title>T &lt;x&gt;</title>" in html
    assert re.search(r"<svg [^>]*class=\"dg\"", html) and "src=" not in html and "http://" not in html.replace("http://www.w3.org/2000/svg", "")
