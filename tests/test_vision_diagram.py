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
        assert hashlib.sha256(svg.encode()).hexdigest() == "e6d9bc348ea6ed7dfa955c546c65c95efe99a0cf0ab325f278367eed32568656"


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


# --------------------------------------------------------------------------- DRAW-RENDERER
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_every_style_order_and_wrap_draws_everything(path):
    ir = _ir(path)
    for style in ("bands", "cell", "spine"):
        for tiers in (1, 2, 3):
            for order in ("flow", "anat"):
                svg = dg._R.render(ir, style, "b", tiers=tiers, order=order)[0]
                assert dg.selfcheck(ir, svg)["failures"] == [], (style, tiers, order)


def test_display_grouping_keeps_names_and_types_membranes():
    R = dg._R
    assert R.name_tokens("periplasmic leaflet of the inner membrane") == R.name_tokens("inner membrane periplasmic leaflet")
    fam = R.display_family
    assert fam("periplasmic leaflet of inner membrane", "membrane") == fam("inner membrane", "membrane") != fam("outer membrane", "membrane")
    assert fam("intracellular", "organelle") == fam("cytoplasm", "cytosol") and fam("nucleus", "organelle") != fam("cytosol", "cytosol")
    assert R.membrane_type("mitochondrial matrix") in R.DOUBLE and R.membrane_type("Golgi apparatus") == "mt-golgi"
    assert R.membrane_type("thylakoid lumen") not in R.DOUBLE and R.membrane_type("cytosol") == ""
    assert R.is_currency("Hydrogen Ion") and R.is_currency(" water ") and not R.is_currency("pyruvate")


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_line_bridges_only_where_lines_cross(path):
    from t2pw.vision.checks import layout_metrics as LM
    m = LM.svg_layout_metrics(dg.render_svg(_ir(path)))
    assert m["hops"] <= m["crossings"] and m["node_overlaps"] == 0


# --------------------------------------------------------------------------- DRAW-RENDERER review round 1
def _tiny(locs, compounds, reactions=(), transports=()):
    """A hand-made IR: one state per location name; reactions (key, state, [left], [right]); transports
    (key, compound, from state, to state)."""
    ref = lambda k: {"entity_type": "compound", "entity_key": k}  # noqa: E731
    return {"pathway": {"name": "test pathway"}, "cell_types": [],
            "subcellular_locations": [{"key": "scl_%d" % i, "name": n} for i, n in enumerate(locs)],
            "biological_states": [{"key": "bs_%d" % i, "name": n + " state", "subcellular_location_key": "scl_%d" % i,
                                   "cell_type_key": None} for i, n in enumerate(locs)],
            "entities": {"compounds": [{"key": k, "name": n} for k, n in compounds]},
            "processes": {"reactions": [{"key": k, "name": k, "biological_state_key": s, "direction": ">",
                                         "left": [ref(x) for x in L], "right": [ref(x) for x in R], "enzymes": []}
                                        for k, s, L, R in reactions],
                          "transports": [{"key": k, "name": k, "transporters": [],
                                          "transport_elements": [dict(ref(c), left_biological_state_key=a, right_biological_state_key=b)]}
                                         for k, c, a, b in transports],
                          "interactions": []}}


@pytest.mark.parametrize("name, double", [("nucleus", True), ("nucleoid", False), ("nucleolus", False)])
def test_double_membrane_only_for_real_envelopes(name, double):
    ir = _tiny([name], [("c1", "alpha"), ("c2", "beta")], reactions=[("rxn_1", "bs_0", ["c1"], ["c2"])])
    svg = dg.render_svg(ir, style="bands")
    assert dg.selfcheck(ir, svg)["failures"] == []
    assert ('class="mline2"' in svg) is double and (dg._R.membrane_type(name) in dg._R.DOUBLE) is double


def test_transported_currency_keeps_its_link_to_the_reactions_that_use_it():
    cpds = [("h", "H+"), ("a", "alpha"), ("b", "beta"), ("c", "gamma")]
    rx = [("rxn_1", "bs_1", ["a", "h"], ["b"]), ("rxn_2", "bs_1", ["b", "h"], ["c"])]
    pumped = _tiny(["cytosol", "mitochondrial matrix"], cpds, rx, transports=[("tr_1", "h", "bs_0", "bs_1")])
    alone = _tiny(["cytosol", "mitochondrial matrix"], cpds, rx)
    for ir, drawings in ((pumped, 1), (alone, 2)):  # cargo: one shared H+ in the matrix; otherwise a local copy per step
        svg = dg.render_svg(ir)
        assert dg.selfcheck(ir, svg)["failures"] == []
        core = svg.split('class="view-core"')[1] if 'class="view-core"' in svg else svg
        assert core.count('data-ent="compound:h" data-comp="mitochondrial matrix"') == drawings


def test_chained_transports_in_one_membrane_keep_flow_order():
    # tr_10 sorts before tr_2 but comes after it in the flow: the spreading edge must not close a cycle
    ir = _tiny(["cytosol", "inner membrane", "periplasm"], [("x", "cargo")],
               transports=[("tr_2", "x", "bs_0", "bs_1"), ("tr_10", "x", "bs_1", "bs_2")])
    m = dg._R.build(ir)
    m._ent = {("compound", "x"): {"key": "x", "name": "cargo"}}
    dg._R.layout(m, "bands")
    t2, t10 = m.nodes[("J", "T2")], m.nodes[("J", "T10")]
    assert t2.layer < t10.layer  # different columns, in flow order
    for style in ("bands", "cell"):
        assert dg.selfcheck(ir, dg._R.render(ir, style, "b")[0])["failures"] == []
