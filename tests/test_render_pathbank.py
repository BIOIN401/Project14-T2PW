"""VIS-3: the PathBank-style result image (t2pw.render.pathbank) and its service hook."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from t2pw.render import pathbank
from t2pw.service import pathway

ROOT = Path(__file__).resolve().parents[1]
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

MINI = """<?xml version="1.0"?>
<super-pathway-visualization>
 <pathway-visualization-contexts><pathway-visualization-context><pathway-visualization>
  <height>800</height><width>1000</width><background-color>#FFFFFF</background-color>
  <pathway>
   <compounds><compound><id>7</id><name>L-Alanine &amp; $x$</name></compound>
              <compound><id>8</id><name>Adenosine triphosphate</name><short-name>ATP</short-name></compound></compounds>
   <proteins><protein><id>7</id><name>Alanine aminotransferase</name><gene-name>GPT</gene-name></protein></proteins>
  </pathway>
  <compound-locations>
   <compound-location><id>1</id><compound-id>7</compound-id><visualization-template-id>81</visualization-template-id>
     <hidden>false</hidden><x>100</x><y>100</y><zindex>10</zindex><width>200</width><height>190</height></compound-location>
   <compound-location><id>2</id><compound-id>8</compound-id><visualization-template-id>42</visualization-template-id>
     <hidden>false</hidden><x>400</x><y>100</y><zindex>10</zindex><width>50</width><height>30</height></compound-location>
   <compound-location><id>3</id><compound-id>8</compound-id><visualization-template-id>42</visualization-template-id>
     <hidden>true</hidden><x>400</x><y>300</y><zindex>10</zindex><width>50</width><height>30</height></compound-location>
  </compound-locations>
  <protein-locations>
   <protein-location><id>1</id><protein-id>7</protein-id><visualization-template-id>6</visualization-template-id>
     <hidden>false</hidden><x>500</x><y>400</y><zindex>8</zindex><label-type>gene</label-type>
     <width>160</width><height>80</height></protein-location>
  </protein-locations>
  <edges>
   <edge><id>1</id><path>M1000 415 C1000 390 999 385 1000 355</path><visualization-template-id>5</visualization-template-id>
     <hidden>false</hidden><zindex>18</zindex><option:start_arrow>true</option:start_arrow><option:end_arrow>false</option:end_arrow></edge>
   <edge><id>2</id><path>M300 300 C350 300 400 300 450 300</path><visualization-template-id>5</visualization-template-id>
     <hidden>false</hidden><zindex>18</zindex><options>{"end_arrow": true, "start_flat_arrow": false}</options></edge>
   <edge><id>3</id><path>M540 839 C540 839 540 839 540 839</path><visualization-template-id>5</visualization-template-id>
     <hidden>false</hidden><zindex>18</zindex><options>{"end_arrow": true}</options></edge>
  </edges>
  <transport-visualizations><transport-visualization><id>1</id>
   <transport_compound_visualizations><transport-compound-visualization><id>1</id><compound-location-id>1</compound-location-id>
     <edge-id>2</edge-id><side>Left</side></transport-compound-visualization></transport_compound_visualizations>
  </transport-visualization></transport-visualizations>
  <drawable-element-locations>
   <drawable-element-location><id>1</id><x>600</x><y>500</y><scale-x>0.5</scale-x><scale-y>0.5</scale-y><degree>30</degree>
     <zindex>2</zindex><drawable-image-id>1</drawable-image-id><width>490</width><height>240</height></drawable-element-location>
   <drawable-element-location><id>2</id><x>10</x><y>10</y><scale-x>1</scale-x><scale-y>1</scale-y><degree>0</degree>
     <zindex>2</zindex><drawable-image-id>987654</drawable-image-id><width>100</width><height>100</height></drawable-element-location>
  </drawable-element-locations>
  <membrane-visualizations><membrane-visualization><id>1</id>
   <path>M50 100 C50 75 75 50 100 50 L900 50 L900 750 L50 750 z</path><visualization-template-id>84</visualization-template-id>
   <zindex>6</zindex></membrane-visualization></membrane-visualizations>
  <label-locations><label-location><id>1</id><visualization-template-id>235</visualization-template-id>
   <label-text>Mitochondria</label-text><x>60</x><y>40</y><zindex>20</zindex><scale-x>1.5</scale-x><scale-y>1.5</scale-y>
   <width>200</width><height>15</height></label-location></label-locations>
  <zoom-visualizations><zoom-visualization><id>1</id><p1x>800</p1x><p1y>100</p1y><p2x>300</p2x><p2y>400</p2y>
   <p3x>700</p3x><p3y>700</p3y><color>#FFEBEB</color><zindex>4</zindex></zoom-visualization></zoom-visualizations>
 </pathway-visualization></pathway-visualization-context></pathway-visualization-contexts>
</super-pathway-visualization>
"""


@pytest.fixture
def mini(tmp_path: Path) -> Path:
    p = tmp_path / "mini.pwml"
    p.write_text(MINI, encoding="utf-8")  # undeclared option: prefix, as in PathBank's own files
    return p


def _model(path: Path):
    root = pathbank.read_xml_with_repairs(str(path))
    return pathbank._items(root, pathbank.find_first_visualization(root))


def test_chevron_matches_pathbank_arrow_path():
    # PathBank PW000001.svg: edge "M1000 415 C1000 390 ..." with start_arrow draws
    # arrow_path "M 985 389.0192 L 1000 415 L 1015 389.0192" (two 30-unit legs at +-30 degrees)
    pts = pathbank.chevron((1000, 415), (0, 25))
    assert pts[1] == (1000, 415)
    assert sorted((round(x, 3), round(y, 3)) for x, y in (pts[0], pts[2])) == [(985.0, 389.019), (1015.0, 389.019)]
    flat = pathbank.chevron((1140, 424), (0, -10), flat=True)
    assert sorted(round(x) for x, _ in flat) == [1125, 1140, 1155] and {round(y) for _, y in flat} == {424}


def test_arrow_options_read_both_encodings(mini: Path):
    root = pathbank.read_xml_with_repairs(str(mini))
    edges = {e.findtext("id"): e for e in root.iter() if pathbank._local(e.tag) == "edge"}
    assert pathbank._arrow_options(edges["1"]) == {"start_arrow": True, "end_arrow": False}
    assert pathbank._arrow_options(edges["2"]) == {"end_arrow": True, "start_flat_arrow": False}


def test_parse_names_hidden_and_unknown_art(mini: Path):
    root = pathbank.read_xml_with_repairs(str(mini))
    names = pathbank._definitions(root)
    assert names[("compound", "8")] == ("Adenosine triphosphate", "ATP")
    assert names[("protein", "7")] == ("Alanine aminotransferase", "GPT")
    items, w, h = _model(mini)
    # 3 edges + 2 visible compounds + 1 protein + 1 known image (id 987654 skipped) + membrane + label + zoom
    assert len(items) == 10
    assert pathbank._art("987654") is None and pathbank._art("x/../1") is None
    assert pathbank._art("1") is not None  # PathBank's mitochondrion ships as package data


def test_zero_length_edge_has_no_arrow_and_does_not_raise():
    mp = pathbank.svg_path_to_mpl_path("M540 839 C540 839 540 839 540 839")
    assert pathbank._ends(mp) is None


@pytest.mark.parametrize("fmt, magic", [("png", PNG_MAGIC), ("svg", b"<svg")])
def test_render_mini_png_and_svg(mini: Path, fmt: str, magic: bytes):
    data = pathbank.render_pwml(mini, fmt)
    assert magic in data[:400]


def test_render_png_size_is_capped(mini: Path):
    from PIL import Image
    import io

    im = Image.open(io.BytesIO(pathbank.render_pwml(mini, "png", max_px=500)))
    assert max(im.size) <= 500 and abs(im.size[0] / im.size[1] - 1000 / 800) < 0.02


def test_committed_pathbank_reference_renders_with_its_organelle_art():
    ref = ROOT / "reference" / "PW000001.pwml"
    if not ref.exists():
        pytest.skip("reference PWML not in this tree")
    items, _, _ = _model(ref)
    assert items
    assert pathbank.render_pwml(ref).startswith(PNG_MAGIC)


def test_art_catalog_matches_package_files():
    art = ROOT / "src" / "t2pw" / "render" / "pathbank_art"
    cat = json.loads((art / "catalog.json").read_text(encoding="utf-8"))
    assert len(cat) == 41
    for image_id, row in cat.items():
        assert (art / f"{image_id}.png").is_file()
        if row["original"].endswith(".svg"):
            assert (art / "svg" / row["original"]).is_file()


# -- the service hook ------------------------------------------------------------------------------------

@pytest.mark.parametrize("flag", [None, "", "off", "draft", "PATHBANKX"])
def test_flag_off_never_calls_the_renderer(monkeypatch, mini: Path, flag):
    if flag is None:
        monkeypatch.delenv("T2PW_RESULT_RENDER", raising=False)
    else:
        monkeypatch.setenv("T2PW_RESULT_RENDER", flag)
    monkeypatch.setattr(pathbank, "render_pwml", lambda *a, **k: pytest.fail("renderer called with the flag off"))
    assert pathway._pathbank_png(mini) is None  # caller keeps state["refinement_graph_bytes"] unchanged


def test_flag_on_renders_the_pwml(monkeypatch, mini: Path):
    monkeypatch.setenv("T2PW_RESULT_RENDER", " PathBank ")
    assert pathway._pathbank_png(mini).startswith(PNG_MAGIC)


@pytest.mark.parametrize("make", ["none", "missing", "broken"])
def test_flag_on_falls_back_when_pwml_missing_or_broken(monkeypatch, tmp_path: Path, make):
    monkeypatch.setenv("T2PW_RESULT_RENDER", "pathbank")
    path = {"none": None, "missing": tmp_path / "nope.pwml", "broken": tmp_path / "broken.pwml"}[make]
    if make == "broken":
        path.write_text("<super-pathway-visualization><unclosed>", encoding="utf-8")
    assert pathway._pathbank_png(path) is None


def test_flag_on_skips_the_render_once_the_deadline_has_passed(monkeypatch, mini: Path):
    import time

    monkeypatch.setenv("T2PW_RESULT_RENDER", "pathbank")
    assert pathway._pathbank_png(mini, time.time() + 600).startswith(PNG_MAGIC)
    monkeypatch.setattr(pathbank, "render_pwml", lambda *a, **k: pytest.fail("rendered after the deadline"))
    assert pathway._pathbank_png(mini, time.time() - 1) is None


# -- edge colour is the template's, never a role (PathBank: 591/591 template-5 edges are gold) -----------

EDGES_ONLY = """<?xml version="1.0"?>
<super-pathway-visualization><pathway-visualization-contexts><pathway-visualization-context><pathway-visualization>
 <height>500</height><width>500</width>
 <edges>
  <edge><id>1</id><path>M10 10 L100 10</path><visualization-template-id>148</visualization-template-id><zindex>18</zindex></edge>
  <edge><id>2</id><path>M10 20 L100 20</path><visualization-template-id>149</visualization-template-id><zindex>18</zindex></edge>
  <edge><id>3</id><path>M10 30 L100 30</path><visualization-template-id>5</visualization-template-id><zindex>18</zindex></edge>
  <edge><id>4</id><path>M10 40 L100 40</path><visualization-template-id>83</visualization-template-id><zindex>18</zindex></edge>
  <edge><id>5</id><path>M10 50 L100 50</path><visualization-template-id>5</visualization-template-id><zindex>18</zindex></edge>
 </edges>
 <transport-visualizations><transport-visualization><id>1</id><transport_compound_visualizations>
  <transport-compound-visualization><id>1</id><edge-id>4</edge-id></transport-compound-visualization>
  <transport-compound-visualization><id>2</id><edge-id>5</edge-id></transport-compound-visualization>
 </transport_compound_visualizations></transport-visualization></transport-visualizations>
 <interaction-visualizations><interaction-visualization><id>1</id><interaction-id>9</interaction-id>
  <interaction_protein_complex_visualizations><interaction-protein-complex-visualization><id>1</id><edge-id>3</edge-id>
  </interaction-protein-complex-visualization></interaction_protein_complex_visualizations></interaction-visualization>
 </interaction-visualizations>
 <interactions><interaction><id>9</id><interaction-type>Inhibition</interaction-type></interaction></interactions>
</pathway-visualization></pathway-visualization-context></pathway-visualization-contexts></super-pathway-visualization>
"""


def test_edge_colour_comes_from_the_template_only(tmp_path: Path):
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure

    p = tmp_path / "edges.pwml"
    p.write_text(EDGES_ONLY, encoding="utf-8")
    items, _, _ = _model(p)
    colours = []
    for _, _, draw in items:
        ax = Figure().add_subplot()
        draw(ax)
        colours.append(ax.patches[0].get_edgecolor())
    want = ["#FF0000",  # 148 inhibition: red
            "#2AC5E5",  # 149 activation: cyan
            pathbank.GOLD,  # template 5 on an Inhibition interaction: still gold
            "#44ee22",  # 83 transport: green
            pathbank.GOLD]  # template 5 on a transport: gold, as PathWhiz draws it
    assert colours == [to_rgba(c) for c in want]


# -- the run_pathway tail that writes out/pathway.png ---------------------------------------------------

def test_run_pathway_writes_the_pathbank_render_with_the_flag_on(monkeypatch, tmp_path: Path):
    from test_service_run_pathway import World, _run

    World(monkeypatch, tmp_path)
    monkeypatch.setenv("T2PW_RESULT_RENDER", "pathbank")
    seen = []
    monkeypatch.setattr(pathbank, "render_pwml", lambda path, *a, **k: seen.append(Path(path)) or b"NEW-PNG")
    res = _run(tmp_path)
    assert seen == [res.pwml_path]
    assert res.image_path == tmp_path / "out" / pathway.IMAGE_NAME
    assert res.image_path.read_bytes() == b"NEW-PNG"


@pytest.mark.parametrize("flag", ["pathbank", "off"])
def test_run_pathway_keeps_the_draft_graph_when_the_render_raises_or_the_flag_is_off(monkeypatch, tmp_path, flag):
    from test_service_run_pathway import World, _run

    World(monkeypatch, tmp_path)
    monkeypatch.setenv("T2PW_RESULT_RENDER", flag)

    def boom(*a, **k):
        raise RuntimeError("render failed")

    monkeypatch.setattr(pathbank, "render_pwml", boom)
    res = _run(tmp_path)
    assert res.image_path.read_bytes() == b"\x89PNG"  # World's refinement_graph_bytes, unchanged
