"""V7b: layout metrics on tiny hand-made PWML / SVG fixtures with one known defect each."""
import json
from pathlib import Path

import pytest

import t2pw.vision.diagram as dg
from t2pw.vision.checks import layout_metrics as LM

DATA = Path(__file__).parent / "data" / "vision" / "diagram"


# --------------------------------------------------------------------------- fixture builder
def _el(tag, **kids):
    return "<%s>%s</%s>" % (tag, "".join("<%s>%s</%s>" % (k.replace("_", "-"), v, k.replace("_", "-")) for k, v in kids.items()), tag)


def cloc(i, comp, x, y, state="s", w=160, h=60, hidden="false"):
    return _el("compound-location", id=i, compound_id=comp, biological_state_id=state, x=x, y=y, width=w, height=h, hidden=hidden)


def reaction(i, left, right):
    side = lambda tag, ids: "<%s>%s</%s>" % (tag, "".join(_el("e", element_type="Compound", element_id=c) for c in ids), tag)
    return "<reaction><id>%s</id>%s%s</reaction>" % (i, side("reaction-left-elements", left), side("reaction-right-elements", right))


def rviz(rid, locs, enzymes=(), state="s"):
    cv = "".join(_el("reaction-compound-visualization", compound_location_id=l) for l in locs)
    ev = "".join(_el("reaction-enzyme-visualization", protein_complex_visualization_id=e) for e in enzymes)
    return "<reaction-visualization><reaction-id>%s</reaction-id><biological-state-id>%s</biological-state-id>%s%s</reaction-visualization>" % (rid, state, cv, ev)


def pwml(*, compounds=("alpha", "beta", "gamma"), cloc_=(), reactions=(), rviz_=(), ploc=(), pcv=(), edges=(), labels=(), membranes=()):
    comp = "".join(_el("compound", id=i + 1, name=n) for i, n in enumerate(compounds))
    pl = "".join(_el("protein-location", id=i, x=x, y=y, width=150, height=70) for i, x, y in ploc)
    pv = "".join("<protein-complex-visualization><id>%s</id>%s</protein-complex-visualization>" % (
        i, "".join(_el("protein-complex-protein-visualization", protein_location_id=p) for p in ps)) for i, ps in pcv)
    ed = "".join(_el("edge", id=i, path=p) for i, p in edges)
    lb = "".join(_el("label-location", label_text=t, x=x, y=y, scale_x=1, visualization_template_id=15) for t, x, y in labels)
    mb = "".join(_el("membrane-visualization", path=p) for p in membranes)
    return ("<pathway-visualizations><pathway-visualization><width>2000</width><height>1000</height>"
            "<compounds>%s</compounds><compound-locations>%s</compound-locations><protein-locations>%s</protein-locations>"
            "<protein-complex-visualizations>%s</protein-complex-visualizations><reactions>%s</reactions>"
            "<reaction-visualizations>%s</reaction-visualizations><edges>%s</edges><label-locations>%s</label-locations>"
            "<membrane-visualizations>%s</membrane-visualizations></pathway-visualization></pathway-visualizations>"
            % (comp, "".join(cloc_), pl, pv, "".join(reactions), "".join(rviz_), ed, lb, mb)).encode("utf-8")


def two_step(x1, x2, y2=0):
    """alpha -> beta (reaction 1, drawn at x1) then beta -> gamma (reaction 2, drawn at x2, y2): beta is drawn twice."""
    return pwml(cloc_=[cloc(1, 1, x1, 0), cloc(2, 2, x1, 100), cloc(3, 2, x2, y2), cloc(4, 3, x2, y2 + 100)],
                reactions=[reaction(1, [1], [2]), reaction(2, [2], [3])], rviz_=[rviz(1, [1, 2]), rviz(2, [3, 4])])


# --------------------------------------------------------------------------- PWML
def test_overlap_known_pair():
    m = LM.pwml_layout_metrics(pwml(cloc_=[cloc(1, 1, 0, 0), cloc(2, 2, 100, 0), cloc(3, 3, 400, 0)]))
    assert m["nodes"] == 3 and m["node_overlaps"] == 1 and m["examples"]["overlaps"] == [("1", "2")]
    assert LM.pwml_layout_metrics(pwml(cloc_=[cloc(1, 1, 0, 0), cloc(2, 2, 160, 0)]))["node_overlaps"] == 0  # touching is not overlapping
    assert LM.pwml_layout_metrics(pwml(cloc_=[cloc(1, 1, 0, 0), cloc(2, 2, 100, 0, hidden="true")]))["node_overlaps"] == 0  # hidden is not drawn


def test_crossing_known_pair():
    edges = [("1", "M0 0 L100 100"), ("2", "M0 100 L100 0"), ("3", "M500 0 L600 100"), ("4", "M0 0 L0 50")]  # 4 shares an end point with 1
    m = LM.pwml_layout_metrics(pwml(edges=edges))
    assert m["edges"] == 4 and m["edge_crossings"] == 1 and m["examples"]["crossings"] == [("1", "2")]


def test_reading_order_violation_and_forward_cases():
    back = LM.pwml_layout_metrics(two_step(800, 100))  # the consumer of beta sits left of its producer
    assert (back["precedence_pairs"], back["reading_order_violations"]) == (1, 1)
    fwd = LM.pwml_layout_metrics(two_step(100, 800))
    assert (fwd["precedence_pairs"], fwd["reading_order_violations"]) == (1, 0)
    wrap = LM.pwml_layout_metrics(two_step(800, 100, y2=200))  # wrapped to a row 100+ px lower: reads forward
    assert (wrap["precedence_pairs"], wrap["reading_order_violations"]) == (1, 0)


def test_currency_compound_is_not_a_reading_order_claim():
    m = LM.pwml_layout_metrics(pwml(compounds=("alpha", "ATP", "gamma"), cloc_=[cloc(1, 1, 800, 0), cloc(2, 2, 800, 100), cloc(3, 2, 100, 0), cloc(4, 3, 100, 100)],
                                    reactions=[reaction(1, [1], [2]), reaction(2, [2], [3])], rviz_=[rviz(1, [1, 2]), rviz(2, [3, 4])]))
    assert m["precedence_pairs"] == 0 and m["reading_order_violations"] == 0


def test_duplicated_compound_in_one_compartment():
    m = LM.pwml_layout_metrics(two_step(100, 800))  # beta drawn at two places in state s
    assert (m["duplicated_compound_locations"], m["duplicated_compound_keys"], m["examples"]["duplicated"]) == (1, 1, [["beta", 2]])
    assert m["used_compound_locations"] == 4 and m["distinct_compound_state"] == 3
    other = pwml(cloc_=[cloc(1, 1, 0, 0), cloc(2, 2, 0, 100), cloc(3, 2, 800, 0, state="t"), cloc(4, 3, 800, 100, state="t")],  # beta in two compartments
                 reactions=[reaction(1, [1], [2]), reaction(2, [2], [3])], rviz_=[rviz(1, [1, 2]), rviz(2, [3, 4], state="t")])
    assert LM.pwml_layout_metrics(other)["duplicated_compound_locations"] == 0


def test_enzyme_distance():
    def one(px):
        return pwml(cloc_=[cloc(1, 1, 0, 0)], reactions=[reaction(1, [1], [])], rviz_=[rviz(1, [1], enzymes=["v1"])],
                    ploc=[(9, px, 0)], pcv=[("v1", [9])])
    far, near = LM.pwml_layout_metrics(one(400)), LM.pwml_layout_metrics(one(100))
    assert (far["enzymes_drawn"], far["enzyme_distance_gt200"], far["enzyme_distance_max"]) == (1, 1, 395)  # (475,35) to (80,30)
    assert near["enzyme_distance_gt200"] == 0 and near["enzyme_distance_max"] == 95
    assert LM.pwml_layout_metrics(pwml(cloc_=[cloc(1, 1, 0, 0)], reactions=[reaction(1, [1], [])], rviz_=[rviz(1, [1])]))["enzymes_drawn"] == 0


def test_label_fit_and_placement():
    box = "M0 0 L2000 0 L2000 1000 L0 1000 Z"
    m = LM.pwml_layout_metrics(pwml(membranes=[box], labels=[("ATP", 10, 10), ("W" * 28, 10, 200), ("__auto_state__", 10, 400), ("far", 5000, 10)]))
    assert m["labels"] == 4 and m["labels_too_wide"] == 1 and m["examples"]["labels_too_wide"] == ["W" * 28]
    assert m["internal_labels"] == 1 and m["labels_outside_box"] == 1
    assert LM.pwml_layout_metrics(pwml(membranes=[box], cloc_=[cloc(1, 1, 20, 40)], labels=[("ATP", 10, 10)]))["label_node_overlaps"] == 1


def test_label_lines_wrap_like_the_template():
    assert LM.label_lines("aa bb cc") == ["aa bb cc"]
    assert LM.label_lines("x" * 60) == ["x" * 28, "x" * 28, "xxxx"]
    assert LM.label_lines("w " * 20) == ["w " * 13 + "w", "w w w w w w"]  # 27 characters fit, the 15th word wraps


def test_not_a_pwml_raises():
    with pytest.raises(ValueError):
        LM.pwml_layout_metrics(b"<a/>")


# --------------------------------------------------------------------------- SVG
SVG = ('<svg xmlns="http://www.w3.org/2000/svg"><g class="view-core"><g class="edges">'
       '<path class="e prod" d="M0,50 L100,50"/><path class="e prod" d="M50,0 L50,100"/>'
       '<path class="e cat" data-p="R1" data-ek="cat" data-e="protein:E" d="M10,300 L400,300"/>'
       '<path class="e cat" data-p="T1" data-ek="cat" data-e="protein:P" d="M0,500 L500,500"/>'
       '</g><g class="nodes">'
       '<g class="n protein" data-ent="protein:E"><rect x="0" y="280" width="20" height="40"/><text>E</text></g>'
       '<g class="jn reaction" data-p="R1" data-k="r1"><rect x="390" y="280" width="20" height="40"/></g>'
       '<g class="n protein" data-ent="protein:P"><rect x="0" y="480" width="20" height="40"/><text>A very long na…</text></g>'
       '<g class="jn transport" data-p="T1" data-k="t1"><rect x="490" y="480" width="20" height="40"/></g>'
       '</g></g></svg>')


def test_svg_hand_made():
    m = LM.svg_layout_metrics(SVG)
    assert m["crossings"] == 1 and m["bends"] == 0 and m["edges"] == 4 and m["drawings"] == 2
    assert (m["enzyme_links"], m["enzyme_links_gt200"], m["enzyme_link_max"]) == (1, 1, 390)  # the transporter pill is not an enzyme link
    assert m["node_overlaps"] == 0 and m["labels_truncated"] == 1
    assert LM.svg_layout_metrics(SVG.replace('x="390"', 'x="10"'))["node_overlaps"] == 1  # the reaction box moved onto the enzyme box


def test_svg_rounded_corner_path_gives_the_original_vertices():
    assert LM._svg_path_points("M0.0,0.0 L41.0,0.0 Q50.0,0.0 50.0,9.0 L50.0,100.0") == [(0, 0), (50, 0), (50, 100)]
    assert LM._svg_path_points("M1,2 L3,4") == [(1, 2), (3, 4)]


@pytest.mark.parametrize("path", sorted(DATA.glob("*.json")), ids=lambda p: p.stem)
def test_svg_metrics_equal_the_model_metrics(path):
    """Reuse, not a second definition: parsing our SVG gives what diagram_metrics measures on the model it was drawn from."""
    ir = json.loads(path.read_text(encoding="utf-8"))
    a, b = dg.diagram_metrics(ir), LM.svg_layout_metrics(dg.render_svg(ir))
    for k in ("crossings", "bends", "enzyme_links", "enzyme_links_gt200"):
        assert a[k] == b[k], k
    assert abs(a["enzyme_link_max"] - b["enzyme_link_max"]) <= 2


# --------------------------------------------------------------------------- gallery (no Docker, no browser)
def test_gallery_assemble_writes_page_and_small_png(tmp_path):
    from PIL import Image

    from t2pw.vision.checks import gallery as GA
    work = tmp_path / "w"
    d = work / "14"
    d.mkdir(parents=True)
    Image.effect_noise((1600, 900), 80).convert("RGB").save(d / "ours.png")  # noisy: forces the size loop
    (d / "info.json").write_text(json.dumps({"idx": 14, "organism": "Borrelia <b>", "pmcid": "PMC1", "size": {"nr": 1},
                                             "metrics": {"pwml": {"nodes": 4, "node_overlaps": 0}, "svg": {"crossings": 2}}}), encoding="utf-8")
    out = tmp_path / "g"
    page = GA.assemble(work, out).read_text(encoding="utf-8")
    assert (out / "14_ours.png").stat().st_size <= GA.MAX_PNG
    assert "14_ours.png" in page and "no PathWhiz screenshot" in page and "Borrelia &lt;b&gt;" in page and "node_overlaps" in page


def test_gallery_refuses_a_harness_that_is_not_the_t2pw_vis_copy(tmp_path):
    from t2pw.vision.checks import gallery as GA
    (tmp_path / "run_import.py").write_text('PROJECT = "t2pw-pwi"\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        GA._check_harness(tmp_path)
    (tmp_path / "run_import.py").write_text('PROJECT = "t2pw-vis"\n', encoding="utf-8")
    GA._check_harness(tmp_path)
    assert len(GA.DEFAULT_PICK) == 6 and GA._idx("3, 4") == [3, 4] and GA._idx(None) == list(GA.DEFAULT_PICK)


# --------------------------------------------------------------------------- compartment boxes and aspect ratio (review of the gallery)
def test_compartment_box_overlap_unlabeled_and_duplicate():
    sq = lambda x0, y0, x1, y1: "M%d %d L%d %d L%d %d L%d %d Z" % (x0, y0, x1, y0, x1, y1, x0, y1)
    m = LM.pwml_layout_metrics(pwml(membranes=[sq(0, 0, 400, 400), sq(300, 300, 700, 700), sq(1000, 0, 1400, 400)],
                                    labels=[("ER", 10, 10), ("ER", 310, 310)]))  # two boxes named ER, the third has no label, A and B intersect
    assert (m["compartment_boxes"], m["compartment_box_overlaps"], m["unlabeled_compartment_boxes"], m["duplicate_compartment_boxes"]) == (3, 1, 1, 1)
    assert m["examples"]["box_overlaps"] == [(0, 1)] and m["examples"]["unlabeled_boxes"] == [2] and m["examples"]["duplicate_boxes"] == [("er", 2)]
    ok = LM.pwml_layout_metrics(pwml(membranes=[sq(0, 0, 400, 400), sq(500, 0, 900, 400)], labels=[("ER", 10, 10), ("Golgi", 510, 10), ("__auto_state__", 20, 200)]))
    assert (ok["compartment_box_overlaps"], ok["unlabeled_compartment_boxes"], ok["duplicate_compartment_boxes"]) == (0, 0, 0)  # generated names are not shown names


def test_svg_aspect_ratio_and_totals():
    wide = SVG.replace('<svg xmlns="http://www.w3.org/2000/svg">', '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 200">')
    m = LM.svg_layout_metrics(wide)
    assert (m["aspect_ratio"], m["canvas_w"], m["canvas_h"]) == (5.0, 1000.0, 200.0) and LM.svg_layout_metrics(SVG)["aspect_ratio"] is None
    t = LM._sum([dict(aspect_ratio=5.0, crossings=1), dict(aspect_ratio=2.0, crossings=2), dict(aspect_ratio=None, crossings=0)])
    assert (t["crossings"], t["runs_aspect_gt_4"], t["aspect_ratio_max"]) == (3, 1, 5.0)


# --------------------------------------------------------------------------- DRAW-RENDERER readability counts
def test_svg_readability_counts_hand_made():
    block = '<g class="n compound" data-ent="compound:X"><rect x="40" y="40" width="20" height="20"/><text x="50" y="54" text-anchor="middle">X</text></g>'
    svg = SVG.replace('<g class="nodes">', '<g class="nodes">' + block).replace(
        '</g></g></svg>', '</g><g class="hops"><path class="hop" d="M45,50 A5,5 0 0 1 55,50"/></g></g></svg>')
    m = LM.svg_layout_metrics(svg)
    assert (m["long_edges"], m["non_straight"], m["edge_length"]) == (1, 0, 1090)  # 100 + 100 + 390 + 500 px
    assert m["node_edge_overlaps"] == 2  # both crossing lines run through X, which is neither line's end
    assert m["hops"] == 1 and m["duplicate_names"] == 0 and m["score"] > LM.svg_layout_metrics(SVG)["score"]
    twice = svg.replace(block, block + block.replace('x="40" y="40"', 'x="200" y="40"').replace('x="50" y="54"', 'x="210" y="54"'))
    assert LM.svg_layout_metrics(twice)["duplicate_names"] == 1
    assert m["edge_overlaps"] == 0
    on_top = SVG.replace('<path class="e prod" d="M50,0 L50,100"/>', '<path class="e prod" d="M50,0 L50,100"/><path class="e prod" d="M30,50 L90,50"/>')
    assert LM.svg_layout_metrics(on_top)["edge_overlaps"] == 1  # 60 px on top of the first line, no end node in common
