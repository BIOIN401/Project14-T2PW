"""VIS-7: subunit labels, the edge-path guard, Mycobacterium envelope, interaction arcs (src/t2pw/pwml/layout.py)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from t2pw.pwml import cell_art  # noqa: E402
from t2pw.pwml.layout import _arc, _path_points, _shorten_long_paths, relayout  # noqa: E402
from test_pwml_vis2_cell_art import _bacterium  # noqa: E402
from test_pwml_vis5_layout import _chain, _ids, _pl  # noqa: E402


def _label_types(names):
    """Chain of 3 reactions; complex 300 has two named subunits (100, 199), 301 one (101), 302 two with an Unknown (102, 198)."""
    comps, prots, pcvs, edges, rvs = _chain(3)
    prots.append(_pl(198, 1))
    member = lambda pcv, loc: {"id": pcv * 10 + loc, "protein-location-id": loc}  # noqa: E731
    for pcv in pcvs:
        pcv["protein_complex_protein_visualizations"] = {
            300: [member(300, 100), member(300, 199)], 301: [member(301, 101)],
            302: [member(302, 102), member(302, 198)], 399: [member(399, 199)]}[pcv["id"]]
    items = {"biological-states": [{"id": 1, "name": "cytosol state", "subcellular-location-id": 11}],
             "subcellular-locations": [{"id": 11, "name": "cytosol"}],
             "proteins": [{"id": i, "name": n} for i, n in names.items()],
             "compound-locations": comps, "protein-locations": prots, "protein-complex-visualizations": pcvs,
             "edges": edges, "reaction-visualizations": rvs}
    relayout(items, _ids())
    return {p["id"]: p["label-type"] for p in prots}


def test_two_named_subunits_are_labelled_by_subunit_one_subunit_and_unknown_stay_protein():
    lt = _label_types({100: "EntA", 199: "EntB", 101: "EntC", 102: "EntD", 198: "Unknown"})
    assert lt[100] == lt[199] == "subunit"
    assert lt[101] == "protein"
    assert lt[102] == lt[198] == "protein"


def test_edge_path_over_255_characters_is_redrawn_between_its_end_points():
    pts = " ".join(f"L{100 + 37 * i} {50 + 41 * (i % 7)}" for i in range(40))
    long = f"M100 50 {pts} L5000 900"
    edges = [{"id": 1, "path": long, "options": "{}"}, {"id": 2, "path": "M0 0 C10 0 20 0 30 0", "options": "{}"}]
    assert len(long) > 255
    _shorten_long_paths({"edges": edges})
    assert len(edges[0]["path"]) <= 255
    pts = _path_points(edges[0]["path"])
    assert pts[0] == (100, 50) and pts[-1] == (5000, 900)
    assert edges[1]["path"] == "M0 0 C10 0 20 0 30 0"  # a short one is left alone


def test_mycobacterium_outer_membrane_is_not_a_gram_negative_envelope():
    items = _bacterium(name="Mycobacterium tuberculosis")
    relayout(items, _ids())
    assert not [m for m in items["membrane-visualizations"] if m["visualization-template-id"] == cell_art.PEPTIDOGLYCAN_TEMPLATE]
    gram = _bacterium()
    relayout(gram, _ids())
    assert [m for m in gram["membrane-visualizations"] if m["visualization-template-id"] == cell_art.PEPTIDOGLYCAN_TEMPLATE]
    assert not cell_art.gram_negative(["outer_membrane"], ["Mycobacterium smegmatis"])
    assert cell_art.gram_negative(["outer_membrane"], ["Escherichia coli"])


def test_arc_is_one_cubic_between_the_two_end_points_standing_off_along_the_normals():
    path, c1, c2 = _arc((100, 300), (0, -1), (900, 320), (0, -1), 110)
    assert path == "M100 300 C100 190 900 210 900 320" and (c1, c2) == ((100, 190), (900, 210))
