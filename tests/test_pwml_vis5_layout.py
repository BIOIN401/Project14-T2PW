"""VIS-5: layout v2 follows PathBank's PWML conventions (docs/webserver/results/VIS-5.md, STRUCTURE_DIFF)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.layout import relayout  # noqa: E402


def _ids():
    n = [1000]

    def nxt():
        n[0] += 1
        return n[0]
    return nxt


def _cl(i, state):
    return {"id": i, "compound-id": i, "biological-state-id": state, "visualization-template-id": 81, "hidden": False,
            "x": 0, "y": 0, "zindex": 10, "font-size": "regular", "width": "200", "height": "190"}


def _pl(i, state):
    return {"id": i, "protein-id": i, "biological-state-id": state, "visualization-template-id": 2, "hidden": False,
            "x": 0, "y": 0, "zindex": 8, "label-type": "protein", "font-size": "regular", "width": "150", "height": "70"}


def _chain(n, state=1, first_id=1):
    """n reactions in a row in one state: compound i -> compound i+1 by enzyme 100+i (two enzymes on reaction 0)."""
    comps = [_cl(first_id + i, state) for i in range(n + 1)]
    prots = [_pl(100 + i, state) for i in range(n)] + [_pl(199, state)]
    pcvs = [{"id": 300 + i, "protein_complex_protein_visualizations": [{"id": 400 + i, "protein-location-id": 100 + i}]}
            for i in range(n)] + [{"id": 399, "protein_complex_protein_visualizations": [{"id": 499, "protein-location-id": 199}]}]
    edges, rvs = [], []
    for i in range(n):
        edges += [{"id": 500 + 2 * i, "path": "M0 0", "hidden": False, "options": '{"start_arrow": false}'},
                  {"id": 501 + 2 * i, "path": "M0 0", "hidden": False, "options": '{"start_arrow": true}'}]
        rvs.append({"id": 600 + i, "reaction-id": i, "biological-state-id": state,
                    "reaction_compound_visualizations": [
                        {"compound-location-id": first_id + i, "edge-id": 500 + 2 * i, "side": "Left"},
                        {"compound-location-id": first_id + i + 1, "edge-id": 501 + 2 * i, "side": "Right"}],
                    "reaction_enzyme_visualizations": [{"protein-complex-visualization-id": 300 + i}]
                    + ([{"protein-complex-visualization-id": 399}] if i == 0 else [])})
    return comps, prots, pcvs, edges, rvs


def test_long_chain_snakes_and_isozymes_stack_and_membranes_are_named_on_their_line():
    comps, prots, pcvs, edges, rvs = _chain(14)
    items = {"biological-states": [{"id": 1, "name": "ER state", "subcellular-location-id": 11}],
             "subcellular-locations": [{"id": 11, "name": "endoplasmic reticulum"}],
             "compound-locations": comps, "protein-locations": prots, "protein-complex-visualizations": pcvs,
             "nucleic-acid-locations": [{"id": 700, "nucleic-acid-id": 1, "biological-state-id": 1, "hidden": False,
                                         "visualization-template-id": 2, "x": 0, "y": 0, "zindex": 8,
                                         "width": "150", "height": "70"}],
             "edges": edges, "reaction-visualizations": rvs}
    w, h = relayout(items, _ids())
    # a 14-step chain folds into rows instead of one 9000-unit line
    assert w < 2.5 * h, (w, h)
    ys = sorted({c["y"] for c in comps})
    assert ys[-1] - ys[0] > 400
    # the second enzyme of reaction 0 overlaps its first enzyme, a little lower (PathBank's stack)
    p0, p9 = next(p for p in prots if p["id"] == 100), next(p for p in prots if p["id"] == 199)
    assert 0 < p9["y"] - p0["y"] < 70 and abs(p9["x"] - p0["x"]) < 40
    # the ER is named on its membrane line: the label's box is centred on the box's top line
    (mem,) = items["membrane-visualizations"]
    top = min(int(v) for v in re.findall(r"-?[0-9]+", mem["path"])[1::2])
    (lb,) = items["label-locations"]
    sc = float(lb["scale-y"])
    assert abs(lb["y"] + (25 + 15.5) * sc - top) <= 1
    # a nucleic-acid location is drawn with PathWhiz's Nucleic Acid template, not the protein template 2
    assert items["nucleic-acid-locations"][0]["visualization-template-id"] == 12
