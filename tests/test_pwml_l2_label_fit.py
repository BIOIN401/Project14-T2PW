"""L2 (correctness F10): compartment labels stay inside their rectangle (layout v2, src/t2pw/pwml/layout.py)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml import layout  # noqa: E402


def _loc(i, state):
    return {"id": i, "compound-id": i, "biological-state-id": state, "visualization-template-id": 81, "hidden": False,
            "x": 0, "y": 0, "zindex": 10, "font-size": "regular", "width": "160", "height": "60"}


def test_long_compartment_label_is_laid_out_as_stacked_labels_above_the_first_node():
    n = [1000]

    def new_id():
        n[0] += 1
        return n[0]

    items = {"biological-states": [{"id": 1, "name": "Bacillus subtilis cytoplasm state"}],
             "compound-locations": [_loc(1, 1), _loc(2, 1)],
             "edges": [{"id": 70, "path": "M0 0", "hidden": False, "options": None},
                       {"id": 71, "path": "M0 0", "hidden": False, "options": None}],
             "reaction-visualizations": [{"id": 80, "reaction-id": 1, "biological-state-id": 1,
                                          "reaction_compound_visualizations": [
                                              {"compound-location-id": 1, "edge-id": 70, "side": "Left"},
                                              {"compound-location-id": 2, "edge-id": 71, "side": "Right"}],
                                          "reaction_enzyme_visualizations": []}]}
    layout.relayout(items, new_id)
    labels = items["label-locations"]
    assert [lb["label-text"] for lb in labels] == ["Bacillus subtilis", "cytoplasm"]
    assert labels[1]["y"] - labels[0]["y"] == layout.LABEL_STEP
    top = min(it["y"] for it in items["compound-locations"])
    assert labels[-1]["y"] + (40 + 16) * 2 < top  # the stack ends above the first node


def test_wrapped_label_that_pathwhiz_would_draw_outside_its_rectangle_is_split_into_one_line_pieces():
    assert layout.label_parts("Bacillus subtilis cytoplasm") == ["Bacillus subtilis", "cytoplasm"]  # one 27-character line, 210 wide
    assert layout.label_parts("Bacillus subtilis extracellular") == ["Bacillus subtilis extracellular"]  # PathWhiz wraps it itself
    assert layout.label_parts("plastid") == ["plastid"]
    for text in ("human carnitine biosynthesis", "endoplasmic reticulum of A. membranaceus", "x" * 60, "cytoplasmic leaflet of IM"):
        parts = layout.label_parts(text)
        assert all(layout._text_w(p) <= layout.LABEL_FIT and len(layout._label_chunks(p)) == 1 for p in parts), parts
        assert "".join("".join(parts).split()) == "".join(text.split())  # nothing lost
