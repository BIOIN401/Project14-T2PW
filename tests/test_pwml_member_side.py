"""A reaction member's `side` is its role (substrate = Left, product = Right), not the serpentine row direction.

PathWhiz reads `side` as a role (SBML / SBGN export, orientation, Reaction#find_element). The row direction still
places the stacks (geometry); it must not leak into the attribute.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.validate import discover_structure_signature  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402

IR = json.loads((ROOT / "tests" / "fixtures" / "pwml_layout" / "ir_run4.json").read_text(encoding="utf-8"))
KINDS = (("reaction-compound-visualization", "compound-location", "compound-location-id", "compound-id", "Compound"),
         ("reaction-nucleic-acid-visualization", "nucleic-acid-location", "nucleic-acid-location-id", "nucleic-acid-id", "NucleicAcid"),
         ("reaction-element-collection-visualization", "element-collection-location", "element-collection-location-id",
          "element-collection-id", "ElementCollection"))


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


def _members(root):
    """(side, roles from the reaction's own element lists) per reaction member."""
    pv = root.find(".//pathway-visualization")
    roles = {}
    for r in pv.iter("reaction"):
        d = {}
        for lst, side in (("reaction-left-elements", "Left"), ("reaction-right-elements", "Right")):
            for e in r.iterfind(f"{lst}/*"):
                d.setdefault((e.findtext("element-type"), e.findtext("element-id")), set()).add(side)
        roles[r.findtext("id")] = d
    rows = []
    for rv in pv.iter("reaction-visualization"):
        for vtag, ltag, lfield, efield, etype in KINDS:
            locs = {el.findtext("id"): el.findtext(efield) for el in pv.iter(ltag)}
            for m in rv.iter(vtag):
                role = roles.get(rv.findtext("reaction-id"), {}).get((etype, locs.get(m.findtext(lfield))), set())
                rows.append((m.findtext("side"), role))
    return rows


def test_reaction_member_side_is_the_role_in_both_layouts(monkeypatch):
    for layout in (None, "v2"):
        rows = _members(_build(monkeypatch, layout))
        single = [(s, r) for s, r in rows if len(r) == 1]
        assert len(single) >= 16  # the fixture has real reactions, so the check is not vacuous
        assert [s for s, r in single if s not in r] == [], layout


def test_the_fixture_has_right_to_left_rows(monkeypatch):
    """Guards the test above: a row that runs right to left exists, so the old code would have flipped it."""
    from t2pw.pwml import writer

    seen = []
    orig = writer._reaction_visual_side

    def spy(layout, substrate_side):
        seen.append(int(layout.get("row-dir", 1)))
        return orig(layout, substrate_side)

    monkeypatch.setattr(writer, "_reaction_visual_side", spy)
    _build(monkeypatch, None)
    assert -1 in seen
