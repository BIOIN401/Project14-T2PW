"""Layout v2 draws a chain as one connected picture in pathway order (correctness F3, F4).

F3: one location per (compound, biological state) for every reaction that uses it, so a product feeds visibly into
the next reaction; currency compounds stay one per reaction. F4: a cycle is broken at the edge that returns to the
earlier process, so the chain reads left to right. Only locations and visualizations change; v1 keeps its reuse.
"""
from __future__ import annotations

import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.layout import _Block, _layered  # noqa: E402
from t2pw.pwml.validate import discover_structure_signature  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder, _is_currency_compound_record  # noqa: E402

IR = json.loads((ROOT / "tests" / "fixtures" / "pwml_layout" / "ir_run4.json").read_text(encoding="utf-8"))
VIZ = ("bound-visualizations", "compound-locations", "element-collection-locations", "nucleic-acid-locations",
       "protein-locations", "protein-complex-visualizations", "edges", "reaction-visualizations",
       "reaction-coupled-transport_visualizations", "transport-visualizations", "interaction-visualizations",
       "sub-pathway-visualizations", "vacuous-compound-visualizations", "vacuous-edge-visualizations",
       "vacuous-nucleic-acid-visualizations", "vacuous-element-collection-visualizations",
       "vacuous-protein-visualizations", "drawable-element-locations", "membrane-visualizations", "label-locations",
       "zoom-visualizations")


def _build(monkeypatch, layout):
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
    monkeypatch.setenv("T2PW_PWML_LAYOUT", layout)
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000",
                           height=1400, width=3200, background_color="#FFFFFF",
                           ref=str(ROOT / "reference" / "PW000001.pwml"))
    sig = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    root = DeterministicPwmlBuilder(extraction=copy.deepcopy(IR), signature=sig, args=args).build().root
    return etree.fromstring(etree.tostring(root)).find(".//pathway-visualization")


def _compound_uses(pv):
    """Per reaction: [(compound location id, compound id, state id, side)] from the reaction visualizations."""
    locs = {el.findtext("id"): el for el in pv.iter("compound-location")}
    uses = {}
    for rv in pv.iter("reaction-visualization"):
        rows = []
        for m in rv.iter("reaction-compound-visualization"):
            loc = locs[m.findtext("compound-location-id")]  # every member points at a location that exists
            rows.append((loc.findtext("id"), loc.findtext("compound-id"), loc.findtext("biological-state-id"), m.findtext("side")))
        uses[rv.findtext("reaction-id")] = rows
    return uses


def _currency(pv):
    out = set()
    for c in pv.iter("compound"):
        rec = {k: c.findtext(k) for k in ("name", "kegg-id", "chebi-id")}
        if _is_currency_compound_record(rec):
            out.add(c.findtext("id"))
    return out


def test_v2_draws_one_location_per_compound_and_state_for_the_chain(monkeypatch):
    pv = _build(monkeypatch, "v2")
    uses, currency = _compound_uses(pv), _currency(pv)
    by_identity = defaultdict(set)
    for rows in uses.values():
        assert len({loc for loc, *_ in rows}) == len(rows)  # a reaction never gets one location twice
        for loc, cid, st, _ in rows:
            if cid not in currency:
                by_identity[(cid, st)].add(loc)
    shared = [k for k, v in by_identity.items() if sum(1 for rows in uses.values() for r in rows if (r[1], r[2]) == k) > 1]
    assert len(shared) >= 5  # the fixture's chain intermediates, incl. 2,3-dihydroxybenzoate used by three reactions
    assert {k: len(v) for k, v in by_identity.items() if len(v) != 1} == {}


def test_v2_chain_is_one_drawn_component(monkeypatch):
    pv = _build(monkeypatch, "v2")
    uses, currency = _compound_uses(pv), _currency(pv)
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x
    for rid, rows in uses.items():
        for loc, cid, *_ in rows:
            if cid not in currency:
                parent[find("L" + loc)] = find("R" + rid)
    assert len({find("R" + rid) for rid in uses}) == 1


def test_v2_chain_reads_left_to_right(monkeypatch):
    pv = _build(monkeypatch, "v2")
    names = {c.findtext("id"): c.findtext("name") for c in pv.iter("compound")}
    x = {}
    used = {loc for rows in _compound_uses(pv).values() for loc, *_ in rows}
    for loc in pv.iter("compound-location"):
        if loc.findtext("id") not in used:  # base locations no process draws (PathWhiz's viewer drops them)
            continue
        x.setdefault(names[loc.findtext("compound-id")], []).append(int(loc.findtext("x")))
    chain = ["chorismate", "isochorismate", "2,3-dihydro-2,3-dihydroxybenzoate", "2,3-dihydroxybenzoic acid",
             "2,3-dihydroxybenzoyl-AMP", "enterobactin"]
    xs = [x[n] for n in chain]
    assert all(len(v) == 1 for v in xs), xs
    xs = [v[0] for v in xs]
    # isochorismate and its product share one enzyme (pc_4) in the fixture, so they may stand in the same column
    assert xs[0] < xs[1] <= xs[2] < xs[3] < xs[4] < xs[5], dict(zip(chain, xs))


def _content(pv, keep_viz):
    """Everything but the drawing, plus each reaction's members dereferenced to (compound, state, side, edge kind)."""
    pv = copy.deepcopy(pv)
    for tag in ("width", "height"):
        pv.find(tag).text = "#"
    locs = {el.findtext("id"): (el.findtext("compound-id"), el.findtext("biological-state-id"), el.findtext("hidden"),
                                el.findtext("visualization-template-id")) for el in pv.iter("compound-location")}
    edges = {el.findtext("id"): (el.findtext("hidden"), el.findtext("visualization-template-id"),
                                 {k: v for k, v in json.loads(el.findtext("options") or "{}").items() if not k.endswith("_path")}.__repr__())
             for el in pv.iter("edge")}
    members = Counter()
    for rv in pv.iter("reaction-visualization"):
        for m in rv.iter("reaction-compound-visualization"):
            members[(rv.findtext("reaction-id"), locs[m.findtext("compound-location-id")], m.findtext("side"),
                     edges[m.findtext("edge-id")])] += 1
    for tag in VIZ:
        el = pv.find(tag)
        if el is not None and tag not in keep_viz:
            pv.remove(el)
    return etree.tostring(pv, method="c14n"), members


def test_v2_changes_only_the_drawing(monkeypatch):
    a, ma = _content(_build(monkeypatch, "v1"), ())
    b, mb = _content(_build(monkeypatch, "v2"), ())
    assert a == b
    assert ma == mb and sum(ma.values()) >= 16


def test_layered_breaks_a_cycle_at_the_edge_back_to_the_earlier_process():
    blocks = {k: _Block(k, 1, [{}], 100, 50, [(0, 0)]) for k in ("b", "a")}
    flow = [("a", "b"), ("b", "a")]
    _layered(blocks, ["b", "a"], flow, {"a": 0, "b": 1})
    assert blocks["a"].x < blocks["b"].x
    _layered(blocks, ["a", "b"], flow, {"b": 0, "a": 1})
    assert blocks["b"].x < blocks["a"].x
