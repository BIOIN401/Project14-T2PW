"""Reaction-member `side` against the member's role, read from the PWML alone.

usage: side_pwml.py OUT.json DIR [DIR ...]
Role, two ways: (a) the reaction's own element lists (<reaction-left-elements>/<reaction-right-elements>, the biology),
(b) the arrow the writer puts on product edges. A member is a mismatch when its `side` names the other list.
"""
import json
import sys
from pathlib import Path

from lxml import etree

KINDS = (("reaction-compound-visualization", "compound-location", "compound-location-id", "compound-id", "Compound"),
         ("reaction-nucleic-acid-visualization", "nucleic-acid-location", "nucleic-acid-location-id", "nucleic-acid-id", "NucleicAcid"),
         ("reaction-element-collection-visualization", "element-collection-location", "element-collection-location-id",
          "element-collection-id", "ElementCollection"))


def scan(path):
    pv = etree.parse(str(path)).getroot().find(".//pathway-visualization")
    rx = {}
    for r in pv.iter("reaction"):
        sides = {}
        for lst, side in (("reaction-left-elements", "Left"), ("reaction-right-elements", "Right")):
            for e in r.iterfind(f"{lst}/*"):
                sides.setdefault((e.findtext("element-type"), e.findtext("element-id")), set()).add(side)
        rx[r.findtext("id")] = sides
    edges = {e.findtext("id"): e for e in pv.iter("edge")}
    rows = []
    for rv in pv.iter("reaction-visualization"):
        sides = rx.get(rv.findtext("reaction-id"), {})
        for vtag, ltag, lfield, efield, etype in KINDS:
            locs = {l.findtext("id"): l.findtext(efield) for l in pv.iter(ltag)}
            for m in rv.iter(vtag):
                ent = locs.get(m.findtext(lfield))
                role = sides.get((etype, ent), set())
                e = edges.get(m.findtext("edge-id"))
                arrow = e is not None and '"start_arrow": true' in (e.findtext("options") or "")
                rows.append({"reaction": rv.findtext("reaction-id"), "vis_member": m.findtext("id"), "type": etype,
                             "entity": ent, "side": m.findtext("side"), "role": sorted(role), "product_arrow": arrow})
    return rows


out, dirs = sys.argv[1], sys.argv[2:]
res = {}
tot = {"members": 0, "mismatch_vs_element_lists": 0, "both_sides": 0, "not_in_lists": 0, "mismatch_vs_arrow": 0}
for d in dirs:
    for f in sorted(Path(d).glob("*.pwml"), key=lambda p: (len(p.stem), p.stem)):
        rows = scan(f)
        mm = [r for r in rows if len(r["role"]) == 1 and r["side"] not in r["role"]]
        tot["members"] += len(rows)
        tot["mismatch_vs_element_lists"] += len(mm)
        tot["both_sides"] += sum(1 for r in rows if len(r["role"]) == 2)
        tot["not_in_lists"] += sum(1 for r in rows if not r["role"])
        tot["mismatch_vs_arrow"] += sum(1 for r in rows if (r["side"] == "Left") == r["product_arrow"])
        res[f"{Path(d).parent.name}/{f.name}"] = {"members": len(rows), "mismatch": len(mm), "rows": mm}
Path(out).write_text(json.dumps({"totals": tot, "files": res}, indent=1), encoding="utf-8")
print(json.dumps(tot))
print("files with a mismatch:", sum(1 for v in res.values() if v["mismatch"]))
