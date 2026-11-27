"""Content identity of two PWML sets with the layout stripped.

usage: content_check.py DIR_A DIR_B OUT.json [strict]

VIS-7 (F1): the two VIS-5 rules are one-directional and exact: only base (DIR_A) label-type protein vs head (DIR_B) subunit,
and only base template 2 vs head template 12 on a nucleic-acid location, are read as equal; the reverse, or any other value,
counts. `strict` drops those two rules too (a strict diff to read alongside the rule-based one). VIS-1's 157->81 and 83->5
rules are unchanged.

For every NAME.pwml in DIR_A, DIR_B/NAME.pwml must exist and be identical after removing layout:
x, y, width, height of every element (locations, bound-visualizations, the pathway-visualization canvas);
edge <path> and the *_path arrow strings inside edge <options> (derived from the path; VIS-1 copy: also a null one,
since a degenerate base path gave null where the new geometry gives a real head; the key set and every arrow flag are
still compared); a compound location's box template 157 read as 81 (VIS-1 redraws drug boxes as compound boxes) and an edge template 83 read as 5 (VIS-1 draws transport edges with
PathBank's transport template); and the
membrane-visualizations and label-locations sections (drawn compartment outlines and their labels).
VIS-2 copy: also the drawable-element-locations and zoom-visualizations sections (PathBank cell and organelle images
and zoom boxes, visualization records only); the canvas size was already masked (width/height above). Everything else (every element,
id, name, reference, state, location, reaction, transport, interaction, hidden flag, template, zindex,
arrow on/off, reaction-member side) is compared as canonical XML. Exit 0 only when every file matches.
"""
import difflib
import json
import sys
from pathlib import Path

from lxml import etree

GEOM = {"x", "y", "width", "height"}
STRIP_SECTIONS = {"membrane-visualizations", "label-locations", "drawable-element-locations", "zoom-visualizations"}  # VIS-2: + the cell / organelle art


def base_values(path):
    """(location id -> label-type) of protein locations and (id -> template) of nucleic-acid locations of the base file."""
    root = etree.parse(str(path)).getroot()
    out = {}
    for el in root.iter():
        if isinstance(el.tag, str) and etree.QName(el).localname in ("protein-location", "nucleic-acid-location"):
            k = {etree.QName(c).localname: (c.text or "") for c in el if isinstance(c.tag, str)}
            out[(etree.QName(el).localname, k.get("id"))] = (k.get("label-type"), k.get("visualization-template-id"))
    return out


def strip(path, head=False, strict=False, base=None):
    root = etree.parse(str(path)).getroot()
    for el in list(root.iter()):
        if not isinstance(el.tag, str):
            continue
        tag = etree.QName(el).localname
        parent = el.getparent()
        ptag = etree.QName(parent).localname if parent is not None else ""
        if tag in STRIP_SECTIONS:
            for c in list(el):
                el.remove(c)
            el.text = None
        elif tag in GEOM and len(el) == 0:
            el.text = "#"
        elif tag == "path" and ptag == "edge":
            el.text = "#"
        elif tag == "options" and ptag == "edge" and el.text:
            try:
                o = json.loads(el.text)
                if isinstance(o, dict):
                    el.text = json.dumps({k: ("#" if k.endswith("_path") else v) for k, v in o.items()}, sort_keys=True)
            except ValueError:
                pass
        elif ptag == "edge" and tag.endswith("_path"):  # option:*_path element form
            el.text = "#"
        elif tag == "visualization-template-id" and ptag == "compound-location" and el.text == "157":
            el.text = "81"  # VIS-1 copy: v2 draws the drug box (157) as the compound box (81); any other template change still counts
        elif head and not strict and tag == "label-type" and ptag == "protein-location" and el.text == "subunit"                 and (base or {}).get(("protein-location", parent.findtext("id")), (None,))[0] == "protein":
            el.text = "protein"  # VIS-5 copy: a subunit of a multi-subunit complex labelled with its own protein name (PathBank's label-type), not the complex name
        elif head and not strict and tag == "visualization-template-id" and ptag == "nucleic-acid-location" and el.text == "12"                 and (base or {}).get(("nucleic-acid-location", parent.findtext("id")), (None, None))[1] == "2":
            el.text = "2"  # VIS-5 copy (STRUCTURE_DIFF G6): a nucleic-acid location drawn with PathWhiz's "Nucleic Acid" template 12 instead of the protein template 2
        elif tag == "visualization-template-id" and ptag == "edge" and el.text == "83":
            el.text = "5"  # VIS-1 copy: v2 draws transport edges with PathBank's transport template (83) instead of 5
    return etree.tostring(root, method="c14n").decode("utf-8")


def main():
    a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    strict = len(sys.argv) > 4 and sys.argv[4] == "strict"
    rows, bad = [], 0
    for fa in sorted(a.glob("*.pwml"), key=lambda p: (len(p.stem), p.stem)):
        fb = b / fa.name
        if not fb.exists():
            rows.append({"file": fa.name, "same": False, "diff": "missing in B"})
            bad += 1
            continue
        sa, sb = strip(fa, False, strict), strip(fb, True, strict, base_values(fa))
        same = sa == sb
        row = {"file": fa.name, "same": same}
        if not same:
            bad += 1
            row["diff"] = "".join(list(difflib.unified_diff(sa.splitlines(True), sb.splitlines(True), n=1))[:40])
        rows.append(row)
    extra = sorted(p.name for p in b.glob("*.pwml") if not (a / p.name).exists())
    res = {"files": len(rows), "different": bad, "only_in_b": extra, "rows": rows}
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    sys.exit(0 if bad == 0 and not extra else 1)


if __name__ == "__main__":
    main()
