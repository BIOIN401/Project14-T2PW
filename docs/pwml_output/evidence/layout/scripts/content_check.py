"""Content identity of two PWML sets with the layout stripped.

usage: content_check.py DIR_A DIR_B OUT.json

For every NAME.pwml in DIR_A, DIR_B/NAME.pwml must exist and be identical after removing layout:
x, y, width, height of every element (locations, bound-visualizations, the pathway-visualization canvas);
edge <path> and the non-null *_path arrow strings inside edge <options> (derived from the path); and the
membrane-visualizations and label-locations sections (drawn compartment outlines and their labels). Everything else (every element,
id, name, reference, state, location, reaction, transport, interaction, hidden flag, template, zindex,
arrow on/off, reaction-member side) is compared as canonical XML. Exit 0 only when every file matches.
"""
import difflib
import json
import sys
from pathlib import Path

from lxml import etree

GEOM = {"x", "y", "width", "height"}
STRIP_SECTIONS = {"membrane-visualizations", "label-locations"}


def strip(path):
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
                    el.text = json.dumps({k: ("#" if k.endswith("_path") and v is not None else v) for k, v in o.items()}, sort_keys=True)
            except ValueError:
                pass
        elif ptag == "edge" and tag.endswith("_path"):  # option:*_path element form
            el.text = "#"
    return etree.tostring(root, method="c14n").decode("utf-8")


def main():
    a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    rows, bad = [], 0
    for fa in sorted(a.glob("*.pwml"), key=lambda p: (len(p.stem), p.stem)):
        fb = b / fa.name
        if not fb.exists():
            rows.append({"file": fa.name, "same": False, "diff": "missing in B"})
            bad += 1
            continue
        sa, sb = strip(fa), strip(fb)
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
