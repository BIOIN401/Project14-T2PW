"""Proof that the side fix changes nothing but reaction-member `side` attributes.

usage: side_only_check.py BEFORE_DIR AFTER_DIR OUT.json
For every NAME.pwml: (1) with the `side` of the reaction-member visualizations blanked in both files, the two files are
byte-identical (raw bytes, layout NOT stripped); (2) the number of changed `side` values is counted. That the new values equal
the role is proved by layout/scripts/side_pwml.py (mismatch 0).
The comparison is on raw XML, stronger than the layout-stripped content_check.py (which also compares side).
Exit 0 only when every file passes both.
"""
import json
import sys
from pathlib import Path

from lxml import etree


MEMBER_TAGS = {"reaction-compound-visualization", "reaction-nucleic-acid-visualization",
               "reaction-element-collection-visualization"}


def blank(path):
    """(bytes with reaction-member side blanked, list of (member tag, member id, side))"""
    root = etree.parse(str(path)).getroot()
    sides = []
    for el in root.iter():
        if isinstance(el.tag, str) and el.tag in MEMBER_TAGS:
            s = el.find("side")
            sides.append((el.tag, el.findtext("id"), s.text if s is not None else None))
            if s is not None:
                s.text = "#"
    return etree.tostring(root, method="c14n"), sides


def main():
    a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    rows, bad = [], 0
    tot = {"files": 0, "members": 0, "side_changed": 0, "files_changed": 0}
    if not any(a.glob("*.pwml")):
        sys.exit("no input files in " + str(a))
    for fa in sorted(a.glob("*.pwml"), key=lambda p: (len(p.stem), p.stem)):
        fb = b / fa.name
        if not fb.exists():
            rows.append({"file": fa.name, "ok": False, "why": "missing in after"})
            bad += 1
            continue
        ba, sa = blank(fa)
        bb, sb = blank(fb)
        same_rest = ba == bb
        changed = [(x, y) for x, y in zip(sa, sb) if x != y]
        # the rest of the file must be identical, with the layout NOT stripped
        ok = same_rest and len(sa) == len(sb)
        rows.append({"file": fa.name, "ok": ok, "members": len(sa), "side_changed": len(changed),
                     "rest_identical_raw": same_rest, "raw_identical_whole_file": fa.read_bytes() == fb.read_bytes()})
        tot["files"] += 1
        tot["members"] += len(sa)
        tot["side_changed"] += len(changed)
        tot["files_changed"] += 1 if changed else 0
        if not ok:
            bad += 1
    extra = sorted(p.name for p in b.glob("*.pwml") if not (a / p.name).exists())
    res = dict(tot, failing_files=bad, only_in_after=extra, rows=rows)
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    sys.exit(0 if bad == 0 and not extra else 1)


if __name__ == "__main__":
    main()
