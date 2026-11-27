"""VIS-9 content check: usage cc9.py BASE_DIR HEAD_DIR OUT.json
For every NAME.pwml in BASE, HEAD/NAME.pwml must equal it as canonical XML after removing only the <pwc-id> elements HEAD
added to a <compound> that has none in BASE. Nothing else is masked (no geometry, no template rules). A pwc-id present in
BASE must be present and identical in HEAD. Lists every added pwc-id with the compound id and name. Exit 0 iff all equal."""
import json, sys
from pathlib import Path
from lxml import etree

def comps(root):
    return [c for c in root.iter("compound") if c.getparent() is not None and c.getparent().tag == "compounds"]

def main():
    a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    rows, bad, added = [], 0, []
    for fa in sorted(a.glob("*.pwml")):
        fb = b / fa.name
        if not fb.exists():
            rows.append({"file": fa.name, "same": False, "why": "missing in head"}); bad += 1; continue
        ra, rb = etree.parse(str(fa)).getroot(), etree.parse(str(fb)).getroot()
        ca, cb = comps(ra), comps(rb)
        why = None
        if len(ca) != len(cb):
            why = "compound count %d != %d" % (len(ca), len(cb))
        else:
            for x, y in zip(ca, cb):
                px, py = x.find("pwc-id"), y.find("pwc-id")
                if px is None and py is not None:
                    added.append({"file": fa.name, "id": y.findtext("id"), "name": y.findtext("name"), "hmdb": y.findtext("hmdb-id") or "", "pwc_id": py.text})
                    y.remove(py)
                elif px is not None and (py is None or px.text != py.text):
                    why = "existing pwc-id changed or removed: %s" % px.text
        if why is None and etree.tostring(ra, method="c14n") != etree.tostring(rb, method="c14n"):
            why = "content differs beyond added pwc-id"
        rows.append({"file": fa.name, "same": why is None, "why": why, "added": sum(1 for x in added if x["file"] == fa.name)})
        bad += why is not None
    res = {"files": len(rows), "different": bad, "added_pwc_ids": len(added), "rows": rows, "added": added}
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("files", "different", "added_pwc_ids")}))
    sys.exit(0 if bad == 0 else 1)

main()
