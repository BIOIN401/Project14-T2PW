"""Structure survey of PWML files: section order, record counts, field sets and value stats per visualization record type."""
import sys, re, json, pathlib
from collections import Counter, defaultdict
from lxml import etree
sys.path.insert(0, "C:/t/vis-pw5/src")
from t2pw.pwml.render import read_xml_with_repairs
L = lambda e: e.tag.rsplit("}", 1)[-1].split(":")[-1] if isinstance(e.tag, str) else ""
REC = ["compound-location","protein-location","nucleic-acid-location","element-collection-location","protein-complex-visualization",
       "reaction-visualization","transport-visualization","interaction-visualization","edge","membrane-visualization",
       "drawable-element-location","zoom-visualization","label-location","bound-visualization","sub-pathway-visualization",
       "text-location","pathway-visualization"]
def survey(p):
    r = read_xml_with_repairs(p)
    out = {}
    pv = next(e for e in r.iter() if L(e) == "pathway-visualization")
    out["pv_children"] = [L(c) for c in pv]
    out["top"] = [L(c) for c in r]
    for rt in REC:
        recs = [e for e in r.iter() if L(e) == rt]
        if not recs: continue
        fields = Counter()
        vals = defaultdict(Counter)
        for e in recs:
            for c in e:
                f = L(c); fields[f] += 1
                if len(c): vals[f]["<%d children>" % len(c)] += 1; continue
                t = (c.text or "").strip()
                if f in ("x","y","p1x","p1y","p2x","p2y","p3x","p3y","id","path","options"):
                    if f == "path": vals[f]["segs=%d" % t.count("C")] += 1
                    if f == "options": vals[f][t[:70]] += 1
                    continue
                vals[f][t[:40]] += 1
        out[rt] = {"n": len(recs), "fields": dict(fields), "vals": {f: v.most_common(8) for f, v in vals.items()}}
    return out
res = {pathlib.Path(p).stem: survey(p) for p in sys.argv[2:]}
json.dump(res, open(sys.argv[1], "w"), indent=1)
for name, s in res.items():
    print("=====", name, "pv children:", s["pv_children"])
    for rt in REC:
        if rt in s: print(f"  {rt:32s} n={s[rt]['n']}")
