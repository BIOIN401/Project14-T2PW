import sys
sys.path.insert(0, "C:/t/vis-pw5/src")
from t2pw.pwml.render import read_xml_with_repairs
L = lambda e: e.tag.rsplit("}", 1)[-1].split(":")[-1] if isinstance(e.tag, str) else ""
r = read_xml_with_repairs(sys.argv[1])
uses = {}
for v in r.iter():
    for m in v:
        for c in m.iter():
            if L(c) == "edge-id": uses[(c.text or "").strip()] = (L(v), L(m))
for e in r.iter():
    if L(e) == "edge":
        d = {L(k): (k.text or "").strip() for k in e}
        if d.get("visualization-template-id") == "10" or "C" not in d.get("path", "") or d.get("path","").count("C") > 1:
            print(d.get("id"), d.get("visualization-template-id"), d.get("hidden"), uses.get(d.get("id")), d.get("path")[:90])
