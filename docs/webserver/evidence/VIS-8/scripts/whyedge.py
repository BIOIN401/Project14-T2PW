"""usage: whyedge.py PWML EDGE_ID... -- reaction visualization(s) holding the edge, with members, enzymes and edge paths"""
import sys
from lxml import etree
L = lambda e: e.tag.rsplit("}", 1)[-1].split(":")[-1] if isinstance(e.tag, str) else ""
def d(e): return {L(k): (k.text or "").strip() for k in e}
r = etree.parse(sys.argv[1], etree.XMLParser(recover=True, huge_tree=True)).getroot()
edges = {d(e)["id"]: d(e) for e in r.iter() if L(e) == "edge"}
cl = {d(e)["id"]: d(e) for e in r.iter() if L(e) == "compound-location"}
comp = {d(e).get("id"): d(e).get("name") for e in r.iter() if L(e) == "compound"}
for rv in r.iter():
    if L(rv) not in ("reaction-visualization", "transport-visualization"): continue
    eids = [x.text.strip() for x in rv.iter() if L(x) == "edge-id"]
    if not set(eids) & set(sys.argv[2:]): continue
    print(L(rv), d(rv).get("id"), "reaction/transport", d(rv).get("reaction-id") or d(rv).get("transport-id"))
    for m in rv.iter():
        if L(m).endswith("compound-visualization") and L(m) != "reaction-compound-visualizations":
            k = d(m); loc = cl.get(k.get("compound-location-id"), {})
            e = edges.get(k.get("edge-id"), {})
            print("  compound", comp.get(loc.get("compound-id")), "side", k.get("side"), "edge", k.get("edge-id"), "hidden", e.get("hidden"), e.get("path"), "loc", loc.get("x"), loc.get("y"))
        if L(m) in ("reaction-enzyme-visualization", "transport-transporter-visualization"):
            print("  enzyme/transporter viz", d(m))
