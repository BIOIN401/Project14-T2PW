"""Every drawn protein location whose protein is named Unknown: label-type, the complex it belongs to and that complex's
name (what PathWhiz shows for label-type protein), and the reaction it catalyses."""
import sys, pathlib
sys.path.insert(0, "C:/t/vis-pw5/src")
from t2pw.pwml.render import read_xml_with_repairs
L = lambda e: e.tag.rsplit("}", 1)[-1].split(":")[-1] if isinstance(e.tag, str) else ""
def kids(e): return {L(c): c for c in e}
def t(e, n):
    c = kids(e).get(n); return (c.text or "").strip() if c is not None else ""
for p in sys.argv[1:]:
    r = read_xml_with_repairs(p)
    prot = {t(e, "id"): t(e, "name") for e in r.iter() if L(e) == "protein"}
    cplx = {t(e, "id"): t(e, "name") for e in r.iter() if L(e) == "protein-complex"}
    loc_cplx = {}
    pcv_of_loc = {}
    for v in r.iter():
        if L(v) == "protein-complex-visualization":
            for m in v.iter():
                if L(m) == "protein-location-id":
                    loc_cplx[(m.text or "").strip()] = cplx.get(t(v, "protein-complex-id"))
                    pcv_of_loc[(m.text or "").strip()] = t(v, "id")
    print("==", pathlib.Path(p).stem)
    for e in r.iter():
        if L(e) == "protein-location" and t(e, "hidden") != "true":
            n = prot.get(t(e, "protein-id"))
            if n and n.strip().lower() == "unknown":
                print(f"   loc {t(e,'id')} label-type={t(e,'label-type')} complex={loc_cplx.get(t(e,'id'))!r}")
