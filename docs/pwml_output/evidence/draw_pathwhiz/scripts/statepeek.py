"""Per biological state of a PWML: which drawn locations and process visualizations sit in it.
usage: statepeek.py FILE.pwml"""
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
pv = root.find(".//pathway-visualization")
locname = {l.findtext("id"): l.findtext("name") for l in root.iter("subcellular-location")}
names = {}
for kind in ("compound", "protein", "nucleic-acid", "element-collection"):
    for e in root.iter(kind):
        names[(kind, e.findtext("id"))] = e.findtext("name")
for st in root.iter("biological-state"):
    sid = st.findtext("id")
    print("STATE", sid, repr(st.findtext("name")), "loc=", locname.get(st.findtext("subcellular-location-id")))
    for kind in ("compound", "protein", "nucleic-acid", "element-collection"):
        for l in pv.iter(kind + "-location"):
            if l.findtext("biological-state-id") == sid:
                print("   ", kind, l.findtext("id"), names.get((kind, l.findtext(kind + "-id"))), "hidden" if l.findtext("hidden") == "true" else "")
    for tag in ("transport-visualization", "reaction-visualization"):
        for v in pv.iter(tag):
            if v.findtext("biological-state-id") == sid:
                print("    ", tag, v.findtext("id"))
for tv in pv.iter("transport-visualization"):
    print("TV", tv.findtext("id"), [(m.findtext("compound-location-id"), m.findtext("side")) for m in tv.iter("transport-compound-visualization")],
          "transporters", [m.findtext("protein-complex-visualization-id") for m in tv.iter("transport-transporter-visualization")])
