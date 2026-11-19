"""Print each transport of a PWML with its element states (tuple + location name) and transporters."""
import sys
import xml.etree.ElementTree as ET


def t(e, k):
    x = e.find(k)
    return (x.text or "").strip() if x is not None and x.text else ""


for path in sys.argv[1:]:
    pv = ET.parse(path).getroot().find(".//pathway-visualization")
    root = ET.parse(path).getroot()
    loc = {t(e, "id"): t(e, "name") for e in root.iter("subcellular-location")}
    st = {}
    for e in root.iter("biological-state"):
        st[t(e, "id")] = (t(e, "name"), t(e, "species-id"), loc.get(t(e, "subcellular-location-id"), t(e, "subcellular-location-id")), t(e, "cell-type-id"), t(e, "tissue-id"))
    print("==", path)
    for k, v in st.items():
        print("  state", k, v)
    names = {}
    for tag in ("compound", "protein-complex", "protein", "nucleic-acid", "element-collection"):
        for e in root.iter(tag):
            names[(tag, t(e, "id"))] = t(e, "name")
    for tr in root.iter("transport"):
        print("  TR", t(tr, "id"), t(tr, "name"))
        for el in tr.iter("transport-element"):
            print("     el", t(el, "element-type"), t(el, "element-id"), "L", t(el, "left-biological-state-id"), st.get(t(el, "left-biological-state-id"), ("?",))[2:3], "R", t(el, "right-biological-state-id"), st.get(t(el, "right-biological-state-id"), ("?",))[2:3])
        for el in tr.iter("transport-transporter"):
            print("     tt", t(el, "protein-complex-id"), names.get(("protein-complex", t(el, "protein-complex-id"))), "pid", t(el, "protein-id"), "bs", t(el, "biological-state-id"))
