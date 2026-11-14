import sys
sys.path.insert(0, "C:/t/p5scratch/writerA")
from pathlib import Path
from lxml import etree
E = Path("C:/t/fp/docs/pwml_output/evidence")
def states_of(path):
    root = etree.parse(str(path)).getroot(); rows = {}
    for sec in ("tissues", "cell-types", "subcellular-locations"):
        for n in root.findall(f".//{sec}/{sec[:-1]}"):
            rows[(sec, (n.findtext("id") or "").strip())] = ((n.findtext("name") or "").casefold(), (n.findtext("ontology-id") or "").strip() or None)
    res = []
    for st in root.findall(".//biological-states/biological-state"):
        c = [(st.findtext("species-id") or "").strip() or None]
        for sec, tag in (("tissues", "tissue-id"), ("cell-types", "cell-type-id"), ("subcellular-locations", "subcellular-location-id")):
            v = (st.findtext(tag) or "").strip(); c.append(rows.get((sec, v), ("?" + v, None)) if v else None)
        res.append(tuple(c))
    return res
def nb(sts): return sum(1 for k, s in enumerate(sts) if any(j != k and all(x is None or x == y for x, y in zip(s, t)) for j, t in enumerate(sts)))
OFF = E / "final_payload_p4/d3/replay/off/pwml"; WA = Path("C:/t/ev-fp/p5/replay/writerA_on/pwml"); D3 = E / "final_payload_p4/d3/replay/on/pwml"
for f in sorted(WA.glob("*.pwml"), key=lambda p: int(p.stem)):
    a, b, c = nb(states_of(OFF / f.name)), nb(states_of(f)), nb(states_of(D3 / f.name))
    if a != b or b != c: print(f.stem, "off", a, "writerA", b, "d3", c)
print("----")
for i in (3, 10):
    for lab, d in (("off", OFF), ("wA", WA)):
        print(i, lab)
        for s in states_of(d / f"{i}.pwml"): print("   ", s)
