import sys
from pathlib import Path
from lxml import etree
WA = Path("C:/t/ev-fp/p5/replay/writerA_on/pwml")

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

def nb(sts):
    return sum(1 for k, s in enumerate(sts) if any(j != k and all(x is None or x == y for x, y in zip(s, t)) for j, t in enumerate(sts)))

# order of creation matters for the importer: a state omitting a component binds to an EARLIER created state only
def nb_ordered(sts):
    return sum(1 for k, s in enumerate(sts) if any(j < k and all(x is None or x == y for x, y in zip(s, t)) for j, t in enumerate(sts)))

tot = 0
bind, nobind = [], []
for f in sorted(WA.glob("*.pwml"), key=lambda p: int(p.stem)):
    sts = states_of(f)
    a, b = nb(sts), nb_ordered(sts)
    (bind if a else nobind).append(f.stem)
    if a:
        print(f.stem, "states", len(sts), "bindable", a, "bindable_to_earlier", b)
print("BIND", " ".join(bind))
print("NOBIND_COUNT", len(nobind))
print("NOBIND", " ".join(nobind))
