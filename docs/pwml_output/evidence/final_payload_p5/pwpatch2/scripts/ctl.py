import sys
from lxml import etree
f = sys.argv[1]
root = etree.parse(f, etree.XMLParser(recover=True)).getroot()
sts = []
for st in root.findall(".//biological-states/biological-state"):
    g = lambda t: (st.findtext(t) or "").strip() or None
    sts.append((g("id"), g("species-id"), g("tissue-id"), g("cell-type-id"), g("subcellular-location-id"), g("pwbs-id")))
print("states", len(sts))
print("no pwbs-id", sum(1 for s in sts if s[5] is None))
print("distinct pwbs", len({s[5] for s in sts if s[5]}))
print("distinct comp tuples", len({s[1:5] for s in sts}))
from collections import Counter
c = Counter(s[1:5] for s in sts)
print("tuples with >1 states", [(k, v) for k, v in c.items() if v > 1][:10])
cp = Counter(s[5] for s in sts if s[5])
print("pwbs shared", [(k, v) for k, v in cp.items() if v > 1][:10])
# wildcard-bindable to earlier (given pwbs equal) : count states which, ignoring pwbs, are subset of earlier
n = 0
for k, s in enumerate(sts):
    for j in range(k):
        t = sts[j]
        if all(x is None or x == y for x, y in zip(s[1:5], t[1:5])) and (s[5] is None or s[5] == t[5]):
            n += 1; break
print("bindable to earlier incl pwbs equality", n)
