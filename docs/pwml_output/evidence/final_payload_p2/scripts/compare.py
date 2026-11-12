import json, sys, glob, collections
from pathlib import Path
from lxml import etree

A, B = sys.argv[1:3]
R = Path("C:/t/fp/docs/pwml_output/evidence/final_payload_p2/replay")


def load(label):
    d = {}
    for f in glob.glob(str(R / label / "results_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            d[r["idx"]] = r
    return d


def sig(e):
    return (e.tag, tuple(sorted(e.attrib.items())), (e.text or "").strip())


def xdiff(pa, pb):
    ca = collections.Counter(sig(e) for e in etree.parse(str(pa)).iter())
    cb = collections.Counter(sig(e) for e in etree.parse(str(pb)).iter())
    out = ["-A %s %s" % (s[0], str(dict(s[1]))[:80]) for s in (ca - cb)]
    out += ["+B %s %s" % (s[0], str(dict(s[1]))[:80]) for s in (cb - ca)]
    return out


a, b = load(A), load(B)
tot = collections.defaultdict(collections.Counter)
for i in sorted(set(a) | set(b)):
    ra, rb = a.get(i), b.get(i)
    g = (ra or rb)["group"]
    if not ra or not rb:
        st = "missing_in_" + ("A" if not ra else "B")
    elif ra["pwml_written"] and rb["pwml_written"]:
        st = "identical" if ra["sha256"] == rb["sha256"] else "differs"
    elif ra["pwml_written"] or rb["pwml_written"]:
        st = "only_" + ("A" if ra["pwml_written"] else "B") + "_wrote"
    else:
        st = "neither_wrote"
    tot[g][st] += 1
    if st == "differs":
        d = xdiff(R / A / "pwml" / ("%d.pwml" % i), R / B / "pwml" / ("%d.pwml" % i))
        print("idx %d %s differs: %d element deltas; first: %s" % (i, g, len(d), d[:6]))
    elif st not in ("identical", "neither_wrote"):
        print("idx %d %s %s" % (i, g, st))
for g, c in sorted(tot.items()):
    print(g, dict(c))
