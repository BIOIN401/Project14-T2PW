"""ORCH-769b F3: for each reaction the tip quarantines on the chlorophyll leg, the surviving reaction(s)
with the same inputs and outputs, and the catalysts on each side. Stdlib only.

    python twin_probe.py <fixture final_mapped.json.gz> <tip canonical.json>
"""
import gzip
import json
import sys

fm = json.loads(gzip.decompress(open(sys.argv[1], "rb").read()).decode("utf-8"))
canon = json.loads(open(sys.argv[2], encoding="utf-8").read())
canon = canon.get("payload", canon) if isinstance(canon, dict) and "processes" not in canon else canon


def names(xs):
    return [x.get("entity") if isinstance(x, dict) else x for x in xs or []]


def sig(r):
    f = lambda xs: tuple(sorted(str(x.get("entity") if isinstance(x, dict) else x).casefold() for x in xs or []))  # noqa: E731
    return f(r.get("inputs")), f(r.get("outputs"))


surv = canon["processes"]["reactions"]
print("surviving reactions:", len(surv), "interactions:", len(canon["processes"].get("interactions") or []))
for i in (17, 18, 21, 22, 23):
    r = fm["processes"]["reactions"][i]
    print(f"\nr{i} {r['name']!r}\n  enzymes={names(r.get('enzymes'))}\n  modifiers={names(r.get('modifiers'))}")
    for s in surv:
        if sig(s) == sig(r):
            print(f"  twin {s['name']!r}: enzymes={names(s.get('enzymes'))} modifiers={names(s.get('modifiers'))}")
