"""PathWhiz import errors per showcase: VIS-2 final arm vs a VIS-5 run (error keys with the record id stripped)."""
import json, re, sys
def load(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        name = re.sub(r"^(after_show_|v5_)", "", r["file"].replace("\\", "/").rsplit("/", 1)[-1])
        errs = {}
        for k, v in (r.get("errors") or {}).items():
            kk = re.sub(r"_\d+$", "", k)
            for m in v:
                errs[(kk, m)] = errs.get((kk, m), 0) + 1
        out[name] = (r.get("success"), errs, r.get("created"))
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
for n in sorted(b):
    if n not in a:
        print(n, "only in new:", b[n][0], sorted(b[n][1].items())); continue
    print(n, "success", a[n][0], "->", b[n][0], "| errors same" if a[n][1] == b[n][1] else f"| before {sorted(a[n][1].items())} after {sorted(b[n][1].items())}")
    ca, cb = a[n][2] or {}, b[n][2] or {}
    diff = {k: (ca.get(k), cb.get(k)) for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
    print("   created diff:", diff)
