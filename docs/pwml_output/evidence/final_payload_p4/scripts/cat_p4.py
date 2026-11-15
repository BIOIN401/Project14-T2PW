import json, sys, collections
def cat(k, m):
    b = k.rsplit("_", 1)[0] if k.rsplit("_", 1)[-1].isdigit() else k
    if b in ("subcellular_location", "cell_type", "tissue"): return "vocabulary:" + b
    if "nucleic_acid" in b: return "nucleic acid"
    if b == "transport_transporter_visualization": return "transporter visualization blank"
    if b.startswith("transport"): return "transport family"
    if "unique biological states" in m: return "left=right"
    if b == "interaction": return "interactions"
    return "other"
for name, path in (a.split("=", 1) for a in sys.argv[1:]):
    rs = [json.loads(l) for l in open(path, encoding="utf-8")]
    c = collections.Counter(); f = collections.defaultdict(set); oth = collections.Counter()
    for i, r in enumerate(rs):
        for k, v in r["errors"].items():
            for m in v:
                ct = cat(k, m)
                if ct == "transport family" and "unique biological states" in m: ct = "left=right"
                if ct == "transport family" and "transporter" in k and "blank" in m: ct = "transporter visualization blank"
                c[ct] += 1; f[ct].add(i)
                if ct == "other": oth[k.rsplit("_", 1)[0] if k.rsplit("_", 1)[-1].isdigit() else k + ": " + m] += 1
    print("==", name, len(rs), "files; clean", sum(1 for r in rs if not r["errors"]), "success", sum(bool(r["success"]) for r in rs), "msgs", sum(c.values()))
    for k, v in sorted(c.items()): print("  %-34s %3d msgs %2d files" % (k, v, len(f[k])))
    for k, v in oth.most_common(): print("     other>", v, k)
    ex = sum((collections.Counter(r.get("expected") or {}) for r in rs), collections.Counter()); cr = sum((collections.Counter(r.get("created") or {}) for r in rs), collections.Counter())
    print("  transports exp/created", ex["transports"], cr["transports"], "| interactions", ex["interactions"], cr["interactions"])
