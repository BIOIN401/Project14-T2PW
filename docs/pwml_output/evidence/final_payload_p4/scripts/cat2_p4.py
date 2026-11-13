"""P3-comparable categories (same definitions as FINAL-PAYLOAD-P3.md s4). Usage: cat2_p4.py name=results.jsonl ..."""
import json, sys, collections
def base(k): return k.rsplit("_", 1)[0] if k.rsplit("_", 1)[-1].isdigit() else k
for name, path in (a.split("=", 1) for a in sys.argv[1:]):
    rs = [json.loads(l) for l in open(path, encoding="utf-8")]
    c = collections.Counter(); f = collections.defaultdict(set); keys = collections.defaultdict(collections.Counter)
    for i, r in enumerate(rs):
        for k, v in r["errors"].items():
            b = base(k)
            for m in v:
                if b in ("subcellular_location", "cell_type", "tissue"): ct = "vocabulary"
                elif b.startswith("nucleic_acid") or b == "interaction_nucleic_acid_visualization": ct = "nucleic acid"
                elif b.startswith("transport"):
                    ct = "transport family"
                else: ct = "interactions" if b.startswith("interaction") else "other"
                c[ct] += 1; f[ct].add(i); keys[ct][b + ": " + m[:70]] += 1
                if "unique biological states" in m: c["  of which left=right"] += 1
                if b == "transport_transporter_visualization": c["  of which transporter visualization blank"] += 1
    print("==", name, "clean", sum(1 for r in rs if not r["errors"]), "of", len(rs), "msgs", sum(c[x] for x in ("vocabulary", "nucleic acid", "transport family", "interactions", "other")))
    for k in sorted(c): print("  %-44s %3d  files %s" % (k, c[k], len(f[k]) if k in f else ""))
    for ct in keys:
        if ct in ("nucleic acid", "interactions", "other", "transport family"):
            for k, v in keys[ct].most_common(): print("     [%s] %3d %s" % (ct, v, k))
