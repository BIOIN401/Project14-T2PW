"""analyze_import.py NAME=path/to/results.jsonl ... -> per-set cause table, clean files, created vs in-file"""
import collections, json, sys

def cause(key, msg):
    k = key.rsplit("_", 1)[0] if key.rsplit("_", 1)[-1].isdigit() else key
    if "Ontology" in msg: return "ontology"
    if "Spontaneous" in msg: return "spontaneous"
    if k.startswith("bound_visualization"): return "bound_visualization"
    if k == "transport_compound_visualization" or "Compound location" in msg: return "compound_location"
    if k.startswith("interaction"): return "interaction"
    if k.startswith("transport"): return "transport"
    if k.startswith(("protein_complex", "protein", "species", "reaction_enzyme", "protein_location")): return "complex/species"
    return "other"

out = {}
for a in sys.argv[1:]:
    name, path = a.split("=", 1)
    rs = [json.loads(l) for l in open(path, encoding="utf-8")]
    c = collections.Counter(); files = collections.defaultdict(set); other = collections.Counter()
    clean = 0; tot = 0; te = tc = ie = ic = 0; succ = 0
    for i, r in enumerate(rs):
        n = 0
        succ += bool(r.get("success"))
        for k, v in r["errors"].items():
            for m in v:
                cs = cause(k, m); c[cs] += 1; files[cs].add(i); n += 1
                if cs == "other": other[(k.rsplit("_", 1)[0] if k.rsplit("_", 1)[-1].isdigit() else k) + ": " + m] += 1
        tot += n; clean += n == 0
        e = r.get("expected") or {}; cr = r.get("created") or {}; cm = r.get("created_main") or {}
        te += e.get("transports", 0); tc += cm.get("transports", cr.get("transports", 0))
        ie += e.get("interactions", 0); ic += cr.get("interactions", 0)
    out[name] = {"files": len(rs), "success_true": succ, "clean_files": clean, "error_messages": tot,
                 "by_cause_messages": dict(c), "by_cause_files": {k: len(v) for k, v in files.items()},
                 "transports_in_file": te, "transports_created": tc, "interactions_in_file": ie, "interactions_created": ic,
                 "other_detail": dict(other.most_common(12))}
print(json.dumps(out, indent=1))
