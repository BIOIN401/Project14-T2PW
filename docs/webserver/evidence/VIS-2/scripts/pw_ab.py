"""PathWhiz import A/B: results.jsonl of the base and after harness runs -> table + json.
usage: pw_ab.py BASE_DIR AFTER_DIR OUT.json"""
import collections, json, os, re, sys

B, A, OUT = sys.argv[1:4]
KEYS = ["compounds", "proteins", "protein_complexes", "reactions", "transports", "interactions", "compound_locations",
        "nucleic_acids", "transport_transporters", "biological_states", "label_locations",
        "membrane_visualizations", "drawable_element_locations", "zoom_visualizations"]  # VIS-2: + the art tables


def load(d):
    rows = {}
    for l in open(os.path.join(d, "results.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        name = os.path.basename(r["file"]).rsplit(".", 1)[0].split("_", 1)[1]  # drop the arm prefix
        if not name.startswith("extra_"):  # files with no other arm (B2) are reported apart
            rows[name] = r
    return rows


def errs(r):
    c = collections.Counter()
    for k, msgs in (r.get("errors") or {}).items():
        for m in msgs:
            c[(re.sub(r"_\d+$", "", k), m)] += 1
    return c


rb, ra = load(B), load(A)
out = {"files": {}, "totals": {}}
tb, ta = collections.Counter(), collections.Counter()
eb, ea = collections.Counter(), collections.Counter()
print("%-28s %-8s %s" % ("file", "success", "created (base->after) | errors base->after"))
for f in sorted(set(rb) | set(ra)):
    b, a = rb.get(f, {}), ra.get(f, {})
    cb, ca = b.get("created") or {}, a.get("created") or {}
    row = {"success": [b.get("success"), a.get("success")], "created": {}, "expected": {},
           "errors": [sum(errs(b).values()), sum(errs(a).values())]}
    for k in KEYS:
        row["created"][k] = [cb.get(k, 0), ca.get(k, 0)]
        row["expected"][k] = [(b.get("expected") or {}).get(k), (a.get("expected") or {}).get(k)]
        tb[k] += cb.get(k, 0)
        ta[k] += ca.get(k, 0)
    eb.update(errs(b))
    ea.update(errs(a))
    row["down"] = [k for k in KEYS if ca.get(k, 0) < cb.get(k, 0)]
    out["files"][f] = row
    ch = ", ".join("%s %d->%d" % (k, *row["created"][k]) for k in KEYS if row["created"][k][0] != row["created"][k][1])
    print("%-28s %-8s %s | %d->%d %s" % (f, row["success"], ch or "same", *row["errors"], ("DOWN " + str(row["down"])) if row["down"] else ""))
out["totals"] = {"created_base": dict(tb), "created_after": dict(ta), "errors_base": sum(eb.values()), "errors_after": sum(ea.values()),
                 "messages_base": {"%s: %s" % k: n for k, n in eb.most_common()}, "messages_after": {"%s: %s" % k: n for k, n in ea.most_common()},
                 "files_with_a_created_count_down": [f for f, r in out["files"].items() if r["down"]]}
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
print("created base ", dict(tb))
print("created after", dict(ta))
print("errors", sum(eb.values()), "->", sum(ea.values()))
for k in sorted(set(eb) | set(ea)):
    if eb[k] != ea[k]:
        print("   %3d -> %3d  %s: %s" % (eb[k], ea[k], *k))
