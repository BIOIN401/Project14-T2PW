import json

old = json.load(open("C:/t/render-corr/web/metrics.json", encoding="utf-8"))["per_pathway"]
new = {str(r["idx"]): r for r in json.load(open("C:/t/vis-runs/scratch/layout/metrics97.json", encoding="utf-8"))["runs"]}
print(len(old), len(new), set(old) ^ set(new))
rows = []
for k in sorted(old, key=int):
    o, n = old[k], new[k]["svg"]
    d = {m: n[m] - o[m] for m in ("crossings", "bends", "enzyme_links", "edges", "drawings")}
    if any(d.values()):
        try:
            ir = json.load(open("C:/t/render-corr/ir/%s/pwml_ir.json" % k, encoding="utf-8"))
            P = ir["processes"]
            c = (len(P.get("reactions", [])), len(P.get("transports", [])), len(P.get("interactions", [])))
        except Exception as e:
            c = str(e)[:40]
        rows.append((k, d, c, o["style"]))
for r in rows:
    print(r)
print(len(rows), "runs differ;", sum(1 for k in old if all(new[k]["svg"][m] == old[k][m] for m in ("crossings", "bends", "enzyme_links", "edges", "drawings"))), "identical")
pw = {str(r["idx"]): r["pwml"] for r in json.load(open("C:/t/vis-runs/scratch/layout/metrics97.json", encoding="utf-8"))["runs"]}
print("reading order per run (viol/pairs) where viol:", {k: (v["reading_order_violations"], v["precedence_pairs"]) for k, v in pw.items() if v["reading_order_violations"]})
