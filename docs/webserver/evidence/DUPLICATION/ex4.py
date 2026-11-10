import json, sys, glob
cases = [("runs_verify/2026-09-16_2152/papers/PMC13170813/strict", ["hordatine A", "hordatine C"]),
         ("runs_verify/2026-09-01_1612/papers/PMC12096016/strict", ["EntB", "holo-EntB", "EntE", "enterobactin synthase"]),
         ("runs_verify/2026-09-06_1425/papers/PMC12376012/strict", ["L-serine", "serine"]),
         ("runs_smoke/2026-09-08_1528/papers/PMC6112128/strict", ["compound 1", "compound 2"]),
         ("runs_smoke/2026-09-07_2323/papers/PMC11487621/strict", ["CoA-SH", "coenzyme A"])]
for d, names in cases:
    p = json.load(open(d + "/final_mapped.json", encoding="utf-8"))
    for b in ("compounds", "proteins", "protein_complexes"):
        for r in p["entities"].get(b, []):
            if r.get("name") in names:
                ids = r.get("mapped_ids") or {}
                print(d[-34:], b, r["name"], {k: v for k, v in ids.items() if k in ("chebi", "kegg", "hmdb", "uniprot", "pubchem", "pathbank_compound_id")}, "comps", [c.get("name") for c in r.get("components", [])][:4])
