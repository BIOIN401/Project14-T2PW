"""Paper classification + class x kind funnel. Output out/paper_classes.csv, out/class_kind.json"""
import json, collections, statistics, csv
OUT = "C:/t/ev-oa/funnel/out/"
rows = [json.loads(l) for l in open(OUT + "runs_funnel.jsonl")]
for r in rows:
    if "e2e" in r["path"]:
        r["paper"] = "PMC7059786"
    r["group"] = ("OTHER40" if r["qok"] is not True else "RS" if r["mode"] == "research" else "P1" if r.get("pwml") else "P0")
canon = [r for r in rows if r["qok"] is True]

def classify(R, T, I):
    tot = R + T + I
    if tot < 3: return "other (sparse: <3 processes at Stage 1)"
    if T >= 2 and T / tot >= 0.4: return "transport-heavy"
    if R / tot >= 0.5: return "metabolic"
    if I / tot >= 0.5: return "signalling"
    return "other (mixed)"

import re as _re
def name_class(n):
    n = (n or "").lower()
    if _re.search(r"transport|uptake|secretion|efflux|channel|taste|d-alanylation", n): return "transport-heavy"
    if _re.search(r"biosynthe|metabol|catabol|degrad|synthesis|cycle|glycoly", n): return "metabolic"
    if _re.search(r"signal|transduction|wnt|hippo|operon|regulat|pcp", n): return "signalling"
    return "other"
papers = collections.defaultdict(list)
for r in canon: papers[r["paper"]].append(r)
pc = {}
tab = []
for p, rs in sorted(papers.items()):
    med = lambda k: statistics.median(r["s1"][k] for r in rs)
    R_, T_, I_ = med("reactions"), med("transports"), med("interactions")
    cl = classify(R_, T_, I_)
    names = collections.Counter(r.get("pathway_name") for r in rs if r.get("pathway_name"))
    ctx = collections.Counter(r.get("context_type") for r in rs if r.get("context_type"))
    pc[p] = cl
    tab.append({"paper": p, "n_canon_runs": len(rs), "s1_median_R": R_, "s1_median_T": T_, "s1_median_I": I_, "class": cl,
                "pathway_name_mode": (names.most_common(1) or [("",)])[0][0], "context_type_mode": (ctx.most_common(1) or [("",)])[0][0]})
    tab[-1]["name_keyword_class"] = name_class(tab[-1]["pathway_name_mode"])
with open(OUT + "paper_classes.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(tab[0].keys())); w.writeheader(); w.writerows(tab)
print(collections.Counter(pc.values()))
for t in tab: print(t["paper"], t["n_canon_runs"], t["s1_median_R"], t["s1_median_T"], t["s1_median_I"], t["class"], "|", t["pathway_name_mode"], "|", t["context_type_mode"])

# class x kind (pathwhiz canon: P1+P0 ; research separately)
res = {}
for g in ("P", "RS"):
    for cl in sorted(set(pc.values())):
        rs = [r for r in canon if pc[r["paper"]] == cl and ((r["group"] in ("P1", "P0")) if g == "P" else r["group"] == "RS")]
        if not rs: continue
        d = {"n_runs": len(rs), "n_papers": len({r["paper"] for r in rs})}
        for kind in ("reactions", "transports", "interactions"):
            s1 = sum(r["s1"][kind] for r in rs); m = sum(r["m"][kind] for r in rs); f = sum(r["f"][kind] for r in rs)
            fps = sum(r["proc_origin"].get(f"f:{kind}:paper_stated", 0) for r in rs)
            mps = sum(r["proc_origin"].get(f"m:{kind}:paper_stated", 0) for r in rs)
            if g == "P":
                pw = {"reactions": "reactions", "transports": "transports", "interactions": "interactions"}[kind]
                drawn = sum(r["pwml"].get(pw, 0) for r in rs if r.get("pwml"))
            else:
                drawn = None
            d[kind] = {"S1": s1, "M": m, "M_paper_stated": mps, "F": f, "F_paper_stated": fps, "PWML": drawn}
        res[f"{g}|{cl}"] = d
json.dump(res, open(OUT + "class_kind.json", "w"), indent=1)
for k, v in res.items():
    print(k, v["n_runs"], v["n_papers"], {kd: v[kd] for kd in ("reactions", "transports", "interactions")})
