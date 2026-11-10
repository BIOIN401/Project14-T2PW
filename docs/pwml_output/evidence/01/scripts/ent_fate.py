"""Fate of each merged_payload entity (pathwhiz canonical runs): kept / merged-duplicate / removed (closure reason) / other. Also research runs for comparison."""
import json, os, re, collections
R = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
rows = [json.loads(l) for l in open("C:/t/ev-oa/funnel/out/runs_funnel.jsonl")]
def norm(s):
    s = str(s or "").casefold(); s = re.sub(r"\([^)]*\)", " ", s); return re.sub(r"[^a-z0-9]+", " ", s).strip()
class UF:
    def __init__(s): s.p = {}
    def f(s, x):
        s.p.setdefault(x, x)
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b): s.p[s.f(a)] = s.f(b)
res = collections.defaultdict(collections.Counter)
for r in rows:
    if r["qok"] is not True: continue
    d = os.path.dirname(R + r["path"] if not r["path"].startswith("C:") else r["path"])
    mg = json.load(open(d + "/merged_payload.json", encoding="utf-8")); fm = json.load(open(d + "/final_mapped.json", encoding="utf-8"))
    rem = json.load(open(d + "/removed_entity_report.json", encoding="utf-8")) if os.path.exists(d + "/removed_entity_report.json") else {}
    uf = UF()
    for x in mg["processes"].get("interactions", []):
        if str(x.get("relationship")).upper() == "SAME_AS": uf.u(norm(x.get("entity_1")), norm(x.get("entity_2")))
    fnames = collections.defaultdict(set)
    for b in ("compounds", "proteins", "protein_complexes", "nucleic_acids"):
        for e in fm["entities"].get(b, []):
            if isinstance(e, dict) and e.get("name"): fnames[b].add(uf.f(norm(e["name"])))
    allf = set().union(*fnames.values()) if fnames else set()
    remd = {(x.get("bucket"), uf.f(norm(x.get("name")))): x.get("reason") for x in rem.get("removed_entities", [])}
    grp = "research" if r["mode"] == "research" else "pathwhiz"
    for b in ("compounds", "proteins", "protein_complexes", "nucleic_acids"):
        seen = set()
        for e in mg["entities"].get(b, []):
            if not isinstance(e, dict) or not e.get("name"): continue
            k = uf.f(norm(e["name"]))
            if k in seen:
                res[(grp, b)]["M-internal duplicate (alias-resolved name)"] += 1; continue
            seen.add(k)
            if k in allf: res[(grp, b)]["kept in F (same name or alias-merged)"] += 1
            elif (b, k) in remd: res[(grp, b)]["removed: " + str(remd[(b, k)])] += 1
            else: res[(grp, b)]["absent from F, not in removed report"] += 1
for k in sorted(res):
    t = sum(res[k].values())
    print(k, t, {a: f"{v} ({v/t:.0%})" for a, v in res[k].most_common()})
