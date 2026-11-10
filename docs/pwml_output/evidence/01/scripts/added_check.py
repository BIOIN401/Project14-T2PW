import json, os, re, collections
R = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
rows = [json.loads(l) for l in open("C:/t/ev-oa/funnel/out/runs_funnel.jsonl")]
def nz(s): return re.sub(r"[^a-z0-9]+", " ", str(s or "").casefold()).strip()
c = collections.Counter()
for r in rows:
    if r["qok"] is not True or r["date"] < "2026-09-01" and False: continue
    d = os.path.dirname(R + r["path"] if not r["path"].startswith("C:") else r["path"])
    mg = json.load(open(d + "/merged_payload.json", encoding="utf-8"))
    src = (mg.get("source_text_index") or {}).get("normalized") or ""
    if len(src) < 1000:
        c["no source text in merged_payload"] += 1; continue
    src = nz(src)
    for k in ("reactions", "interactions", "transports"):
        for x in mg["processes"].get(k, []):
            L = x.get("provenance_lineage") or []
            if L: continue
            if k == "interactions" and str(x.get("relationship")).upper() == "SAME_AS": continue
            ev = nz(x.get("evidence"))
            if not ev: c[(k, "no evidence")] += 1; continue
            c[(k, "evidence quote found verbatim in paper text" if ev in src else "evidence NOT found verbatim")] += 1
for k, v in sorted(c.items(), key=str): print(v, k)
