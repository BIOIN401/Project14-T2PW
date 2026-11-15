"""Run-to-run variation. Reads final_mapped.json / stage1_payload.json per run (read-only). Writes var_runs.json
Key choices: entities=(bucket,casefold name); reactions=sorted pair of side-sets (name sets); transports=(cargo,transporter set);
interactions=(unordered endpoint pair, relationship) SAME_AS excluded."""
import csv, json, os, re, sys, itertools, statistics as st, collections
R = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
rows = list(csv.DictReader(open("C:/t/oa/docs/pwml_output/evidence/01/tidy_runs.csv")))
def nm(s): return re.sub(r"\s+", " ", str(s or "").strip().casefold())
def J(p):
    try: return json.load(open(p, encoding="utf-8"))
    except Exception: return None
def absp(p): return p if re.match(r"^[A-Za-z]:", p) else R + p
EB = ("compounds", "proteins", "protein_complexes", "nucleic_acids")
def eid(b, r):
    if b == "proteins": return r.get("uniprot") or r.get("pathbank_protein_id")
    if b == "compounds": return r.get("hmdb_id") or r.get("pathbank_compound_id") or r.get("chebi_id")
    return None
def keys(p):
    out = {k: set() for k in ("compounds","proteins","protein_complexes","nucleic_acids","reactions","transports","interactions")}
    ids = {}
    if not p: return None, None
    E = p.get("entities") or {}
    for b in EB:
        for r in E.get(b) or []:
            if isinstance(r, dict) and r.get("name"):
                out[b].add(nm(r["name"]))
                i = eid(b, r)
                if i not in (None, ""): ids[(b, nm(r["name"]))] = str(i)
    P = p.get("processes") or {}
    for r in P.get("reactions") or []:
        if not isinstance(r, dict): continue
        a = tuple(sorted(nm(x) for x in (r.get("inputs") or []))); b = tuple(sorted(nm(x) for x in (r.get("outputs") or [])))
        out["reactions"].add(tuple(sorted([a, b])))
    for r in P.get("transports") or []:
        if not isinstance(r, dict): continue
        t = tuple(sorted(nm(x.get("protein") or x.get("entity") if isinstance(x, dict) else x) for x in (r.get("transporters") or [])))
        out["transports"].add((nm(r.get("cargo")), t))
    for r in P.get("interactions") or []:
        if not isinstance(r, dict): continue
        rel = nm(r.get("relationship"))
        if rel == "same_as": continue
        out["interactions"].add((tuple(sorted([nm(r.get("entity_1")), nm(r.get("entity_2"))])), rel))
    return out, ids
def era(d, path):
    if not d: return "2026-11web"
    return d[:7] if d < "2026-10-20" else "2026-10late"
runs = []
for x in rows:
    if x["qok"] != "True": continue
    fp = absp(x["path"]); d = os.path.dirname(fp)
    F, fid = keys(J(fp)); S, sid = keys(J(os.path.join(d, "stage1_payload.json")))
    if F is None: continue
    runs.append(dict(path=x["path"], paper=x["paper"], mode=x["mode"], date=x["date"], era=era(x["date"], x["path"]), pwml=bool(x["pwml_file"]),
                     F={k: sorted(map(repr, v)) for k, v in F.items()}, S={k: sorted(map(repr, v)) for k, v in (S or {}).items()} if S else None, fid={b+"|"+n:v for (b,n),v in fid.items()}))
json.dump(runs, open("C:/t/ev-oa/accvar/var_runs.json", "w"))
print(len(runs))
