"""Canonical-run and TCA name-mode probes. 1 req/s, cached. Output summary_extra.json"""
import sys; sys.argv = ["x", "none"]
exec(open("C:/t/cm/docs/pwml_output/evidence/08/s2_query.py", encoding="utf-8").read().split("which = sys.argv")[0])
import glob
R = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
SP = "http://vocabularies.wikipathways.org/wp#"
PFX = ('PREFIX dc: <http://purl.org/dc/elements/1.1/> PREFIX dcterms: <http://purl.org/dc/terms/> PREFIX wp: <http://vocabularies.wikipathways.org/wp#> '
       'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> ')
def sparql(q):
    t = get("https://sparql.wikipathways.org/sparql?" + urllib.parse.urlencode({"query": PFX + q}), "application/sparql-results+json")
    try: return [{k: v["value"] for k, v in b.items()} for b in json.loads(t)["results"]["bindings"]]
    except Exception: return []
def wp_find(kw_list, organism):
    f = " || ".join(f'CONTAINS(LCASE(?t),"{k}")' for k in kw_list)
    return sparql('SELECT DISTINCT ?pw ?t ?o WHERE { ?pw a wp:Pathway ; dc:title ?t ; wp:organismName ?o . FILTER((%s) && ?o="%s") }' % (f, organism))
def wp_edges(pwuri):
    rows = sparql('SELECT DISTINCT ?l1 ?l2 WHERE { ?i wp:source ?a ; wp:target ?b ; dcterms:isPartOf <%s> . ?a a wp:Metabolite ; rdfs:label ?l1 . ?b a wp:Metabolite ; rdfs:label ?l2 }' % pwuri)
    return [(r["l1"], r["l2"]) for r in rows]
def delivered(pid):
    p = glob.glob(R + "runs_*/**/" + pid + "/*/final_mapped.json", recursive=True)
    best = max(p, key=lambda x: len(json.load(open(x, encoding="utf-8"))["processes"]["reactions"]))
    d = json.load(open(best, encoding="utf-8")); u = {}
    for r in d["processes"]["reactions"]: u[(tuple(sorted(r["inputs"])), tuple(sorted(r["outputs"])))] = r
    return list(u)
def tokens(n): return {norm(n)}
def covers(dl, a, b):  # delivered row has a on input side (or output) and b on the other
    na, nb = norm(a), norm(b)
    return any((na in {norm(x) for x in i} and nb in {norm(x) for x in o}) or (nb in {norm(x) for x in i} and na in {norm(x) for x in o}) for i, o in dl)
out = {}
CASES = {"PMC5130987": (["chlorophyll", "porphyrin"], "Arabidopsis thaliana"), "PMC12376012": (["sphingolipid", "ceramide"], "Homo sapiens"),
         "PMC11487621": (["coenzyme a", "pantothenate", "vitamin b5"], "Bacillus subtilis")}
for pid, (kw, org) in CASES.items():
    dl = delivered(pid); o = {"delivered_unique_rows": len(dl)}
    # Rhea confirmation of delivered rows (literal names, no alias)
    conf = 0; ex = []
    for i, ou in dl:
        if not i or not ou: continue
        q = f'"{i[0]}" AND "{ou[0]}"'
        t = get("https://www.rhea-db.org/rhea?" + urllib.parse.urlencode({"query": q, "columns": "rhea-id,equation", "format": "tsv", "limit": 10}))
        ok = False
        for ln in t.splitlines()[1:]:
            p = ln.split("\t")
            if len(p) > 1 and " = " in p[1]:
                L, Rr = p[1].split(" = ", 1)
                ls = [{norm(x)} for x in L.split(" + ")]; rs = [{norm(x)} for x in Rr.split(" + ")]
                if (norm(i[0]) in set().union(*ls) and norm(ou[0]) in set().union(*rs)) or (norm(ou[0]) in set().union(*ls) and norm(i[0]) in set().union(*rs)): ok = True
        conf += ok
    o["rhea_confirmed_by_literal_name"] = conf
    pws = wp_find(kw, org); o["wp_pathways"] = [(p["pw"].split("/")[-1], p["t"]) for p in pws]
    best = None
    for p in pws[:6]:
        e = wp_edges(p["pw"]); m = sum(covers(dl, a, b) for a, b in e)
        o.setdefault("wp_detail", []).append({"id": p["pw"].split("/")[-1], "title": p["t"], "metabolite_edges": len(e), "edges_matched_by_delivered": m,
                                              "edges_not_delivered": [f"{a} -> {b}" for a, b in e if not covers(dl, a, b)][:12]})
    out[pid] = o
    print(pid, json.dumps(o)[:300], flush=True)
# Reactome human sphingolipid de novo pathway coverage vs delivered
t = json.loads(get("https://reactome.org/ContentService/search/query?" + urllib.parse.urlencode({"query": "sphingolipid de novo biosynthesis", "types": "Pathway", "species": "Homo sapiens", "rows": 3})))
pw = [e["stId"] + " " + e["name"] for g in t["results"] for e in g["entries"]]
out["reactome_sphingo_search"] = pw
# TCA name mode
tca = {}
for sp in ("Homo sapiens", "Escherichia coli"):
    t = json.loads(get("https://reactome.org/ContentService/search/query?" + urllib.parse.urlencode({"query": "TCA cycle", "types": "Pathway", "species": sp, "rows": 5})))
    tca["reactome_" + sp] = [e["stId"] + " " + re.sub("<[^>]+>", "", e["name"]) for g in t.get("results", []) for e in g["entries"]]
    tca["wp_" + sp] = [(p["pw"].split("/")[-1], p["t"], len(wp_edges(p["pw"]))) for p in wp_find(["tca cycle", "krebs", "citric acid"], sp)]
    tca["wp_ecoli_or_human_note"] = ""
ev = json.loads(get("https://reactome.org/ContentService/data/pathway/R-HSA-71403/containedEvents"))
tca["reactome_R-HSA-71403_contained_events"] = [(e.get("stId"), e.get("schemaClass"), e.get("displayName")) for e in ev if isinstance(e, dict)]
tca["rhea_by_ec"] = {}
for ec, nm in [("2.3.3.1", "citrate synthase"), ("4.2.1.3", "aconitase"), ("1.1.1.42", "isocitrate DH"), ("1.2.4.2", "OGDH"), ("6.2.1.5", "succinyl-CoA synthetase"),
               ("1.3.5.1", "succinate DH"), ("4.2.1.2", "fumarase"), ("1.1.1.37", "malate DH")]:
    t = get("https://www.rhea-db.org/rhea?" + urllib.parse.urlencode({"query": "ec:" + ec, "columns": "rhea-id,equation", "format": "tsv", "limit": 6}))
    tca["rhea_by_ec"][ec + " " + nm] = t.splitlines()[1:4]
out["tca"] = tca
json.dump(out, open("C:/t/cm/docs/pwml_output/evidence/08/summary_extra.json", "w"), indent=1)
