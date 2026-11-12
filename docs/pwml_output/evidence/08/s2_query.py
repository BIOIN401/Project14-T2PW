"""Query public APIs (Reactome ContentService, Rhea REST, WikiPathways SPARQL) for each gold reaction.
Usage: python -I s2_query.py rhea|wp|reactome   (1 request/s, cached in C:/t/ev-kpdb/raw). No KEGG. No personal email."""
import subprocess, sys, json, re, time, hashlib, os, urllib.request, urllib.parse
RAW = "C:/t/ev-kpdb/raw/"; UA = "t2pw-assessment/0.1 (research)"
GOLD = json.load(open("C:/t/cm/docs/pwml_output/evidence/08/gold_reactions.json", encoding="utf-8"))
ALIAS = json.load(open("C:/t/cm/docs/pwml_output/evidence/08/aliases.json", encoding="utf-8"))
_last = [0.0]
def get(url, accept=None):
    f = RAW + hashlib.md5(url.encode()).hexdigest() + ".txt"
    if os.path.exists(f): return open(f, encoding="utf-8").read()
    w = 1.1 - (time.time() - _last[0]);  time.sleep(max(0, w))
    cmd = ["curl", "-sL", "-m", "60", "-A", UA] + (["-H", "Accept: " + accept] if accept else []) + [url]
    r = subprocess.run(cmd, capture_output=True); txt = r.stdout.decode("utf-8", "replace") if r.returncode == 0 else "ERR curl " + str(r.returncode)
    _last[0] = time.time(); open(f, "w", encoding="utf-8").write(txt); return txt
def norm(s):
    s = re.sub(r"\s*\[[^\]]*\]$", "", s.lower()); s = s.replace("ic acid", "ate")
    return re.sub(r"[^a-z0-9]", "", s)
def aliases(n):
    return {norm(a) for a in [n] + ALIAS.get(n.lower(), [])}
def side_has(names_norm, gold_name):  # names_norm: list of sets of norm synonyms per participant
    a = aliases(gold_name); return any(a & s for s in names_norm)
def hit(g, ins_sets, outs_sets):
    fw = all(side_has(ins_sets, x) for x in g["in"]) and all(side_has(outs_sets, x) for x in g["out"])
    bw = all(side_has(outs_sets, x) for x in g["in"]) and all(side_has(ins_sets, x) for x in g["out"])
    return "fwd" if fw else ("rev" if bw else None)
which = sys.argv[1]; res = {}
for pid, v in GOLD.items():
    for r in v["rx"]:
        key = f"{pid}#{r['i']}"; hits = []
        if which == "rhea":
            q = " AND ".join(f'"{(ALIAS.get(x.lower()) or [x])[0]}"' for x in (r["in"][:1] + r["out"][:1]))
            t = get("https://www.rhea-db.org/rhea?" + urllib.parse.urlencode({"query": q, "columns": "rhea-id,equation,ec,uniprot", "format": "tsv", "limit": 25}))
            for ln in t.splitlines()[1:]:
                p = ln.split("\t")
                if len(p) < 2 or " = " not in p[1]: continue
                L, R = p[1].split(" = ", 1)
                ls = [aliases(x) for x in L.split(" + ")]; rs = [aliases(x) for x in R.split(" + ")]
                h = hit(r, ls, rs)
                if h: hits.append({"id": p[0], "eq": p[1], "dir": h, "ec": p[2] if len(p) > 2 else ""})
        elif which == "wp":
            sa = sorted({a for x in r["in"][:1] for a in [x.lower()] + [y.lower() for y in ALIAS.get(x.lower(), [])]})
            sb = sorted({a for x in r["out"][:1] for a in [x.lower()] + [y.lower() for y in ALIAS.get(x.lower(), [])]})
            fa = " || ".join(f'lcase(str(?l1))="{n}"' for n in sa); fb = " || ".join(f'lcase(str(?l2))="{n}"' for n in sb)
            fa2 = fa.replace("?l1", "?l2"); fb2 = fb.replace("?l2", "?l1")
            q = ('PREFIX dc: <http://purl.org/dc/elements/1.1/> PREFIX dcterms: <http://purl.org/dc/terms/> PREFIX wp: <http://vocabularies.wikipathways.org/wp#> '
                 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT DISTINCT ?pw ?t ?o WHERE { ?i a ?ty ; wp:source ?a ; wp:target ?b ; dcterms:isPartOf ?pw . '
                 '?a rdfs:label ?l1 . ?b rdfs:label ?l2 . ?pw dc:title ?t ; wp:organismName ?o . FILTER((%s) && (%s)) } LIMIT 40' % (fa, fb))
            t = get("https://sparql.wikipathways.org/sparql?" + urllib.parse.urlencode({"query": q}), "application/sparql-results+json")
            try:
                for b in json.loads(t)["results"]["bindings"]:
                    hits.append({"id": b["pw"]["value"].split("/")[-1].split("_")[0], "title": b["t"]["value"], "species": b["o"]["value"]})
            except Exception: hits.append({"err": t[:100]})
        elif which == "reactome":
            ids = []
            A = lambda x: (ALIAS.get(x.lower()) or [x])
            qs = [A(r["out"][0])[-1], A(r["in"][0])[-1], A(r["in"][0])[-1] + " " + A(r["out"][0])[-1], A(r["out"][0])[0]]
            for nm in dict.fromkeys(qs):
                for sp in (["Homo sapiens"] if pid in ("PMC12856317", "PMC12782028") else []) + [None]:
                    prm = {"query": nm, "types": "Reaction", "rows": 15}
                    if sp: prm["species"] = sp
                    t = get("https://reactome.org/ContentService/search/query?" + urllib.parse.urlencode(prm))
                    try:
                        for g in json.loads(t)["results"]:
                            for e in g["entries"]:
                                if e["stId"] not in ids: ids.append(e["stId"])
                    except Exception: pass
            for sid in ids[:60]:
                t = get("https://reactome.org/ContentService/data/query/enhanced/" + sid)
                try: d = json.loads(t)
                except Exception: continue
                def sets(l): return [{norm(n) for n in ([e.get("displayName", "")] + (e.get("name") or [])) if n} for e in l if isinstance(e, dict)]
                h = hit(r, sets(d.get("input", [])), sets(d.get("output", [])))
                if h: hits.append({"id": sid, "name": d.get("displayName"), "species": (d.get("speciesName") or ""), "dir": h})
        res[key] = hits
        print(which, key, len(hits), flush=True)
json.dump(res, open(f"C:/t/cm/docs/pwml_output/evidence/08/hits_{which}.json", "w"), indent=1)
