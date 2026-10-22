"""Read-only: do KF147 Men proteins exist in UniParc / proteomes (outside UniProtKB)?"""
import json, sys, urllib.parse, urllib.request

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "orch761-readonly-probe"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.headers.get("X-Total-Results"), json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, None, {"error": e.read().decode("utf-8", "replace")[:300]}

out = {}
for g in ["menF", "menD", "menB", "menA"]:
    q = f"taxonomy_id:684738 AND gene:{g}"
    s, t, d = get("https://rest.uniprot.org/uniparc/search?" + urllib.parse.urlencode({"query": q, "format": "json", "size": 5}))
    hits = [r.get("uniParcId") for r in d.get("results", [])] if isinstance(d, dict) else []
    out[q] = {"status": s, "total": t, "hits": hits, "err": d.get("error") if isinstance(d, dict) else None}
    print("uniparc", q, s, t, hits, (d.get("error") or "")[:200] if isinstance(d, dict) else "")
s, t, d = get("https://rest.uniprot.org/proteomes/search?" + urllib.parse.urlencode({"query": "organism_id:684738", "format": "json", "size": 5}))
pr = [(r.get("id"), r.get("proteomeType"), r.get("proteinCount"), (r.get("redundantTo") or "")) for r in d.get("results", [])] if isinstance(d, dict) else []
out["proteomes organism_id:684738"] = {"status": s, "total": t, "rows": pr}
print("proteomes 684738", s, t, pr)
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
