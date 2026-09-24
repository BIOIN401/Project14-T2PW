import json, urllib.request, urllib.parse
OUT = "C:/t/mining-pilot-runs3/bio-audit/"
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "t2pw-bio-audit/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode("utf-8", "replace")
base = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=lite&pageSize=2&query="
rows = [json.loads(l) for l in open("C:/t/mining-pilot-runs3/pilot-003-jasmonate-ath/papers.jsonl", encoding="utf-8")]
res = []
for r in rows:
    if r["acquisition_source"] != "ncbi":
        continue
    q = f"EXT_ID:{r['pmid']} AND SRC:MED"
    try:
        d = json.loads(get(base + urllib.parse.quote(q)))
        hit = (d.get("resultList", {}).get("result") or [{}])[0]
        res.append({"paper_id": r["paper_id"], "rec_pmid": r["pmid"], "rec_pmcid": r["pmcid"], "rec_title": r["title"][:90],
                    "epmc_pmcid": hit.get("pmcid"), "epmc_title": (hit.get("title") or "")[:90]})
    except Exception as e:
        res.append({"paper_id": r["paper_id"], "error": str(e)})
json.dump(res, open(OUT + "ncbi_identity_check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("checked", len(res))
