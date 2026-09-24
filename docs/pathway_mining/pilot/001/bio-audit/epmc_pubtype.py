import json, urllib.request, urllib.parse
ids = ["PMC13364333", "PMC9947334", "PMC12280011", "PMC11300881", "PMC13096701"]
out = {}
for pmcid in ids:
    q = urllib.parse.urlencode({"query": f"PMCID:{pmcid}", "format": "json", "resultType": "core", "pageSize": 1})
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + q
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.load(r)
    res = (d.get("resultList") or {}).get("result") or []
    if not res:
        out[pmcid] = None
        continue
    x = res[0]
    out[pmcid] = {
        "title": x.get("title"),
        "pubType": (x.get("pubTypeList") or {}).get("pubType"),
        "pmid": x.get("pmid"),
        "journal": (x.get("journalInfo") or {}).get("journal", {}).get("title"),
        "year": x.get("pubYear"),
        "meshOrganisms": [m.get("descriptorName") for m in ((x.get("meshHeadingList") or {}).get("meshHeading") or []) if m.get("descriptorName") in ("Arabidopsis", "Marchantia")],
    }
json.dump(out, open("C:/t/mining-pilot-runs/bio-audit/epmc_pubtype.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
