import json, urllib.request, urllib.parse, re
OUT = "C:/t/mining-pilot-runs2/bio-audit/"
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "t2pw-bio-audit/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode("utf-8", "replace")
base = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=lite&pageSize=3&query="
qs = {
 "pmid12777050": "EXT_ID:12777050 AND SRC:MED",
 "title_cesium": 'TITLE:"Cesium Inhibits Plant Growth through Jasmonate Signaling"',
 "title_gsh": 'TITLE:"Glutathione and Its Biosynthetic Intermediates Alleviate Cesium Stress"',
 "title_pat1": 'TITLE:"GRAS-domain transcription factor PAT1 regulates jasmonic acid biosynthesis"',
 "pmc4844422": "PMCID:PMC4844422",
 "pmid31834385": "EXT_ID:31834385 AND SRC:MED",
 "pmid28818464": "EXT_ID:28818464 AND SRC:MED",
}
res = {}
for k, q in qs.items():
    try:
        d = json.loads(get(base + urllib.parse.quote(q)))
        res[k] = [{f: r.get(f) for f in ("pmid","pmcid","doi","title","pubYear","authorString")} for r in d.get("resultList",{}).get("result",[])]
    except Exception as e:
        res[k] = str(e)
json.dump(res, open(OUT+"epmc_ids2.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
try:
    xml = get("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11300881/fullTextXML")
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml))
    open(OUT+"nad_fulltext.txt","w",encoding="utf-8").write(txt)
    print("nad chars", len(txt))
except Exception as e:
    print("nad err", e)
print("done")
