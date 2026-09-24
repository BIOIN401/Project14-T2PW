import json, urllib.request, urllib.parse, sys, re
OUT = "C:/t/mining-pilot-runs2/bio-audit/"
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "t2pw-bio-audit/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode("utf-8", "replace")
base = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=lite&pageSize=5&query="
queries = {
    "pmc411572": "PMCID:PMC411572",
    "title_13lox": 'TITLE:"Biochemical Characterization of 13-Lipoxygenases of Arabidopsis thaliana"',
    "pmc6628814": "PMCID:PMC6628814",
    "pmc3000969": "PMCID:PMC3000969",
    "title_lox2ser600": 'TITLE:"Pseudophosphorylation of Arabidopsis jasmonate biosynthesis enzyme lipoxygenase 2"',
    "pmc4793799": "PMCID:PMC4793799",
    "title_gh310": 'TITLE:"Arabidopsis GH3.10 conjugates jasmonates"',
    "pmc1955694": "PMCID:PMC1955694",
    "pmc5104190": "PMCID:PMC5104190",
    "pmc5338657": "PMCID:PMC5338657",
    "title_nad": 'TITLE:"NAD+ deficiency primes defense metabolism"',
}
res = {}
for k, q in queries.items():
    try:
        d = json.loads(get(base + urllib.parse.quote(q)))
        res[k] = [{f: r.get(f) for f in ("id","source","pmid","pmcid","doi","title","pubYear","journalTitle","authorString")} for r in d.get("resultList",{}).get("result",[])]
    except Exception as e:
        res[k] = str(e)
json.dump(res, open(OUT+"epmc_ids.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("ids done")
# full text of the real 13-LOX paper
pmc = None
for r in res.get("title_13lox") or []:
    if isinstance(r, dict) and r.get("pmcid"):
        pmc = r["pmcid"]; break
print("13-LOX pmcid", pmc)
if pmc:
    try:
        xml = get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML")
        open(OUT+"lox13_fulltext.xml","w",encoding="utf-8").write(xml)
        txt = re.sub(r"<[^>]+>", " ", xml); txt = re.sub(r"\s+", " ", txt)
        open(OUT+"lox13_fulltext.txt","w",encoding="utf-8").write(txt)
        print("fulltext chars", len(txt))
    except Exception as e:
        print("fulltext err", e)
