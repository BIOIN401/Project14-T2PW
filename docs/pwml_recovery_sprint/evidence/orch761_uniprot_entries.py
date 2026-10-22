"""Read-only: fetch named UniProtKB entries and print name, genes, EC, organism, review status."""
import json, sys, time, urllib.request
ACC = sys.argv[2:]
out = {}
for a in ACC:
    req = urllib.request.Request(f"https://rest.uniprot.org/uniprotkb/{a}.json", headers={"User-Agent": "orch761-readonly-probe"})
    with urllib.request.urlopen(req, timeout=60) as r:
        e = json.load(r)
    pd = e.get("proteinDescription", {})
    rec = (pd.get("recommendedName") or {})
    name = (rec.get("fullName") or {}).get("value") or ((pd.get("submissionNames") or [{}])[0].get("fullName") or {}).get("value")
    ecs = [x.get("value") for x in rec.get("ecNumbers", [])] + [x.get("value") for s in pd.get("submissionNames", []) for x in s.get("ecNumbers", [])]
    genes = []
    for g in e.get("genes", []):
        if g.get("geneName"): genes.append(g["geneName"]["value"])
        genes += ["syn:" + s["value"] for s in g.get("synonyms", [])]
        genes += ["locus:" + s["value"] for s in g.get("orderedLocusNames", [])]
    org = e.get("organism", {})
    out[a] = {"type": e.get("entryType"), "name": name, "ec": ecs, "genes": genes, "organism": org.get("scientificName"), "taxid": org.get("taxonId"),
              "length": (e.get("sequence") or {}).get("length")}
    print(a, json.dumps(out[a], ensure_ascii=False))
    time.sleep(0.2)
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
