"""Read-only: which UniProtKB accessions share the exact sequence of each KF147 Men protein?

For every UniParc entry (UPI) carrying a KF147 (taxon 684738) Men gene, list its
active UniProtKB cross-references. A shared UPI means an identical amino-acid
sequence, so the UniProtKB accession denotes the same polypeptide.
"""
import json, sys, time, urllib.parse, urllib.request

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "orch761-readonly-probe"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)

GENES = ["menF", "menD", "menX", "menH", "menB", "menE", "menC", "yhdB", "menA", "menG", "ubiE", "menI", "yhdA", "preA"]
out = {}
for g in GENES:
    q = f"taxonomy_id:684738 AND gene:{g}"
    try:
        d = get("https://rest.uniprot.org/uniparc/search?" + urllib.parse.urlencode({"query": q, "format": "json", "size": 10}))
    except Exception as e:
        print(g, "ERR", e); continue
    out[g] = []
    for r in d.get("results", []):
        upi = r.get("uniParcId")
        try:
            full = get(f"https://rest.uniprot.org/uniparc/{upi}?format=json")
        except Exception as e:
            print(g, upi, "ERR", e); continue
        xrefs = full.get("uniParcCrossReferences", []) or []
        kf = [x for x in xrefs if str((x.get("organism") or {}).get("taxonId")) == "684738"]
        kf_genes = sorted({x.get("geneName") for x in kf if x.get("geneName")})
        kf_names = sorted({x.get("proteinName") for x in kf if x.get("proteinName")})[:2]
        kb = [(x.get("id"), x.get("database"), (x.get("organism") or {}).get("taxonId"), x.get("active"))
              for x in xrefs if str(x.get("database", "")).startswith("UniProtKB") and x.get("active")]
        seqlen = (full.get("sequence") or {}).get("length")
        rec = {"upi": upi, "len": seqlen, "kf147_genes": kf_genes, "kf147_names": kf_names, "active_uniprotkb": kb[:12]}
        out[g].append(rec)
        print(g, upi, "len", seqlen, "KF147 genes", kf_genes, kf_names, "| active UniProtKB:", kb[:8])
        time.sleep(0.2)
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
