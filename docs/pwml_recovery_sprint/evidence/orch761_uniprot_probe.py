"""Read-only UniProt REST probe for ORCH-761. Plain urllib, no t2pw import.

Usage: python uniprot_probe.py OUT.json
Runs a fixed list of queries and records (query, http status, hit count, top hits).
"""
import json, sys, time, urllib.parse, urllib.request

FIELDS = "accession,protein_name,gene_names,organism_name,reviewed,organism_id"
BASE = "https://rest.uniprot.org/uniprotkb/search"

def search(query, size=25):
    url = BASE + "?" + urllib.parse.urlencode({"query": query, "format": "json", "size": size, "fields": FIELDS})
    req = urllib.request.Request(url, headers={"User-Agent": "orch761-readonly-probe"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            total = r.headers.get("X-Total-Results")
            data = json.load(r)
            status = r.status
    except urllib.error.HTTPError as e:
        return {"query": query, "status": e.code, "error": e.read().decode("utf-8", "replace")[:300]}
    hits = []
    for it in data.get("results", []):
        genes = []
        for g in it.get("genes", []) or []:
            if g.get("geneName"): genes.append(g["geneName"]["value"])
            for s in g.get("synonyms", []) or []: genes.append("syn:" + s["value"])
            for s in g.get("orderedLocusNames", []) or []: genes.append("locus:" + s["value"])
        pd = it.get("proteinDescription", {})
        name = (pd.get("recommendedName") or {}).get("fullName", {}).get("value") \
            or ((pd.get("submissionNames") or [{}])[0].get("fullName", {}) or {}).get("value")
        org = it.get("organism", {})
        hits.append({"acc": it.get("primaryAccession"), "reviewed": "unreviewed" not in it.get("entryType", "").lower(),
                     "name": name, "genes": genes, "organism": org.get("scientificName"), "taxid": org.get("taxonId")})
    return {"query": query, "status": status, "total": total, "hits": hits}

LL_MEN = ["menA", "menB", "menC", "menD", "menE", "menF", "menH", "menI", "menG", "ubiE", "menX", "mqnA", "preA", "ubiA"]
ENT = ["entA", "entB", "entC", "entD", "entE", "entF"]

QUERIES = []
# 1. exactly what T2PW sent (menaquinone, from initial_stage3_gate_report queries_tried)
for g in ["MenF", "MenD", "MenB", "MenA", "MenG"]:
    QUERIES.append(("t2pw_exact_organism_name", g, f'(protein_name:"{g}" OR gene:"{g}") AND organism_name:"Lactococcus lactis subsp. lactis KF147"'))
# 2. taxonomy-scoped variants
for g in LL_MEN:
    QUERIES.append(("taxonomy_id_1358_lineage", g, f'gene:{g} AND taxonomy_id:1358'))
    QUERIES.append(("organism_id_684738_KF147", g, f'gene:{g} AND organism_id:684738'))
for g in LL_MEN:
    QUERIES.append(("reviewed_1358", g, f'gene:{g} AND taxonomy_id:1358 AND reviewed:true'))
# organism_name variants that are spelled the UniProt way
for g in ["menF", "menD"]:
    QUERIES.append(("organism_name_strain_paren", g, f'gene:{g} AND organism_name:"Lactococcus lactis subsp. lactis (strain KF147)"'))
    QUERIES.append(("organism_name_species", g, f'gene:{g} AND organism_name:"Lactococcus lactis"'))
# 3. E. coli Ent
for g in ENT:
    QUERIES.append(("ecoli_k12_83333", g, f'gene:{g} AND organism_id:83333 AND reviewed:true'))
    QUERIES.append(("ecoli_t2pw_shape", g, f'(protein_name:"{g}" OR gene:"{g}") AND organism_name:"Escherichia coli"'))
# 4. proteomes for L. lactis strains named in the paper (KF147) and reference strains
QUERIES.append(("kf147_any_men", "*", 'organism_id:684738 AND (gene:men* OR protein_name:menaquinone)'))

def main(out):
    results = []
    for tag, gene, q in QUERIES:
        r = search(q)
        r["tag"], r["gene"] = tag, gene
        results.append(r)
        print(f"{tag:28s} {gene:6s} status={r.get('status')} total={r.get('total')} "
              + "; ".join(f"{h['acc']}{'*' if h['reviewed'] else ''} {','.join(x for x in h['genes'] if not x.startswith('locus:'))[:20]} {h['organism'][:45]} ({h['taxid']})" for h in r.get('hits', [])[:4]))
        time.sleep(0.25)
    json.dump(results, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

if __name__ == "__main__":
    main(sys.argv[1])
