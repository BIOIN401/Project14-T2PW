"""Second read-only UniProt probe: taxonomy facts and name-based L. lactis lookups."""
import json, sys, time, urllib.parse, urllib.request
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from orch761_uniprot_probe import search  # same helper, same fields

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "orch761-readonly-probe"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.headers.get("X-Total-Results"), json.load(r)

out = {}
for tid in ["684738", "1358", "1359", "272623", "416870", "1360", "562", "83333"]:
    try:
        _, t = get(f"https://rest.uniprot.org/taxonomy/{tid}?format=json")
        out[f"taxonomy_{tid}"] = {k: t.get(k) for k in ("taxonId", "scientificName", "rank", "parent", "statistics", "active", "mnemonic")}
        out[f"taxonomy_{tid}"]["parent"] = (t.get("parent") or {}).get("scientificName")
    except Exception as e:
        out[f"taxonomy_{tid}"] = {"error": str(e)}
    print(tid, json.dumps(out[f"taxonomy_{tid}"], ensure_ascii=False))
    time.sleep(0.2)

Q = [
 ("all_entries_684738", 'organism_id:684738'),
 ("all_entries_1358_lineage", 'taxonomy_id:1358'),
 ("all_entries_1359_lineage", 'taxonomy_id:1359'),
 ("menA_name_1358", 'taxonomy_id:1358 AND protein_name:"1,4-dihydroxy-2-naphthoate octaprenyltransferase"'),
 ("menA_name_1358b", 'taxonomy_id:1358 AND protein_name:"naphthoate" AND protein_name:"prenyltransferase"'),
 ("menI_name_1358", 'taxonomy_id:1358 AND protein_name:"1,4-dihydroxy-2-naphthoyl-CoA hydrolase"'),
 ("menI_thioesterase_1358", 'taxonomy_id:1358 AND protein_name:thioesterase'),
 ("menG_ubiE_name_1358", 'taxonomy_id:1358 AND protein_name:"demethylmenaquinone methyltransferase"'),
 ("menH_name_1358", 'taxonomy_id:1358 AND protein_name:"2-succinyl-6-hydroxy-2,4-cyclohexadiene-1-carboxylate synthase"'),
 ("menC_name_1358", 'taxonomy_id:1358 AND protein_name:"o-succinylbenzoate synthase"'),
 ("menB_name_1358", 'taxonomy_id:1358 AND protein_name:"1,4-dihydroxy-2-naphthoyl-CoA synthase"'),
 ("menE_name_1358", 'taxonomy_id:1358 AND protein_name:"2-succinylbenzoate--CoA ligase"'),
 ("menF_name_1358", 'taxonomy_id:1358 AND protein_name:"isochorismate synthase"'),
 ("men_genes_IL1403", 'organism_id:272623 AND (gene:men* OR gene:ubiE OR gene:preA)'),
 ("men_genes_MG1363", 'organism_id:416870 AND (gene:men* OR gene:ubiE OR gene:preA)'),
 ("menA_any_lactococcus", 'gene:menA AND taxonomy_id:1357'),
 ("menI_any_lactococcus", 'gene:menI AND taxonomy_id:1357'),
]
res = []
for tag, q in Q:
    r = search(q, size=25); r["tag"] = tag; res.append(r)
    print(f"{tag:26s} total={r.get('total')} " + "; ".join(
        f"{h['acc']}{'*' if h['reviewed'] else ''} {','.join(h['genes'])[:40]} [{(h['name'] or '')[:45]}] ({h['taxid']})" for h in r.get('hits', [])[:12]))
    time.sleep(0.25)
out["queries"] = res
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
