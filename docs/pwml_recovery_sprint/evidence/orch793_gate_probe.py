import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, sys.argv[1] + "/src")
from t2pw.mapping import map_ids
W = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/"
p784 = json.load(open(W + "orch784run/data/id_mapping_cache.json", encoding="utf-8"))["proteins"]
p785 = json.load(open(W + "orch785run/data/id_mapping_cache.json", encoding="utf-8"))["proteins"]
for name, org, entry in (("Fur", "Escherichia coli", p784["api-fallback-v2::fur::tax:562>562"]),
                         ("c-Jun", "Homo sapiens", p784["api-fallback-v2::cjun::tax:9606>9606"]),
                         ("PIF1", "Arabidopsis thaliana", p785["api-v9::pif1::tax:3702>3702::::{}::[]"]),
                         ("TAK1", "Homo sapiens", p785["api-v9::tak1::tax:9606>9606::::{}::[]"]),
                         ("Kir2.1", "Mus musculus", p785["api-v9::kir21::tax:10090>10090::::{}::[]"])):
    out = map_ids._apply_name_plausibility_gate(json.loads(json.dumps(entry)), name, kind="protein", organism=org)
    v = out.get("identity_verdict") or {}
    print(name, out.get("status"), out.get("mapped_ids"), v.get("reason"), json.dumps(v.get("checks")),
          "collapsed=", v.get("collapsed_duplicate_accessions"), "outranked=", v.get("outranked_by_primary_gene_name"),
          "competing=", v.get("competing_accessions"))
