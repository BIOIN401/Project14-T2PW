import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"runs scanned {r['scanned']}, with an out-of-scope Stage-1 row {r['with_out_of_scope']}, drops {r['drops_neutral']}, dropped rows matching gold {r['gold_matching_drops']}, gold lost {r['gold_lost']}\n")
print("| paper | run | Stage-2 row dropped | Stage-1 row (index) | gold matched by dropped row | final_mapped gold lost | precision before -> after |")
print("|---|---|---|---|---|---|---|")
for x in r["runs"]:
    run = x["run"].replace("\\", "/").replace("C:/t/acc-runs/", "").replace("C:/t/t2pw-server-data/work/", "server ").split("/papers/")[0]
    for d in x["drops"]:
        pb, pa = x.get("precision_before"), x.get("precision_after")
        print(f"| {x['paper']} | {run} | {d['stage2']} | {d['stage1']} ({d['stage1_index']}) | {x.get('dropped_rows_matching_gold', 'n/a')} | {x.get('gold_lost', 'no final_mapped')} | {pb} -> {pa} |")
