import json, collections, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
c = collections.Counter()
for x in rows:
    if x["by_verifier"]:
        continue
    c[(x["paper"], x["name"], str(x["biological_state"]), tuple(sorted({f"{e['provenance']}/{e['confidence']}" for e in x["enzymes"]})),
       tuple(sorted({f"{l['stage']}/{l['origin']}/{l['paper_explicit']}/{l['support']}/{l['review_required']}" for l in x["lineage"]})),
       x["locked_reaction_id"], tuple(x["keys"]) == tuple(sorted(x["keys"])) and ("biological_state" in x["keys"]))] += 1
for k, n in sorted(c.items()):
    print(n, k)
