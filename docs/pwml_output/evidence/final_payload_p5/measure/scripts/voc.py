import json, collections
for n, p in (("a", "m03-out"), ("b", "m04-out")):
    c = collections.Counter(); f = set()
    for l in open("C:/t/p5scratch/measure/import/%s/results.jsonl" % p, encoding="utf-8"):
        r = json.loads(l)
        for k, v in r["errors"].items():
            b = k.rsplit("_", 1)[0] if k.rsplit("_", 1)[-1].isdigit() else k
            if b in ("subcellular_location", "cell_type", "tissue"): c[b] += len(v); f.add(r["file"].split("\\")[-1])
    print(n, dict(c), sorted(f))
