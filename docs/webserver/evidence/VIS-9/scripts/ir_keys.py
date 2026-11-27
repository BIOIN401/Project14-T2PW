import json, sys, collections
ir = json.load(open(sys.argv[1], encoding="utf-8"))
comps = ir["entities"]["compounds"]
keys = collections.Counter()
for c in comps:
    keys.update(c.keys())
print(len(comps), dict(keys))
for c in comps[: int(sys.argv[2]) if len(sys.argv) > 2 else 3]:
    d = {k: v for k, v in c.items() if k not in ("description",)}
    print(json.dumps(d, ensure_ascii=False)[:3000])
    print("---")
