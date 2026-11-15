import collections, json, re, sys
for a in sys.argv[1:]:
    name, path = a.split("=", 1)
    c = collections.Counter(); files = collections.defaultdict(set)
    for i, l in enumerate(open(path, encoding="utf-8")):
        r = json.loads(l)
        for k, v in r["errors"].items():
            if k.startswith(("transport", "interaction", "bound", "protein", "compound_loc")):
                kk = re.sub(r"_\d+$", "", k)
                for m in v:
                    c[(kk, m)] += 1; files[(kk, m)].add(i)
    print("==", name)
    for (k, m), n in c.most_common(): print(f"  {n:4d} msgs {len(files[(k,m)]):3d} files  {k}: {m}")
