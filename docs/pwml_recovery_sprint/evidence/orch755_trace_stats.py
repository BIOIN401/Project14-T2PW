import json, glob, collections, sys
tot = collections.defaultdict(lambda: [0, 0])
for t in glob.glob(sys.argv[1]):
    for line in open(t, encoding="utf-8"):
        r = json.loads(line)
        if r.get("kind") == "model_attempt":
            b = tot[r.get("stage")]; b[0] += 1
            b[1] += r.get("finish_reason") == "length" and not r.get("content_chars")
print(sys.argv[1], "|", " | ".join(f"{s}: {e}/{n} empty" for s, (n, e) in sorted(tot.items(), key=lambda x: -x[1][0])))
