"""Group importer messages of the 8 showcase + 6 V7b imports by (object kind, message)."""
import collections, glob, json, re, sys

files = {}
for p in glob.glob("C:/t/showcase/*/pathwhiz_import.json"):
    files["show:" + p.replace("\\", "/").split("/")[-2]] = json.load(open(p, encoding="utf-8"))
v = json.load(open("C:/t/pwml-fix/docs/vision/evidence/V7b/pathwhiz_import_results.json", encoding="utf-8"))
for k, r in v.items():
    if isinstance(r, dict):
        files["v7b:" + k] = r
tot = collections.Counter()
for name, r in sorted(files.items()):
    c = collections.Counter()
    for key, msgs in (r.get("errors") or {}).items():
        kind = re.sub(r"_\d+$", "", key)
        for m in msgs:
            c[(kind, m)] += 1
            tot[(kind, m)] += 1
    print("==", name, sum(c.values()))
    for (k, m), n in sorted(c.items()):
        print("   %3d %s: %s" % (n, k, m))
print("== TOTAL", sum(tot.values()))
for (k, m), n in tot.most_common():
    print("   %3d %s: %s" % (n, k, m))
