import collections
import json
import sys

fix = json.load(open(sys.argv[1], encoding="utf-8"))
c = collections.Counter()
for key, row in fix.items():
    kind, path = key.split("|", 1)
    leaf = path.replace("\\", "/").split("/")[-1]
    root = next((r for r in ("legs3/", "legs4/", "legs4g/", "legs5g/") if r in path.replace("\\", "/")), "pp")
    if kind == "input":
        c[("input", root, leaf, row["post_audit"])] += 1
    if kind == "gemma_shape":
        c[("gemma_fix", row.get("post_audit"), "name_rejections>=1" if row.get("name_rejections", 0) >= 1 else "none")] += 1
    if kind in ("llm_patch", "gap_patch", "det_patch"):
        c[(kind, "fix_name_rejections", row.get("name_rejections", 0))] += 1
for k, n in sorted(c.items()):
    print(k, n)
