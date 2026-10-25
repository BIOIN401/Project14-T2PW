"""ORCH-772 dev check: the unclassified omission reasons and one leg's full omission list, from the tip corpus JSON."""
import json
import sys
from pathlib import Path

doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tally = {}
for row in doc["legs"]:
    cov = (row["replay"] or {}).get("pwml_coverage") or {}
    for n, ki, group, reason in cov.get("omitted_by_reason") or []:
        if ki == "unclassified" and group != "no_pwml":
            tally[(group, reason)] = tally.get((group, reason), 0) + n
        if row["id"] == sys.argv[2]:
            print("LEG", row["id"], n, ki, group, reason)
    if row["id"] == sys.argv[2]:
        print("LEG cov", {k: cov[k] for k in ("drawn", "total", "by_kind", "by_design_withheld", "not_counted")})
for (group, reason), n in sorted(tally.items(), key=lambda kv: -kv[1]):
    print(n, group, reason)
pw = [r for r in doc["legs"] if (r["replay"] or {}).get("pwml_written")]
print("PWML legs:", len(pw), "with drawn == 0:", [r["id"] for r in pw if r["replay"]["pwml_coverage"]["drawn"] == 0])
print("not counted total:", sum(((r["replay"] or {}).get("pwml_coverage") or {}).get("not_counted", 0) for r in doc["legs"]),
      "merged duplicates:", sum(((r["replay"] or {}).get("pwml_coverage") or {}).get("merged_duplicates", 0) for r in doc["legs"]))
