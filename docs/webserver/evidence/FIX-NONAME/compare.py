"""Diff two replay outputs. usage: python -I compare.py out_base.json out_fix.json"""

import collections
import json
import sys

base = json.load(open(sys.argv[1], encoding="utf-8"))
fix = json.load(open(sys.argv[2], encoding="utf-8"))
assert set(base) == set(fix), "case sets differ"

tally = collections.Counter()
diffs = []
for key in sorted(base):
    kind = key.split("|", 1)[0]
    b, f = base[key], fix[key]
    tally[(kind, "total")] += 1
    if "exception" in b or "exception" in f:
        tally[(kind, "exception")] += 1
        if b != f:
            diffs.append((key, "exception_differs", b, f))
        continue
    if kind == "input":
        continue
    if b.get("matches_recorded_output"):
        tally[(kind, "base_reproduces_recorded_output")] += 1
    if b.get("post_audit", "").startswith("fail:"):
        tally[(kind, "base_post_audit_fail")] += 1
    if f.get("post_audit", "").startswith("fail:"):
        tally[(kind, "fix_post_audit_fail")] += 1
    if b == f:
        tally[(kind, "identical")] += 1
        continue
    allowed = b["post_audit"].startswith("fail:") and "entity_missing_name" in b["post_audit"] and f["post_audit"] == "ok"
    tally[(kind, "differs_allowed" if allowed else "differs_NOT_ALLOWED")] += 1
    diffs.append((key, "allowed" if allowed else "NOT_ALLOWED", b, f))

for (kind, what), n in sorted(tally.items()):
    print(f"{kind:12s} {what:34s} {n}")
print("not-allowed diffs:", sum(1 for d in diffs if d[1] != "allowed"))
for key, why, b, f in diffs:
    if why != "allowed":
        print(why, key, b, f)
