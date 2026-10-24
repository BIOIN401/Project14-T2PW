"""ORCH-769b: every leg whose replay record differs between two ORCH-768 corpus runs, field by field.

    python corpus_delta.py <base.json> <tip.json>

Stdlib only. Prints the legs whose replay record changed in ANY field (outcome or not), then, for each,
the differing top-level replay keys and the quarantine/final-gate/export/release detail at base and tip.
"""
import json
import sys


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


base, tip = load(sys.argv[1]), load(sys.argv[2])
b = {r["id"]: r for r in base["legs"]}
t = {r["id"]: r for r in tip["legs"]}
assert set(b) == set(t)
VOLATILE = {"files"}  # per-file sha256 of what the replay wrote; reported separately


def strip(r):
    return {k: v for k, v in (r or {}).items() if k not in VOLATILE}


changed = []
files_only = []
for i in sorted(b):
    rb, rt = b[i].get("replay"), t[i].get("replay")
    if rb == rt:
        continue
    if strip(rb) == strip(rt):
        files_only.append(i)
        continue
    changed.append(i)

print(f"legs: {len(b)}; replay record identical: {len(b) - len(changed) - len(files_only)}; "
      f"only written-file hashes differ: {len(files_only)}; verdict fields differ: {len(changed)}")
for i in files_only:
    print("  files only:", i)
for i in changed:
    rb, rt = b[i]["replay"] or {}, t[i]["replay"] or {}
    print(f"\n== {i}\n  class: {b[i]['class']}  ->  {t[i]['class']}")
    keys = sorted(k for k in set(rb) | set(rt) if k not in VOLATILE and rb.get(k) != rt.get(k))
    print("  differing keys:", keys)
    for k in keys:
        print(f"  [{k}] base: {json.dumps(rb.get(k), sort_keys=True, default=str)[:900]}")
        print(f"  [{k}] tip : {json.dumps(rt.get(k), sort_keys=True, default=str)[:900]}")
