"""ORCH-769c: which corpus legs changed in ANY recorded field between two harness runs, not only
pass/fail (``compare`` checks pass rate, broken legs and gold recall; this checks everything else).

Usage: python corpus_leg_diff.py <base.json> <tip.json> <leg id to print in full>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

base, tip = (json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[1:3])
b = {r["id"]: r for r in base["legs"]}
t = {r["id"]: r for r in tip["legs"]}
assert set(b) == set(t)
changed = {}
for i in sorted(b):
    rb, rt = b[i].get("replay") or {}, t[i].get("replay") or {}
    keys = sorted(k for k in set(rb) | set(rt) if rb.get(k) != rt.get(k))
    if keys or {k: v for k, v in b[i].items() if k != "replay"} != {k: v for k, v in t[i].items() if k != "replay"}:
        changed[i] = keys
print(f"legs: {len(b)}; legs with any changed field: {len(changed)}")
for i, keys in changed.items():
    print(f"  {i}: {keys}")
leg = sys.argv[3]
for side, rec in (("base", b[leg]), ("tip", t[leg])):
    r = rec.get("replay") or {}
    print(f"\n{side} {leg}:")
    for k in ("failure_stage", "pwml_written", "refinement_error", "export_if_refinement_opened", "export", "pwml_sha256",
              "pwml_matches_leg", "quarantine", "final_gate", "network_refused"):
        print(f"  {k}: {json.dumps(r.get(k), sort_keys=True, default=str)[:400]}")
    print(f"  files: {sorted((r.get('files') or {}))}")
