"""C-107: compare two corpus verdict dumps and report BOTH DIRECTIONS separately.

C-107 section 3 and REV-107 B5: "a net figure alone is a fail". This prints the
newly-REFUSED set and the newly-ADMITTED set as two independent counts, each
with quoted examples, and never prints a net.

Usage::  <python> c107_corpus_diff.py <base.json> <tip.json> [examples]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

base = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tip = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
n_ex = int(sys.argv[3]) if len(sys.argv) > 3 else 12

only_base = set(base) - set(tip)
only_tip = set(tip) - set(base)
print(f"rows at base : {len(base)}    rows at tip : {len(tip)}")
print(f"population drift: keys only at base = {len(only_base)}, only at tip = {len(only_tip)}")
if only_base or only_tip:
    print("  !! the row population changed; the two directions below cover the shared keys only")

shared = sorted(set(base) & set(tip))
newly_refused = [k for k in shared if base[k] and not tip[k]]
newly_admitted = [k for k in shared if not base[k] and tip[k]]

print()
print(f"BASE accepted : {sum(1 for k in shared if base[k])} of {len(shared)}")
print(f"TIP  accepted : {sum(1 for k in shared if tip[k])} of {len(shared)}")
print()
print("=" * 90)
print(f"DIRECTION 1 -- NEWLY REFUSED (accepted at base, refused at tip): {len(newly_refused)}")
print("=" * 90)
for k in newly_refused[:n_ex]:
    cont, bucket, name, role, ev = k.split("|", 4)
    print(f"  [{cont}/{bucket} role={role!r}] {name!r}")
    print(f"      {ev[:300]!r}{' ...(%d chars)' % len(ev) if len(ev) > 300 else ''}")
if len(newly_refused) > n_ex:
    print(f"  ... and {len(newly_refused) - n_ex} more")

print()
print("=" * 90)
print(f"DIRECTION 2 -- NEWLY ADMITTED (refused at base, accepted at tip): {len(newly_admitted)}")
print("=" * 90)
for k in newly_admitted[:n_ex]:
    cont, bucket, name, role, ev = k.split("|", 4)
    print(f"  [{cont}/{bucket} role={role!r}] {name!r}")
    print(f"      {ev[:300]!r}{' ...(%d chars)' % len(ev) if len(ev) > 300 else ''}")
if len(newly_admitted) > n_ex:
    print(f"  ... and {len(newly_admitted) - n_ex} more")

print()
print(f"SUMMARY  newly_refused={len(newly_refused)}  newly_admitted={len(newly_admitted)}"
      f"  (reported separately, by C-107 section 3 / REV-107 B5)")
