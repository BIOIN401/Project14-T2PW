"""ORCH-797c: which c011 leg fields move between the BEFORE fixture and the tip's seam document.

Usage: python orch797c_c011probe.py   (cwd = the worktree; PYTHONPATH = its src)
Builds ``_document(None)`` exactly as ``test_c011_freeze_seam_golden_equivalence`` does and prints, per leg, every
field whose value differs from the fixture (C-052's two added keys ignored), with both values for small fields.
"""

import json
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[3] / "tests"
sys.path.insert(0, str(TESTS))
import test_c011_freeze_seam_golden_equivalence as C  # noqa: E402

after = C._document(None)
before = json.loads(C._fixture_bytes())
moved = 0
for name in sorted(before["legs"]):
    b, a = before["legs"][name], after["legs"][name]
    diffs = [f for f in sorted(set(b) | set(a)) if f not in C._C052_KEYS and b.get(f) != a.get(f)]
    if not diffs:
        continue
    moved += 1
    print(f"LEG {name}: {diffs}")
    for f in diffs:
        bv, av = b.get(f), a.get(f)
        if isinstance(bv, dict) and isinstance(av, dict):
            for k in sorted(set(bv) | set(av)):
                if bv.get(k) != av.get(k):
                    print(f"   {f}.{k}: {json.dumps(bv.get(k))[:200]} -> {json.dumps(av.get(k))[:200]}")
        else:
            print(f"   {f}: {json.dumps(bv)[:200]} -> {json.dumps(av)[:200]}")
print("legs with any moved field:", moved)
