"""REV-F150 probe B -- exhaustive deep-diff of two acceptance reports.

INSTRUMENT-SENSITIVITY MEASUREMENT, not a re-score. Both reports were produced
from the SAME committed artifacts under
``runs_verify/2026-08-28_1816`` (T-107's run tree, named explicitly). Nothing was
re-run; only the gold instrument differs:

    A : gold blob aee8cb4f1da3d417f36206407867585622b741c0  (pre-edit)
    B : gold blob 36f4b7b690b577f72882c3045ca6728d1ec8d9d1  (post-edit)

The point of this probe is the NEGATIVE result: every leaf that did NOT move.
"A mover you did not predict is a finding, not a footnote", so the whole report
is walked to leaves rather than spot-checked at the headline numbers.

Usage:
    <py> revf150_probeB_report_diff.py <report-A.json> <report-B.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

A = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
B = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

#: Keys whose value is the gold file's own identity/date and which MUST differ
#: (or may differ) purely because a different gold file was loaded. Recorded and
#: reported, never silently skipped.
EXPECTED_INSTRUMENT_KEYS = {"gold_path", "gold_sha256", "gold_version", "generated_at"}


def walk(node, path="") -> dict:
    """Flatten to leaf paths. Lists are indexed, so a reordering shows up."""
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(walk(v, f"{path}/{k}"))
    elif isinstance(node, list):
        out[f"{path}#len"] = len(node)
        for i, v in enumerate(node):
            out.update(walk(v, f"{path}/{i}"))
    else:
        out[path] = node
    return out


fa, fb = walk(A), walk(B)
keys = sorted(set(fa) | set(fb))

added, removed, changed, instrument = [], [], [], []
for k in keys:
    if k not in fa:
        added.append((k, fb[k]))
    elif k not in fb:
        removed.append((k, fa[k]))
    elif fa[k] != fb[k]:
        (instrument if any(k.endswith("/" + e) or k.endswith(e)
                           for e in EXPECTED_INSTRUMENT_KEYS) else changed).append(
            (k, fa[k], fb[k]))

print("=" * 78)
print("REV-F150 probe B -- INSTRUMENT-SENSITIVITY MEASUREMENT")
print("  same artifacts: runs_verify/2026-08-28_1816   (T-107's run tree)")
print("  A = pre-edit  gold aee8cb4f1da3d417f36206407867585622b741c0")
print("  B = post-edit gold 36f4b7b690b577f72882c3045ca6728d1ec8d9d1")
print("  NO LEG WAS RE-RUN. T-107's verdict is untouched and is not restated here.")
print("=" * 78)
print(f"\nleaf paths compared : {len(keys)}")
print(f"  identical         : {len(keys) - len(added) - len(removed) - len(changed) - len(instrument)}")
print(f"  changed           : {len(changed)}")
print(f"  added in B        : {len(added)}")
print(f"  removed in B      : {len(removed)}")
print(f"  instrument-identity keys (gold path/hash/date) : {len(instrument)}")

print("\n--- instrument-identity differences (expected: a different gold file) ---")
for k, x, y in instrument:
    print(f"  {k}\n      A={x!r}\n      B={y!r}")

print("\n--- CHANGED leaves ---")
for k, x, y in changed:
    print(f"  {k}\n      A={x!r}\n      B={y!r}")
if not changed:
    print("  (none)")

print("\n--- ADDED leaves (present only under post-edit gold) ---")
for k, v in added:
    print(f"  {k} = {v!r}")
if not added:
    print("  (none)")

print("\n--- REMOVED leaves (present only under pre-edit gold) ---")
for k, v in removed:
    print(f"  {k} = {v!r}")
if not removed:
    print("  (none)")

# -- which PAPERS and which ENTITY TYPES were touched at all? ----------------
print("\n--- blast radius: every paper id appearing in any diff line ---")
papers = set()
buckets = set()
for k, *_ in added + removed + changed:
    for token in k.split("/"):
        if token.startswith("PMC"):
            papers.add(token)
    for b in ("compounds", "proteins", "complexes", "reactions", "entities"):
        if f"/{b}/" in k or k.endswith(f"/{b}"):
            buckets.add(b)
for _k, x, y in changed:
    for val in (x, y):
        if isinstance(val, str) and "PMC" in val:
            for tok in val.replace(":", " ").replace("/", " ").split():
                if tok.startswith("PMC"):
                    papers.add(tok)
for _k, v in added:
    if isinstance(v, str) and "PMC" in v:
        for tok in v.replace(":", " ").replace("/", " ").split():
            if tok.startswith("PMC"):
                papers.add(tok)
print(f"  papers touched  : {sorted(papers) or '(none)'}")
print(f"  buckets touched : {sorted(buckets) or '(none)'}")

# -- the headline acceptance numbers, side by side --------------------------
print("\n--- headline numbers, A vs B ---")
INTERESTING = ("false_real", "priority", "coverage", "accepted", "raw",
               "placeholder", "withheld", "unsupported", "referential",
               "release", "disposition", "status", "ok")
seen = set()
for k in keys:
    low = k.lower()
    if not any(t in low for t in INTERESTING):
        continue
    va, vb = fa.get(k, "<absent>"), fb.get(k, "<absent>")
    if isinstance(va, (dict, list)) or isinstance(vb, (dict, list)):
        continue
    if va == vb and not isinstance(va, (int, float)):
        continue
    if k in seen:
        continue
    seen.add(k)
    flag = "   <-- MOVED" if va != vb else ""
    print(f"  {k}\n      A={va!r}  B={vb!r}{flag}")

print("\n=== probe B complete ===")
