"""C-121 correction round -- REV-121 finding 2, measured rather than accepted.

The guard scans FOUR ``element_locations`` buckets (the gate's own table);
``ensure_autostates`` assigns a state to only TWO of them. So a payload whose only
unassigned visible row is a nucleic-acid or element-collection row would fire the
guard, be mutated, and still fail the gate.

This measures whether the archive exercises that gap, so the asymmetry can be
documented as a latent limitation with a number behind it rather than an assertion.
Read-only. Nothing is written.

    <py> c121_bucket_gap_census.py [--tree <import root>] [--repo <archive root>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
DEFAULT_TREE = HERE.parents[3]
DEFAULT_REPO = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")

ap = argparse.ArgumentParser()
ap.add_argument("--tree", default=str(DEFAULT_TREE))
ap.add_argument("--repo", default=str(DEFAULT_REPO))
args = ap.parse_args()

TREE = Path(args.tree).resolve()
REPO = Path(args.repo).resolve()
sys.path.insert(0, str(TREE / "src"))
import t2pw  # noqa: E402

print("EXPECTED tree :", TREE.as_posix())
print("RESOLVED t2pw :", Path(t2pw.__file__).as_posix())
if Path(t2pw.__file__).resolve().parent != (TREE / "src" / "t2pw").resolve():
    print("MEASUREMENT_TREE_REFUSED")
    raise SystemExit(98)

from t2pw.pipeline.process_normalizer import _VISIBLE_LOCATION_BUCKETS  # noqa: E402

#: The buckets ``ensure_autostates`` actually iterates when it assigns a state.
#: Read from that function's own source so this census cannot drift from it.
ASSIGNED = ("compound_locations", "protein_locations")
UNASSIGNED = tuple(b for b in _VISIBLE_LOCATION_BUCKETS if b not in ASSIGNED)
print("scanned by the guard   :", list(_VISIBLE_LOCATION_BUCKETS))
print("assigned by ensure_autostates:", list(ASSIGNED))
print("the gap                :", list(UNASSIGNED))

legs = sorted(
    fm
    for fm in REPO.rglob("final_mapped.json")
    if ".claude" not in fm.parts
    and ".git" not in fm.parts
    and ".pytest_tmp_baseline" not in fm.parts
    and "papers" in fm.parts
)
print("production legs:", len(legs))

present = {b: [] for b in UNASSIGNED}
missing_state = {b: [] for b in UNASSIGNED}
for fm in legs:
    payload = json.loads(fm.read_text(encoding="utf-8"))
    el = payload.get("element_locations") or {}
    rel = fm.relative_to(REPO).as_posix()
    for bucket in UNASSIGNED:
        rows = [r for r in (el.get(bucket) or []) if isinstance(r, dict)]
        if rows:
            present[bucket].append((rel, len(rows)))
        for row in rows:
            if not str(row.get("biological_state") or "").strip():
                missing_state[bucket].append(rel)

for bucket in UNASSIGNED:
    print("")
    print(bucket + ": legs carrying rows =", len(present[bucket]))
    for rel, n in present[bucket]:
        print("    " + rel + "  rows=" + str(n))
    print(bucket + ": legs with such a row MISSING a state =",
          len(set(missing_state[bucket])))
    for rel in sorted(set(missing_state[bucket])):
        print("    " + rel)

# Scope reconciliation. REV-121 / ORCH-737 reported 3 and 7; this census reports 4
# and 10 over ALL 154 production legs. The difference is scope, so both slices are
# printed and the docstring can name the one it means.
print("")
print("SCOPE BREAKDOWN")
for bucket in UNASSIGNED:
    strict = [rel for rel, _ in present[bucket] if "/strict/" in rel]
    research = [rel for rel, _ in present[bucket] if "/research/" in rel]
    with_gate = [
        rel for rel, _ in present[bucket]
        if (REPO / rel).parent.joinpath("pwml_required_field_gate_report.json").exists()
    ]
    print("  " + bucket + ": all=" + str(len(present[bucket]))
          + " strict=" + str(len(strict))
          + " research=" + str(len(research))
          + " with_committed_gate_report=" + str(len(with_gate)))

total_gap = sum(len(set(missing_state[b])) for b in UNASSIGNED)
print("")
print("SUMMARY")
for bucket in UNASSIGNED:
    print("  " + bucket + ": present_on=" + str(len(present[bucket]))
          + " missing_state_on=" + str(len(set(missing_state[bucket]))))
print("  legs that would exercise the gap:", total_gap, "(0 = latent, not live)")
print("BUCKET GAP CENSUS DONE")
