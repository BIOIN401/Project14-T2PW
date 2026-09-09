"""C-121 IMPLEMENTER census -- G5. Reproduces the orchestrator's pre-dispatch
census (``c121_census_guard.py`` / ``c121_census_crosstab.py``) against the
CANDIDATE build instead of the base tree, and against the PRODUCTION guard that
now ships in ``process_normalizer`` rather than a copy of the predicate.

Two questions, one script:

1. Does the shipped guard fire on exactly the five F-192-positive production legs
   and perturb no others?  (cross-tabulated against each leg's own COMMITTED
   ``pwml_required_field_gate_report.json``, exactly as the orchestrator did)
2. For every leg the guard does NOT fire on, would an unguarded
   ``ensure_autostates`` re-run have changed the payload?  That difference is the
   regression the guard prevents, and it must be reported, not assumed.

Read-only on the archive. Nothing is written anywhere.

    <py> c121_impl_census.py [--tree <import root>] [--repo <archive root>]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve()
DEFAULT_TREE = HERE.parents[3]
DEFAULT_REPO = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")

ap = argparse.ArgumentParser()
ap.add_argument("--tree", default=str(DEFAULT_TREE), help="checkout to IMPORT t2pw from")
ap.add_argument("--repo", default=str(DEFAULT_REPO), help="checkout holding the run archive")
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
print("ARCHIVE root  :", REPO.as_posix())

from t2pw.pipeline.process_normalizer import (  # noqa: E402
    autostate_restoration_required,
    ensure_autostates,
    restore_autostates_if_required,
)

F192_CODES = {"no_biological_states", "visible_entity_missing_location_state"}

legs = sorted(
    fm
    for fm in REPO.rglob("final_mapped.json")
    if ".claude" not in fm.parts
    and ".git" not in fm.parts
    and ".pytest_tmp_baseline" not in fm.parts
    and "papers" in fm.parts
)
print("production legs:", len(legs))

rows = []
for fm in legs:
    payload = json.loads(fm.read_text(encoding="utf-8"))
    fires = autostate_restoration_required(payload)

    # the shipped entry point must agree with the predicate, and must be a
    # BYTE no-op whenever the predicate is quiet
    guarded = deepcopy(payload)
    before = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    acted = restore_autostates_if_required(guarded)
    after_guarded = json.dumps(guarded, sort_keys=True, ensure_ascii=False)
    guarded_changes = before != after_guarded

    unguarded = deepcopy(payload)
    ensure_autostates(unguarded)
    unguarded_changes = before != json.dumps(unguarded, sort_keys=True, ensure_ascii=False)

    gate = fm.parent / "pwml_required_field_gate_report.json"
    codes: set = set()
    gate_ok = None
    if gate.exists():
        g = json.loads(gate.read_text(encoding="utf-8"))
        gate_ok = g.get("ok")
        codes = {e.get("code") for e in (g.get("errors") or []) if isinstance(e, dict)}
    rows.append(
        dict(
            leg=fm.relative_to(REPO).as_posix(),
            fires=fires,
            acted=acted,
            guarded_changes=guarded_changes,
            unguarded_changes=unguarded_changes,
            gate_ok=gate_ok,
            codes=codes,
            has_gate=gate.exists(),
        )
    )

tab: Counter = Counter()
for r in rows:
    tab[("fires" if r["fires"] else "quiet") + "/" + ("F192" if (r["codes"] & F192_CODES) else "noF192")] += 1

print("\n=== CROSS-TAB (production legs) ===")
for k in ("fires/F192", "fires/noF192", "quiet/F192", "quiet/noF192"):
    print(f"  {k:16} {tab.get(k, 0)}")

fire_rows = [r for r in rows if r["fires"]]
print(f"\n=== GUARD FIRES ({len(fire_rows)}) ===")
for r in fire_rows:
    print(f"  {r['leg']}")
    print(f"      gate_ok={r['gate_ok']} codes={sorted(c for c in r['codes'])} acted={r['acted']}")

quiet_changed = [r for r in rows if not r["fires"] and r["guarded_changes"]]
print(f"\n=== QUIET LEGS THE GUARDED PATH PERTURBED ({len(quiet_changed)}) -- MUST BE 0 ===")
for r in quiet_changed:
    print("  ", r["leg"])

unguarded_only = [r for r in rows if not r["fires"] and r["unguarded_changes"]]
print(f"\n=== UNGUARDED WOULD CHANGE, GUARD QUIET ({len(unguarded_only)}) ===")
print("    ^^ the regression the guard prevents")
for r in unguarded_only:
    print("  ", r["leg"])

fp = [r for r in rows if r["fires"] and not (r["codes"] & F192_CODES)]
fn = [r for r in rows if not r["fires"] and (r["codes"] & F192_CODES)]
print(f"\nfalse positives (fires, no F-192 code): {len(fp)}")
for r in fp:
    print(f"   {r['leg']} gate_ok={r['gate_ok']} codes={sorted(c for c in r['codes'])} gate_present={r['has_gate']}")
print(f"false negatives (F-192 code, guard quiet): {len(fn)}")
for r in fn:
    print(f"   {r['leg']} gate_ok={r['gate_ok']} codes={sorted(c for c in r['codes'])}")
print(f"legs with no committed gate report: {sum(1 for r in rows if not r['has_gate'])}")

disagree = [r for r in rows if r["fires"] != r["acted"] or r["acted"] != r["guarded_changes"]]
print(f"\npredicate/entry-point disagreements: {len(disagree)} (must be 0)")
for r in disagree:
    print("  ", r["leg"], r["fires"], r["acted"], r["guarded_changes"])

print("\nSUMMARY")
print(f"  legs={len(rows)} fires={len(fire_rows)} quiet_perturbed={len(quiet_changed)} "
      f"unguarded_would_change_quiet={len(unguarded_only)} fp={len(fp)} fn={len(fn)}")
print("CENSUS DONE")
