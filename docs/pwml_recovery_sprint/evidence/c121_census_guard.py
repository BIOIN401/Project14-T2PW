"""C-121 pre-dispatch census -- READ ONLY, base tree.

For every archived leg, answer one question: WOULD A GUARDED POST-SWEEP
RE-ESTABLISHMENT FIRE? The guard under test:

    fire  <=>  exportable content survives
               AND ( zero biological_states
                     OR some surviving visible element-location row carries no
                        biological_state )

Then, for every leg where it does NOT fire, confirm a bare re-run of
ensure_autostates WOULD have changed the payload -- i.e. measure exactly how
much damage an UNGUARDED fix would do. That difference is the regression surface.
"""
import json, sys
from copy import deepcopy
from pathlib import Path

REPO = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
sys.path.insert(0, str(REPO / "src"))
import t2pw
print("RESOLVED t2pw:", t2pw.__file__)
from t2pw.pipeline.process_normalizer import ensure_autostates

LOC_BUCKETS = ("compound_locations", "protein_locations",
               "nucleic_acid_locations", "element_collection_locations")

legs = []
for fm in REPO.rglob("final_mapped.json"):
    if ".claude" in fm.parts or ".git" in fm.parts:
        continue
    parts = fm.parts
    if "papers" not in parts:
        continue
    i = parts.index("papers")
    if len(parts) < i + 3:
        continue
    legs.append(fm)
legs.sort()
print(f"legs discovered: {len(legs)}")

def state_names(p):
    return [r.get("name") for r in (p.get("biological_states") or []) if isinstance(r, dict)]

def loc_rows(p):
    el = p.get("element_locations") or {}
    out = []
    for b in LOC_BUCKETS:
        for r in (el.get(b) or []):
            if isinstance(r, dict):
                out.append((b, r))
    return out

def has_content(p):
    pr = p.get("processes") or {}
    return bool((pr.get("reactions") or []) or (pr.get("transports") or [])
                or (pr.get("interactions") or []))

fire = []
unguarded_only = []
inert = []
unreadable = []

for fm in legs:
    try:
        p = json.loads(fm.read_text(encoding="utf-8"))
    except Exception as exc:
        unreadable.append((fm, repr(exc)[:80])); continue
    if not isinstance(p, dict):
        unreadable.append((fm, "not an object")); continue

    states = state_names(p)
    rows = loc_rows(p)
    rows_missing_state = [(b, r) for b, r in rows
                          if not str(r.get("biological_state") or "").strip()]
    content = has_content(p)
    guard_fires = content and (len(states) == 0 or bool(rows_missing_state))

    before = json.dumps(p, sort_keys=True, ensure_ascii=False)
    after_p = deepcopy(p)
    ensure_autostates(after_p)
    after = json.dumps(after_p, sort_keys=True, ensure_ascii=False)
    unguarded_changes = (before != after)

    rel = fm.relative_to(REPO).as_posix()
    rec = dict(leg=rel, states=states, n_rows=len(rows),
               n_rows_missing_state=len(rows_missing_state), content=content,
               unguarded_changes=unguarded_changes)
    if guard_fires:
        fire.append(rec)
    elif unguarded_changes:
        unguarded_only.append(rec)
    else:
        inert.append(rec)

print(f"\nunreadable: {len(unreadable)}")
for fm, why in unreadable: print("   ", fm.relative_to(REPO).as_posix(), why)

print(f"\n=== GUARD FIRES ({len(fire)}) -- these are the F-192-positive legs ===")
for r in fire:
    print(f"  {r['leg']}")
    print(f"      states={r['states']} rows={r['n_rows']} rows_missing_state={r['n_rows_missing_state']}")

print(f"\n=== UNGUARDED WOULD CHANGE, GUARD DOES NOT FIRE ({len(unguarded_only)}) ===")
print("    ^^ this is EXACTLY the regression an unguarded ensure_autostates re-run causes")
for r in unguarded_only:
    print(f"  {r['leg']}  states={r['states']}")

print(f"\n=== INERT EITHER WAY ({len(inert)}) ===")
print("\nCENSUS DONE")
