"""C-121 pre-dispatch cross-tabulation -- READ ONLY.

Does the proposed guard fire on EXACTLY the archived legs whose COMMITTED
required-field gate report failed on an F-192 code, and on no others?
"""
import json, sys
from collections import Counter
from pathlib import Path

REPO = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
sys.path.insert(0, str(REPO / "src"))
import t2pw
print("RESOLVED t2pw:", t2pw.__file__)

F192_CODES = {"no_biological_states", "visible_entity_missing_location_state"}
LOC_BUCKETS = ("compound_locations", "protein_locations",
               "nucleic_acid_locations", "element_collection_locations")

legs = []
for fm in REPO.rglob("final_mapped.json"):
    if ".claude" in fm.parts or ".git" in fm.parts or ".pytest_tmp_baseline" in fm.parts:
        continue
    if "papers" not in fm.parts:
        continue
    legs.append(fm)
legs.sort()
print("production legs:", len(legs))

rows = []
for fm in legs:
    p = json.loads(fm.read_text(encoding="utf-8"))
    states = [r.get("name") for r in (p.get("biological_states") or []) if isinstance(r, dict)]
    el = p.get("element_locations") or {}
    locrows = [r for b in LOC_BUCKETS for r in (el.get(b) or []) if isinstance(r, dict)]
    missing = [r for r in locrows if not str(r.get("biological_state") or "").strip()]
    pr = p.get("processes") or {}
    content = bool((pr.get("reactions") or []) or (pr.get("transports") or [])
                   or (pr.get("interactions") or []))
    fires = content and (len(states) == 0 or bool(missing))

    gate = fm.parent / "pwml_required_field_gate_report.json"
    codes = set()
    gate_ok = None
    if gate.exists():
        g = json.loads(gate.read_text(encoding="utf-8"))
        gate_ok = g.get("ok")
        codes = {e.get("code") for e in (g.get("errors") or []) if isinstance(e, dict)}
    rows.append((fm.relative_to(REPO).as_posix(), fires, gate_ok, codes, gate.exists()))

def cell(fires, f192):
    return ("fires" if fires else "quiet") + "/" + ("F192" if f192 else "noF192")

tab = Counter()
for rel, fires, ok, codes, has in rows:
    f192 = bool(codes & F192_CODES)
    tab[cell(fires, f192)] += 1

print("\n=== CROSS-TAB (production legs with a committed gate report) ===")
for k in ("fires/F192", "fires/noF192", "quiet/F192", "quiet/noF192"):
    print(f"  {k:16} {tab.get(k,0)}")

print("\n--- fires/F192 : the legs the fix is FOR ---")
for rel, fires, ok, codes, has in rows:
    if fires and (codes & F192_CODES):
        print(f"  {rel}\n      gate_ok={ok} codes={sorted(codes)}")

print("\n--- fires/noF192 : FALSE POSITIVES (guard would perturb a leg not failing on F-192) ---")
for rel, fires, ok, codes, has in rows:
    if fires and not (codes & F192_CODES):
        print(f"  {rel}\n      gate_ok={ok} codes={sorted(codes)} gate_report_present={has}")

print("\n--- quiet/F192 : FALSE NEGATIVES (F-192 failure the guard would NOT fix) ---")
for rel, fires, ok, codes, has in rows:
    if not fires and (codes & F192_CODES):
        print(f"  {rel}\n      gate_ok={ok} codes={sorted(codes)}")

missing_gate = [r for r in rows if not r[4]]
print(f"\nlegs with NO committed gate report: {len(missing_gate)}")
for rel, fires, ok, codes, has in missing_gate:
    print(f"  {rel}  guard_fires={fires}")
print("\nCROSSTAB DONE")
