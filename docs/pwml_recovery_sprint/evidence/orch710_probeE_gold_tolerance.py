"""Probe E — PMC12444477: does the tolerance flag contradict its own rationale?

The rationale scopes the tolerance to seven named hard-to-resolve proteins and
says verbatim it "does NOT extend to the nine core Raetz enzymes".  The flag is a
single boolean consumed at bench/semantic.py:1417 as
`elif not case.unknown_backed_proteins_acceptable:` -- there is no per-entity
scoping anywhere.  This probe asks: do the rows the flag actually excuses
INCLUDE the ones the rationale says must not be excused?
"""
import json, os, io, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

gold = json.load(io.open("src/t2pw/bench/gold/pinned_v1.json", encoding="utf-8"))["cases"]
c = next(x for x in gold if x.get("paper_id") == "PMC12444477")
pinned = json.load(io.open(os.environ["PINNED_JSON"], encoding="utf-8"))["placeholders"]
rows = [p for p in pinned if p["leg"].startswith("PMC12444477")]

print("=" * 78); print("E1. THE FLAG AND THE RATIONALE, verbatim")
print("=" * 78)
print(f"  unknown_backed_proteins_acceptable = {c.get('unknown_backed_proteins_acceptable')!r}")
print(f"  unknown_backed_rationale:\n      {c.get('unknown_backed_rationale')}")

print(); print("=" * 78); print("E2. WHAT THE RATIONALE SCOPES IN vs OUT")
print("=" * 78)
tolerated = ["LapA", "YciS", "LapB", "YciM", "Ght", "LabP", "LpxG", "YhcB", "lipoprotein"]
print(f"  rationale SCOPES IN  (tolerance applies) : {tolerated}")
print(f"  rationale SCOPES OUT (must NOT apply)    : the nine core Raetz enzymes")
print(f"  gold expected_enzymes ({len(c.get('expected_enzymes') or [])}):")
for e in c.get("expected_enzymes") or []:
    nm = e.get("name") if isinstance(e, dict) else e
    print(f"      {nm}")

print(); print("=" * 78); print("E3. THE ROWS THE FLAG ACTUALLY EXCUSES on this case")
print("=" * 78)
exp_names = {str(e.get("name") if isinstance(e, dict) else e) for e in (c.get("expected_enzymes") or [])}
print(f"  pinned placeholder rows on PMC12444477/strict: {len(rows)}")
overlap = []
for r in rows:
    nm = str(r["name"])
    is_core = nm in exp_names
    is_tol = any(t.lower() in nm.lower() for t in tolerated)
    if is_core: overlap.append(nm)
    print(f"      {nm:34s} sentinel={str(r['sentinel']):5s} "
          f"IN_expected_enzymes={str(is_core):5s} matches_rationale_tolerated={is_tol}")
print()
print(f"  *** rows excused by the flag that the rationale scopes OUT (core enzymes): {len(overlap)} ***")
print(f"      {overlap}")
print()
if overlap:
    print("  VERDICT: CONTRADICTION CONFIRMED.  The single boolean excuses rows the")
    print("           rationale says in words it must not excuse.  There is no per-entity")
    print("           scoping in the schema, so the rationale cannot be honoured as written.")
else:
    print("  VERDICT: NO CONTRADICTION on the pinned artifacts.")

print(); print("=" * 78); print("E4. HOW EACH FIELD IS CONSUMED")
print("=" * 78)
print("  unknown_backed_proteins_acceptable:")
print("      goldset.py:397  dataclass field, default False")
print("      goldset.py:721  parsed:  bool(raw.get('unknown_backed_proteins_acceptable', False))")
print("      goldset.py:468  round-tripped in to_dict()")
print("      semantic.py:1417  elif not case.unknown_backed_proteins_acceptable:  -> emits")
print("                        'unknown_backed_protein_not_acceptable' finding per row")
print("      semantic.py:1453  selects the summary wording only")
print("      render.py:211     printed as unknown_ok=<flag>")
print("  unknown_backed_rationale:")
print("      NOT REFERENCED ANYWHERE IN src/ EXCEPT parsing/round-trip  <-- see E5")
