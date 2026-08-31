import sys, json, io
from pathlib import Path
REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "src"))
from t2pw.bench.goldset import load_gold_set, pinned_gold_set_path
gs = load_gold_set(pinned_gold_set_path())
cases = {c.paper_id: c for c in gs.cases}
print("=== F-150 half 1: the delta spelling gap ===")
c = cases["PMC12180156"]
for term in ("5-aminolevulinic acid", "delta-aminolevulinic acid",
             "\u03b4-aminolevulinic acid", "protoporphyrin IX", "ALA"):
    hit = c.forbidden_match(term)
    print(f"  forbidden_match({term!r:34s}) -> {getattr(hit,'name',None)!r}")
print("\n  forbidden_identifiers[0].name    :", c.forbidden_identifiers[0].name)
print("  forbidden_identifiers[0].aliases :", list(c.forbidden_identifiers[0].aliases))

print("\n=== F-150 half 2: supported_reactions_complete across ALL gold cases ===")
n_true = n_false = n_missing = 0
for pid, case in sorted(cases.items()):
    v = getattr(case, "supported_reactions_complete", "MISSING")
    if v is True: n_true += 1
    elif v is False: n_false += 1
    else: n_missing += 1
    mx = getattr(case, "max_retained_reactions", None)
    print(f"  {pid:14s} supported_reactions_complete={str(v):8s} max_retained_reactions={mx}")
print(f"\n  TRUE={n_true}  FALSE={n_false}  MISSING={n_missing}   (total {len(cases)})")
print("  -> Priority 2 is evaluable only through max_retained_reactions" if n_true == 0 else "  -> some case sets it")
