"""Measure the exact F-132 contradiction population: which positive coverage
anchors are ALSO listed in the same case's forbidden_identifiers."""
from __future__ import annotations
import json, sys
from pathlib import Path
TREE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(TREE / "src"))
from t2pw.bench.goldset import load_gold_set, normalize_name  # noqa: E402
print("MEASURED_TREE t2pw =", sys.modules["t2pw"].__file__)

gs = load_gold_set()
cases = list(getattr(gs, "cases", gs) or [])
print("cases:", len(cases))

POSITIVE = ("expected_pathway_anchors", "expected_enzymes", "acceptable_enzymes",
            "expected_substrates", "expected_products")
total_overlap = 0
affected_papers = []
for c in cases:
    fids = list(getattr(c, "forbidden_identifiers", ()) or ())
    fnorm = {}
    for f in fids:
        nm = getattr(f, "identifier", None) or getattr(f, "name", None) or str(f)
        fnorm[normalize_name(str(nm))] = str(nm)
    if not fnorm:
        continue
    hits = []
    for field in POSITIVE:
        for t in (getattr(c, field, ()) or ()):
            tn = getattr(t, "name", None) or str(t)
            n = normalize_name(str(tn))
            if n in fnorm:
                hits.append((field, str(tn), fnorm[n]))
    if hits:
        affected_papers.append(c.paper_id)
        total_overlap += len(hits)
        print(f"\n--- {c.paper_id} --- forbidden={len(fids)}")
        for field, term, forb in hits:
            print(f"    OVERLAP  {field:26s} '{term}'   <-- forbidden as '{forb}'")

print()
print("=" * 70)
print("F-132 CONTRADICTION POPULATION")
print("=" * 70)
print("affected papers:", len(affected_papers), json.dumps(affected_papers))
print("total overlapping (field, term) pairs:", total_overlap)
print()
print("Per-case forbidden_identifiers inventory (for the charter):")
for c in cases:
    fids = list(getattr(c, "forbidden_identifiers", ()) or ())
    names = [str(getattr(f, "identifier", None) or getattr(f, "name", None) or f) for f in fids]
    print(f"  {c.paper_id}: {len(names)} -> {json.dumps(names)}")
