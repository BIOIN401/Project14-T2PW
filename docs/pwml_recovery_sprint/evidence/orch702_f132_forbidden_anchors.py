"""F-132: how often does Stage 0 draw a requested-core term that the SAME gold case
lists in forbidden_identifiers?  Each such term then counts as unmatched and lowers
coverage_ratio -- the pipeline scored down for obeying the gold."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(ROOT/"src"))
from t2pw.bench.goldset import load_gold_set, pinned_gold_set_path
gold = load_gold_set(pinned_gold_set_path())
listed = subprocess.run(["git","ls-files","*quarantine_report.json"],
                        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
tot_terms = tot_forbidden = legs = 0
for line in sorted(l.strip() for l in listed.stdout.splitlines() if l.strip()):
    leg = (ROOT/line).parent
    paper, mode = leg.parent.name, leg.name
    case = gold.by_id(paper)
    if case is None: continue
    cov = (json.loads((ROOT/line).read_text(encoding="utf-8")).get("coverage") or {})
    unmatched = cov.get("unmatched_terms") or []
    if not unmatched: continue
    hits = []
    for t in unmatched:
        fb = case.forbidden_match(t)
        if fb is None:  # anchors carry parenthetical glosses; retry on the head
            fb = case.forbidden_match(str(t).split("(")[0].strip())
        if fb is not None:
            hits.append((t, fb.kind))
    legs += 1; tot_terms += len(unmatched); tot_forbidden += len(hits)
    if hits:
        print(f"{leg.parents[2].name:18s} {paper} {mode:9s} "
              f"coverage={cov.get('coverage_ratio')!r} {len(hits)}/{len(unmatched)} unmatched are GOLD-FORBIDDEN")
        for t, k in hits: print(f"      {t!r}  kind={k}")
print(f"\nlegs with unmatched terms: {legs}   unmatched terms total: {tot_terms}   "
      f"of which gold-forbidden: {tot_forbidden}")
