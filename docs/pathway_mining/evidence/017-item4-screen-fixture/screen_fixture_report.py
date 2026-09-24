"""Offline: the mining screen over pilot-001's 21 papers, per round vocabulary. No network."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/t/mining/tests")
import test_mining_screen_rescue as T  # noqa: E402
from t2pw.mining.models import FrontierItem  # noqa: E402
from t2pw.mining.queries import rescue_terms  # noqa: E402

config = T._config(Path(sys.argv[1]))
rows = []
for rnd in (1, 2):
    frontier = [FrontierItem(frontier_item_id=f"fr_{i}", kind=f["kind"], target=f["target"])
                for i, f in enumerate(T.FIXTURE["frontier_by_round"][str(rnd)])]
    terms = rescue_terms(config, frontier)
    papers = [p for p in T.FIXTURE["papers"] if p["discovered_round"] == rnd]
    result = T._acquire(config, papers, rnd, run_terms=terms)
    exp = {p["paper_id"]: p["expectation"] for p in papers}
    for lp in result.papers:
        rows.append((rnd, lp.record.paper_id, exp[lp.record.paper_id], "INCLUDED", lp.record.inclusion_reasons[0]))
    for r in result.excluded:
        rows.append((rnd, r.paper_id, exp[r.paper_id], "excluded", r.exclusion_reasons[0][:90]))
    print(f"round {rnd}: {len(terms)} run terms: {terms[:12]}{' ...' if len(terms) > 12 else ''}")
for row in sorted(rows, key=lambda r: (r[3], r[2])):
    print(" | ".join(str(x) for x in row))
Path(sys.argv[2]).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
