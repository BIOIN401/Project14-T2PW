"""Stdlib-only: CPO's name_gate / rejected_mapped_ids / lineage in final_mapped.json."""
import json
from pathlib import Path
LEG = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict")
p = json.loads((LEG / "final_mapped.json").read_text(encoding="utf-8"))
r = next(r for r in p["entities"]["proteins"] if r.get("name") == "CPO")
mm = r["mapping_meta"]
for k in ("name_gate", "rejected_mapped_ids", "identity_verdict", "queries_tried", "resolved_name", "alias_source", "candidates"):
    print(k, "=", json.dumps(mm.get(k), ensure_ascii=False)[:1500])
for i, l in enumerate(r.get("provenance_lineage") or []):
    print("lineage", i, l.get("stage"), "|", l.get("reason"))
