"""ORCH-759e: diff ``llm-view-base.json`` against ``llm-view-tip.json`` (written by ``llm_view_dump.py``)."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
base = json.loads((HERE / "llm-view-base.json").read_text(encoding="utf-8"))
tip = json.loads((HERE / "llm-view-tip.json").read_text(encoding="utf-8"))
sys.stdout.reconfigure(encoding="utf-8")
for case in base:
    b, t = base[case], tip[case]
    print(f"== {case}")
    print("  report key paths only at tip :", sorted(set(t["report_key_paths"]) - set(b["report_key_paths"])))
    print("  report key paths only at base:", sorted(set(b["report_key_paths"]) - set(t["report_key_paths"])))
    bv = {json.dumps(r, sort_keys=True) for r in b["llm_view"]}
    tv = {json.dumps(r, sort_keys=True) for r in t["llm_view"]}
    print(f"  LLM-view records: base {len(b['llm_view'])}, tip {len(t['llm_view'])}, identical {len(bv & tv)}")
    for rec in sorted(tv - bv):
        row = json.loads(rec)
        old = next((json.loads(x) for x in bv - tv if set(json.loads(x)) == set(row)), None)
        if old is not None:
            print("  changed record, fields:", {k: (old[k], row[k]) for k in row if old.get(k) != row[k]})
        else:
            print("  record only at tip:", {k: row[k] for k in ("entity_type", "name", "status", "reason") if k in row} or rec[:300])
    for rec in sorted(bv - tv):
        row = json.loads(rec)
        if not any(set(json.loads(x)) == set(row) for x in tv - bv):
            print("  record only at base:", rec[:300])
