import json
import logging
from pathlib import Path

logging.disable(logging.CRITICAL)
from t2pw.pwml.ir import build_pwml_ir
from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution


class _DownDb:
    last_error = "harvest_db_down"

    def available(self):
        return False


leg = "runs/2026-07-27_1623/papers/PMC12312563__structures-of-listeria-monocytogenes-mend-in-th/strict/final_mapped.json"
staged = json.loads(Path(leg).read_text(encoding="utf-8"))
run_prefreeze_resolution(staged, strict_db=True, db_resolver=_DownDb(), name_index=None)
ir, report = build_pwml_ir(staged, db_resolver=_DownDb(), strict_db=True, name_index=None)
print([w.get("code") for w in report["warnings"]])
print([(w.get("reason"), w.get("raw_interaction_type"), w.get("pointer")) for w in report["warnings"] if w.get("code") == "interaction_withheld_from_pwml"])
print([(r["name"], r["export_status"], (r.get("withheld") or {}).get("reason")) for r in ir["processes"]["interactions"]])
