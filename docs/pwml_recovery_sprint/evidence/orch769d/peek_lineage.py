"""ORCH-769d: which rows of the Wnt/Ca2+ leg's final_mapped.json carry an
audit_repair lineage entry (the rows an accepted patch op wrote). Read-only."""
import json
import sys
from pathlib import Path

LEG = Path(sys.argv[1])
fm = json.loads((LEG / "final_mapped.json").read_text(encoding="utf-8"))
for section in ("entities", "processes"):
    for bucket, rows in (fm.get(section) or {}).items():
        if not isinstance(rows, list):
            continue
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            lin = row.get("provenance_lineage") or []
            audit = [e for e in lin if isinstance(e, dict) and e.get("stage") == "audit_repair"]
            if audit:
                print(section, bucket, i, ascii(row.get("name")), "syn=", ascii(row.get("synonyms")),
                      "| audit entries:", [ascii(e.get("reason"))[:60] for e in audit])
print("stages seen on protein rows:",
      sorted({e.get("stage") for r in fm["entities"]["proteins"] for e in (r.get("provenance_lineage") or []) if isinstance(e, dict)}))
