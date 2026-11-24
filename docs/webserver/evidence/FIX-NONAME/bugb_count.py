"""Count stored leg payloads where the deterministic audit adds a compound that is already a nucleic acid.

usage: python -I bugb_count.py <src>
"""

import glob
import json
import sys

sys.path.insert(0, sys.argv[1])
from t2pw.curation.audit_json_llm import audit_payload  # noqa: E402

total = hit = 0
for root in ("legs3", "legs4", "legs4g", "legs5g"):
    for path in sorted(glob.glob(f"C:/t/acc-runs/{root}/**/merged_payload.json", recursive=True)):
        payload = json.load(open(path, encoding="utf-8"))
        nucleic = {str(r.get("name", "")).strip().lower() for r in payload.get("entities", {}).get("nucleic_acids", []) or [] if isinstance(r, dict)}
        dups = [
            op["value"].get("name")
            for op in audit_payload(payload, use_llm=False)["patch"]
            if op["op"] == "add" and op["path"] == "/entities/compounds/-"
            and str(op["value"].get("name", "")).strip().lower() in nucleic
        ]
        total += 1
        if dups:
            hit += 1
            print(path.replace("\\", "/").split("acc-runs/")[1].split("/papers")[0], dups)
print(f"{hit} of {total} merged payloads get a compound row duplicating a nucleic acid")
