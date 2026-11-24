"""Bug B probe: does the deterministic audit add a compound row for a nucleic-acid reference?

usage: python -I bugb_probe.py <src> <merged_payload.json>
"""

import json
import sys

sys.path.insert(0, sys.argv[1])
from t2pw.curation.audit_json_llm import audit_payload  # noqa: E402

payload = json.load(open(sys.argv[2], encoding="utf-8"))
nucleic = {str(r.get("name", "")).strip().lower() for r in payload["entities"].get("nucleic_acids", []) if isinstance(r, dict)}
ops = audit_payload(payload, use_llm=False)["patch"]
for op in ops:
    if op["op"] == "add" and op["path"] in ("/entities/compounds/-", "/entities/proteins/-"):
        name = str(op["value"].get("name", ""))
        print(op["path"], repr(name), "ALSO_A_NUCLEIC_ACID" if name.strip().lower() in nucleic else "")
