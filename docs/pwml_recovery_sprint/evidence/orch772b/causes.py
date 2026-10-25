"""ORCH-772b dev check: per no-PWML replayed leg, the replay's recorded blocking items and how the KI table labels them."""
import json
import sys
from pathlib import Path

from t2pw.app.reaction_evidence import known_issue_id

doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in doc["legs"]:
    r = row["replay"]
    if not r or r.get("error") or r["pwml_written"]:
        continue
    stage, ex = r["failure_stage"], r.get("export") or {}
    if stage == "final_pre_export_gate_lifecycle":
        items = r["quarantine"]["refusal"] or [r["driver"]["gate_reason"]]
    elif stage == "final_pre_export_stage3_gates":
        items = [reason for _p, reason in r["final_gate"]["errors"]]
    elif stage == "upstream_contract_report":
        items = r["driver"]["codes"]
    elif stage == "pwml_export":
        items = ex.get("required_gate_codes") or ex.get("ir_blocking") or ex.get("ir_validation") or ex.get("stage3_contract_codes") or [ex.get("error")]
    else:
        items = [stage]
    items = list(dict.fromkeys(str(i) for i in items if i))
    kis = [known_issue_id(i) for i in items]
    print(f"{row['id']:58s} {stage:34s} n={len(items)} first={kis[0] if kis else '-'} joined={known_issue_id('; '.join(items))} "
          f"distinct={sorted(set(kis))} err={ex.get('error', '')[:70]!r}")
    print(f"{'':60s}{'; '.join(items)[:220]}")
