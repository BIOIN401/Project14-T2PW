"""ORCH-759c scratch: capture base-code facts on the chlorophyll-strict leg (prints only)."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import t2pw.pipeline.strict_quarantine as sq

LEG = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict")
raw = (LEG / "final_mapped.json").read_bytes()
print("final_mapped sha256 =", hashlib.sha256(raw).hexdigest(), "bytes", len(raw))
fm = json.loads(raw.decode("utf-8"))
qrep = json.loads((LEG / "quarantine_report.json").read_text(encoding="utf-8"))
ctx = qrep["coverage"]["requested_context"]
print("requested_context =", json.dumps(ctx, ensure_ascii=False))
res = sq.quarantine_and_close(deepcopy(fm), strict_db=True, pathway_context=ctx, mode="pathwhiz")
r = res.quarantine_report
for k in ("admitted_payload_hash", "decision_input_hash", "resulting_payload_hash", "ok", "refusal_reasons", "counts"):
    print(k, "=", r.get(k))
print("release.status =", (r.get("release") or {}).get("status"))
adm = sorted(f"{a['pointer']}|{a['state']}" for a in res.admissions)
print("admissions n =", len(adm), "sha =", hashlib.sha256("\n".join(adm).encode()).hexdigest())
rem = sorted(f"{e['bucket']}|{e['name']}|{e['reason']}" for e in res.removed_entity_report["removed_entities"])
print("removed n =", len(rem), "sha =", hashlib.sha256("\n".join(rem).encode()).hexdigest())
for line in rem:
    print("  removed:", line)
print("surviving NA =", sorted(sq._row_name(x) for x in res.payload["entities"]["nucleic_acids"]))
print("removed_entity_report keys =", sorted(res.removed_entity_report))
