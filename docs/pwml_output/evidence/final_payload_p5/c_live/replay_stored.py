"""Replay the repair step on the four stored defective Stage-1 payloads (TRACE.md section 2) with the live model.

usage: python replay_stored.py <out_dir>
One model call per case (the call only happens when the stored payload has a deficient transport).
Writes <out_dir>/replay_<case>.json (the repair report plus before/after transports) and prints one line per case.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "C:/t/fp/src")
import t2pw.llm.client  # noqa: F401,E402  (loads .env, override=True)

for k in ("T2PW_FINAL_PAYLOAD", "T2PW_OFFLINE_CURATOR", "T2PW_TRANSPORT_SIDES"):
    os.environ.pop(k, None)

from t2pw.service import transport_sides as ts  # noqa: E402

R = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
CASES = {
    "1_TubI_export": "runs_smoke/2026-09-08_1528/papers/PMC6112128/strict",
    "2_ferric_enterobactin_import": "runs_verify/2026-08-21_2057/papers/PMC12096016/research",
    "3_enterobactin_secretion_0827": "runs_verify/2026-08-27_1341/papers/PMC12096016/strict",
    "4_enterobactin_secretion_0902": "runs_verify/2026-09-02_2052/papers/PMC12096016/strict",
}
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for case, rel in CASES.items():
    payload = json.loads(Path(R + rel, "stage1_payload.json").read_text(encoding="utf-8"))
    tp = Path(R + rel, "01_source_text.txt")
    text = (tp if tp.exists() else Path(R + rel).parent / "01_source_text.txt").read_text(encoding="utf-8")
    t0 = time.time()
    before = copy.deepcopy(payload)
    fixed, report = ts.repair_transport_sides(payload, text)
    trans = lambda p: [{k: t.get(k) for k in ("name", "cargo", "from_biological_state", "to_biological_state")}
                       for t in (p.get("processes") or {}).get("transports") or []]
    rec = {"case": case, "seconds": round(time.time() - t0, 1), "unchanged_input": payload == before, "report": report,
           "transports_before": trans(before), "transports_after": trans(fixed),
           "states_before": [s.get("name") for s in before.get("biological_states") or []],
           "states_after": [s.get("name") for s in fixed.get("biological_states") or []]}
    (out / f"replay_{case}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    print(case, "calls", report["calls"], "deficient", len(report["deficient"]), "repairs", len(report["repairs"]),
          "rejected", len(report["rejected"]), "error", report["error"])
