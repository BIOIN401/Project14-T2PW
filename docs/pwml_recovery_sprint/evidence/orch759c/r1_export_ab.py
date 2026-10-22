"""ORCH-759c round 1: on the 8 legs the location rule changes, does the EXPORT outcome move?

    r1_export_ab.py dump <out.json>      # once with base src, once with tip src
    r1_export_ab.py compare <base.json> <tip.json>

Per leg: the quarantine verdict; then, when quarantine is ok, the strict final gate
(as freeze_canonical_payload runs it) and Chunk E's exporter approximation
(``_required_contract_codes``, ``_ir_codes``). Read-only, except the dump file.
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
P = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
LEGS = [
    P / ".claude/worktrees/orch759c/runs/2026-07-28_0919/papers/PMC12444477__the-regulation-of-lipid-a-biosynthesis/strict",
    P / "runs_papers/ORCH-759/legs/wnt_pcp-strict/2026-10-22_1432/papers/PMC10425814/strict",
    P / "runs_verify/2026-08-25_1216/papers/PMC12856317/strict",
    P / "runs_verify/2026-08-28_1816/papers/PMC12856317/strict",
    P / "runs_verify/2026-09-02_2052/papers/PMC12444477/strict",
    P / "runs_verify/2026-09-02_2052/papers/PMC12856317/strict",
    P / "runs_verify/2026-09-06_1425/papers/PMC11172790/strict",
    P / "runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict",
]


def dump(out):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_strict_quarantine_real_artifact_replay import _ir_codes, _required_contract_codes
    from t2pw.pipeline.process_normalizer import GateValidationError, run_strict_post_normalization_gates
    from t2pw.pipeline.strict_quarantine import quarantine_and_close

    record = {}
    for leg in LEGS:
        payload = json.loads((leg / "final_mapped.json").read_text(encoding="utf-8"))
        q = leg / "quarantine_report.json"
        ctx = (json.loads(q.read_text(encoding="utf-8")).get("coverage") or {}).get("requested_context") if q.is_file() else None
        r = quarantine_and_close(payload, strict_db=True, pathway_context=ctx, mode="pathwhiz")
        row = {"ok": r.ok, "refusal": r.refusal_reasons, "release": (r.quarantine_report.get("release") or {}).get("status")}
        if r.ok:
            try:
                details = run_strict_post_normalization_gates(deepcopy(r.payload), enforce_all_proteins_connected=True)
                row["final_gate"] = {"ok": True, "connectivity": details.get("connectivity")}
            except GateValidationError as exc:
                row["final_gate"] = {"ok": False, "errors": sorted(str(e.get("reason") or e.get("code")) for e in (exc.details or {}).get("errors") or [])}
            row["contract_codes"] = sorted(_required_contract_codes(r.payload))
            row["ir_codes"] = sorted(_ir_codes(r.payload))
        record[str(leg.relative_to(P))] = row
    Path(out).write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"DUMP legs={len(record)} -> {out}")


def compare(base_path, tip_path):
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    tip = json.loads(Path(tip_path).read_text(encoding="utf-8"))
    moved = 0
    for leg in base:
        b, t = base[leg], tip[leg]
        same = b == t
        moved += not same
        print(f"{'SAME' if same else 'MOVED'} {leg}")
        for key in ("ok", "refusal", "release", "final_gate", "contract_codes", "ir_codes"):
            if b.get(key) != t.get(key):
                print(f"   {key}: {json.dumps(b.get(key))[:400]} -> {json.dumps(t.get(key))[:400]}")
            elif key in ("final_gate", "contract_codes") and key in t:
                print(f"   {key} (both): {json.dumps(t.get(key))[:300]}")
    print(f"COMPARE legs={len(base)} moved={moved}")


if __name__ == "__main__":
    dump(sys.argv[2]) if sys.argv[1] == "dump" else compare(sys.argv[2], sys.argv[3])
