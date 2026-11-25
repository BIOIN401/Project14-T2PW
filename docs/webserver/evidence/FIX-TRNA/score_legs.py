"""usage: python -I score_legs.py  -> per-leg table for C:/t/acc-runs/legs-trna (run with PYTHONPATH unset; sets paths itself)."""
import json
import re
import sys
from pathlib import Path

sys.path[:0] = ["C:/t/fix-trna/src", "C:/t/acc-runs/scripts"]
import accscore as S  # noqa: E402

TREE = "C:/t/fix-trna"
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/t/acc-runs/legs-trna")
rows = []
for leg in sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith("partial")):
    lp = S.leg_paths(leg)
    rec = {"leg": leg.name}
    if lp is None:
        rec["note"] = "no run dir"
        rows.append(rec)
        continue
    paper, sd, text = lp
    res = (sd / "RESULT.txt")
    rtxt = res.read_text(encoding="utf-8", errors="replace") if res.exists() else ""
    m = re.search(r"^RESULT:\s*(\S+)", rtxt, re.M)
    rec["result"] = m.group(1) if m else "?"
    m = re.search(r"^stage\s*:\s*(.+)$", rtxt, re.M)
    rec["fail_stage"] = m.group(1).strip() if m and rec["result"] != "PASS" else ""
    m = re.search(r"^message\s*:\s*(.+)$", rtxt, re.M)
    rec["message"] = m.group(1).strip()[:200] if m and rec["result"] != "PASS" else ""
    rec["pwml"] = bool(list(sd.parent.glob("**/*.pwml")) or list(sd.glob("**/*.pwml")))
    fm = sd / "final_mapped.json"
    mp = sd / "merged_payload.json"
    pj = fm if fm.exists() else mp
    rec["payload"] = pj.name if pj.exists() else None
    if pj.exists():
        pay = json.load(open(pj, encoding="utf-8"))
        ent = pay.get("entities", {})
        rec["n_reactions"] = len((pay.get("processes") or {}).get("reactions") or [])
        rec["trna_compounds"] = [c.get("name") for c in ent.get("compounds", []) if "trna" in str(c.get("name", "")).lower()]
        rec["trna_nucleic_acids"] = [c.get("name") for c in ent.get("nucleic_acids", []) if "trna" in str(c.get("name", "")).lower()]
        rec["trna_reactions"] = [r.get("name") for r in (pay.get("processes") or {}).get("reactions", []) if "trna" in json.dumps(r).lower()]
        if pj.name == "final_mapped.json":
            try:
                sc = S.score_payload(S.CHL, pay, text, (), TREE)
                rec["recall"] = sc["recall"]
                rec["precision"] = sc["precision"]
                rec["matched"] = sc["matched"]
                rec["gold_rows_scored"] = sc["rows"]
            except Exception as e:  # keep going
                rec["score_error"] = repr(e)[:200]
    rows.append(rec)
print(json.dumps(rows, indent=1))
json.dump(rows, open("C:/t/vis-runs/scratch/fixtrna/legs_score.json", "w"), indent=1)
