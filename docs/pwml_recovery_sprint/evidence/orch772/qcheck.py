"""ORCH-772 dev check: per replayed leg, recorded vs replayed quarantined[] counts and replayed ok."""
import json
from pathlib import Path

PRIMARY = Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
man = json.loads(Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch772/docs/pwml_recovery_sprint/evidence/orch768_corpus_manifest.json").read_text(encoding="utf-8"))
legs = {l["id"]: l for l in man["legs"]}
tip = json.loads(Path("C:/t/orch772/tip1.json").read_text(encoding="utf-8"))
rows = []
for row in tip["legs"]:
    r = row["replay"]
    if not r or r.get("error"):
        continue
    leg = legs[row["id"]]
    rec = json.loads((PRIMARY / leg["inputs"]["quarantine_report.json"]["path"]).read_text(encoding="utf-8"))
    out_q = Path("C:/t/orch772/out-tip1") / __import__("re").sub(r"[^A-Za-z0-9._-]+", "_", row["id"]) / "outputs" / "quarantine_report.json"
    rep = json.loads(out_q.read_text(encoding="utf-8")) if out_q.is_file() else None
    rows.append((r["checkpoint"], row["id"], len(rec.get("quarantined") or []), rec.get("ok"),
                 None if rep is None else len(rep.get("quarantined") or []), None if rep is None else rep.get("ok"),
                 r["quarantine"]["ok"]))
for kind in ("canonical", "pre_freeze"):
    sub = [x for x in rows if x[0] == kind]
    print(kind, len(sub), "recorded>0:", sum(x[2] > 0 for x in sub), "replayed>0:", sum((x[4] or 0) > 0 for x in sub),
          "replayed refused:", sum(x[6] is False for x in sub), "no outputs file:", sum(x[4] is None for x in sub))
    for x in sub:
        if (x[4] or 0) > 0 or x[6] is False or (kind == "pre_freeze"):
            print("  ", x)
