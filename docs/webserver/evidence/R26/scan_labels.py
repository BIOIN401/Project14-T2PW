"""Read-only: scope_membership labels and Stage-2 reply sizes per past run."""
import collections
import json
from pathlib import Path

BASE = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
rows = []
for root in ("runs_papers", "runs_validation", "runs_verify"):
    for s1 in (BASE / root).rglob("stage1_payload.json"):
        d = s1.parent
        try:
            p1 = json.loads(s1.read_text(encoding="utf-8"))
        except Exception:
            continue
        rx = (p1.get("processes") or {}).get("reactions") or []
        labels = collections.Counter(str(r.get("scope_membership", "<absent>")).strip().lower() for r in rx)
        stage2 = []
        trace = d / "LEG_TRACE.jsonl"
        if trace.exists():
            for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("kind") == "model_attempt" and "stage 2" in str(rec.get("stage", "")).lower():
                    stage2.append(rec.get("content_chars"))
        rows.append((d.parent.name, d.name, str(d.relative_to(BASE))[:60], dict(labels), stage2))

labelled = [r for r in rows if set(r[3]) - {"<absent>"}]
print(len(rows), "payloads,", len(labelled), "with any label")
for r in sorted(labelled):
    print(r)
