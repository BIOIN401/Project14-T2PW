"""Read-only scan of past paper runs: out-of-scope reactions and Stage-1 -> merged growth."""
import json
import sys
from pathlib import Path

BASE = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
SECTIONS = ("compounds", "proteins", "protein_complexes", "subcellular_locations")


def count(p):
    ent = p.get("entities") or {}
    proc = p.get("processes") or {}
    out = {k: len(ent.get(k) or []) for k in SECTIONS}
    for k, v in proc.items():
        if isinstance(v, list):
            out[k] = len(v)
    out["states"] = len(p.get("biological_states") or [])
    return out


rows = []
for root in ("runs_papers", "runs_validation", "runs_verify"):
    for s1 in (BASE / root).rglob("stage1_payload.json"):
        d = s1.parent
        try:
            p1 = json.loads(s1.read_text(encoding="utf-8"))
        except Exception:
            continue
        rx = (p1.get("processes") or {}).get("reactions") or []
        oos = [r.get("name") for r in rx if str(r.get("scope_membership", "")).strip().lower() == "out_of_scope"]
        mp = d / "merged_payload.json"
        grew = {}
        if mp.exists():
            try:
                c1, c2 = count(p1), count(json.loads(mp.read_text(encoding="utf-8")))
                grew = {k: c2[k] - c1.get(k, 0) for k in c2 if c2[k] != c1.get(k, 0)}
            except Exception:
                grew = {"err": 1}
        pwml = [x.name for x in d.glob("*.pwml")]
        secs = None
        res = d / "RESULT.txt"
        status = ""
        if res.exists():
            t = res.read_text(encoding="utf-8", errors="replace")
            status = t.splitlines()[0]
            for line in t.splitlines():
                if line.startswith("wall time"):
                    secs = line.split(":", 1)[1].strip()
        rows.append((len(oos), str(d.relative_to(BASE)), d.parent.name, status, secs, len(rx), grew, bool(pwml)))

rows.sort(key=lambda r: -r[0])
for r in rows:
    if r[0] or "--all" in sys.argv:
        print(r)
print(len(rows), "stage1 payloads;", sum(1 for r in rows if r[0]), "with out_of_scope")
