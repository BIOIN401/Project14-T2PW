"""Trace the 5 unknown names through ORCH-767 chlorophyll strict leg artifacts. Read-only."""
import json, sys
from pathlib import Path

LEG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/"
    "ORCH-767/legs/chlorophyll-strict/2026-10-23_2328/papers/PMC5130987/strict")
NAMES = ["PIF1", "GBP", "FAR1", "CPO1", "PPO1", "CPO", "PPO", "FHY3"]


def load(n):
    p = LEG / n
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}/{i}")
    else:
        yield path, o


def payload_of(d):
    if d is None:
        return None
    for k in ("payload", "mapped", "data"):
        if isinstance(d, dict) and k in d and isinstance(d[k], dict) and "entities" in d[k]:
            return d[k]
    return d


def summarize(label, d):
    print(f"\n===== {label} =====")
    if d is None:
        print("missing"); return
    d = payload_of(d)
    ents = (d.get("entities") or {}) if isinstance(d, dict) else {}
    for nm in NAMES:
        hits = []
        for b, rows in ents.items():
            if isinstance(rows, list):
                for i, r in enumerate(rows):
                    if isinstance(r, dict) and str(r.get("name", "")).strip().lower() == nm.lower():
                        hits.append(f"{b}[{i}]")
        print(f"  entity {nm}: {hits or '-'}")
    procs = d.get("processes") or {}
    for nm in NAMES:
        refs = []
        for pb, rows in procs.items():
            if not isinstance(rows, list):
                continue
            for i, r in enumerate(rows):
                for p, v in walk(r):
                    if isinstance(v, str) and v.strip().lower() == nm.lower():
                        refs.append(f"{pb}[{i}]{p}")
        print(f"  refs {nm}: {refs or '-'}")


for f in ["stage1_payload.json", "merged_payload.json", "final_mapped.json"]:
    summarize(f, load(f))

print("\n===== keys =====")
for f in ["final_mapped.json", "merged_payload.json", "stage1_payload.json", "quarantine_report.json"]:
    d = load(f)
    print(f, list(d.keys())[:30] if isinstance(d, dict) else type(d))
