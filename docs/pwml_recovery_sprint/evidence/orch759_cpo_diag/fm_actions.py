"""Stdlib-only: find every CPO mention outside entities/processes in final_mapped.json (mapping reports/actions)."""
import json
from pathlib import Path
LEG = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict")
p = json.loads((LEG / "final_mapped.json").read_text(encoding="utf-8"))
print("top keys:", list(p.keys()))
def scan(o, path):
    if isinstance(o, dict):
        if any(isinstance(v, str) and ("CPO" in v) for v in o.values()) and not path.startswith("/entities") and not path.startswith("/processes"):
            print(path, json.dumps(o, ensure_ascii=False)[:700])
            return
        for k, v in o.items():
            scan(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            scan(v, f"{path}/{i}")
scan(p, "")
