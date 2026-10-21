import json, glob, os
def size(p):
    try:
        d = json.load(open(p, encoding="utf-8")); pr = d.get("processes", {}) if isinstance(d, dict) else {}
        return len(d.get("entities", {}).get("compounds", []) if isinstance(d.get("entities"), dict) else []), len(pr.get("reactions", [])), len(pr.get("interactions", [])), os.path.getsize(p)
    except Exception as e: return str(e)[:40]
for g in ["runs_verify/2026-09-16_2152/papers/*/*/stage1_payload.json", "runs_verify/2026-09-06_1425/papers/*/*/stage1_payload.json", "runs_papers/ORCH-755/legs/*/*/papers/*/*/stage1_payload.json"]:
    rows = [size(p) for p in glob.glob(g)]; ok = [r for r in rows if isinstance(r, tuple)]
    if ok: print(g.split("/")[1], "n=", len(ok), "median bytes", sorted(r[3] for r in ok)[len(ok)//2], "median reactions", sorted(r[1] for r in ok)[len(ok)//2], "max bytes", max(r[3] for r in ok))
