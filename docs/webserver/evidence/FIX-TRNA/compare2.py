"""usage: python -I compare2.py replay_wt.json replay_main.json  (fixes compare.py: report issues live in report["errors"]/["warnings"], not report["issues"]; identity ignores report.llm)."""
import json, sys

wt = json.load(open(sys.argv[1], encoding="utf-8"))
mn = json.load(open(sys.argv[2], encoding="utf-8"))

def strip(s):
    d = json.loads(s); d["report"].pop("llm", None); return d

print("payloads scanned:", len(wt))
print("flag-off (worktree) vs original main-tree code, patch+report minus report.llm identical:",
      sum(strip(wt[p]["off"]) == strip(mn[p]["off"]) for p in wt), "of", len(wt))
print("   (full strings differ on all 69 only through report.llm.provider: env-dependent, local vs openrouter)")
diff = [p for p in wt if wt[p]["off"] != wt[p]["on"]]
print("payloads whose audit result differs flag off vs on:", len(diff))
for p in diff:
    off, on = json.loads(wt[p]["off"]), json.loads(wt[p]["on"])
    payload = json.load(open(p, encoding="utf-8"))
    print("\n==", p)
    ops = lambda r: {json.dumps(o, sort_keys=True) for o in r["patch"]}
    print("  ops only flag off:", [json.loads(o) for o in sorted(ops(off) - ops(on))])
    print("  ops only flag on :", [json.loads(o) for o in sorted(ops(on) - ops(off))])
    for lvl in ("errors", "warnings"):
        a = {json.dumps(i, sort_keys=True) for i in off["report"][lvl]}
        b = {json.dumps(i, sort_keys=True) for i in on["report"][lvl]}
        print(f"  {lvl} only flag off:", [json.loads(i) for i in sorted(a - b)])
        print(f"  {lvl} only flag on :", [json.loads(i) for i in sorted(b - a)])
    print("  summary off:", off["report"]["summary"])
    print("  summary on :", on["report"]["summary"])
    for o in ops(off) - ops(on):
        o = json.loads(o)
        n = (o.get("value") or {}).get("name")
        for idx, r in enumerate(payload["processes"].get("reactions", [])):
            names = lambda L: [c if isinstance(c, str) else c.get("name") for c in L]
            if n in names(r.get("inputs", [])): print(f"  reaction[{idx}] {r.get('name')!r}: {n!r} is an input")
            if n in names(r.get("outputs", [])): print(f"  reaction[{idx}] {r.get('name')!r}: {n!r} is an output")
