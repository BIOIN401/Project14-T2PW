"""usage: python -I compare.py replay_wt.json replay_main.json  -> prints identity + every difference."""
import json
import re
import sys

wt = json.load(open(sys.argv[1], encoding="utf-8"))
mn = json.load(open(sys.argv[2], encoding="utf-8"))
print("payloads scanned:", len(wt))
print("flag-off (worktree) == original main-tree code, byte-identical:", sum(wt[p]["off"] == mn[p]["off"] for p in wt), "of", len(wt))
print("main-tree on==off (flag ignored there):", sum(mn[p]["off"] == mn[p]["on"] for p in mn))
diff = [p for p in wt if wt[p]["off"] != wt[p]["on"]]
print("payloads whose audit result differs flag off vs on:", len(diff))
res = []
for p in diff:
    off = json.loads(wt[p]["off"])
    on = json.loads(wt[p]["on"])
    payload = json.load(open(p, encoding="utf-8"))
    ops_off = [json.dumps(o, sort_keys=True) for o in off["patch"]]
    ops_on = [json.dumps(o, sort_keys=True) for o in on["patch"]]
    only_off = [json.loads(o) for o in ops_off if o not in ops_on]
    only_on = [json.loads(o) for o in ops_on if o not in ops_off]
    rep_off = json.dumps(off["report"], sort_keys=True)
    rep_on = json.dumps(on["report"], sort_keys=True)
    print("\n==", p.split("acc-runs/")[-1].split("/papers")[0] if "acc-runs" in p else p)
    print("  ops only with flag off:", json.dumps([(o["op"], o["path"], o.get("value"), o.get("reason")) for o in only_off]))
    print("  ops only with flag on :", json.dumps([(o["op"], o["path"], o.get("value"), o.get("reason")) for o in only_on]))
    # issues differences
    def issues(r):
        out = []
        for lvl in ("errors", "warnings"):
            for i in (r["report"].get("issues", {}) or {}).get(lvl, []) if isinstance(r["report"].get("issues"), dict) else []:
                out.append((lvl, json.dumps(i, sort_keys=True)))
        return out
    io, in_ = issues(off), issues(on)
    print("  issues only off:", [x for x in io if x not in in_][:6])
    print("  issues only on :", [x for x in in_ if x not in io][:6])
    print("  report keys differ:", [k for k in set(off["report"]) | set(on["report"]) if json.dumps(off["report"].get(k), sort_keys=True) != json.dumps(on["report"].get(k), sort_keys=True)])
    names = {o["value"]["name"] for o in only_off if o["op"] == "add" and isinstance(o.get("value"), dict)}
    for n in sorted(names):
        for idx, r in enumerate(payload["processes"].get("reactions", [])):
            ins = [c if isinstance(c, str) else c.get("name") for c in r.get("inputs", [])]
            outs = [c if isinstance(c, str) else c.get("name") for c in r.get("outputs", [])]
            if n in ins or n in outs:
                print(f"  reaction[{idx}] {r.get('name')!r}: {n!r} as {'input' if n in ins else 'output'}")
        for t in payload["processes"].get("transports", []) or []:
            if n in json.dumps(t):
                print("  transport:", t.get("name") or json.dumps(t)[:120])
