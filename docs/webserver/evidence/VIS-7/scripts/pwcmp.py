"""usage: pwcmp.py OLD_RESULTS NEW_RESULTS -- per file: success, error counts and error messages (class: message) old vs new."""
import json, sys, pathlib
def load(p):
    out = {}
    for l in pathlib.Path(p).read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        n = pathlib.Path(r["file"].replace("\\", "/")).stem.replace("v5_", "").replace("ref_", "")
        errs = r.get("errors") or {}
        flat = sorted(f"{k}: {v}" for k, v in (errs.items() if isinstance(errs, dict) else enumerate(errs)))
        out[n] = (r.get("success"), flat, r.get("screenshot"), r.get("created_main"))
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
bad = 0
for n in sorted(set(a) | set(b)):
    oa, ob = a.get(n), b.get(n)
    same = oa and ob and oa[0] == ob[0] and oa[1] == ob[1]
    new = sorted(set(ob[1]) - set(oa[1])) if oa and ob else []
    gone = sorted(set(oa[1]) - set(ob[1])) if oa and ob else []
    print(f"{n:24s} success {oa and oa[0]} -> {ob and ob[0]}  errors {oa and len(oa[1])} -> {ob and len(ob[1])}  {'SAME' if same else 'DIFF'}  shot {bool(ob and ob[2])}  created_main_same={oa and ob and oa[3]==ob[3]}")
    for x in new: print("   NEW :", x[:200]); bad += 1
    for x in gone: print("   GONE:", x[:200])
print("new error lines:", bad)
