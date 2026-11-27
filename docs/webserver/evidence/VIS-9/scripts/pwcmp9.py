"""PathWhiz import A/B: VIS-7 (a3 + a3b retry) vs VIS-9 run(s). Per file: success, error lines (ids and SQL values cut),
created record counts. usage: pwcmp9.py OUT.txt NEW_results.jsonl [NEW2_results.jsonl ...]"""
import json, re, sys
from pathlib import Path
def load(paths):
    out = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            name = Path(r["file"].replace("\\", "/")).stem.replace("v5_", "").replace("ref_", "")
            errs = {}
            for k, v in (r.get("errors") or {}).items():
                for m in v:
                    key = re.sub(r"_\d+$", "", k) + ": " + re.sub(r"VALUES \(.*", "VALUES (...)", m)[:140]
                    errs[key] = errs.get(key, 0) + 1
            out[name] = {"success": r.get("success"), "errors": errs, "created": r.get("created") or {}, "screenshot": r.get("screenshot")}
    return out
old = load(["C:/t/vis-pw7-work/pw/a3/results.jsonl", "C:/t/vis-pw7-work/pw/a3b/results.jsonl"])
new = load(sys.argv[2:])
lines, new_err = [], 0
for name in sorted(new):
    o, n = old.get(name), new[name]
    if o is None:
        lines.append(f"{name}: no VIS-7 row"); continue
    d = [f"  {o['errors'].get(k, 0)} -> {n['errors'].get(k, 0)}  {k}" for k in sorted(set(o["errors"]) | set(n["errors"])) if o["errors"].get(k) != n["errors"].get(k)]
    new_err += sum(max(0, n["errors"].get(k, 0) - o["errors"].get(k, 0)) for k in n["errors"])
    c = [f"  created {k}: {o['created'].get(k)} -> {n['created'].get(k)}" for k in sorted(set(o["created"]) | set(n["created"])) if o["created"].get(k) != n["created"].get(k)]
    lines.append(f"{name}: success {o['success']} -> {n['success']}; errors {sum(o['errors'].values())} -> {sum(n['errors'].values())}; screenshot {'yes' if n['screenshot'] else 'no'}")
    lines += d + c
lines.append(f"NEW_ERROR_LINES {new_err}")
Path(sys.argv[1]).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
