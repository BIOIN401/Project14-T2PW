"""R7 live check on a server job folder: verifier report, duplicates, gold, lock prune, dropped names, rename keys."""
import json, sys, glob, os, collections
sys.path.insert(0, "C:/t/t2pw-server/src")
from t2pw.pipeline.name_keys import reference_key as rk
o = sys.argv[1]
J = lambda f: json.load(open(os.path.join(o, f), encoding="utf-8"))
v = J("reaction_verifier_report.json"); f = J("final_mapped.json")
print("flags", v.get("flags"), "| model", v.get("model"), "| calls", v.get("calls"), "| s", v.get("seconds"), "| error", v.get("error"), "| lock removed", v.get("lock_manifest_entries_removed"))
r = v["reactions"]
print("omitted", len(r.get("omitted") or []), "| merged", len(r.get("merged") or []), "| repaired", len(r.get("repaired") or []), "| flagged", len(r.get("flagged") or []), "| appended", r.get("appended"))
def nm(x): return x.get("name") if isinstance(x, dict) else x
rx = (f.get("processes") or {}).get("reactions") or []
sigs = []
for x in rx:
    ins = sorted(rk(nm(i)) for i in (x.get("inputs") or []) if nm(i)); outs = sorted(rk(nm(i)) for i in (x.get("outputs") or []) if nm(i))
    sigs.append((tuple(ins), tuple(outs)))
    print("  ", [nm(i) for i in x.get("inputs") or []], "->", [nm(i) for i in x.get("outputs") or []])
print("reactions", len(rx), "| duplicate signatures", [s for s, c in collections.Counter(sigs).items() if c > 1])
# renames/additions: proposed vs written reference_key
bad = 0; seen = 0
for row in r.get("rows") or []:
    for ch in (row.get("applied_changes") or row.get("changes") or []):
        if isinstance(ch, dict) and ch.get("proposed") and ch.get("written"):
            seen += 1; bad += rk(ch["proposed"]) != rk(ch["written"])
print("rename/add pairs seen", seen, "| reference_key mismatches", bad)
# dropped entity names must not appear anywhere in final
dropped = [d.get("name") if isinstance(d, dict) else d for d in (v.get("entities_removed") or r.get("entities_removed") or [])]
txt = json.dumps(f)
print("dropped entities", dropped, "| still referenced:", [d for d in dropped if d and json.dumps(d)[1:-1] in txt])
print("ir report errors:", [e.get("code") for e in (J("pwml_ir_report.json").get("errors") or []) if isinstance(e, dict)][:10])
