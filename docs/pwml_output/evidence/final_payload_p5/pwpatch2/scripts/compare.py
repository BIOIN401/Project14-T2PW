import json, sys, os
from pathlib import Path
B = Path("C:/t/ev-fp/p5/pwpatch2/before/results.jsonl")
A = Path("C:/t/ev-fp/p5/pwpatch2/after/results.jsonl")
sys.path.insert(0, "C:/t/p5scratch/pwpatch2")

def load(p):
    d = {}
    for l in p.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l); d[os.path.basename(r["file"]).replace("\\", "/")] = r
    return d

def flat(errs):
    if not errs: return []
    if isinstance(errs, dict):
        out = []
        for k, v in sorted(errs.items()): out.append("%s=%s" % (k, json.dumps(v, sort_keys=True)))
        return out
    return [str(errs)]

b, a = load(B), load(A)
bind = set(open("C:/t/p5scratch/pwpatch2/bind.txt").read().split())
rows = []
changed_nobind = 0
for f in b:
    rb, ra = b[f], a.get(f)
    if ra is None:
        print(f, "MISSING in after"); continue
    stem = f.replace(".pwml", "")
    kind = "bind" if stem in bind else ("control" if f.startswith("PW") else "nobind")
    same = (rb["success"], flat(rb["errors"]), rb["created"], rb["created_main"]) == (ra["success"], flat(ra["errors"]), ra["created"], ra["created_main"])
    keys = ["biological_states", "transports", "transport_transporters", "reaction_coupled_transports", "reaction_coupled_transport_enzymes", "pathway_elements", "element_states"]
    def pick(r): return {k: r["created"].get(k, 0) for k in keys}
    print("%-14s %-8s same=%s success %s->%s msgs %d->%d bs %s->%s transports %s->%s" % (
        f, kind, same, rb["success"], ra["success"], len(flat(rb["errors"])), len(flat(ra["errors"])),
        rb["created"].get("biological_states", 0), ra["created"].get("biological_states", 0),
        rb["created"].get("transports", 0), ra["created"].get("transports", 0)))
    if not same:
        if kind != "bind": changed_nobind += 1
        tabs = sorted(set(rb["created"]) | set(ra["created"]))
        for t in tabs:
            if rb["created"].get(t, 0) != ra["created"].get(t, 0): print("      table", t, rb["created"].get(t, 0), "->", ra["created"].get(t, 0))
        eb, ea = set(flat(rb["errors"])), set(flat(ra["errors"]))
        for e in sorted(eb - ea): print("      msg only before:", e[:300])
        for e in sorted(ea - eb): print("      msg only after: ", e[:300])
print("files", len(b), "changed among non-bind/control:", changed_nobind)
