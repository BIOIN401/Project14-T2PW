"""aligned payload vs IR reaction diff (by reaction name) with per-reaction detail"""
import json, sys, collections
sys.path.insert(0, "C:/t/corr-assess/scripts")
import a3_models as M
B = M.B
sample = json.load(open(f"{B}/sample.json", encoding="utf-8"))
out = {}
for s in sample:
    i = s["idx"]
    p = json.load(open(f"{B}/base/{i}/final_payload.json", encoding="utf-8"))
    ir = json.load(open(f"{B}/ir/{i}/pwml_ir.json", encoding="utf-8"))
    P = M.from_payload(p); I = M.from_ir(ir)
    # alias map by position for compounds
    pc = [c["name"] for c in p["entities"]["compounds"]]; ic = [c["name"] for c in ir["entities"]["compounds"]]
    alias = {M.nn(a): M.nn(b) for a, b in zip(pc, ic)} if len(pc) == len(ic) else {}
    renames = [(a, b) for a, b in zip(pc, ic) if M.nn(a) != M.nn(b)]
    byname = collections.defaultdict(list)
    for r in I["rxn"]: byname[(r["name"] or "").casefold()].append(r)
    res = dict(renames=renames, aligned=0, identical=0, diffs=[], missing_in_ir=[], quarantined=[])
    for r in P["rxn"]:
        cands = byname.get((r["name"] or "").casefold(), [])
        if not cands:
            res["missing_in_ir"].append(r["name"]); continue
        c = cands.pop(0)
        res["aligned"] += 1
        L = frozenset(alias.get(x, x) for x in r["L"]); R = frozenset(alias.get(x, x) for x in r["R"])
        d = {}
        if L != c["L"]: d["L"] = (sorted(L), sorted(c["L"]))
        if R != c["R"]: d["R"] = (sorted(R), sorted(c["R"]))
        if r["E"] != c["E"]: d["E"] = (sorted(r["E"]), sorted(c["E"]), "payload mods:", r.get("MOD"))
        if d: res["diffs"].append((r["name"], d))
        else: res["identical"] += 1
    res["extra_in_ir"] = [r["name"] for rs in byname.values() for r in rs]
    q = p.get("quarantined_locked_reactions") or []
    res["quarantined"] = [x.get("reaction_name") + " | " + str(x.get("reason")) for x in q]
    out[i] = res
json.dump(out, open(f"{B}/a3/p_ir_aligned.json", "w", encoding="utf-8"), indent=1, default=str)
for i, r in out.items():
    print(i, "aligned", r["aligned"], "identical", r["identical"], "diffs", len(r["diffs"]), "missing_in_ir", len(r["missing_in_ir"]), "extra", len(r["extra_in_ir"]), "quar", len(r["quarantined"]), "renames", r["renames"])
