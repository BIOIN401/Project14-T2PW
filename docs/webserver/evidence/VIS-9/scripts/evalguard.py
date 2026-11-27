"""Offline evaluation of the pwc-id guard on IRs, checked against the DB rows fetched by dbcheck.py (not used by the writer)."""
import json, re, sys, collections
rows = json.load(open(sys.argv[1], encoding="utf-8"))["rows"]
def norm(s):
    return re.sub(r"[^0-9a-z]+", "", str(s or "").casefold())
C = collections.Counter(); fails = []; passes = []
for p in sys.argv[2:]:
    ir = json.load(open(p, encoding="utf-8"))
    for c in (ir.get("entities") or {}).get("compounds") or []:
        pid = c.get("pathbank_compound_id") or c.get("pw_compound_id")
        if not pid: continue
        pid = int(pid)
        mm = c.get("mapping_meta") or {}
        res = (mm.get("resolution") or {}).get("status")
        C["res:" + str(res)] += 1
        C["mm_pid_eq:" + str(str(mm.get("pathbank_compound_id")) == str(pid))] += 1
        names = {norm(mm.get("resolved_name"))}
        for cand in mm.get("candidates") or []:
            if str(cand.get("pathbank_compound_id")) == str(pid):
                names |= {norm(cand.get("name")), norm(cand.get("short_name"))}
        names.discard("")
        dbr = c.get("db_row") or {}
        ours = c.get("name") if not dbr.get("name") else dbr.get("name")
        ok = norm(ours) in names
        syn = {norm(s) for s in c.get("synonyms") or []}
        ok_syn = ok or bool(syn & names)
        true = rows.get(str(pid), {})
        C["name_ok:" + str(ok)] += 1
        C["name_ok_or_syn:" + str(ok_syn)] += 1
        C["ours_eq_dbname:" + str(norm(ours) == norm(true.get("name")))] += 1
        rec = (p.split("/")[-2] if "show" in p or "item6" in p else p.split("/")[-1][:8], pid, ours, true.get("name"), res, sorted(names)[:3])
        (passes if ok else fails).append(rec)
print(dict(C))
print("FAILS (name guard)", len(fails))
for f in fails: print("  ", f)
