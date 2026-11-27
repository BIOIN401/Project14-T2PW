"""Offline evaluation of the proposed guard (T1 name, T2 external id) vs DB truth (dbcheck rows). Not used by the writer."""
import json, re, sys, collections
rows = json.load(open(sys.argv[1], encoding="utf-8"))["rows"]
def norm(s):
    s = str(s or "").casefold().strip()
    s = re.sub(r"^(l|d|\(r\)|\(s\))-", "", s)
    s = re.sub(r"ic acid$", "ate", s)
    return re.sub(r"[^0-9a-z]+", "", s)
def pb_names(c, pid):
    out = set(); mm = c.get("mapping_meta") or {}
    dr = c.get("db_row") or {}
    if str(dr.get("id")) == str(pid): out.add(dr.get("name"))
    if str(mm.get("pathbank_compound_id")) == str(pid): out.add(mm.get("resolved_name"))
    for cand in mm.get("candidates") or []:
        if str(cand.get("pathbank_compound_id")) == str(pid): out.add(cand.get("name"))
    for cand in (c.get("db_match") or {}).get("candidates") or []:
        if str(cand.get("id")) == str(pid): out.add(cand.get("name"))
    return {norm(x) for x in out if x} - {""}
KEYS = {"hmdb": "hmdb_id", "kegg": "kegg_id", "chebi": "chebi_id"}
def ext_ok(c, pid):
    mm = c.get("mapping_meta") or {}
    for cand in mm.get("candidates") or []:
        if str(cand.get("pathbank_compound_id")) != str(pid): continue
        k = cand.get("matched_on")
        if k in KEYS:
            ours = str(c.get(KEYS[k]) or (c.get("mapped_ids") or {}).get(k) or "").replace("CHEBI:", "")
            theirs = str((cand.get("mapped_ids") or {}).get(k) or "").replace("CHEBI:", "")
            if ours and ours == theirs: return k
    return None
tot = collections.Counter(); out = []
for p in sys.argv[2:]:
    ir = json.load(open(p, encoding="utf-8"))
    tag = p.replace("\\", "/").split("/")[-2]
    for c in (ir.get("entities") or {}).get("compounds") or []:
        tot["compounds"] += 1
        pid = c.get("pathbank_compound_id") or c.get("pw_compound_id")
        if not pid: tot["no_pb_id"] += 1; continue
        pid = int(pid); dr = c.get("db_row") or {}
        ours = dr.get("name") or c.get("name")
        t1 = norm(ours) in pb_names(c, pid)
        t2 = None if t1 else ext_ok(c, pid)
        verdict = "T1" if t1 else ("T2:" + t2 if t2 else "skip")
        tot[verdict.split(":")[0]] += 1
        true = rows.get(str(pid), {})
        out.append((tag, pid, ours, true.get("name"), verdict, true.get("pwc_id") == "PW_C%06d" % pid))
print(dict(tot))
for o in out: print("  ", o)
