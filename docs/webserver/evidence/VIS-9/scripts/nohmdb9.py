"""Mirror of the writer's VIS-9 pwc-id decision on IR records (no writer import): per document, existing (trusted, held)
pwc-ids, new derived ones, refusals; and how many new ones lack an HMDB id."""
import json, re, sys, collections
SKIP = {98041, 98042, 265982, 266704, 268987, 268993, 269024, 276999, 277933, 277957, 278008, 285452, 40747}
def norm(s):
    s = str(s or "").casefold().strip(); s = re.sub(r"^(l|d|\(r\)|\(s\))-", "", s); s = re.sub(r"ic acid$", "ate", s)
    return re.sub(r"[^0-9a-z]+", "", s)
def ival(v):
    try: return int(float(str(v).strip()))
    except (TypeError, ValueError): return None
C = collections.Counter()
for p in sys.argv[1:]:
    written = set()
    for c in (json.load(open(p, encoding="utf-8")).get("entities") or {}).get("compounds") or []:
        C["records"] += 1
        dr = c.get("db_row") if isinstance(c.get("db_row"), dict) else {}
        mi = c.get("mapped_ids") if isinstance(c.get("mapped_ids"), dict) else {}
        dm = c.get("db_match") if isinstance(c.get("db_match"), dict) else {}
        trusted = bool(dr) or str(c.get("db_status") or "").casefold() == "matched" or str(dm.get("status") or "").casefold() == "matched"
        held = trusted and (dr.get("pwc_id") or c.get("pwc_id") or mi.get("pwc_id"))
        if held:
            C["existing"] += 1; written.add(held); continue
        pid = ival(c.get("pw_compound_id") or c.get("pathbank_compound_id"))
        if not pid: C["no_pathbank_id"] += 1; continue
        if pid in SKIP: C["refused_skiplist"] += 1; continue
        mm = c.get("mapping_meta") or {}
        names = [dr.get("name")] if ival(dr.get("id")) == pid else []
        if ival(mm.get("pathbank_compound_id")) == pid: names.append(mm.get("resolved_name"))
        names += [x.get("name") for x in mm.get("candidates") or [] if isinstance(x, dict) and ival(x.get("pathbank_compound_id")) == pid]
        names += [x.get("name") for x in dm.get("candidates") or [] if isinstance(x, dict) and ival(x.get("id")) == pid]
        ours = norm(dr.get("name") or c.get("name"))
        if not ours or ours not in {norm(n) for n in names if n}: C["refused_name"] += 1; continue
        pwc = "PW_C%06d" % pid
        if pwc in written: C["refused_repeat"] += 1; continue
        written.add(pwc); C["new"] += 1
        C["new_no_hmdb" if not (dr.get("hmdb_id") or c.get("hmdb_id") or mi.get("hmdb")) else "new_with_hmdb"] += 1
print(dict(C))
