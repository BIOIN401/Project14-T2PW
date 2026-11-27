import json, sys, collections
C = collections.Counter()
for p in sys.argv[1:]:
    for c in (json.load(open(p, encoding="utf-8")).get("entities") or {}).get("compounds") or []:
        pid = c.get("pathbank_compound_id")
        if not pid: continue
        mi = c.get("mapped_ids") or {}
        st = c.get("db_status")
        C[(st, bool(mi.get("pwc_id")), bool((c.get("db_row") or {}).get("pwc_id")), bool(c.get("pwc_id")))] += 1
        if mi.get("pwc_id") and mi["pwc_id"] != "PW_C%06d" % int(pid): C["held_ne_derived"] += 1
for k, v in sorted(C.items(), key=str): print(k, v)
