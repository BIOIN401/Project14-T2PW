"""Our written pubchem-cid vs PathBank's pubchem_cid for the same PathBank compound id (dbcheck rows). Read-only evaluation."""
import json, sys, collections
rows = json.load(open(sys.argv[1], encoding="utf-8"))["rows"]
C = collections.Counter(); diff = []
for p in sys.argv[2:]:
    for c in (json.load(open(p, encoding="utf-8")).get("entities") or {}).get("compounds") or []:
        pid = c.get("pathbank_compound_id")
        dr = c.get("db_row") or {}
        ours = dr.get("pubchem_cid") or c.get("pubchem_cid") or (c.get("mapped_ids") or {}).get("pubchem")
        if not pid or not ours: continue
        t = (rows.get(str(pid)) or {}).get("pubchem_cid")
        k = "same" if str(t) == str(ours) else ("pb_none" if not t else "differs")
        C[k] += 1
        if k == "differs": diff.append((p.replace("\\", "/").split("/")[-2], pid, c.get("name"), ours, t))
print(dict(C)); [print("  ", d) for d in diff]
