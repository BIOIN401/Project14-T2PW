"""Survey compound structure keys in IRs: pathbank id present, db_row pwc_id vs PW_C%06d(id), hmdb, status."""
import json, sys, glob, collections
paths = sys.argv[1:]
C = collections.Counter()
mism = []
for p in paths:
    ir = json.load(open(p, encoding="utf-8"))
    for c in (ir.get("entities") or {}).get("compounds") or []:
        C["compounds"] += 1
        pid = c.get("pathbank_compound_id") or c.get("pw_compound_id")
        st = c.get("db_status")
        C["status:" + str(st)] += 1
        if pid:
            C["with_pb_id"] += 1
            if not c.get("hmdb_id") and not (c.get("db_row") or {}).get("hmdb_id") and not (c.get("mapped_ids") or {}).get("hmdb"):
                C["with_pb_id_no_hmdb"] += 1
        else:
            C["no_pb_id"] += 1
            if c.get("chebi_id") or (c.get("mapped_ids") or {}).get("chebi"):
                C["no_pb_id_has_chebi"] += 1
            if c.get("pubchem_cid") or (c.get("mapped_ids") or {}).get("pubchem"):
                C["no_pb_id_has_cid"] += 1
            if c.get("kegg_id") or (c.get("mapped_ids") or {}).get("kegg"):
                C["no_pb_id_has_kegg"] += 1
        def rows(c):
            dr = c.get("db_row")
            if isinstance(dr, dict) and dr: yield dr
            dm = c.get("db_match") or {}
            for cand in dm.get("candidates") or []:
                if isinstance(cand.get("db_row"), dict): yield cand["db_row"]
        for dr in rows(c):
            if dr.get("id") and dr.get("pwc_id"):
                C["db_rows_with_pwc"] += 1
                if dr["pwc_id"] != "PW_C%06d" % int(dr["id"]):
                    mism.append((p, dr["id"], dr["pwc_id"]))
print(dict(C))
print("pwc mismatches", len(mism), mism[:10])
