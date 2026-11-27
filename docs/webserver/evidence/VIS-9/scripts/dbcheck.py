"""Read-only measurement (SELECT only, repo helper): how often PathBank's compounds.pwc_id equals PW_C%06d(id),
and the real pwc_id / name / structure fields for every PathBank compound id our showcase + corpus IRs carry.
Usage: dbcheck.py OUT.json IR..."""
import json, sys
sys.path.insert(0, "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/src")
from t2pw.mapping.map_ids import PathBankDbResolver
out = sys.argv[1]
ids = set()
for p in sys.argv[2:]:
    ir = json.load(open(p, encoding="utf-8"))
    for c in (ir.get("entities") or {}).get("compounds") or []:
        v = c.get("pathbank_compound_id") or c.get("pw_compound_id")
        if v:
            ids.add(int(v))
db = PathBankDbResolver.from_env()
if db is None:
    print("NO_DB_CONFIG"); sys.exit(2)
cols = [r.get("Field") for r in db._query("SHOW COLUMNS FROM compounds", tuple())]
print("columns", cols)
tot = db._query("SELECT COUNT(*) AS n, SUM(pwc_id = CONCAT('PW_C', LPAD(id, 6, '0'))) AS eq, SUM(pwc_id IS NULL OR pwc_id = '') AS nopwc FROM compounds", tuple())
print("table", tot)
want = [c for c in ("id", "name", "pwc_id", "hmdb_id", "kegg_id", "chebi_id", "pubchem_cid", "moldb_smiles", "moldb_formula", "moldb_inchikey") if c in cols]
rows = {}
idl = sorted(ids)
for i in range(0, len(idl), 200):
    chunk = idl[i:i + 200]
    q = "SELECT %s FROM compounds WHERE id IN (%s)" % (", ".join(want), ", ".join(["%s"] * len(chunk)))
    for r in db._query(q, tuple(chunk)):
        rows[str(r["id"])] = {k: (None if r.get(k) is None else str(r.get(k))) for k in want}
print("ids asked", len(idl), "rows", len(rows))
mism = [r for r in rows.values() if r.get("pwc_id") != "PW_C%06d" % int(r["id"])]
print("our ids with pwc != derived:", len(mism), [(r["id"], r["pwc_id"], r["name"]) for r in mism][:20])
json.dump({"columns": cols, "table": [{k: str(v) for k, v in t.items()} for t in tot], "rows": rows}, open(out, "w", encoding="utf-8"), indent=1)
