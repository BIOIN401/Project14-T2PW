"""Read-only: for PathBank rows whose pwc_id != PW_C%06d(id), who owns the derived pwc_id? And structure coverage of our ids."""
import json, sys
sys.path.insert(0, "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/src")
from t2pw.mapping.map_ids import PathBankDbResolver
db = PathBankDbResolver.from_env()
mis = db._query("SELECT id, name, pwc_id FROM compounds WHERE pwc_id <> CONCAT('PW_C', LPAD(id, 6, '0'))", tuple())
print("mismatched rows", len(mis))
owned = db._query("SELECT a.id AS aid, a.name AS aname, a.pwc_id AS apwc, b.id AS bid, b.name AS bname FROM compounds a JOIN compounds b ON b.pwc_id = CONCAT('PW_C', LPAD(a.id, 6, '0')) WHERE a.pwc_id <> CONCAT('PW_C', LPAD(a.id, 6, '0'))", tuple())
print("derived pwc owned by another compound:", len(owned))
for r in owned[:15]:
    print("  ", r["aid"], repr(r["aname"]), r["apwc"], "-> derived owned by", r["bid"], repr(r["bname"]))
mx = db._query("SELECT MAX(id) AS m FROM compounds", tuple())
print("max id", mx)
json.dump({"mismatched": [{k: str(v) for k, v in r.items()} for r in mis], "owned": [{k: str(v) for k, v in r.items()} for r in owned], "max_id": str(mx[0]["m"])},
          open(sys.argv[1], "w", encoding="utf-8"), indent=1)
rows = json.load(open(sys.argv[2], encoding="utf-8"))["rows"]
n = len(rows); s = sum(1 for r in rows.values() if r.get("moldb_smiles")); k = sum(1 for r in rows.values() if r.get("moldb_inchikey"))
print("our ids", n, "with moldb_smiles", s, "with inchikey", k)
