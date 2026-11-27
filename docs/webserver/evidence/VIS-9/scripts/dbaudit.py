"""Read-only audit: for each PathBank id we carry, other PathBank rows with the same MolDB InChIKey (a structure shared by
differently-named rows hints that one structure is wrong). usage: dbaudit.py pb_rows.json OUT.json"""
import json, re, sys
sys.path.insert(0, "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/src")
from t2pw.mapping.map_ids import PathBankDbResolver
rows = json.load(open(sys.argv[1], encoding="utf-8"))["rows"]
db = PathBankDbResolver.from_env()
keys = sorted({r["moldb_inchikey"] for r in rows.values() if r.get("moldb_inchikey")})
q = "SELECT id, name, moldb_inchikey, moldb_formula FROM compounds WHERE moldb_inchikey IN (%s)" % ", ".join(["%s"] * len(keys))
share = {}
for r in db._query(q, tuple(keys)):
    share.setdefault(r["moldb_inchikey"], []).append((int(r["id"]), str(r["name"])))
n = lambda s: re.sub(r"[^0-9a-z]", "", s.casefold())
out = []
for pid, r in sorted(rows.items(), key=lambda x: int(x[0])):
    others = [(i, nm) for i, nm in share.get(r.get("moldb_inchikey"), []) if i != int(pid)]
    if others:
        out.append({"id": pid, "name": r["name"], "formula": r.get("moldb_formula"), "same_inchikey_rows": others[:8], "n": len(others)})
for o in out: print(o["id"], repr(o["name"]), o["formula"], o["n"], [x[1] for x in o["same_inchikey_rows"]][:5])
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
