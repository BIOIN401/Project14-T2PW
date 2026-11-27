"""One-off audit (dev time, no network): PathBank's MolDB InChIKey for each PathBank id our showcase + corpus IRs carry
(db/pb_rows.json, read-only SELECT) against ChEBI's standard InChIKey for the same compound, from the local ChEBI fill
cache (C:/t/showcase/_work/fillcache, earlier clean-model fill runs). Compares the first InChIKey block (connectivity;
charge / protonation is not in it) and the second (stereo). usage: structaudit.py OUT.json"""
import json, sys
from pathlib import Path
rows = json.load(open("C:/t/vis-pw9-work/db/pb_rows.json", encoding="utf-8"))["rows"]
chebi = {}
for f in Path("C:/t/showcase/_work/fillcache").rglob("*.json"):
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        continue
    v = d.get("value")
    if not isinstance(v, dict) or "/chebi/" not in str(d.get("url")):
        continue
    st = v.get("default_structure") or {}
    ik = st.get("standard_inchi_key") if isinstance(st, dict) else None
    cd = v.get("chemical_data") or {}
    if v.get("id") and ik:
        chebi[str(v["id"])] = {"inchikey": ik, "formula": cd.get("formula"), "name": v.get("name")}
out, n = [], {"ids": len(rows), "with_pb_key_and_chebi": 0, "conn_mismatch": 0, "stereo_mismatch": 0}
for pid, r in sorted(rows.items(), key=lambda x: int(x[0])):
    c = chebi.get(str(r.get("chebi_id") or "").replace("CHEBI:", ""))
    if not (r.get("moldb_inchikey") and c):
        continue
    n["with_pb_key_and_chebi"] += 1
    a = r["moldb_inchikey"].replace("InChIKey=", "").split("-"); b = c["inchikey"].split("-")
    kind = "conn" if a[0] != b[0] else ("stereo" if a[1] != b[1] else None)
    if kind:
        n[kind + "_mismatch"] += 1
        out.append({"id": pid, "pb_name": r["name"], "pb_formula": r.get("moldb_formula"), "pb_key": r["moldb_inchikey"],
                    "chebi": r.get("chebi_id"), "chebi_name": c["name"], "chebi_formula": c["formula"], "chebi_key": c["inchikey"], "kind": kind})
print(n, "chebi records", len(chebi))
for o in out: print(o["kind"], o["id"], repr(o["pb_name"]), o["pb_formula"], "| ChEBI", o["chebi"], repr(o["chebi_name"]), o["chebi_formula"])
json.dump({"summary": n, "mismatches": out}, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
