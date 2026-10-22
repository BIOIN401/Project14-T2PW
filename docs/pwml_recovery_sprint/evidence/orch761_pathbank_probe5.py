"""ORCH-761 read-only PathBank probe, pass 5: do the pwp-ids T2PW writes already name other complexes?

writer.py emits pwp-id PW_P{document id:06d} (PW_P040000...) and ProteinComplex#set_pathwhiz_id only
fills a blank pwp_id, so an imported complex keeps whatever the PWML carried. SELECT only.
"""
from __future__ import annotations

import json
import sys

from t2pw.mapping.map_ids import PathBankDbResolver

R = PathBankDbResolver.from_env()
if R is None or not R.available():
    print("DB NOT CONFIGURED / DRIVER MISSING")
    sys.exit(2)


def q(label, sql, params=()):
    assert sql.lstrip().upper().startswith("SELECT"), "read-only probe: SELECT only"
    rows = R._query(sql, tuple(params))
    print(f"\n### {label}")
    print("SQL:", " ".join(sql.split()), "| params:", list(params))
    if not rows and R.last_error:
        print("  (error)", str(R.last_error).split("(using password")[0][:300])
        R.last_error = ""
    for row in rows[:40]:
        print("  ", json.dumps(row, default=str, ensure_ascii=False))
    return rows


ids = [f"PW_P04000{i}" for i in range(6)]
q("existing complexes already carrying the pwp-ids written for document ids 40000-40005", "SELECT id, name, species_id, pwp_id "
  f"FROM protein_complexes WHERE pwp_id IN ({','.join(['%s'] * len(ids))}) ORDER BY pwp_id", ids)
q("unique index on protein_complexes.pwp_id?", "SELECT index_name, non_unique, column_name FROM information_schema.statistics "
  "WHERE table_schema = DATABASE() AND table_name = %s", ["protein_complexes"])
q("duplicate pwp_ids already present", "SELECT COUNT(*) AS duplicated_pwp_ids FROM (SELECT pwp_id FROM protein_complexes "
  "WHERE pwp_id IS NOT NULL AND pwp_id <> %s GROUP BY pwp_id HAVING COUNT(*) > 1) t", [""])
print("\nDONE")
