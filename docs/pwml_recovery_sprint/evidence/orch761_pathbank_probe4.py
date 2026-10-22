"""ORCH-761 read-only PathBank probe, pass 4: the column defaults PathWhiz's importer relies on.

Checks that the IsOwned default owner (user_id 11) exists and what species_id 1 is, because a
PWML protein / complex whose species-id is empty falls back to that column default.
Only the id column of users is read. SELECT only; password never read or printed.
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


q("column defaults on proteins / protein_complexes / species", "SELECT table_name, column_name, column_default, is_nullable "
  "FROM information_schema.columns WHERE table_schema = DATABASE() AND table_name IN (%s, %s, %s) "
  "AND column_name IN (%s, %s) ORDER BY table_name, column_name",
  ["proteins", "protein_complexes", "species", "user_id", "species_id"])
q("default owner user 11 exists (id only)", "SELECT id FROM users WHERE id = %s", [11])
q("species id 1 (the default species_id)", "SELECT id, name, taxonomy_id FROM species WHERE id = %s", [1])
q("rows already carrying P0ADI4 / P11454 (what PathWhiz find_by would meet)", "SELECT id, name, uniprot_id, gene_name, species_id "
  "FROM proteins WHERE uniprot_id IN (%s, %s) ORDER BY id", ["P0ADI4", "P11454"])
q("complexes named like the 3-draw wrapper or its stem in E. coli (species 3)", "SELECT id, name, species_id FROM protein_complexes "
  "WHERE species_id = %s AND (LOWER(name) = LOWER(%s) OR LOWER(name) = LOWER(%s))", [3, "enterobactin synthase complex", "enterobactin synthase"])
print("\nDONE")
