"""ORCH-761 read-only PathBank probe, pass 2.

pymysql %-formats the statement whenever params is a tuple (even an empty one), so
pass 1 lost every query with a literal LIKE '%...%'. Pass 2 passes all patterns as
parameters. SELECT only, T2PW's own client, password never read or printed.
"""
from __future__ import annotations

import json
import sys

from t2pw.mapping.map_ids import PathBankDbResolver

R = PathBankDbResolver.from_env()
if R is None or not R.available():
    print("DB NOT CONFIGURED / DRIVER MISSING")
    sys.exit(2)


def q(label, sql, params=(), limit=80):
    assert sql.lstrip().upper().startswith("SELECT"), "read-only probe: SELECT only"
    rows = R._query(sql, tuple(params))
    print(f"\n### {label}")
    print("SQL:", " ".join(sql.split()), "| params:", list(params))
    if not rows and R.last_error:
        print("  (error)", str(R.last_error).split("(using password")[0][:300])
        R.last_error = ""
    for row in rows[:limit]:
        print("  ", json.dumps(row, default=str, ensure_ascii=False))
    if len(rows) > limit:
        print(f"   ... {len(rows)} rows total")
    return rows


REAL = "p.uniprot_id IS NOT NULL AND TRIM(p.uniprot_id) <> '' AND p.uniprot_id <> 'Unknown'"
PV_JOIN = ("JOIN pathway_visualizations pv ON pv.pathway_id = pw.id "
           "JOIN protein_locations pl ON pl.pathway_visualization_id = pv.id JOIN proteins p ON p.id = pl.protein_id ")
SPV_JOIN = "LEFT JOIN super_pathway_visualizations spv ON spv.named_for_id = pw.id AND spv.named_for_type = 'Pathway' "

lacto = q("species rows naming Lactococcus", "SELECT id, name, common_name, taxonomy_id, classification FROM species "
          "WHERE name LIKE %s ORDER BY id", ["%Lactococc%"])
ids = [int(r["id"]) for r in lacto]
m = ",".join(["%s"] * len(ids))
q("E. coli species rows", "SELECT id, name, taxonomy_id FROM species WHERE name LIKE %s ORDER BY id LIMIT 30", ["Escherichia coli%"])
q("proteins per Lactococcus species", f"SELECT p.species_id, COUNT(*) AS n, SUM({REAL}) AS real_uniprot FROM proteins p "
  f"WHERE p.species_id IN ({m}) GROUP BY p.species_id", ids)
q("all proteins in Lactococcus species (rows)", "SELECT p.id, p.name, p.gene_name, p.uniprot_id, p.species_id FROM proteins p "
  f"WHERE p.species_id IN ({m}) ORDER BY p.species_id, p.gene_name", ids, limit=200)
q("Men genes / Men enzyme names in Lactococcus species", "SELECT p.id, p.name, p.gene_name, p.uniprot_id, p.species_id FROM proteins p "
  f"WHERE p.species_id IN ({m}) AND (LOWER(p.gene_name) LIKE %s OR p.name LIKE %s OR p.name LIKE %s OR p.name LIKE %s OR p.name LIKE %s)",
  ids + ["men%", "%naphtho%", "%succinyl%", "%menaquinone%", "%isochorismate%"])
q("pathways for Lactococcus species", "SELECT pw.id AS pathway_id, pw.name, pw.species_id, spv.pw_id, spv.smpdb_id FROM pathways pw "
  + SPV_JOIN + f"WHERE pw.species_id IN ({m}) ORDER BY pw.species_id, pw.name", ids, limit=120)

q("Men-pathway enzyme names, any species: count by name", "SELECT p.name, COUNT(*) AS n, SUM(" + REAL + ") AS real_uniprot FROM proteins p "
  "WHERE p.name LIKE %s OR p.name LIKE %s OR p.name LIKE %s OR p.name LIKE %s GROUP BY p.name ORDER BY n DESC LIMIT 30",
  ["%naphthoate%", "%naphthoyl%", "%succinylbenzoate%", "%2-succinyl-%"])
q("Men gene rows, any species: count by gene", "SELECT LOWER(p.gene_name) AS gene, COUNT(*) AS n, SUM(" + REAL + ") AS real_uniprot, "
  "COUNT(DISTINCT p.species_id) AS n_species FROM proteins p WHERE LOWER(p.gene_name) IN (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) GROUP BY LOWER(p.gene_name)",
  ["mena", "menb", "menc", "mend", "mene", "menf", "menh", "meni", "meng", "ubie"])

q("pathways named menaquinone / enterobactin (PW accession via spv.pw_id)",
  "SELECT pw.id AS pathway_id, pw.name, pw.species_id, s.name AS species, s.taxonomy_id, spv.pw_id, spv.smpdb_id "
  "FROM pathways pw LEFT JOIN species s ON s.id = pw.species_id " + SPV_JOIN +
  "WHERE pw.name LIKE %s OR pw.name LIKE %s OR pw.name LIKE %s ORDER BY pw.name, pw.id", ["%enaquinone%", "%nterobactin%", "%nterochelin%"], limit=120)
q("proteins drawn in E. coli (species 3) enterobactin pathways", "SELECT DISTINCT pw.id AS pathway_id, spv.pw_id, p.id AS protein_id, p.name, "
  "p.gene_name, p.uniprot_id, p.species_id FROM pathways pw " + PV_JOIN + SPV_JOIN +
  "WHERE pw.name LIKE %s AND pw.species_id = 3 ORDER BY pw.id, p.gene_name", ["%nterobactin%"])
q("menaquinone pathways: protein identity per pathway", "SELECT pw.id AS pathway_id, pw.name, s.name AS species, s.classification, "
  f"COUNT(DISTINCT p.id) AS n, COUNT(DISTINCT CASE WHEN {REAL} THEN p.id END) AS real_uniprot, "
  "COUNT(DISTINCT CASE WHEN p.uniprot_id = 'Unknown' THEN p.id END) AS unknown FROM pathways pw LEFT JOIN species s ON s.id = pw.species_id "
  + PV_JOIN + "WHERE pw.name LIKE %s GROUP BY pw.id, pw.name, s.name, s.classification ORDER BY s.classification, s.name", ["%enaquinone%"], limit=120)
q("E. coli (species 3) menaquinone pathway proteins", "SELECT DISTINCT pw.id AS pathway_id, spv.pw_id, p.id AS protein_id, p.name, p.gene_name, "
  "p.uniprot_id, p.species_id FROM pathways pw " + PV_JOIN + SPV_JOIN +
  "WHERE pw.name LIKE %s AND pw.species_id = 3 ORDER BY pw.id, p.gene_name", ["%enaquinone%"])
q("Unknown-sentinel proteins drawn in prokaryote pathways: how many pathways", "SELECT COUNT(DISTINCT pw.id) AS pathways_with_unknown, "
  "COUNT(DISTINCT p.id) AS unknown_rows FROM pathways pw JOIN species s ON s.id = pw.species_id " + PV_JOIN +
  "WHERE p.uniprot_id = %s AND s.classification = %s", ["Unknown", "Prokaryote"])
q("protein species vs pathway species in prokaryote menaquinone pathways", "SELECT (p.species_id = pw.species_id) AS same_species, COUNT(DISTINCT p.id) AS n "
  "FROM pathways pw JOIN species s ON s.id = pw.species_id " + PV_JOIN + "WHERE pw.name LIKE %s AND s.classification = %s GROUP BY same_species",
  ["%enaquinone%", "Prokaryote"])
print("\nDONE")
