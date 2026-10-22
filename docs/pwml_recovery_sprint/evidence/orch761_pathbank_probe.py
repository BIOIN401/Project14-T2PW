"""ORCH-761 read-only PathBank probe.

Reuses T2PW's own client (PathBankDbResolver.from_env -> _query), so the
connection settings and credentials are exactly the ones the pipeline uses. Every
statement is a SELECT; q() refuses anything else before it reaches the driver.
The password is never read by this script and never printed.

Run from the primary checkout root, through the bounded wrapper (see ORCH-761-RESULT.md).
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
    for row in rows[:80]:
        print("  ", json.dumps(row, default=str, ensure_ascii=False))
    if len(rows) > 80:
        print(f"   ... {len(rows)} rows total")
    return rows


def species_sql():
    return ("SELECT id, name, common_name, taxonomy_id FROM species WHERE LOWER(name)=LOWER(%s) "
            "OR LOWER(common_name)=LOWER(%s) OR taxonomy_id=%s OR LOWER(name) LIKE LOWER(%s) "
            "OR LOWER(common_name) LIKE LOWER(%s) LIMIT 40")


REAL = "p.uniprot_id IS NOT NULL AND TRIM(p.uniprot_id) <> '' AND p.uniprot_id <> 'Unknown'"
DRUG = "p.drugbank_id IS NOT NULL AND TRIM(p.drugbank_id) <> ''"
NEITHER = "(p.uniprot_id IS NULL OR TRIM(p.uniprot_id) = '') AND (p.drugbank_id IS NULL OR TRIM(p.drugbank_id) = '')"

q("proteins columns", "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns "
  "WHERE table_schema = DATABASE() AND table_name = 'proteins' ORDER BY ordinal_position")
q("protein identity coverage (all rows)",
  f"SELECT COUNT(*) AS total, SUM({REAL}) AS real_uniprot, SUM(p.uniprot_id = 'Unknown') AS uniprot_literal_unknown, "
  f"SUM({DRUG}) AS has_drugbank, SUM({NEITHER}) AS has_neither, SUM(p.name = 'Unknown') AS name_unknown FROM proteins p")
q("protein identity coverage, proteins drawn in any pathway (protein_locations)",
  f"SELECT COUNT(DISTINCT p.id) AS total, COUNT(DISTINCT CASE WHEN {REAL} THEN p.id END) AS real_uniprot, "
  f"COUNT(DISTINCT CASE WHEN {DRUG} THEN p.id END) AS has_drugbank, "
  "COUNT(DISTINCT CASE WHEN p.uniprot_id = 'Unknown' THEN p.id END) AS unknown_sentinel, "
  f"COUNT(DISTINCT CASE WHEN {NEITHER} THEN p.id END) AS has_neither "
  "FROM proteins p JOIN protein_locations pl ON pl.protein_id = p.id")
q("identity coverage by species classification",
  f"SELECT s.classification, COUNT(*) AS total, SUM({REAL}) AS real_uniprot, SUM(p.uniprot_id = 'Unknown') AS unknown_sentinel, "
  f"SUM({DRUG}) AS has_drugbank, SUM({NEITHER}) AS has_neither "
  "FROM proteins p LEFT JOIN species s ON s.id = p.species_id GROUP BY s.classification")
q("uniprot_id strings that are not accession-shaped",
  "SELECT uniprot_id, COUNT(*) AS n FROM proteins WHERE uniprot_id IS NOT NULL AND uniprot_id <> '' "
  "AND uniprot_id NOT REGEXP '^[A-Z0-9]{6}([A-Z0-9]{4})?$' GROUP BY uniprot_id ORDER BY n DESC LIMIT 25")
q("drugbank_id sample", f"SELECT p.id, p.name, p.uniprot_id, p.drugbank_id, p.species_id FROM proteins p WHERE {DRUG} LIMIT 10")
q("rows with neither id (sample)", "SELECT p.id, p.name, p.uniprot_id, p.drugbank_id, p.gene_name, p.species_id, s.name AS species "
  f"FROM proteins p LEFT JOIN species s ON s.id = p.species_id WHERE {NEITHER} LIMIT 15")
q("accessions carried by more than one protein row", "SELECT COUNT(*) AS accessions_on_more_than_one_row, SUM(n) AS rows_involved FROM ("
  "SELECT uniprot_id, COUNT(*) AS n FROM proteins WHERE uniprot_id IS NOT NULL AND uniprot_id <> '' AND uniprot_id <> 'Unknown' "
  "GROUP BY uniprot_id HAVING COUNT(*) > 1) t")
q("Unknown sentinel count by species", "SELECT p.species_id, s.name, COUNT(*) AS n FROM proteins p "
  "LEFT JOIN species s ON s.id = p.species_id WHERE p.name = 'Unknown' GROUP BY p.species_id, s.name ORDER BY n DESC LIMIT 15")

q("species rows: Lactococcus, lactis, coli, plantarum, and the taxa UniProt uses",
  "SELECT id, name, common_name, taxonomy_id, classification FROM species "
  "WHERE name LIKE '%Lactococc%' OR name LIKE '%lactis%' OR name LIKE '%coli%' OR name LIKE '%plantarum%' "
  "OR taxonomy_id IN ('1358','1359','1360','684738','272623','416870','562','83333','511145','316407','1590') ORDER BY id")
q("species count by classification", "SELECT classification, COUNT(*) AS n FROM species GROUP BY classification")
q("T2PW _find_species_ids SQL (map_ids.py:1004-1015) for the KF147 organism string", species_sql(),
  ["Lactococcus lactis subsp. lactis KF147"] * 3 + ["%Lactococcus lactis subsp. lactis KF147%"] * 2)
q("same SQL for the canonicalised name Lactococcus lactis", species_sql(),
  ["Lactococcus lactis"] * 3 + ["%Lactococcus lactis%"] * 2)
q("same SQL for taxonomy id 684738 (what the species row carries)", species_sql(), ["684738"] * 3 + ["%684738%"] * 2)
q("proteins whose species is Lactococcus", "SELECT s.id AS species_id, s.name, s.taxonomy_id, COUNT(p.id) AS n_proteins, "
  f"SUM({REAL}) AS real_uniprot FROM species s LEFT JOIN proteins p ON p.species_id = s.id "
  "WHERE s.name LIKE '%Lactococc%' GROUP BY s.id, s.name, s.taxonomy_id")

MEN = ["mena", "menb", "menc", "mend", "mene", "menf", "menh", "meni", "meng", "ubie", "menx", "yhdb", "yhda", "prea"]
ENT = ["enta", "entb", "entc", "entd", "ente", "entf", "entg", "enth", "ents", "fepa", "fes", "tolc"]


def marks(n):
    return ",".join(["%s"] * n)


q("Men genes, any species", "SELECT p.id, p.name, p.gene_name, p.uniprot_id, p.drugbank_id, p.species_id, s.name AS species, s.taxonomy_id "
  f"FROM proteins p LEFT JOIN species s ON s.id = p.species_id WHERE LOWER(p.gene_name) IN ({marks(len(MEN))}) "
  "ORDER BY s.name, p.gene_name", MEN)
q("Men-pathway enzyme names, any species", "SELECT p.id, p.name, p.gene_name, p.uniprot_id, p.species_id, s.name AS species "
  "FROM proteins p LEFT JOIN species s ON s.id = p.species_id WHERE p.name LIKE '%naphthoate%' OR p.name LIKE '%naphthoyl%' "
  "OR p.name LIKE '%succinylbenzoate%' OR p.name LIKE '%menaquinone%' OR p.name LIKE '%2-succinyl-5-enolpyruvyl%' "
  "OR p.name LIKE '%2-succinyl-6-hydroxy%' ORDER BY s.name, p.name LIMIT 200")
q("Ent genes, any species", "SELECT p.id, p.name, p.gene_name, p.uniprot_id, p.drugbank_id, p.species_id, s.name AS species, s.taxonomy_id "
  f"FROM proteins p LEFT JOIN species s ON s.id = p.species_id WHERE LOWER(p.gene_name) IN ({marks(len(ENT))}) "
  "ORDER BY s.name, p.gene_name LIMIT 200", ENT)
ACC = ["P15047", "P0ADI4", "P0AEJ2", "P19925", "P10378", "P11454", "P38051", "P02930",
       "Q9CHJ9", "Q9CHK0", "Q9CHK1", "Q9CHK2", "Q9CHK3", "Q9CHK4", "Q9CHK5", "P49016", "Q9CJ25",
       "A2RM72", "A2RM73", "A2RM74", "A2RM75", "A2RM76", "A2RM77", "A2RHR6", "A2RJA5"]
q("rows carrying the accessions UniProt REST returned", "SELECT p.id, p.name, p.gene_name, p.uniprot_id, p.species_id, s.name AS species "
  f"FROM proteins p LEFT JOIN species s ON s.id = p.species_id WHERE p.uniprot_id IN ({marks(len(ACC))}) ORDER BY p.uniprot_id", ACC)

PV_JOIN = ("JOIN pathway_visualizations pv ON pv.pathway_id = pw.id "
           "JOIN protein_locations pl ON pl.pathway_visualization_id = pv.id JOIN proteins p ON p.id = pl.protein_id ")
SPV_JOIN = "LEFT JOIN super_pathway_visualizations spv ON spv.named_for_id = pw.id AND spv.named_for_type = 'Pathway' "

q("pathways named menaquinone / enterobactin (PW accession via spv.pw_id)",
  "SELECT pw.id AS pathway_id, pw.name, pw.species_id, s.name AS species, s.taxonomy_id, spv.id AS spv_id, spv.pw_id, spv.smpdb_id "
  "FROM pathways pw LEFT JOIN species s ON s.id = pw.species_id " + SPV_JOIN +
  "WHERE pw.name LIKE '%enaquinone%' OR pw.name LIKE '%nterobactin%' OR pw.name LIKE '%nterochelin%' ORDER BY pw.name LIMIT 60")
q("proteins drawn in E. coli enterobactin pathways", "SELECT DISTINCT pw.id AS pathway_id, spv.pw_id, p.id AS protein_id, p.name, p.gene_name, "
  "p.uniprot_id, p.species_id FROM pathways pw " + PV_JOIN + SPV_JOIN +
  "WHERE pw.name LIKE '%nterobactin%' AND pw.species_id = 3 ORDER BY pw.id, p.gene_name LIMIT 80")
q("menaquinone pathways: protein identity per pathway", "SELECT pw.id AS pathway_id, pw.name, s.name AS species, COUNT(DISTINCT p.id) AS n, "
  f"COUNT(DISTINCT CASE WHEN {REAL} THEN p.id END) AS real_uniprot, COUNT(DISTINCT CASE WHEN p.uniprot_id = 'Unknown' THEN p.id END) AS unknown "
  "FROM pathways pw LEFT JOIN species s ON s.id = pw.species_id " + PV_JOIN +
  "WHERE pw.name LIKE '%enaquinone%' GROUP BY pw.id, pw.name, s.name ORDER BY s.name LIMIT 60")
q("menaquinone pathway proteins in prokaryote pathways (rows)", "SELECT DISTINCT pw.id AS pathway_id, s.name AS species, p.id AS protein_id, "
  "p.name, p.gene_name, p.uniprot_id, p.species_id AS protein_species_id FROM pathways pw LEFT JOIN species s ON s.id = pw.species_id "
  + PV_JOIN + "WHERE pw.name LIKE '%enaquinone%' AND s.classification = 'Prokaryote' ORDER BY s.name, pw.id, p.gene_name LIMIT 120")

for cid in (3623, 1143, 1189, 1190, 3929):
    q(f"protein_complex {cid} with components", "SELECT pc.id, pc.name, pc.species_id, pc.pwp_id, pcp.protein_id, pcp.stoichiometry, "
      "p.name AS protein_name, p.gene_name, p.uniprot_id FROM protein_complexes pc "
      "LEFT JOIN protein_complex_proteins pcp ON pcp.protein_complex_id = pc.id LEFT JOIN proteins p ON p.id = pcp.protein_id "
      "WHERE pc.id = %s", [cid])
print("\nDONE")
