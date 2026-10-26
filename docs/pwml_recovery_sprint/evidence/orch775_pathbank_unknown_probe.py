"""ORCH-775 read-only PathBank probe: the Unknown protein sentinel, its use as a reaction enzyme,
how PathBank records enzyme-less reactions, and whether the enzymes the papers name exist.

SELECT only. Credentials come from .env through PathBankDbResolver and are never printed.
"""
from __future__ import annotations

import json
import sys
import time

from t2pw.mapping.map_ids import PathBankDbResolver

R = PathBankDbResolver.from_env({"read_timeout": 240})
if R is None or not R.available():
    print("DB NOT CONFIGURED / DRIVER MISSING")
    sys.exit(2)

U = "Unknown"
RX = "Reaction"
PW = "Pathway"


def ph(n):
    return ",".join(["%s"] * n)


def q(label, sql, params=(), show=40):
    assert sql.lstrip().upper().startswith("SELECT"), "read-only probe: SELECT only"
    t0 = time.time()
    rows = R._query(sql, tuple(params))
    print(f"\n### {label}  ({time.time() - t0:.1f}s, {len(rows)} rows)")
    print("SQL:", " ".join(sql.split()), "| params:", list(params))
    if not rows and R.last_error:
        print("  (error)", str(R.last_error).split("(using password")[0][:300])
        R.last_error = ""
    for row in rows[:show]:
        print("  ", json.dumps(row, default=str, ensure_ascii=False))
    return rows


tables = ["reactions", "reaction_enzymes", "protein_complex_proteins", "protein_complexes", "pathway_elements",
          "super_pathway_visualizations", "pathways", "proteins"]
q("schema: columns of the tables used below",
  "SELECT table_name, GROUP_CONCAT(column_name ORDER BY ordinal_position) AS cols FROM information_schema.columns "
  f"WHERE table_schema = DATABASE() AND table_name IN ({ph(len(tables))}) GROUP BY table_name", tables)

# 1. the sentinel record
q("record 9659", "SELECT p.id, p.name, p.uniprot_id, p.drugbank_id, p.gene_name, p.species_id, s.name AS species, "
  "s.taxonomy_id, p.created_at, p.updated_at FROM proteins p LEFT JOIN species s ON s.id = p.species_id WHERE p.id = %s", [9659])
q("species rows for the paper organisms (taxa 3702 Arabidopsis, 9606 human, 562 and 83333 E. coli, 4513 barley, 10090 mouse)",
  f"SELECT id, name, taxonomy_id FROM species WHERE taxonomy_id IN ({ph(6)}) ORDER BY id",
  [3702, 9606, 562, 83333, 4513, 10090])
q("Unknown protein rows in those species", "SELECT p.id, p.name, p.uniprot_id, p.species_id, p.created_at FROM proteins p "
  f"JOIN species s ON s.id = p.species_id WHERE p.name = %s AND s.taxonomy_id IN ({ph(6)}) ORDER BY p.species_id, p.id",
  [U, 3702, 9606, 562, 83333, 4513, 10090])
q("Unknown protein rows overall", "SELECT COUNT(*) AS n_rows, COUNT(DISTINCT species_id) AS n_species, "
  "SUM(uniprot_id = %s) AS uniprot_literal_unknown FROM proteins WHERE name = %s", [U, U])

# 2. complexes
q("protein complexes NAMED Unknown", "SELECT COUNT(*) AS n, COUNT(DISTINCT species_id) AS n_species FROM protein_complexes WHERE name = %s", [U])
q("examples of complexes named Unknown", "SELECT id, name, species_id, pwp_id FROM protein_complexes WHERE name = %s ORDER BY id LIMIT 8", [U])
q("complexes with at least one Unknown component, and with ONLY Unknown components",
  "SELECT COUNT(*) AS complexes_with_unknown, SUM(u = n) AS complexes_only_unknown FROM "
  "(SELECT pcp.protein_complex_id, SUM(p.name = %s) AS u, COUNT(*) AS n FROM protein_complex_proteins pcp "
  "JOIN proteins p ON p.id = pcp.protein_id GROUP BY pcp.protein_complex_id HAVING SUM(p.name = %s) > 0) t", [U, U])
q("complexes that contain record 9659", "SELECT COUNT(DISTINCT protein_complex_id) AS n FROM protein_complex_proteins WHERE protein_id = %s", [9659])
q("examples: complexes containing 9659", "SELECT pc.id, pc.name, pc.species_id, pc.pwp_id FROM protein_complex_proteins pcp "
  "JOIN protein_complexes pc ON pc.id = pcp.protein_complex_id WHERE pcp.protein_id = %s ORDER BY pc.id LIMIT 25", [9659])

# 3. use as a reaction enzyme
q("reactions whose enzyme complex has an Unknown component",
  "SELECT COUNT(DISTINCT re.reaction_id) AS reactions, COUNT(DISTINCT re.protein_complex_id) AS complexes "
  "FROM reaction_enzymes re JOIN protein_complex_proteins pcp ON pcp.protein_complex_id = re.protein_complex_id "
  "JOIN proteins p ON p.id = pcp.protein_id WHERE p.name = %s", [U])
q("reactions whose enzyme complex contains record 9659",
  "SELECT COUNT(DISTINCT re.reaction_id) AS reactions, COUNT(DISTINCT re.protein_complex_id) AS complexes "
  "FROM reaction_enzymes re JOIN protein_complex_proteins pcp ON pcp.protein_complex_id = re.protein_complex_id "
  "WHERE pcp.protein_id = %s", [9659])
q("pathways drawing a reaction catalysed by an Unknown-component complex",
  "SELECT COUNT(DISTINCT pe.pathway_id) AS pathways FROM pathway_elements pe "
  "JOIN reaction_enzymes re ON re.reaction_id = pe.process_id "
  "JOIN protein_complex_proteins pcp ON pcp.protein_complex_id = re.protein_complex_id "
  "JOIN proteins p ON p.id = pcp.protein_id WHERE pe.process_type = %s AND p.name = %s", [RX, U])
q("pathways drawing a reaction catalysed by a complex containing 9659",
  "SELECT COUNT(DISTINCT pe.pathway_id) AS pathways FROM pathway_elements pe "
  "JOIN reaction_enzymes re ON re.reaction_id = pe.process_id "
  "JOIN protein_complex_proteins pcp ON pcp.protein_complex_id = re.protein_complex_id "
  "WHERE pe.process_type = %s AND pcp.protein_id = %s", [RX, 9659])
q("examples (9659): PW id via spv.pw_id, pathway, species, enzyme complex",
  "SELECT DISTINCT spv.pw_id, pw.id AS pathway_id, pw.name AS pathway, s.name AS species, pc.id AS complex_id, pc.name AS complex, re.reaction_id "
  "FROM protein_complex_proteins pcp JOIN protein_complexes pc ON pc.id = pcp.protein_complex_id "
  "JOIN reaction_enzymes re ON re.protein_complex_id = pc.id "
  "JOIN pathway_elements pe ON pe.process_type = %s AND pe.process_id = re.reaction_id "
  "JOIN pathways pw ON pw.id = pe.pathway_id LEFT JOIN species s ON s.id = pw.species_id "
  "LEFT JOIN super_pathway_visualizations spv ON spv.named_for_type = %s AND spv.named_for_id = pw.id "
  "WHERE pcp.protein_id = %s ORDER BY pw.id LIMIT 15", [RX, PW, 9659])
q("examples (any Unknown component, pathway species 1 human, 3 E. coli, 4 Arabidopsis): PW id, pathway, enzyme complex",
  "SELECT DISTINCT spv.pw_id, pw.id AS pathway_id, pw.name AS pathway, s.name AS species, pc.name AS complex, p.id AS unknown_protein_id "
  "FROM proteins p JOIN protein_complex_proteins pcp ON pcp.protein_id = p.id "
  "JOIN protein_complexes pc ON pc.id = pcp.protein_complex_id JOIN reaction_enzymes re ON re.protein_complex_id = pc.id "
  "JOIN pathway_elements pe ON pe.process_type = %s AND pe.process_id = re.reaction_id "
  "JOIN pathways pw ON pw.id = pe.pathway_id LEFT JOIN species s ON s.id = pw.species_id "
  "LEFT JOIN super_pathway_visualizations spv ON spv.named_for_type = %s AND spv.named_for_id = pw.id "
  f"WHERE p.name = %s AND pw.species_id IN ({ph(3)}) ORDER BY pw.id LIMIT 15", [RX, PW, U, 1, 3, 4])
q("most frequent names of Unknown-component enzyme complexes",
  "SELECT pc.name, COUNT(DISTINCT pc.id) AS complexes FROM proteins p JOIN protein_complex_proteins pcp ON pcp.protein_id = p.id "
  "JOIN protein_complexes pc ON pc.id = pcp.protein_complex_id JOIN reaction_enzymes re ON re.protein_complex_id = pc.id "
  "WHERE p.name = %s GROUP BY pc.name ORDER BY complexes DESC LIMIT 15", [U])

# 4. how PathBank records a reaction with no enzyme
q("reactions by spontaneous flag and enzyme presence",
  "SELECT r.spontaneous, (re.reaction_id IS NULL) AS no_enzyme, COUNT(*) AS n FROM reactions r "
  "LEFT JOIN (SELECT DISTINCT reaction_id FROM reaction_enzymes) re ON re.reaction_id = r.id GROUP BY r.spontaneous, (re.reaction_id IS NULL)")

# 5. do the enzymes the papers name exist in PathBank (gene symbols, by organism)
genes_ath = ["GSA1", "GSA2", "HEMB1", "HEMC", "HEME1", "HEME2", "HEMF1", "CPX1", "PPOX", "PPO1"]
q("Arabidopsis tetrapyrrole enzymes by gene_name", "SELECT p.id, p.name, p.gene_name, p.uniprot_id, p.species_id FROM proteins p "
  f"JOIN species s ON s.id = p.species_id WHERE s.taxonomy_id = %s AND p.gene_name IN ({ph(len(genes_ath))}) ORDER BY p.gene_name",
  [3702, *genes_ath])
genes_hsa = ["CERS1", "CERS2", "CERS3", "CERS4", "CERS5", "CERS6", "SGMS1", "SGMS2", "UGCG", "SMPD1", "SMPD2", "SMPD3", "GBA", "GBA1", "GBA2", "LDHA"]
q("human sphingolipid enzymes by gene_name", "SELECT p.id, p.name, p.gene_name, p.uniprot_id, p.species_id FROM proteins p "
  f"WHERE p.species_id = %s AND p.gene_name IN ({ph(len(genes_hsa))}) ORDER BY p.gene_name, p.id",
  [1, *genes_hsa], show=60)
print("\nDONE")
