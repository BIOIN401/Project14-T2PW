"""ORCH-761 read-only PathBank probe, pass 3: proteins drawn in the named reference pathways.

Pass 2's whole-table pathway/protein join timed out, so this goes pathway id ->
pathway_visualizations -> protein_locations -> proteins one pathway at a time.
SELECT only, T2PW's own client, password never read or printed.
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
    return rows


PATHWAYS = {1817: "Enterobactin Biosynthesis, E. coli (562)", 438675: "Menaquinone biosynthesis, E. coli MG1655 (511145)",
            734611: "Menaquinone biosynthesis, B. subtilis (1423)"}
for pid, label in PATHWAYS.items():
    pvs = q(f"pathway_visualizations for pathway {pid} ({label})", "SELECT id, pathway_id FROM pathway_visualizations WHERE pathway_id = %s", [pid])
    for pv in pvs:
        q(f"proteins drawn in pathway {pid} / pv {pv['id']}", "SELECT DISTINCT p.id, p.name, p.gene_name, p.uniprot_id, p.drugbank_id, "
          "p.species_id, s.name AS protein_species FROM protein_locations pl JOIN proteins p ON p.id = pl.protein_id "
          "LEFT JOIN species s ON s.id = p.species_id WHERE pl.pathway_visualization_id = %s ORDER BY p.gene_name", [pv["id"]])
        q(f"protein complexes drawn in pathway {pid} / pv {pv['id']}", "SELECT DISTINCT pc.id, pc.name, pc.species_id, pc.pwp_id "
          "FROM protein_complex_visualizations pcv JOIN protein_complexes pc ON pc.id = pcv.protein_complex_id "
          "WHERE pcv.pathway_visualization_id = %s ORDER BY pc.name", [pv["id"]])
print("\nDONE")
