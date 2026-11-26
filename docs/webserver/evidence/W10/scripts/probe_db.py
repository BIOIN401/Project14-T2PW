"""Read-only probe: counts and sample rows for the PathBank pathway index (W10). Prints no host/user."""
import sys
sys.path.insert(0, "C:/t/ws-search/src")
from dotenv import load_dotenv

load_dotenv(dotenv_path=sys.argv[1], override=False)
from t2pw.bench.eval_v2.pathbank_gold import connect

try:
    conn = connect()
except Exception as exc:  # noqa: BLE001
    print("BLOCKED", type(exc).__name__)
    raise SystemExit(2)
cur = conn.cursor()
sys.stdout.reconfigure(encoding="utf-8")
P = "named_for_type='Pathway' AND private_pathway=0 AND export=1"
Q = [
    f"SELECT parent_id IS NULL pn, parent_id=id self_, COUNT(*) n FROM super_pathway_visualizations WHERE {P} GROUP BY 1,2",
    f"SELECT smpdb_id IS NULL nosmp, COUNT(*) n FROM super_pathway_visualizations WHERE {P} GROUP BY 1",
    f"SELECT cached_name, COUNT(*) n, MIN(pw_id) first FROM super_pathway_visualizations WHERE {P} AND (cached_name LIKE '%%TCA%%' OR cached_name LIKE '%%Citric%%' OR cached_name LIKE 'Glycolysis%%' OR cached_name LIKE '%%nterobactin%%' OR cached_name LIKE '%%Krebs%%') GROUP BY 1",
    "SELECT id, pw_id, smpdb_id, cached_name, cached_subject, cached_species_id, parent_id, model_id, export, LEFT(cached_description,80) d FROM super_pathway_visualizations WHERE pw_id IN ('PW000779','PW000001','PW000017','PW002437')",
    f"SELECT cached_species_id, COUNT(*) n FROM super_pathway_visualizations WHERE {P} GROUP BY 1 ORDER BY 2 DESC LIMIT 8",
    f"SELECT model_id IS NULL nomodel, COUNT(*) n FROM super_pathway_visualizations WHERE {P} GROUP BY 1",
    "SELECT COUNT(*) n FROM super_pathway_visualizations WHERE pwml_markup_file_name IS NOT NULL",
]
for q in Q:
    cur.execute(q)
    print("##", q[:110])
    for r in cur.fetchall()[:40]:
        print("  ", r)
conn.close()
