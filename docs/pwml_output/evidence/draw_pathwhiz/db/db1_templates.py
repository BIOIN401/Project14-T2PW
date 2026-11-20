"""Read-only: membrane template usage in PathBank, by year of the visualization's last update; label texts by year.
usage: db1_templates.py OUT.json"""
import json
import sys
import time

from t2pw.bench.eval_v2.pathbank_gold import connect

t0 = time.time()
conn = connect({"read_timeout": 600})
cur = conn.cursor()
out = {}
cur.execute("SELECT id, description, hex_color, partial FROM visualization_templates WHERE visualization_type='MembraneVisualization'")
out["membrane_templates"] = cur.fetchall()
cur.execute("SELECT YEAR(pv.updated_at) y, mv.visualization_template_id t, mv.complete_membrane c, COUNT(*) n, "
            "COUNT(DISTINCT mv.pathway_visualization_id) npv FROM membrane_visualizations mv "
            "JOIN pathway_visualizations pv ON pv.id=mv.pathway_visualization_id GROUP BY y, t, c")
out["by_year_template"] = cur.fetchall()
print("membranes", round(time.time() - t0, 1), flush=True)
cur.execute("SELECT YEAR(updated_at) y, COUNT(*) n FROM pathway_visualizations GROUP BY y")
out["pv_by_year"] = cur.fetchall()
cur.execute("SELECT label_text, COUNT(*) n FROM label_locations GROUP BY label_text ORDER BY n DESC LIMIT 150")
out["top_labels"] = cur.fetchall()
print("labels", round(time.time() - t0, 1), flush=True)
cur.execute("SELECT d.id, d.name, COUNT(*) n FROM drawable_element_locations e JOIN drawable_images d ON d.id=e.drawable_image_id "
            "GROUP BY d.id, d.name ORDER BY n DESC LIMIT 60")
out["top_drawables"] = cur.fetchall()
conn.close()
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
print("done", round(time.time() - t0, 1))
