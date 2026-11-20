"""Read-only: global membrane template counts by year the membrane row was created (single-table scan, no join).
usage: db4_global.py OUT.json"""
import json
import sys
import time

from t2pw.bench.eval_v2.pathbank_gold import connect

t0 = time.time()
conn = connect({"read_timeout": 900})
cur = conn.cursor()
cur.execute("SELECT visualization_template_id t, YEAR(created_at) y, COUNT(*) n, SUM(complete_membrane) closed "
            "FROM membrane_visualizations GROUP BY t, y")
rows = cur.fetchall()
conn.close()
json.dump(rows, open(sys.argv[1], "w"), default=str, indent=0)
print("rows", len(rows), round(time.time() - t0, 1))
