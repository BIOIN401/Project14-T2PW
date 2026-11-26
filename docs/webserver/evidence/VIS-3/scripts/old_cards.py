"""The current result-page image (card grid) from a run's final_mapped.json: old_cards.py OUT.png final_mapped.json"""
import json
import sys

from t2pw.pipeline import draft_graph, draft_graph_render

payload = json.loads(open(sys.argv[2], encoding="utf-8").read())
g = draft_graph.build_draft_graph(payload).to_dict()
open(sys.argv[1], "wb").write(draft_graph_render.render_draft_graph_to_png_bytes(g, dpi=100))
print("ok", sys.argv[1])
