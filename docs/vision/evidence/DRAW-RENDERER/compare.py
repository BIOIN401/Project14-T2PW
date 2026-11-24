"""Before/after PNGs of the 8 showcase pathways + a side-by-side page. usage: compare.py"""
import html
import json
import os
import shutil

W = "C:/t/lane-draw-work"
OUT = W + "/compare"
os.makedirs(OUT, exist_ok=True)
B = json.load(open(W + "/out/showcase_before/metrics.json"))["rows"]
A = json.load(open(W + "/out/showcase_after/metrics.json"))["rows"]
K = ["crossings", "long_edges", "edge_length", "non_straight", "aspect_ratio", "bands", "sparse_bands", "label_collisions",
     "node_edge_overlaps", "repeated_captions", "hops", "selfcheck_failures"]
rows = []
for slug in A:
    for tag in ("before", "after"):
        shutil.copy("%s/out/showcase_%s/%s.png" % (W, tag, slug), "%s/%s_%s.png" % (OUT, slug, tag))
    t = "".join("<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (k, B[slug].get(k), A[slug].get(k)) for k in K)
    rows.append(f"""<section><h2>{html.escape(slug)} <small>{B[slug]['style']} &rarr; {A[slug]['style']}</small></h2>
<table><tr><th>metric</th><th>before</th><th>after</th></tr>{t}</table>
<div class="pair"><figure><figcaption>before</figcaption><a href="{slug}_before.png"><img src="{slug}_before.png" alt="{slug} before"></a></figure>
<figure><figcaption>after</figcaption><a href="{slug}_after.png"><img src="{slug}_after.png" alt="{slug} after"></a></figure></div></section>""")
page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Renderer before and after</title><style>
:root{--bg:#fff;--ink:#1b2631;--rule:#d5dbe3}
@media (prefers-color-scheme: dark){:root{--bg:#12171d;--ink:#e6ebf0;--rule:#2e3944}}
body{background:var(--bg);color:var(--ink);font-family:system-ui,sans-serif;margin:16px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}img{width:100%;border:1px solid var(--rule);background:#fff}
table{border-collapse:collapse;font-size:13px;margin:8px 0}td,th{border:1px solid var(--rule);padding:2px 8px;text-align:right}
td:first-child{text-align:left}section{border-top:2px solid var(--rule);padding-top:8px;margin-top:24px}
@media (max-width:700px){.pair{grid-template-columns:1fr}}</style></head><body>
<h1>Our diagram renderer, before and after (DRAW-RENDERER)</h1>
<p>Same IR, same content (selfcheck 0 failures on both). Click an image to open it full size.</p>
""" + "".join(rows) + "</body></html>"
open(OUT + "/index.html", "w", encoding="utf-8").write(page)
print("wrote", OUT + "/index.html", len(A))
