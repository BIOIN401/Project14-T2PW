"""Write the side-by-side page. usage: sbs.py <round>  -> C:/t/vis-pw4-work/sbs-<round>.html (no rendering)"""
import sys, pathlib
W = pathlib.Path("C:/t/vis-pw4-work")
r = sys.argv[1]
REF = pathlib.Path("C:/t/vis-pw3-work/ref")
rows = []
for pw in ("PW000001", "PW000017", "PW000779", "PW012911", "PW127049"):
    rows.append((pw, [("PathBank (its SVG, rendered by Edge)", REF / f"{pw}.png"),
                      ("VIS-3 render (before)", W / "r0" / f"{pw}.png"), (f"VIS-4 render ({r})", W / r / f"{pw}.png")]))
for name in ("ent-item6-final2", "ent-item6-base", "chlorophyll", "lipid-a", "ceramide-sphingolipid",
             "coenzyme-a", "menaquinone", "furochromone", "sour-taste", "enterobactin"):
    rows.append((name, [("VIS-3 render (before)", W / "r0" / f"{name}.png"), (f"VIS-4 render ({r})", W / r / f"{name}.png")]))
html = ["<!doctype html><meta charset=utf-8><title>VIS-4 side by side</title>",
        "<style>body{font:14px sans-serif;margin:16px}figure{display:inline-block;vertical-align:top;margin:0 8px 24px 0;"
        "width:32%}img{width:100%;border:1px solid #999}h2{margin:24px 0 8px}</style>",
        f"<h1>VIS-4: reference vs ours ({r})</h1><p>Click an image for full size.</p>"]
for name, cells in rows:
    html.append(f"<h2>{name}</h2>")
    for cap, p in cells:
        u = p.as_uri()
        html.append(f"<figure><a href='{u}'><img src='{u}' loading=lazy></a><figcaption>{cap}</figcaption></figure>")
(W / f"sbs-{r}.html").write_text("\n".join(html), encoding="utf-8")
print(W / f"sbs-{r}.html")
