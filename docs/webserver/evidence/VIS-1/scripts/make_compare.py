"""Side-by-side page of base vs after renders (and PathWhiz screenshots when present).
usage: make_compare.py AFTER_LABEL  -> C:/t/vis-pw1-work/compare_<label>.html (images referenced by relative path)"""
import html
import sys
from pathlib import Path

W = Path("C:/t/vis-pw1-work")
L = sys.argv[1]
rows = []
for a in sorted((W / "renders" / L).glob("*.png")):
    name = a.name.split("_", 1)[1]
    b = W / "renders" / "base" / f"base_{name}"
    rows.append((f"geometry render: {name}", b, a))
import json


def shots(arm):
    out = {}
    for line in open(W / "pw" / f"{arm}_{L}" / "results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("screenshot"):
            name = r["file"].replace(chr(92), "/").rsplit("/", 1)[-1].split("_", 1)[1]
            out[name] = W / "pw" / f"{arm}_{L}" / r["screenshot"]
    return out


sb, sa = shots("base"), shots("after")
for name in sorted(sa):
    rows.append((f"PathWhiz: {name}", sb.get(name, W / "missing"), sa[name]))
cells = []
for title, b, a in rows:
    rb = b.relative_to(W).as_posix() if b.exists() else ""
    ra = a.relative_to(W).as_posix()
    cells.append(f"<h2>{html.escape(title)}</h2><div class=row><figure><figcaption>base 7bf8588e</figcaption>"
                 f"{'<img src=' + chr(34) + rb + chr(34) + '>' if rb else '(none)'}</figure><figure><figcaption>VIS-1 ({L})</figcaption>"
                 f"<img src=\"{ra}\"></figure></div>")
ref = "".join(f"<figure><figcaption>{p.stem}</figcaption><img src=\"../vis-ref/renders/{p.name}\"></figure>"
              for p in sorted(Path("C:/t/vis-ref/renders").glob("PW*.png")))
page = ("<!doctype html><meta charset=utf-8><title>VIS-1 before and after</title><style>body{font-family:sans-serif;margin:16px}"
        ".row{display:flex;gap:12px}figure{flex:1;margin:0}img{width:100%;border:1px solid #ccc}</style>"
        f"<h1>VIS-1: PathBank edges and flow, base vs {L}</h1>{''.join(cells)}<h2>PathBank reference renders</h2><div class=row>{ref}</div>")
out = W / f"compare_{L}.html"
out.write_text(page, encoding="utf-8")
print(out, len(rows), "pairs")
