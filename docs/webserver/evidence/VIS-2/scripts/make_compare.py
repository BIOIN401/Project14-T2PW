"""Side-by-side page: VIS-1 (base, 49ab4da8) vs VIS-2 renders with the VIS-3 PathBank-style renderer, the PathBank
reference shots, and PathWhiz screenshots when present.
usage: make_compare.py LABEL [PW_LABEL] -> C:/t/vis-pw2-work/compare_<LABEL>.html (images by relative path)"""
import html
import sys
from pathlib import Path

W = Path("C:/t/vis-pw2-work")
L = sys.argv[1]
P = sys.argv[2] if len(sys.argv) > 2 else None
NOTE = {"ent-item6": "server item-6, enterobactin, E. coli (Gram-negative)", "enterobactin": "showcase, E. coli",
        "lipid-a": "showcase, E. coli", "menaquinone": "showcase, Lactococcus lactis (Gram-positive)",
        "coenzyme-a": "showcase, Bacillus subtilis", "chlorophyll": "showcase, Arabidopsis (plant)",
        "ceramide-sphingolipid": "showcase, human (ER)", "7": "replay 7, chlorophyll (plant)", "79": "replay 79, cholesterol (ER in cell)",
        "132": "replay 132, sialic acid (nucleus)", "134": "replay 134, carnitine (mitochondrion)", "114": "replay 114, cell wall bacterium"}


def rel(p):
    return p.relative_to(W).as_posix()


rows = []
for name in NOTE:
    a, b = W / "renders" / L / f"{name}.png", W / "renders" / "base" / f"{name}.png"
    if a.exists():
        rows.append((f"{name}: {NOTE[name]}", b, a))
cells = []
for title, b, a in rows:
    cells.append(f"<h2>{html.escape(title)}</h2><div class=row><figure><figcaption>before (VIS-1, 49ab4da8)</figcaption>"
                 f"{('<img src=' + chr(34) + rel(b) + chr(34) + '>') if b.exists() else '(none)'}</figure>"
                 f"<figure><figcaption>after (VIS-2, {L})</figcaption><img src=\"{rel(a)}\"></figure></div>")
ref = "".join(f"<figure><figcaption>{p.stem}</figcaption><img src=\"{rel(p)}\"></figure>" for p in sorted((W / "refshots").glob("PW*.png")))
pw = ""
if P:
    shots = sorted((W / "pw" / f"after_{P}" / "screenshots").glob("*.png"))
    pw = "<h2>PathWhiz (t2pw-pwdraw harness) drawing of the imported after-arm files</h2>" + "".join(
        f"<figure><figcaption>{p.name}</figcaption><img src=\"{rel(p)}\"></figure>" for p in shots)
page = ("<!doctype html><meta charset=utf-8><title>VIS-2 cell art</title><style>body{font-family:sans-serif;margin:16px}"
        ".row{display:flex;gap:12px}figure{flex:1;margin:0 0 12px}img{width:100%;border:1px solid #ccc}</style>"
        f"<h1>VIS-2: PathBank cell and organelle art, before vs after ({L})</h1>"
        f"<h2>PathBank references (PW000779 Gram-negative, PW000001 animal, PW012911 plant, PW000017)</h2><div class=row>{ref}</div>"
        f"{''.join(cells)}{pw}")
out = W / f"compare_{L}.html"
out.write_text(page, encoding="utf-8")
print("wrote", out, len(rows), "rows")
