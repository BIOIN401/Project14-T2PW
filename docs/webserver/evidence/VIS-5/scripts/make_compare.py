"""VIS-5 side-by-side page: before (base f3241270) / after (LABEL) previews, PathWhiz's own drawing of the after PWML
(checkpoint PW_LABEL) and the PathBank reference of the same cell type.
usage: make_compare.py LABEL PW_LABEL [PW_LABEL ...] -> C:/t/vis-pw5-work/compare.html (relative image paths)"""
import html
import json
import sys
from pathlib import Path

W = Path("C:/t/vis-pw5-work")
L, PWS = sys.argv[1], sys.argv[2:]
REF = Path("C:/t/vis-pw2-work/refshots")  # PathBank's own SVGs of the reference pathways, screenshotted (VIS-1)
NOTE = {"ent-item6": ("server job item-6: enterobactin, E. coli (Gram-negative)", "PW000779"),
        "enterobactin": ("enterobactin, E. coli", "PW000779"), "lipid-a": ("lipid A, E. coli", "PW000779"),
        "menaquinone": ("menaquinone, Lactococcus lactis", "PW000779"),
        "coenzyme-a": ("coenzyme A, Bacillus subtilis", "PW000779"),
        "chlorophyll": ("chlorophyll, Arabidopsis (plant)", "PW012911"),
        "ceramide-sphingolipid": ("ceramide / sphingolipid, human (ER, Golgi, lysosome)", "PW000001"),
        "sour-taste": ("sour taste, mouse (cells, vesicle, synapse)", "PW000017"),
        "furochromone": ("furochromone, plant (no compartment art)", "PW012911")}


def rel(p: Path) -> str:
    try:
        return p.relative_to(W).as_posix()
    except ValueError:
        return p.as_uri()


shots = {}  # pathway -> (checkpoint, png)
for pl in PWS:
    for line in (W / "pw" / pl / "results.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("screenshot") and (W / "pw" / pl / r["screenshot"]).exists():
            name = Path(r["file"].replace("\\", "/")).stem.replace("v5_", "").replace("ref_", "")
            shots[name] = (pl, W / "pw" / pl / r["screenshot"])


def fig(cap, p):
    return (f"<figure><figcaption>{html.escape(cap)}</figcaption>"
            + (f"<a href=\"{rel(p)}\"><img loading=lazy src=\"{rel(p)}\"></a>" if p and p.exists() else "<p>(none)</p>")
            + "</figure>")


parts = []
for name, (note, ref) in NOTE.items():
    pw = shots.get(name)
    parts.append(f"<h2>{html.escape(name)}: {html.escape(note)}</h2><div class=row>"
                 + fig("before: base f3241270, our preview renderer", W / "renders" / "base" / f"{name}.png")
                 + fig(f"after: {L}, our preview renderer", W / "renders" / L / f"{name}.png")
                 + fig(f"after, drawn by PathWhiz ({pw[0]})" if pw else "PathWhiz: not in a checkpoint", pw[1] if pw else None)
                 + fig(f"PathBank reference {ref} (PathBank's own drawing)", REF / f"{ref}.png") + "</div>")
ref_pw = shots.get("PW000779")
page = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'><title>VIS-5 layout</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#111}"
        ".row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}figure{margin:0 0 12px}"
        "figcaption{font-size:13px;color:#444;margin-bottom:4px}img{width:100%;border:1px solid #ccc}</style>"
        f"<h1>VIS-5: our PWML laid out like PathBank's (before / after / PathWhiz / PathBank)</h1>"
        "<p>Before and after columns are our preview renderer drawing our PWML; the PathWhiz column is PathWhiz's own viewer "
        "drawing the imported after PWML (drawable art and compound structures are placeholders in that dev stack). "
        "Click an image for full size.</p>"
        + (fig("PathBank PW000779 imported into the same PathWhiz stack (reference for the PathWhiz column)", ref_pw[1]) if ref_pw else "")
        + "".join(parts))
(W / "compare.html").write_text(page, encoding="utf-8")
print("wrote", W / "compare.html", len(parts), "rows,", len(shots), "PathWhiz shots")
