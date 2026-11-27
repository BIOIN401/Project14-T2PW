"""VIS-8 side-by-side page: VIS-7 preview (b0), VIS-8 preview (LABEL), VIS-8 drawn by PathWhiz, PathBank reference.
usage: make_compare8.py LABEL PW_LABEL [PW_LABEL ...] -> C:/t/vis-pw8-work/compare.html (relative image paths)"""
import html, json, sys
from pathlib import Path
W = Path("C:/t/vis-pw8-work")
L, PWS = sys.argv[1], sys.argv[2:]
REF = Path("C:/t/vis-pw2-work/refshots")
NOTE = {"ent-item6": ("server job item-6: enterobactin, E. coli (Gram-negative)", "PW000779"),
        "enterobactin": ("enterobactin, E. coli", "PW000779"), "lipid-a": ("lipid A, E. coli", "PW000779"),
        "menaquinone": ("menaquinone, Lactococcus lactis", "PW000779"),
        "coenzyme-a": ("coenzyme A, Bacillus subtilis", "PW000779"),
        "chlorophyll": ("chlorophyll, Arabidopsis (plant)", "PW012911"),
        "ceramide-sphingolipid": ("ceramide / sphingolipid, human (ER, Golgi, lysosome)", "PW000001"),
        "sour-taste": ("sour taste, mouse (cells, vesicle, synapse)", "PW000017"),
        "furochromone": ("furochromone, plant (no compartment art)", "PW012911")}
m8 = json.loads((W / "metrics" / f"m_{L}_all.json").read_text()) if (W / "metrics" / f"m_{L}_all.json").exists() else {}
m7 = json.loads((W / "metrics" / "m_b0_all.json").read_text()) if (W / "metrics" / "m_b0_all.json").exists() else {}
rel = lambda p: p.relative_to(W).as_posix() if str(p).startswith(str(W)) else p.as_uri()
shots = {}
for pl in PWS:
    for line in (W / "pw" / pl / "results.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("screenshot") and (W / "pw" / pl / r["screenshot"]).exists():
            name = Path(r["file"].replace("\\", "/")).stem.replace("v5_", "").replace("ref_", "")
            shots[name] = (pl, W / "pw" / pl / r["screenshot"])
def fig(cap, p):
    return (f"<figure><figcaption>{html.escape(cap)}</figcaption>" + (f"<a href=\"{rel(p)}\"><img loading=lazy src=\"{rel(p)}\"></a>" if p and p.exists() else "<p>(none)</p>") + "</figure>")
def met(m):
    return (f"canvas {m['w']}x{m['h']} (aspect {m['aspect']}), visible edges {m['edges']}, crossings {m['edge_cross']}, "
            f"interaction edges {m['int_n']} total length {m['int_total']} max {m['int_max']}, label at 1100 px wide: {m['label_px_1100']} px") if m else ""
parts = []
for name, (note, ref) in NOTE.items():
    pw = shots.get(name)
    parts.append(f"<h2>{html.escape(name)}: {html.escape(note)}</h2><div class=row>"
                 + fig("VIS-7 (before), our preview renderer. " + met(m7.get(name)), W / "renders" / "b0" / f"{name}.png")
                 + fig(f"VIS-8 ({L}), our preview renderer. " + met(m8.get(name)), W / "renders" / L / f"{name}.png")
                 + fig(f"VIS-8, drawn by PathWhiz ({pw[0]})" if pw else "PathWhiz: no screenshot", pw[1] if pw else None)
                 + fig(f"PathBank reference {ref} (PathBank's own drawing)", REF / f"{ref}.png") + "</div>")
ref_pw = shots.get("PW000779")
page = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'><title>VIS-8 edges and readability</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#111}.row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}"
        "figure{margin:0 0 12px}figcaption{font-size:13px;color:#444;margin-bottom:4px}img{width:100%;border:1px solid #ccc}</style>"
        "<h1>VIS-8: VIS-7 / VIS-8 / VIS-8 in PathWhiz / PathBank</h1>"
        "<p>First two columns: our preview renderer drawing our PWML. Third: PathWhiz's viewer (our patched importer) drawing the imported VIS-8 PWML "
        "(art and compound structures are placeholders in that dev stack). Fourth: PathBank's own drawing of a reference pathway of the same cell type. Click for full size.</p>"
        + (fig("PathBank PW000779 imported into the same PathWhiz stack", ref_pw[1]) if ref_pw else "") + "".join(parts))
(W / "compare.html").write_text(page, encoding="utf-8")
print("wrote", W / "compare.html", len(parts), "rows,", len(shots), "PathWhiz shots")
