"""VIS-9 page: per showcase, PathWhiz's drawing of the VIS-7 PWML (a3/a3b) next to the VIS-9 PWML (PW labels), with the
count of compounds that gained a pwc-id; plus the MolDB contact sheets of every added pwc-id.
usage: make_compare.py CC_SHOW_JSON PW_LABEL [PW_LABEL ...] -> C:/t/vis-pw9-work/compare.html"""
import html, json, sys
from pathlib import Path
W = Path("C:/t/vis-pw9-work"); W7 = Path("C:/t/vis-pw7-work")
cc = json.load(open(sys.argv[1], encoding="utf-8"))
def shots(base, labels):
    out = {}
    for pl in labels:
        f = base / "pw" / pl / "results.jsonl"
        if not f.exists(): continue
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("screenshot") and (base / "pw" / pl / r["screenshot"]).exists():
                out[Path(r["file"].replace("\\", "/")).stem.replace("v5_", "").replace("ref_", "")] = base / "pw" / pl / r["screenshot"]
    return out
old, new = shots(W7, ["a3", "a3b"]), shots(W, sys.argv[2:])
uri = lambda p: p.as_uri() if not str(p).startswith(str(W)) else p.relative_to(W).as_posix()
fig = lambda cap, p: (f"<figure><figcaption>{html.escape(cap)}</figcaption>" + (f"<a href='{uri(p)}'><img loading=lazy src='{uri(p)}'></a>" if p else "<p>(no screenshot)</p>") + "</figure>")
parts = []
for row in cc["rows"]:
    s = row["file"][:-5]
    names = [a["name"] for a in cc["added"] if a["file"] == row["file"]]
    parts.append(f"<h2>{html.escape(s)}: {len(names)} compounds gained a pwc-id</h2><p class=n>{html.escape(', '.join(names))}</p><div class=row>"
                 + fig("VIS-7 PWML in PathWhiz (dev-DB pwc ids: wrong or no pictures)", old.get(s)) + fig("VIS-9 PWML in PathWhiz", new.get(s)) + "</div>")
sheets = "".join(f"<a href='moldb/sheet_{i}.png'><img src='moldb/sheet_{i}.png' style='width:100%'></a>" for i in (1, 2, 3))
page = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'><title>VIS-9 structures</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#111}.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}"
        "figure{margin:0 0 12px}figcaption{font-size:13px;color:#444}img{width:100%;border:1px solid #ccc}.n{font-size:13px;color:#555}</style>"
        "<h1>VIS-9: compound structures in PathWhiz</h1><p>PathWhiz draws a compound from MolDB by its pwc-id only. VIS-9 writes the "
        "PathBank pwc-id for compounds written under their mapped PathBank id when the names agree. Left: VIS-7 PWML; right: VIS-9 PWML; "
        "same patched importer, clean dev DB.</p>" + "".join(parts)
        + "<h2>MolDB pictures of every pwc-id VIS-9 adds or keeps on the showcases (spot check, labelled with our name)</h2>" + sheets
        + (fig("PathBank PW000779 in the same stack", new.get("PW000779")) if new.get("PW000779") else ""))
(W / "compare.html").write_text(page, encoding="utf-8")
print("wrote compare.html", len(parts), "rows", len(new), "new shots", len(old), "old shots")
