"""Side-by-side: sbs.py OUT_DIR LEFT_DIR RIGHT_DIR LEFT_TITLE RIGHT_TITLE
Writes OUT_DIR/<stem>.png (left | right, each 1000 px wide) and OUT_DIR/index.html."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

out, left, right = (Path(a) for a in sys.argv[1:4])
lt, rt = sys.argv[4], sys.argv[5]
out.mkdir(parents=True, exist_ok=True)
rows = []
for lp in sorted(left.glob("*.png")):
    rp = right / lp.name
    if not rp.exists():
        continue
    ims = []
    for p in (lp, rp):
        im = Image.open(p).convert("RGB")
        im = im.resize((1000, max(1, round(im.height * 1000 / im.width))), Image.LANCZOS)
        ims.append(im)
    h = max(i.height for i in ims) + 30
    c = Image.new("RGB", (2010, h), "white")
    d = ImageDraw.Draw(c)
    for k, (im, t) in enumerate(zip(ims, (lt, rt))):
        c.paste(im, (k * 1010, 30))
        d.text((k * 1010 + 5, 8), f"{lp.stem}: {t}", fill="black")
    c.save(out / lp.name)
    rows.append(f'<h2>{lp.stem}</h2><div class="r"><figure><figcaption>{lt}</figcaption><img src="{lp.as_uri()}"></figure>'
                f'<figure><figcaption>{rt}</figcaption><img src="{rp.as_uri()}"></figure></div>')
(out / "index.html").write_text(
    "<!doctype html><meta charset=utf-8><title>VIS-3 side by side</title><style>body{font-family:sans-serif;margin:16px}"
    ".r{display:flex;gap:12px}figure{flex:1;margin:0}img{width:100%;border:1px solid #ccc}</style>"
    f"<h1>VIS-3: {lt} vs {rt}</h1>" + "".join(rows), encoding="utf-8")
print(len(rows), "pairs")
