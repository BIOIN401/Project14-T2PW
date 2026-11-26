"""Dev-time only: rasterize the art SVGs to transparent PNGs at native size with headless Edge.

usage: raster_art.py ART_DIR OUT_DIR [ids...]
Each SVG is wrapped in an HTML page sized to its native width/height (from _catalog.json)."""
import json
import subprocess
import sys
from pathlib import Path

EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
art, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
prof = Path("C:/t/vis-pw3-work/edge-prof")
cat = json.loads((art / "_catalog.json").read_text())
ids = sys.argv[3:] or list(cat)
for i in ids:
    row = cat[i]
    if row.get("mime") != "image/svg+xml":
        continue
    w, h = (int(round(v)) for v in row["sizes"][0])
    page = out / f"_{i}.html"
    page.write_text(
        f'<!doctype html><html><head><style>html,body{{margin:0;padding:0;background:transparent;overflow:hidden}}'
        f'img{{display:block}}</style></head><body><img src="{(art / row["file"]).as_uri()}" width="{w}" height="{h}"></body></html>')
    png = out / f"{i}.png"
    cmd = [EDGE, "--headless", f"--user-data-dir={prof}", "--disable-gpu", "--hide-scrollbars",
           "--default-background-color=00000000", "--force-device-scale-factor=1",
           f"--window-size={w},{h}", f"--screenshot={png}", page.as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    page.unlink()
    print(i, w, h, png.exists() and png.stat().st_size, r.returncode)
