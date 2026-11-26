"""Render PathBank's own SVGs with headless Edge: edge_ref.py OUT_DIR svg..."""
import re
import subprocess
import sys
from pathlib import Path

EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for f in sys.argv[2:]:
    p = Path(f).resolve()
    head = p.read_text(encoding="utf-8", errors="ignore")[:4000]
    m = re.search(r'<svg[^>]*?width="([\d.]+)"[^>]*?height="([\d.]+)"', head, re.S)
    w, h = (int(float(v)) for v in m.groups())
    s = min(1.0, 1400 / max(w, h))
    png = out / (p.stem + ".png")
    cmd = [EDGE, "--headless", "--user-data-dir=C:/t/vis-pw3-work/edge-prof", "--disable-gpu", "--hide-scrollbars",
           f"--force-device-scale-factor={s:.4f}", f"--window-size={w},{h}", f"--screenshot={png}", p.as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    print(p.stem, w, h, round(s, 3), png.exists() and png.stat().st_size, r.returncode)
