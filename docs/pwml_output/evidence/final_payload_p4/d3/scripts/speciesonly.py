"""Count wildcard (species-only) states: species set, tissue, location and cell type all empty. Usage: speciesonly.py dir"""
import re, sys
from pathlib import Path
B = re.compile(r"<biological-state>(.*?)</biological-state>", re.S)
def f(s, tag):
    m = re.search(r"<%s[^>]*?(?:/>|>([^<]*)</%s>)" % (tag, tag), s)
    return (m.group(1) or "").strip() if m else ""
n = 0; files = []
for p in sorted(Path(sys.argv[1], "pwml").glob("*.pwml"), key=lambda x: int(x.stem)):
    k = sum(1 for b in B.findall(p.read_text(encoding="utf-8")) if f(b, "species-id") and not any(f(b, t) for t in ("tissue-id", "subcellular-location-id", "cell-type-id")))
    n += k
    if k: files.append((p.stem, k))
print(sys.argv[1], "species-only states", n, "files", files)
