"""Side-by-side composite (no rendering): usage composite.py <out.png> <height> <caption=img>..."""
import sys
from PIL import Image, ImageDraw
out, H = sys.argv[1], int(sys.argv[2])
tiles = []
for arg in sys.argv[3:]:
    cap, _, p = arg.partition("=")
    im = Image.open(p).convert("RGB")
    im = im.resize((max(1, int(im.width * H / im.height)), H), Image.LANCZOS)
    tiles.append((cap, im))
W = sum(im.width for _, im in tiles) + 10 * (len(tiles) - 1)
canvas = Image.new("RGB", (W, H + 24), "white")
x = 0
d = ImageDraw.Draw(canvas)
for cap, im in tiles:
    canvas.paste(im, (x, 24))
    d.text((x + 4, 5), cap, fill="black")
    x += im.width + 10
canvas.save(out, optimize=True)
print(out, canvas.size)
