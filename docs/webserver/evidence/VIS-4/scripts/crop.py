"""Crop a region of a PNG for close viewing (no rendering). usage: crop.py <in> <out> x0 y0 x1 y1 [zoom]"""
import sys
from PIL import Image
im = Image.open(sys.argv[1])
x0, y0, x1, y1 = map(int, sys.argv[3:7])
z = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
c = im.crop((x0, y0, x1, y1))
if z != 1.0:
    c = c.resize((int(c.width * z), int(c.height * z)), Image.LANCZOS)
c.save(sys.argv[2])
print(im.size, "->", c.size)
