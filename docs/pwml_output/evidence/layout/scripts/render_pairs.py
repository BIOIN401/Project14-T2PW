"""Draw v1 and v2 PWML files from their stored coordinates with t2pw.pwml.render (the file renderer).

usage: render_pairs.py SRC_ROOT V1_DIR V2_DIR OUT_DIR IDX[,IDX...]
Writes OUT_DIR/<idx>-v1.png and <idx>-v2.png, downscaled to at most MAXPX on the long side.
"""
import os
import sys

os.environ["MPLBACKEND"] = "Agg"
src, v1, v2, out, idxs = sys.argv[1:6]
sys.path.insert(0, src)
from PIL import Image  # noqa: E402

from t2pw.pwml.render import render_pwml  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
MAXPX = 1600
os.makedirs(out, exist_ok=True)
for idx in idxs.split(","):
    for tag, d in (("v1", v1), ("v2", v2)):
        p = os.path.join(out, f"{idx}-{tag}.png")
        render_pwml(os.path.join(d, f"{idx}.pwml"), p)
        im = Image.open(p).convert("RGB")
        s = MAXPX / max(im.size)
        if s < 1:
            im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
        im = im.convert("P", palette=Image.ADAPTIVE, colors=16)
        im.save(p, optimize=True)
        print(p, os.path.getsize(p), flush=True)
