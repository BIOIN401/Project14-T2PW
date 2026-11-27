"""Contact sheet of MolDB pictures (by pwc-id) labelled with our written compound name. usage: montage.py TSV OUTPREFIX"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
T, COLS, PER = 240, 5, 20
d = Path(sys.argv[1]).parent
for page in range(0, len(rows), PER):
    chunk = rows[page:page + PER]
    sheet = Image.new("RGB", (COLS * T, ((len(chunk) + COLS - 1) // COLS) * (T + 30)), "white")
    dr = ImageDraw.Draw(sheet)
    for k, (pid, name) in enumerate(chunk):
        x, y = (k % COLS) * T, (k // COLS) * (T + 30)
        im = Image.open(d / ("PW_C%06d.png" % int(pid))).convert("RGB"); im.thumbnail((T - 10, T - 10))
        sheet.paste(im, (x + 5, y + 28))
        dr.text((x + 4, y + 2), f"{pid}: {name.split(chr(39))[0][:34]}", fill="black")
        dr.rectangle([x, y, x + T - 1, y + T + 29], outline="#999")
    sheet.save(f"{sys.argv[2]}_{page // PER + 1}.png")
print("ok")
