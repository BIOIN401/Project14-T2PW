"""Dev-time: assemble src/t2pw/render/pathbank_art/ from the extracted art (extract_art.py) and its Edge rasters
(raster_art.py).

usage: build_art_pkg.py ART_DIR RAST_DIR PKG_DIR
PKG_DIR/<id>.png      runtime raster (transparent, longest side <= 512)
PKG_DIR/svg/<id>.svg  PathBank's original vector art, byte for byte (ids whose original is SVG)
PKG_DIR/catalog.json  id -> native size, source pathways, original file and its sha256
"""
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image

NAMES = {  # read off the contact sheet; PathBank's own names live in its DB (not queried here)
    "1": "mitochondrion", "3": "rough endoplasmic reticulum", "4": "lysosome-like vesicle (hexagon)",
    "5": "peroxisome / granule", "8": "kidney with adrenal gland", "9": "liver", "10": "heart",
    "11": "blood vessel network", "12": "red oval cell", "13": "neuron", "14": "nucleus", "16": "brain",
    "18": "muscle", "21": "inhibition sign", "27": "intestinal epithelium", "30": "digestive tract",
    "38": "lungs", "47": "cross mark", "48": "accumulation burst", "51": "high concentration burst",
    "52": "low concentration burst", "73": "DNA strand", "75": "lightning", "90": "cell organelle cluster",
    "91": "plant cell organelle cluster", "92": "bacterial chromosome", "97": "pituitary gland",
    "99": "blood vessel (wide)", "108": "red blood cell", "139": "cell cycle", "143": "blood vessel (horizontal)",
    "146": "bacterium", "147": "cheese (food)", "149": "head", "180": "bile duct and gallbladder",
    "204": "generic cell", "205": "crystals", "217": "capsule (drug)", "218": "syringe",
    "221": "blood vessel (vertical)", "247": "brain (sagittal)",
}
art, rast, pkg = (Path(a) for a in sys.argv[1:4])
(pkg / "svg").mkdir(parents=True, exist_ok=True)
src_cat = json.loads((art / "_catalog.json").read_text())
cat = {}
total = 0
for i, row in src_cat.items():
    if "file" not in row:
        continue
    orig = art / row["file"]
    raw = orig.read_bytes()
    if row["file"].endswith(".svg"):
        (pkg / "svg" / row["file"]).write_bytes(raw)
        im = Image.open(rast / f"{i}.png")
    else:
        im = Image.open(orig)
    im = im.convert("RGBA")
    im.thumbnail((512, 512), Image.LANCZOS)
    im.quantize(256, method=Image.Quantize.FASTOCTREE).save(pkg / f"{i}.png", optimize=True)
    w, h = row["sizes"][0]
    cat[i] = {"name": NAMES.get(i, ""), "width": w, "height": h, "pathways": row["pathways"],
              "original": row["file"], "original_sha256": hashlib.sha256(raw).hexdigest()}
    total += (pkg / f"{i}.png").stat().st_size
(pkg / "catalog.json").write_text(json.dumps(dict(sorted(cat.items(), key=lambda kv: int(kv[0]))), indent=1) + "\n")
print(len(cat), "images, rasters", total, "bytes")
