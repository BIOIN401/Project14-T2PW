import sys, re, pathlib
sys.path.insert(0, "C:/t/vis-pw5/src")
from t2pw.pwml.render import read_xml_with_repairs
L = lambda e: e.tag.rsplit("}", 1)[-1].split(":")[-1] if isinstance(e.tag, str) else ""
for p in sys.argv[1:]:
    r = read_xml_with_repairs(p)
    pv = next(e for e in r.iter() if L(e) == "pathway-visualization")
    d = {L(k): (k.text or "").strip() for k in pv if not len(k)}
    xs, ys = [], []
    for e in r.iter():
        t = L(e)
        k = {L(c): (c.text or "").strip() for c in e}
        if t in ("compound-location", "protein-location", "nucleic-acid-location", "label-location") and k.get("hidden") != "true":
            x, y = float(k["x"]), float(k["y"]); w, h = float(k.get("width") or 0), float(k.get("height") or 0)
            if t == "label-location": w, h = 205 * float(k.get("scale-x", 1)), 56 * float(k.get("scale-y", 1))
            xs += [x, x + w]; ys += [y, y + h]
        if t == "membrane-visualization":
            n = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", k["path"])]
            xs += n[0::2]; ys += n[1::2]
        if t == "drawable-element-location":
            x, y, sx = float(k["x"]), float(k["y"]), float(k.get("scale-x") or 1)
            xs += [x, x + float(k["width"]) * sx]; ys += [y, y + float(k["height"]) * sx]
    print(f"{pathlib.Path(p).stem:24s} canvas {d.get('width')}x{d.get('height')}  drawn x {min(xs):.0f}..{max(xs):.0f} y {min(ys):.0f}..{max(ys):.0f}")
