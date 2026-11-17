"""render one file to a small PNG with an independent renderer. usage: kind in out.png ; kind = sbgn | sbml
Run with C:/t/fmt-venv python."""
import sys, os, time, json, traceback
kind, src, out = sys.argv[1:4]
MAXW = 1800
t0 = time.time()
res = {"kind": kind, "src": src, "out": out}
try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    tmp = out + ".tmp.png"
    if kind == "sbgn":
        import momapy.io, momapy.rendering.core as R
        r = momapy.io.read(src, reader="sbgnml-0.2")
        mp = r.obj
        res["momapy_load"] = "ok"
        try:
            res["n_layout_elements"] = len(list(mp.layout.layout_elements))
        except Exception:
            pass
        R.render_map(mp, tmp, format_="png", renderer="skia")
    elif kind == "sbml_momapy":
        import momapy.io
        r = momapy.io.read(src, reader="sbml")
        mp = r.obj
        res["momapy_load"] = "ok"
        res["layout"] = None if mp.layout is None else "present"
        m = mp.model
        res["model_attrs"] = {k: (len(getattr(m, k)) if hasattr(getattr(m, k), "__len__") else None) for k in ("species", "reactions", "compartments") if hasattr(m, k)}
        res["rendered"] = False
        print(json.dumps(res)); sys.exit(0)
    else:
        import sbmlnetwork
        net = sbmlnetwork.load(src)
        net.draw(tmp)
    im = Image.open(tmp)
    res["full_size"] = im.size
    if im.width > MAXW:
        im = im.resize((MAXW, max(1, int(im.height * MAXW / im.width))), Image.LANCZOS)
    im = im.convert("RGB")
    im.save(out, optimize=True)
    os.remove(tmp)
    res["rendered"] = True
    res["bytes"] = os.path.getsize(out)
except Exception as e:
    res["rendered"] = False
    res["error"] = repr(e)[:300]
    res["tb"] = traceback.format_exc()[-600:]
res["seconds"] = round(time.time() - t0, 1)
print(json.dumps(res))
