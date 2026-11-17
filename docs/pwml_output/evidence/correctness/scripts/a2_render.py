"""A2: render the 26 sample IRs with the a2/proto copy of the prototype; screenshot with headless Chrome.
usage: a2_render.py <proto_dir> <out_png_dir> [idx,...] [--style S] [--suffix X]
Writes <out>/<idx><suffix>.png, svg/<idx><suffix>.svg, and <out>/render_summary.json."""
import json, os, re, subprocess, sys, traceback, argparse

ap = argparse.ArgumentParser()
ap.add_argument("proto")
ap.add_argument("out")
ap.add_argument("--idx", default="")
ap.add_argument("--style", default="")
ap.add_argument("--suffix", default="")
ap.add_argument("--maxw", type=int, default=1800)
ap.add_argument("--opts", default="{}")
a = ap.parse_args()
sys.path.insert(0, a.proto)
import t2pw_diagrams as D  # noqa: E402
D.OPTS.update(json.loads(a.opts))

A2 = "C:/t/corr-assess/a2"
CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe"
PROF = A2 + "/chrome-profile"
css = open(os.path.join(a.proto, "diagram.css"), encoding="utf-8").read()
sample = [x["idx"] for x in json.load(open("C:/t/corr-assess/sample.json", encoding="utf-8"))]
sel = [int(x) for x in a.idx.split(",")] if a.idx else sample
os.makedirs(a.out, exist_ok=True)
os.makedirs(A2 + "/svg", exist_ok=True)
summ = {}


def shoot(svg, name):
    W, H = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
    s = min(1.0, a.maxw / W)
    w, h = int(W * s) + 1, int(H * s) + 1
    svg = re.sub(r'width="[\d.]+" height="[\d.]+"', f'width="{w}" height="{h}"', svg, count=1)
    hp = f"{A2}/svg/{name}.html"
    open(hp, "w", encoding="utf-8").write('<!doctype html><html data-theme="light"><meta charset="utf-8"><style>' + css +
                                          ' body{margin:0;background:#fff}</style>' + svg)
    png = os.path.abspath(os.path.join(a.out, name.split("__")[-1] + ".png"))
    p = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--user-data-dir=" + PROF,
                        f"--window-size={w},{min(h, 6000)}", "--screenshot=" + png, "file:///" + hp],
                       capture_output=True, text=True, timeout=120)
    return dict(png=png, ok=os.path.exists(png), rc=p.returncode, W=W, H=H, scale=round(s, 3))


for idx in sel:
    rec = {}
    try:
        ir = D.load(f"C:/t/corr-assess/ir/{idx}/pwml_ir.json")
        k = D.metrics(ir)
        style = a.style or D.pick_style(k)[0]
        rec.update(metrics=k, style=style)
        svg, m, lay = D.render(ir, style, f"d{idx}")
        rec["checks"] = D.checks(m)
        open(f"{A2}/svg/{idx}{a.suffix}.svg", "w", encoding="utf-8").write(svg)
        rec.update(shoot(svg, f"{idx}{a.suffix}"))
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)[:300]
        rec["tb"] = traceback.format_exc()[-1200:]
        # fall back: try the other styles so a picture still exists
        for st in ("bands", "cell", "spine"):
            if st == rec.get("style"):
                continue
            try:
                ir = D.load(f"C:/t/corr-assess/ir/{idx}/pwml_ir.json")
                svg, m, lay = D.render(ir, st, f"d{idx}")
                rec["fallback_style"] = st
                rec.update(shoot(svg, f"{idx}{a.suffix}"))
                break
            except Exception as e2:  # noqa: BLE001
                rec.setdefault("fallback_errors", {})[st] = repr(e2)[:200]
    summ[idx] = rec
    print(idx, rec.get("style"), rec.get("error", "ok"), rec.get("fallback_style", ""), flush=True)
json.dump(summ, open(os.path.join(a.out, "render_summary.json"), "w", encoding="utf-8"), indent=1, default=str)
