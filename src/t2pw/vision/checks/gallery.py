"""Screenshot gallery (docs/vision/PLAN.md row V7): our diagram and PathWhiz's drawing of the SAME PWML, side by side.

    python -m t2pw.vision.checks.gallery ours      --runlist R.json --idx 14,126,... --work W   # PWML + SVG + metrics + our PNGs
    python -m t2pw.vision.checks.gallery pathwhiz  --work W                                     # HEAVY: Docker, project t2pw-vis
    python -m t2pw.vision.checks.gallery assemble  --work W --out docs/vision/evidence/V7b/gallery

``ours``: ``generate`` builds the PWML and the SVG from one view; both are measured (layout_metrics) and the SVG is
rendered to a PNG by Playwright in a SUBPROCESS (this module only imports the standard library at the top, and is itself
the subprocess script: ``python gallery.py --shoot jobs.json`` under the Playwright interpreter; the package never imports
Playwright). ``pathwhiz``: imports the same PWML files into a local PathWhiz through a COPY of the import harness
(``C:/t/vis-runs/pwi``, compose project ``t2pw-vis``, host ports 3040 app / 3347 db, source = the read-only PathWhiz tree
``C:/t/pwpr/09``) and takes its live-viewer screenshot; the stack is brought down afterwards (``docker compose -p t2pw-vis
down``, nothing else is touched). ``assemble``: small PNGs (< 300 KB each) and one HTML page.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PLAYWRIGHT_PY = os.environ.get("T2PW_GALLERY_PLAYWRIGHT_PYTHON", "C:/t/e2e3-venv/Scripts/python.exe")
PWI_DIR = "C:/t/vis-runs/pwi"  # copy of tools/pathwhiz_import with PROJECT t2pw-vis (never the original: project t2pw-pwi)
SMPDB_SRC = "C:/t/pwpr/09"  # PathWhiz source worktree t2pw/09-interaction-element-id-remap, read only (the harness clones it)
PROJECT = "t2pw-vis"
MAX_PNG = 300_000
# Six runs across organisms and sizes (indexes into the q=true entries of C:/t/ev-cm/runlist.json): Borrelia mevalonate
# (1 reaction), Petunia benzyl alcohol (5, plant), B. subtilis coenzyme A (11 + 3 transports), human sphingolipid (15),
# mouse sour taste (7 transports, no reaction: the tangle case of PWML-LAYOUT 5.4), Arabidopsis chlorophyll (23 + 31 interactions).
DEFAULT_PICK = (14, 126, 128, 116, 10, 7)


# --------------------------------------------------------------------------- Playwright (runs under the other interpreter)
def _shoot(jobs: List[Dict[str, Any]]) -> None:
    """jobs: [{svg, png, width}]. One browser, one screenshot of the whole diagram per job."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        for j in jobs:
            svg = Path(j["svg"]).read_text(encoding="utf-8")
            pg.set_content("<!doctype html><body style='margin:0;background:#fff'><style>svg.dg{width:%dpx!important;height:auto!important;"
                           "max-width:none!important;display:block}</style>%s</body>" % (j["width"], svg))
            pg.locator("svg.dg").first.screenshot(path=j["png"])
        b.close()


def _run_shoot(jobs: List[Dict[str, Any]], work: Path) -> None:
    jf = work / "shoot_jobs.json"
    jf.write_text(json.dumps(jobs), encoding="utf-8")
    r = subprocess.run([PLAYWRIGHT_PY, "-I", str(Path(__file__).resolve()), "--shoot", str(jf)], capture_output=True, text=True, timeout=900)
    if r.returncode:
        raise RuntimeError("screenshots failed: " + (r.stdout + r.stderr)[-600:])


# --------------------------------------------------------------------------- stage 1: ours
def _run_list(runlist: str) -> List[Dict[str, Any]]:
    return [x for x in json.loads(Path(runlist).read_text(encoding="utf-8")) if x.get("q")]


def build_ours(runlist: str, idxs: List[int], work: Path, scope: str = "default") -> Dict[str, Any]:
    """<work>/<idx>/{<idx>.pwml, ours.svg, ours.png, info.json}: one view, every format from it, both measured."""
    from t2pw.clean_model.builder import read_run_meta
    from t2pw.vision import generate as G
    from t2pw.vision.checks import layout_metrics as LM
    from t2pw.vision.payload import big_from_run
    rl, jobs, out = _run_list(runlist), [], {}
    for i in idxs:
        d = work / str(i)
        d.mkdir(parents=True, exist_ok=True)
        run = rl[i]["dir"]
        meta = read_run_meta(str(run))
        info: Dict[str, Any] = {"idx": i, "run": str(run), "scope": scope, "pmcid": meta.get("pmcid"), "title": meta.get("title"),
                                "organism": meta.get("organism"), "size": {k: rl[i].get(k) for k in ("ne", "nr", "nt", "ni")}}
        try:
            big, opts = big_from_run(run), G.export_options_from_run(run)
            sc = G.parse_scope(scope)
            pw, sv = G.generate(big, sc, "pwml", export=opts), G.generate(big, sc, "svg", export=opts)
            (d / ("%d.pwml" % i)).write_bytes(pw)
            (d / "ours.svg").write_bytes(sv)
            info["metrics"] = {"pwml": LM.pwml_layout_metrics(pw), "svg": LM.svg_layout_metrics(sv.decode("utf-8"))}
            jobs.append({"svg": str(d / "ours.svg"), "png": str(d / "ours.png"), "width": 1400})
        except Exception as e:  # a refused view has no picture; the gallery says so
            info["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        finally:
            G._MEMO.clear()
            G._KEPT.clear()
        (d / "info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
        out[i] = info
        print("ours %s %s" % (i, "ERROR" if "error" in info else "ok"), flush=True)
    if jobs:
        _run_shoot(jobs, work)
    return out


# --------------------------------------------------------------------------- stage 2: PathWhiz (heavy, Docker)
def _check_harness(pwi: Path) -> None:
    """Refuse any harness whose compose project is not ours: this lane never touches another project's stack."""
    src = (pwi / "run_import.py").read_text(encoding="utf-8")
    if 'PROJECT = "%s"' % PROJECT not in src:
        raise SystemExit("%s is not the %s harness copy (PROJECT differs): refusing to start Docker" % (pwi, PROJECT))


def _teardown(pwi: Path) -> None:
    env = dict(os.environ, SMPDB_SRC=str(pwi / "work" / "smpdb"))  # the compose file requires the variable, even to go down
    subprocess.run(["docker", "compose", "-p", PROJECT, "-f", str(pwi / "docker-compose.yml"), "down"], env=env, check=False, timeout=300)


def run_pathwhiz(work: Path, pwi: str = PWI_DIR, smpdb_src: str = SMPDB_SRC, resume: bool = False) -> Dict[str, Any]:
    """Import every <work>/<idx>/<idx>.pwml, screenshot it in the PathWhiz live viewer, bring the stack down."""
    pwi_p = Path(pwi)
    _check_harness(pwi_p)
    files = sorted(work.glob("*/*.pwml"), key=lambda p: int(p.parent.name))
    if not files:
        raise SystemExit("no PWML under %s: run the ours step first" % work)
    lst, out = work / "pwml_list.txt", work / "pw"
    lst.write_text("\n".join(str(f.resolve()) for f in files) + "\n", encoding="utf-8")
    cmd = [sys.executable, str(pwi_p / "run_import.py"), "--list", str(lst), "--screenshots"] + [str(f.resolve()) for f in files] + [
        "--out", str(out), "--smpdb-src", smpdb_src, "--work", str(pwi_p / "work" / "smpdb"), "--down"] + (["--resume"] if resume else [])
    try:
        rc = subprocess.run(cmd).returncode
    finally:
        _teardown(pwi_p)
    res: Dict[str, Any] = {"exit": rc}
    rp = out / "results.jsonl"
    for line in (rp.read_text(encoding="utf-8").splitlines() if rp.exists() else []):
        row = json.loads(line) if line.strip() else None
        if not row:
            continue
        i = int(Path(row["file"]).parent.name)
        shot = out / row["screenshot"] if row.get("screenshot") else None
        if shot and shot.exists():
            shutil.copyfile(shot, work / str(i) / "pw.png")
        res[str(i)] = {k: row.get(k) for k in ("success", "error_count", "errors", "created_main", "expected", "seconds")} | {"screenshot": bool(shot and shot.exists())}
    (work / "pw_results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return res


# --------------------------------------------------------------------------- stage 3: assemble
def _shrink(src: Path, dst: Path, limit: int = MAX_PNG, width: int = 1200) -> None:
    """Palette PNG, narrowed until it is under ``limit`` bytes (flat-colour drawings quantise well)."""
    from PIL import Image
    im = Image.open(src).convert("RGB")
    w = min(width, im.width)
    while True:
        q = im.resize((w, max(1, round(im.height * w / im.width))), Image.LANCZOS).quantize(colors=96)
        q.save(dst, optimize=True)
        if dst.stat().st_size <= limit or w <= 300:
            return
        w = int(w * 0.85)


_PW_KEYS = ("nodes", "node_overlaps", "edge_crossings", "reading_order_violations", "precedence_pairs", "labels_too_wide", "labels_outside_box",
            "label_node_overlaps", "duplicated_compound_locations", "enzyme_distance_gt200", "enzymes_drawn", "enzyme_distance_max")
_SVG_KEYS = ("crossings", "bends", "enzyme_links_gt200", "enzyme_links", "enzyme_link_max", "node_overlaps", "labels_truncated", "drawings")


def _table(title: str, d: Dict[str, Any], keys) -> str:
    rows = "".join("<tr><td>%s</td><td>%s</td></tr>" % (html.escape(k), html.escape(str(d.get(k)))) for k in keys if k in d)
    return "<table><caption>%s</caption>%s</table>" % (html.escape(title), rows)


def assemble(work: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    pw = json.loads((work / "pw_results.json").read_text(encoding="utf-8")) if (work / "pw_results.json").exists() else {}
    cards = []
    for d in sorted((p for p in work.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
        i = d.name
        info = json.loads((d / "info.json").read_text(encoding="utf-8"))
        cell = []
        for src, name, label in (("ours.png", "%s_ours.png" % i, "T2PW diagram (our renderer)"), ("pw.png", "%s_pathwhiz.png" % i, "PathWhiz live viewer (same PWML)")):
            if (d / src).exists():
                _shrink(d / src, out / name)
                cell.append('<figure><img loading="lazy" src="%s" alt="%s"><figcaption>%s</figcaption></figure>' % (name, html.escape(label), html.escape(label)))
            else:
                why = info.get("error") if src == "ours.png" else ("no PathWhiz screenshot" + ("" if i in pw else " (import not run)"))
                cell.append('<figure class="missing"><p>%s</p><figcaption>%s</figcaption></figure>' % (html.escape(str(why)), html.escape(label)))
        m, r = info.get("metrics") or {}, pw.get(i) or {}
        imp = ""
        if r:
            exp, got = r.get("expected") or {}, r.get("created_main") or {}
            imp = "<table><caption>PathWhiz import</caption><tr><td>imported</td><td>%s, %s message(s)</td></tr>%s</table>" % (
                r.get("success"), r.get("error_count"),
                "".join("<tr><td>%s created/written</td><td>%s/%s</td></tr>" % (html.escape(k), got.get(k, "?"), exp[k]) for k in exp if exp[k]))
        sz = info.get("size") or {}
        cap = "%s &middot; %s &middot; %s &middot; entities %s, reactions %s, transports %s, interactions %s" % (
            html.escape(str(i)), html.escape(str(info.get("organism") or "organism not stated")), html.escape(str(info.get("pmcid") or "")),
            sz.get("ne"), sz.get("nr"), sz.get("nt"), sz.get("ni"))
        cards.append('<section><h2>%s</h2><div class="pair">%s</div><div class="tables">%s%s%s</div></section>' % (
            cap, "".join(cell), _table("PWML coordinates", m.get("pwml") or {}, _PW_KEYS), _table("our SVG", m.get("svg") or {}, _SVG_KEYS), imp))
    css = (":root{--bg:#fff;--ink:#1a1a1a;--line:#d0d0d0}@media(prefers-color-scheme:dark){:root{--bg:#161616;--ink:#e8e8e8;--line:#444}}"
           "body{background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,sans-serif;margin:0 auto;padding:16px;max-width:1500px}"
           "section{border-top:1px solid var(--line);padding:12px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}"
           "img{width:100%;height:auto;background:#fff;border:1px solid var(--line)}figure{margin:0}figcaption{font-size:12px}"
           ".missing p{border:1px dashed var(--line);padding:24px}.tables{display:flex;flex-wrap:wrap;gap:16px;margin-top:8px}"
           "table{border-collapse:collapse}td,caption{padding:1px 8px;text-align:left}caption{font-weight:600}"
           "@media(max-width:700px){.pair{grid-template-columns:1fr}}")
    page = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>T2PW layout gallery</title><style>%s</style></head><body><h1>T2PW layout gallery</h1>"
            "<p>The same PWML drawn by our renderer (left) and by the PathWhiz live viewer (right), with layout metrics. "
            "Metrics are counts, not judgements.</p>%s</body></html>" % (css, "".join(cards)))
    (out / "index.html").write_text(page, encoding="utf-8")
    return out / "index.html"


# --------------------------------------------------------------------------- CLI
def _idx(text) -> List[int]:
    return [int(x) for x in str(text).replace("\n", ",").split(",") if x.strip()] if text else list(DEFAULT_PICK)


def main(argv=None) -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--shoot":  # the Playwright subprocess
        _shoot(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
        return 0
    ap = argparse.ArgumentParser(prog="python -m t2pw.vision.checks.gallery", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("ours", help="PWML + SVG + metrics + our PNGs (needs the Playwright interpreter, no Docker)")
    o.add_argument("--runlist", required=True)
    o.add_argument("--idx", default=None, help="comma-separated indexes into the q=true runlist entries (default: %s)" % ",".join(map(str, DEFAULT_PICK)))
    o.add_argument("--work", required=True)
    o.add_argument("--scope", default="default")
    p = sub.add_parser("pathwhiz", help="HEAVY (Docker, project %s): import the PWML files into PathWhiz and screenshot them" % PROJECT)
    p.add_argument("--work", required=True)
    p.add_argument("--pwi", default=PWI_DIR)
    p.add_argument("--smpdb-src", default=SMPDB_SRC)
    p.add_argument("--resume", action="store_true")
    s = sub.add_parser("assemble", help="small PNGs and the HTML page")
    s.add_argument("--work", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.cmd == "ours":
        build_ours(a.runlist, _idx(a.idx), work, a.scope)
    elif a.cmd == "pathwhiz":
        res = run_pathwhiz(work, a.pwi, a.smpdb_src, a.resume)
        print(json.dumps({k: (v if k == "exit" else {kk: v[kk] for kk in ("success", "error_count", "screenshot")}) for k, v in res.items()}, indent=1))
        return 0 if res.get("exit") == 0 else 1
    else:
        print(assemble(work, Path(a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
