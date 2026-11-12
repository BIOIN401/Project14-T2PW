"""Log in to the local PathWhiz, open the live view of one pathway, save a screenshot and the SVG.

usage: python shot.py BASE_URL PW_ID OUT_PNG [OUT_SVG]
prints one JSON line: {ok, title, svg_bytes, svg_elements, console_errors, notes}
"""
import json, sys, time
from playwright.sync_api import sync_playwright

base, pw_id, png = sys.argv[1], sys.argv[2], sys.argv[3]
svg_path = sys.argv[4] if len(sys.argv) > 4 else None
EMAIL, PASSWORD = "admin@pwi.local", "pwi-local-only"  # local throwaway user the rake task creates
res = {"ok": False, "pw_id": pw_id, "console_errors": [], "notes": []}

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1600, "height": 1000})
    pg.on("console", lambda m: res["console_errors"].append(m.text[:200]) if m.type == "error" else None)
    pg.on("pageerror", lambda e: res["console_errors"].append("pageerror: " + str(e)[:200]))
    try:
        pg.goto(base + "/pathwhiz/users/sign_in", timeout=120000)
        pg.fill("#user_email", EMAIL)
        pg.fill("#user_password", PASSWORD)
        pg.click("input[type=submit]")
        pg.wait_for_load_state("load", timeout=120000)
        pg.goto(base + "/live/view/" + pw_id, timeout=150000, wait_until="domcontentloaded")
        try:
            pg.wait_for_selector("#live_viewer svg", timeout=90000)
            time.sleep(5)  # let Backbone finish drawing
        except Exception as e:
            res["notes"].append("no svg in #live_viewer: " + type(e).__name__)
        res["title"] = pg.title()
        info = pg.evaluate("""() => { const s = document.querySelector('#live_viewer svg');
            return s ? {len: s.outerHTML.length, n: s.querySelectorAll('*').length, html: s.outerHTML} : null }""")
        if info:
            res["svg_bytes"], res["svg_elements"] = info["len"], info["n"]
            if svg_path:
                open(svg_path, "w", encoding="utf-8").write(info["html"])
        pg.screenshot(path=png.replace(".png", "_view.png"), full_page=False)  # as the viewer opens it
        if info:  # second shot: scale the whole drawing into the viewport
            pg.evaluate("""() => { const s = document.querySelector('#live_viewer svg'); const b = s.getBBox();
                s.setAttribute('viewBox', [b.x, b.y, b.width, b.height].join(' '));
                s.setAttribute('preserveAspectRatio', 'xMidYMid meet');
                s.style.width = '100vw'; s.style.height = '90vh'; }""")
            time.sleep(2)
        pg.screenshot(path=png, full_page=False)
        res["ok"] = bool(info)
    except Exception as e:
        res["notes"].append(type(e).__name__ + ": " + str(e)[:200])
        try: pg.screenshot(path=png)
        except Exception: pass
    b.close()
res["console_errors"] = res["console_errors"][:8]
print(json.dumps(res))
