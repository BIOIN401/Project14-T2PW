"""Fetch PathWhiz's own exports (sbml, sbgn, pwml) of the pathway currently loaded in the local app.

usage (Playwright python): a1_export.py PW_ID OUT_PREFIX     -> OUT_PREFIX.sbml / .sbgn / .pwml
prints one JSON line {fmt: {status, bytes, ok}}.
Login is read from the harness screenshot.py at run time (local throwaway user).
"""
import json
import re
import sys

from playwright.sync_api import sync_playwright

pw, prefix = sys.argv[1], sys.argv[2]
base = "http://localhost:3010"
SHOT = open("C:/t/pwi/t2pw/tools/pathwhiz_import/screenshot.py", encoding="utf-8").read()
EMAIL, PASSWORD = re.search(r'EMAIL, PASSWORD = "([^"]+)", "([^"]+)"', SHOT).groups()
MARK = {"sbml": b"<sbml", "sbgn": b"<sbgn", "pwml": b"<"}
res = {}
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto(base + "/pathwhiz/users/sign_in", timeout=120000)
    pg.fill("#user_email", EMAIL)
    pg.fill("#user_password", PASSWORD)
    pg.click("input[type=submit]")
    pg.wait_for_load_state("load", timeout=120000)
    for fmt in ("sbml", "sbgn", "pwml"):
        try:
            r = pg.request.get(f"{base}/live/view/{pw}.{fmt}", timeout=300000)
            body = r.body()
            open(f"{prefix}.{fmt}", "wb").write(body)
            res[fmt] = {"status": r.status, "bytes": len(body), "ok": r.status == 200 and MARK[fmt] in body[:2000]}
        except Exception as e:
            res[fmt] = {"ok": False, "error": type(e).__name__ + ": " + str(e)[:200]}
    b.close()
print(json.dumps(res))
