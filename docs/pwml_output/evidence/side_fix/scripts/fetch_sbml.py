"""Fetch the SBML export of the pathway currently in the local PathWhiz (project t2pw-pwi, localhost:3010).

usage: fetch_sbml.py OUT.sbml [PW_ID=PW000001]
Same login as the harness screenshot.py (local throwaway user). Needs the app up (run_import.py with --screenshots leaves it up).
Prints one JSON line {ok, status, bytes}. Run with the Playwright python (PWI_PLAYWRIGHT_PYTHON).
"""
import json
import sys

from playwright.sync_api import sync_playwright

out = sys.argv[1]
pw = sys.argv[2] if len(sys.argv) > 2 else "PW000001"
base = "http://localhost:3010"
res = {"ok": False}
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto(base + "/pathwhiz/users/sign_in", timeout=120000)
    pg.fill("#user_email", "admin@pwi.local")
    pg.fill("#user_password", "pwi-local-only")
    pg.click("input[type=submit]")
    pg.wait_for_load_state("load", timeout=120000)
    r = pg.request.get(f"{base}/live/view/{pw}.sbml", timeout=300000)
    body = r.body()
    res.update(status=r.status, bytes=len(body), ok=r.status == 200 and b"<sbml" in body)
    open(out, "wb").write(body)
    b.close()
print(json.dumps(res))
