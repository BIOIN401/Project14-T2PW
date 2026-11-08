"""E2E-3 phase 1: drive the real T2PW site in a real browser (Playwright/Chromium).

ONE script, run under bounded_run.py with the throwaway venv's python (C:/t/e2e3-venv).
It starts uvicorn + the worker (project venv, fake run function, no .env, no secrets,
127.0.0.1 only), walks the site, writes screenshots and results.json, shuts both down.
It fixes nothing and edits nothing under src/.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

import urllib.error
import uuid

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path("C:/t/ws-E2E3")
EV = REPO / "docs/webserver/evidence/E2E-3"
SHOTS = EV / "screenshots"
RUN = Path("C:/t/e2e3-run")
PROJ_PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"

R = {"checks": [], "timeline": [], "console": [], "failed_requests": [], "bad_responses": [],
     "keyboard": [], "links": {}, "notes": {}, "section_errors": []}
STEP = ["setup"]
T0 = time.time()
PROCS = {}


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def check(cid, ok, detail=""):
    R["checks"].append({"id": cid, "ok": bool(ok), "detail": str(detail)[:600]})
    log("CHECK", "PASS" if ok else "FAIL", cid, "-", str(detail)[:200])


def shot(page, name, full=True):
    p = SHOTS / f"{name}.png"
    try:
        page.screenshot(path=str(p), full_page=full)
    except Exception as e:  # noqa: BLE001
        log("shot failed", name, e)
    return p


# -- servers -----------------------------------------------------------------------------


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def clean_env(port):
    keep = ["SystemRoot", "SYSTEMROOT", "PATH", "TEMP", "TMP", "USERPROFILE", "COMSPEC", "PATHEXT",
            "WINDIR", "LOCALAPPDATA", "APPDATA", "HOMEDRIVE", "HOMEPATH"]
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update(
        PYTHONUTF8="1", PYTHONPATH=f"{REPO / 'src'};{EV}",
        T2PW_DB=str(RUN / "jobs.sqlite"), T2PW_WORK_ROOT=str(RUN / "work"),
        T2PW_BASE_URL=f"http://127.0.0.1:{port}", T2PW_RATE_KEY_SECRET="dummy-e2e3-not-a-secret",
        T2PW_CURATOR_QUEUE_DIR=str(RUN / "curator_queue"))
    return env


def spawn(name, argv, env, logfile):
    fh = open(logfile, "wb")
    p = subprocess.Popen(argv, cwd=str(RUN), env=env, stdin=subprocess.DEVNULL, stdout=fh,
                         stderr=subprocess.STDOUT)
    PROCS[name] = (p, fh)
    log("started", name, "pid", p.pid)
    return p


def alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def stop_all():
    survivors = []
    for name, (p, fh) in PROCS.items():
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(10)
        finally:
            fh.close()
        if alive(p.pid):
            survivors.append((name, p.pid))
        log("stopped", name, "pid", p.pid, "alive_after", alive(p.pid))
    R["notes"]["survivors_after_stop"] = survivors


def wait_http(url, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


# -- fixtures ----------------------------------------------------------------------------


def make_pdf(path: Path):
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>"]
    out = b"%PDF-1.4\n"
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    x = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offs)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF\n".encode()
    path.write_bytes(out)


def pick_labels():
    d = json.loads((REPO / "data/webserver/organisms.json").read_text(encoding="utf-8"))
    labels = {o["label"] for o in d["organisms"]}
    R["notes"]["organism_count"] = len(labels)
    want = ["Mus musculus", "Homo sapiens", "Saccharomyces cerevisiae", "Escherichia coli"]
    out = []
    for w in want:
        if w in labels:
            out.append(w)
        else:
            out.append(next(l for l in sorted(labels) if l.startswith(w)))
    return out


# -- JS probes ---------------------------------------------------------------------------

JS_OVERFLOW = """() => {
  const de = document.documentElement, vw = de.clientWidth, off = [];
  const scrolls = el => { for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
      const o = getComputedStyle(p).overflowX; if (o === 'auto' || o === 'scroll' || o === 'hidden') return true; } return false; };
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect(); const cs = getComputedStyle(el);
    if (r.width > 0 && cs.position !== 'fixed' && (r.right > vw + 1 || r.left < -1) && !scrolls(el))
      off.push({tag: el.tagName, id: el.id, cls: String(el.className).slice(0, 40), left: Math.round(r.left),
                right: Math.round(r.right), text: (el.innerText || '').slice(0, 30)});
  }
  return {scrollW: de.scrollWidth, clientW: vw, nOff: off.length, off: off.slice(0, 8)};
}"""

JS_CONTRAST = """() => {
  const cv = document.createElement('canvas'); cv.width = cv.height = 1; const cx = cv.getContext('2d', {willReadFrequently: true});
  const rgba = c => { cx.clearRect(0,0,1,1); cx.fillStyle = '#000'; cx.fillStyle = c; cx.fillRect(0,0,1,1);
    const d = cx.getImageData(0,0,1,1).data; return [d[0], d[1], d[2], d[3] / 255]; };
  const lum = ([r,g,b]) => { const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  const bgOf = el => { const layers = [];
    for (let p = el; p; p = p.parentElement) { const c = rgba(getComputedStyle(p).backgroundColor);
      if (c[3] > 0) { layers.push(c); if (c[3] >= 1) break; } }
    let base = [255,255,255]; if (matchMedia('(prefers-color-scheme: dark)').matches && !layers.length) base = [0,0,0];
    for (let i = layers.length - 1; i >= 0; i--) { const [r,g,b,a] = layers[i]; base = [r*a + base[0]*(1-a), g*a + base[1]*(1-a), b*a + base[2]*(1-a)]; }
    return base; };
  const bad = []; let n = 0;
  for (const el of document.querySelectorAll('body *')) {
    if (['SCRIPT','STYLE','OPTION','DATALIST','NOSCRIPT'].includes(el.tagName)) continue;
    const own = [...el.childNodes].some(t => t.nodeType === 3 && t.textContent.trim());
    if (!own) continue; const r = el.getBoundingClientRect(); if (r.width < 1 || r.height < 1) continue;
    const cs = getComputedStyle(el); if (cs.visibility === 'hidden' || cs.display === 'none') continue;
    if (el.closest('.visually-hidden')) continue;
    n++; const fg = rgba(cs.color), bg = bgOf(el);
    const a = fg[3]; const f = [fg[0]*a + bg[0]*(1-a), fg[1]*a + bg[1]*(1-a), fg[2]*a + bg[2]*(1-a)];
    const L1 = lum(f), L2 = lum(bg); const ratio = (Math.max(L1,L2) + 0.05) / (Math.min(L1,L2) + 0.05);
    const px = parseFloat(cs.fontSize), bold = parseInt(cs.fontWeight) >= 700; const large = px >= 24 || (px >= 18.66 && bold);
    if (ratio < (large ? 3 : 4.5)) bad.push({tag: el.tagName, cls: String(el.className).slice(0,30), text: el.innerText.slice(0,30), ratio: +ratio.toFixed(2), fg: f.map(Math.round).join(','), bg: bg.map(Math.round).join(',')});
  }
  return {checked: n, nBad: bad.length, bad: bad.slice(0, 10)};
}"""

JS_FOCUS = """() => { const e = document.activeElement; if (!e) return null; const cs = getComputedStyle(e); const r = e.getBoundingClientRect();
  return {tag: e.tagName, id: e.id, name: e.getAttribute('name') || '', type: e.type || '',
          text: ((e.innerText || e.value || e.getAttribute('aria-label') || '') + '').slice(0, 30),
          outline: cs.outlineStyle + ' ' + cs.outlineWidth + ' ' + cs.outlineColor, shadow: cs.boxShadow.slice(0, 60),
          fv: e.matches(':focus-visible'), inView: r.width > 0 && r.height > 0 && r.top >= 0 && r.bottom <= innerHeight}; }"""

JS_BADGES = """() => { const el = document.getElementById('job-items'); const swapped = el ? !el.__m : null; if (el) el.__m = 1;
  window.__nav = window.__nav || Math.random();
  return {swapped, hasPoll: !!(el && el.hasAttribute('hx-get')), nav: window.__nav,
    cards: [...document.querySelectorAll('article.job-item')].map(a => (a.querySelector('.job-item__title') || {}).innerText + ' | ' +
      ((a.querySelector('.badge') || {}).innerText || '') + ' | ' + ((a.querySelector('p strong') || {}).innerText || ''))}; }"""


# -- page wiring -------------------------------------------------------------------------


FIX = {"on": False}


def wire(ctx):
    def on_console(m):
        if m.type in ("error", "warning"):
            R["console"].append({"step": STEP[0], "type": m.type, "text": m.text[:400],
                                 "loc": f"{m.location.get('url', '')}:{m.location.get('lineNumber', '')}"})

    ctx.on("console", on_console)
    ctx.on("weberror", lambda e: R["console"].append({"step": STEP[0], "type": "pageerror", "text": str(e.error)[:400],
                                                      "loc": "", "stack": str(getattr(e.error, "stack", ""))[:900]}))
    ctx.on("requestfailed", lambda r: R["failed_requests"].append(
        {"step": STEP[0], "url": r.url, "failure": r.failure}))

    def on_resp(r):
        if r.status >= 400:
            R["bad_responses"].append({"step": STEP[0], "status": r.status, "url": r.url,
                                       "retry_after": r.headers.get("retry-after", "")})

    ctx.on("response", on_resp)


def section(name):
    def deco(fn):
        def run(*a, **k):
            STEP[0] = name
            log("=== SECTION", name)
            try:
                return fn(*a, **k)
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                R["section_errors"].append({"section": name, "error": f"{type(e).__name__}: {e}", "tb": tb[-1500:]})
                log("SECTION ERROR", name, e)
                print(tb, flush=True)
        return run
    return deco


# -- sections ----------------------------------------------------------------------------


@section("s1-home-modes")
def s1(page, base):
    page.goto(base + "/")
    shot(page, "01-home-name-mode")
    vis = lambda: {f: page.locator("#" + f).is_visible() for f in ("name_1", "organism_1", "pmids_1", "pdfs_1")}
    v1 = vis()
    check("S1 name mode shows name+organism only", v1 == dict(name_1=True, organism_1=True, pmids_1=False, pdfs_1=False), v1)
    page.click("label[for=mode-2]")
    v2 = vis()
    shot(page, "02-home-pmids-mode")
    check("S1 PMID mode shows pmids field", v2["pmids_1"] and not v2["pdfs_1"], v2)
    page.click("label[for=mode-3]")
    v3 = vis()
    shot(page, "03-home-pdf-mode")
    check("S1 PDF mode shows file field", v3["pdfs_1"] and not v3["pmids_1"], v3)
    page.click("label[for=mode-1]")
    modes = page.eval_on_selector_all("input[name=mode]", "e=>e.map(x=>x.value)")
    R["notes"]["modes_offered"] = modes
    check("S1 pasted-text mode exists", "text" in modes, f"modes offered: {modes} (task list expected 4 incl. pasted text)")
    # open the optional rows
    for n in (2, 3, 4, 5):
        page.click(f"details.disclosure:has(#name_{n}) summary")
    shot(page, "04-home-all-rows-open")
    check("S1 five rows", page.locator("input[id^=name_]").count() == 5, page.locator("input[id^=name_]").count())


@section("s2-organism-search")
def s2(page, base, labels):
    page.goto(base + "/")
    R["notes"]["htmx_version"] = page.evaluate("typeof htmx !== 'undefined' ? htmx.version : null")
    inp = page.locator("#organism_1")
    reqs = []
    page.on("request", lambda r: reqs.append(r.url) if "/organisms" in r.url else None)

    def type_and_wait(text, timeout=6000):
        inp.fill("")
        page.wait_for_timeout(500)
        before = len(reqs)
        inp.click()
        inp.press_sequentially(text, delay=70)
        page.wait_for_timeout(1500)  # NOT wait_for_function: it uses eval(), which the site CSP blocks
        return page.eval_on_selector_all("#organism-options option", "e=>e.map(x=>x.value)"), len(reqs) - before

    # attempt 1 is on a freshly loaded page, which is what a visitor does
    inp.click()
    inp.press_sequentially("escher", delay=70)
    page.wait_for_timeout(1500)
    opts = page.eval_on_selector_all("#organism-options option", "e=>e.map(x=>x.value)")
    R["notes"]["organism_first_typing_options"] = len(opts)
    R["notes"]["organism_first_typing_requests"] = len(reqs)
    check("S2 typing 'escher' on a fresh page fills the datalist", len(opts) > 0,
          f"{len(opts)} options, first: {opts[:3]}; /organisms requests: {len(reqs)}")
    if not opts:
        opts2, n2 = type_and_wait("escher")
        R["notes"]["organism_second_typing"] = {"options": len(opts2), "requests": n2}
        check("S2 typing 'escher' a second time (after clear) fills the datalist", len(opts2) > 0, f"{len(opts2)} options, {n2} requests")
        opts = opts2
    inp.screenshot(path=str(SHOTS / "05-organism-typed.png"))
    R["notes"]["organism_escher_options"] = opts[:10]
    # keyboard choice from the datalist popup (headless Chromium draws no native popup, so informational)
    inp.press("ArrowDown")
    inp.press("Enter")
    R["notes"]["organism_value_after_ArrowDown_Enter(headless, informational)"] = inp.input_value()
    # pick one exactly (what clicking a suggestion does) and check it sticks
    pick = opts[0] if opts else labels[0]
    inp.fill(pick)
    page.locator("#name_1").click()
    page.wait_for_timeout(700)
    check("S2 chosen organism sticks after blur + 700ms", inp.input_value() == pick, inp.input_value())
    # which typed strings reach the server? (fresh load each, 70 ms per key, then 1.6 s)
    exp = {}
    for word in ("e", "escher", "coli", "Mus", "zzzzqq", "Homo"):
        page.goto(base + "/")
        n_req, n_err = len(reqs), len(R["console"])
        page.locator("#organism_1").click()
        page.locator("#organism_1").press_sequentially(word, delay=70)
        page.wait_for_timeout(1600)
        exp[word] = {"requests": len(reqs) - n_req, "console_errors": len(R["console"]) - n_err,
                     "options": page.locator("#organism-options option").count()}
    R["notes"]["organism_typing_experiment"] = exp
    page.goto(base + "/")
    inp = page.locator("#organism_1")
    # partial-word and case-insensitive search
    o3, _ = type_and_wait("COLI", 4000)
    R["notes"]["organism_search_COLI_options"] = o3[:5]
    inp.fill("")
    inp.press_sequentially("zzzzqq", delay=50)
    page.wait_for_timeout(1000)
    R["notes"]["organism_nonsense_options"] = page.locator("#organism-options option").count()
    inp.fill("")
    page.wait_for_timeout(900)
    R["notes"]["organism_options_after_clearing"] = page.locator("#organism-options option").count()
    shot(page, "06-organism-box", full=False)


def errs(page):
    """Just the error messages on the page: the alert and every field error."""
    return " || ".join(page.eval_on_selector_all(".alert--error, .field__error", "e=>e.map(x=>x.innerText.replace(/\s+/g,' ').trim())"))


@section("s3-bad-input")
def s3(page, base, labels, inputs):
    mus = labels[0]

    def server_mode():
        page.evaluate("document.querySelector('form.submit').noValidate = true")

    def submit():
        with page.expect_navigation(wait_until="load"):
            page.click("button[type=submit]")
        R["notes"].setdefault("s3_errors", []).append(errs(page))

    # a. native required validation
    go(page, base + "/")
    page.click("button[type=submit]")
    page.wait_for_timeout(400)
    msg = page.evaluate("document.querySelector('#name_1').validationMessage")
    check("S3a empty submit blocked in browser with a message", page.url.rstrip("/") == base and bool(msg), f"url={page.url} validationMessage={msg!r}")
    shot(page, "07-bad-empty-native", full=False)

    # b. empty, server side. First WITHOUT the Origin workaround: records what Chromium really sends.
    server_mode(); submit()
    txt = page.inner_text("main")
    shot(page, "08a-bad-empty-server-RAW-403")
    check("S3b-raw a plain Chromium form POST /submit is accepted (no workaround)", "Request blocked" not in txt,
          f"{txt[:120]!r}; request headers seen: {R['notes'].get('post_headers')}")
    FIX["on"] = True  # from here on the test makes Chromium send a real Origin (see INJECT_META)
    go(page, base + "/")
    server_mode(); submit()
    txt = page.inner_text("main")
    shot(page, "08-bad-empty-server")
    check("S3b empty submit (server) says what to do", "Add at least one pathway" in txt, txt[:200].replace("\n", " "))

    # c. too long name
    go(page, base + "/")
    page.fill("#name_1", "x" * 250); page.fill("#organism_1", mus)
    server_mode(); submit()
    v = page.input_value("#name_1")
    shot(page, "09-bad-name-too-long")
    check("S3c too-long name: message", "at most 200" in page.inner_text("main"), page.inner_text("main")[:200].replace("\n", " "))
    check("S3c too-long name: form keeps the value", len(v) >= 200, f"kept {len(v)} of 250 chars; organism kept: {page.input_value('#organism_1')!r}")

    # d. bad PMID
    go(page, base + "/")
    page.click("label[for=mode-2]")
    page.fill("#name_1", "Glycolysis"); page.fill("#organism_1", mus); page.fill("#pmids_1", "abc, 123456, 12-34")
    server_mode(); submit()
    shot(page, "10-bad-pmid")
    t = page.inner_text("main")
    check("S3d bad PMID: names the bad tokens", "abc" in t and "not PubMed IDs" in t.replace("\u2019", "'"), t[:300].replace("\n", " "))
    check("S3d bad PMID: mode, values retained",
          page.is_checked("#mode-2") and page.input_value("#pmids_1") == "abc, 123456, 12-34" and page.input_value("#name_1") == "Glycolysis"
          and page.input_value("#organism_1") == mus,
          f"mode2 checked={page.is_checked('#mode-2')} pmids={page.input_value('#pmids_1')!r}")

    # e. organism typo + bad email
    go(page, base + "/")
    page.fill("#name_1", "Glycolysis"); page.fill("#organism_1", "Mus musculu"); page.fill("#email", "not-an-email")
    server_mode(); submit()
    shot(page, "11-bad-organism-email")
    t = page.inner_text("main")
    check("S3e organism typo: suggests alternatives", "Did you mean" in t or "not in the organism list" in t, t[:300].replace("\n", " "))
    check("S3e bad email: message", "valid email" in t, "")
    check("S3e email value retained", page.input_value("#email") == "not-an-email", page.input_value("#email"))

    # f. PDF mode with no file (server)
    go(page, base + "/")
    page.click("label[for=mode-3]")
    page.fill("#name_1", "Glycolysis"); page.fill("#organism_1", mus)
    server_mode(); submit()
    shot(page, "12-bad-pdf-none")
    t = page.inner_text("main")
    check("S3f PDF mode without file: message", "Choose at least one PDF" in t, t[:250].replace("\n", " "))
    check("S3f PDF mode stays selected", page.is_checked("#mode-3"), "")

    # g. non-PDF file (bypasses the picker's accept filter, as drag-and-drop would). Costs a rate slot.
    go(page, base + "/")
    page.click("label[for=mode-3]")
    page.fill("#name_1", "Glycolysis"); page.fill("#organism_1", mus)
    page.set_input_files("#pdfs_1", str(inputs / "notes.txt"))
    submit()
    shot(page, "13-bad-non-pdf")
    t = page.inner_text("main")
    check("S3g non-PDF file: clear message", "not a PDF" in t, t[:300].replace("\n", " "))
    check("S3g non-PDF: other values retained, mode stays", page.is_checked("#mode-3") and page.input_value("#name_1") == "Glycolysis",
          f"name={page.input_value('#name_1')!r}")
    R["notes"]["non_pdf_file_input_after_error"] = page.evaluate("document.querySelector('#pdfs_1').files.length")


@section("s4-main-job")
def s4(page, base, labels, start_worker, ctx):
    go(page, base + "/")
    names = ["Fake Success Glycolysis", "Fake Review Pathway", "Fake Failure Pathway", "Fake Flaky Retry"]
    for i, (nm, org) in enumerate(zip(names, labels), 1):
        if i > 1:
            page.click(f"details.disclosure:has(#name_{i}) summary")
        page.fill(f"#name_{i}", nm)
        page.fill(f"#organism_{i}", org)
    shot(page, "14-form-filled-4-rows")
    with page.expect_navigation(url=re.compile(r"/job/[0-9a-f]{32}$"), wait_until="load"):
        page.click("button[type=submit]")
    sid = page.url.rsplit("/", 1)[1]
    R["notes"]["main_sid"] = sid
    check("S4 submit redirects to /job/<32 hex>", True, page.url)
    shot(page, "15-job-queued")
    # a second tab on the result page of a queued item: its meta refresh should bring it up to date
    p2 = ctx.new_page()
    p2.goto(f"{base}/job/{sid}/3")
    shot(p2, "16-result-page-while-queued")
    p2_before = p2.inner_text("main")[:400]
    page.evaluate(JS_BADGES)  # set marker
    nav0 = page.evaluate("window.__nav")
    status_reqs = []
    page.on("response", lambda r: status_reqs.append((round(time.time() - T0, 1), r.status)) if r.url.endswith("/status") else None)
    # the page should poll with the worker still off
    try:
        with page.expect_response(lambda r: r.url.endswith("/status"), timeout=15000):
            pass
        check("S4 HTMX polls /status by itself (worker still off)", True, "")
    except Exception as e:  # noqa: BLE001
        check("S4 HTMX polls /status by itself (worker still off)", False, f"no /status request within 15 s: {e}")
    start_worker()
    seen, last, swaps, t_start = [], None, 0, time.time()
    shots = 0
    while time.time() - t_start < 200:
        s = page.evaluate(JS_BADGES)
        if s["swapped"]:
            swaps += 1
        key = tuple(s["cards"])
        R["timeline"].append({"t": round(time.time() - T0, 1), "cards": s["cards"], "poll": s["hasPoll"], "swaps": swaps, "nav": s["nav"] == nav0})
        if key != last:
            last = key
            seen.append(key)
            if shots < 9:
                shots += 1
                shot(page, f"17-job-poll-{shots:02d}", full=False)
        if not s["hasPoll"] and all(("Queued" not in c and "Running" not in c) for c in s["cards"]):
            break
        page.wait_for_timeout(1000)
    final = page.evaluate(JS_BADGES)
    shot(page, "18-job-done")
    check("S4 page was never reloaded (window marker survived)", final["nav"] == nav0, "")
    check("S4 DOM was swapped by HTMX", swaps >= 3, f"{swaps} swaps of #job-items")
    states_text = " ".join(" ".join(k) for k in seen)
    check("S4 saw Queued, Running and a finished state without reload",
          "Queued" in states_text and "Running" in states_text and ("Done" in states_text), f"{len(seen)} distinct DOM states")
    R["notes"]["saw_retrying_text"] = any("Retrying" in " ".join(k) or "Queued for a retry" in " ".join(k) for k in seen)
    n_before = len(status_reqs)
    page.wait_for_timeout(13000)
    check("S4 polling stops once all items are finished", len(status_reqs) == n_before and not final["hasPoll"],
          f"{len(status_reqs) - n_before} extra /status requests in 13 s")
    R["notes"]["status_request_count"] = len(status_reqs)
    # second tab: should have refreshed itself by now (meta refresh every 30 s)
    p2.wait_for_timeout(1500)
    t2 = p2.inner_text("main")
    for _ in range(20):
        if "Done" in t2 or "Passed" in t2:
            break
        p2.wait_for_timeout(2000)
        t2 = p2.inner_text("main")
    shot(p2, "19-result-page-self-refreshed")
    check("S4 result page of a queued item refreshes itself to the final state", "Done" in t2, f"before: {p2_before[:100]!r} / now: {t2[:100]!r}")
    p2.close()
    # real click on View results
    page.goto(f"{base}/job/{sid}")
    page.click("article.job-item >> nth=0 >> text=View results")
    page.wait_for_load_state()
    check("S4 'View results' link opens item 1", page.url.endswith(f"/job/{sid}/0"), page.url)
    return sid


def raw_probe(base, inputs, label):
    """A PDF upload sent without a browser (Origin set correctly) to tell a browser problem from a server one."""
    b = "----probe" + uuid.uuid4().hex
    crlf = "\r\n"
    parts = []
    for k, v in (("mode", "pdfs"), ("name_1", "Fake Quick Probe"), ("organism_1", label), ("consent", "1")):
        parts.append(f'--{b}{crlf}Content-Disposition: form-data; name="{k}"{crlf}{crlf}{v}{crlf}'.encode())
    parts.append(f'--{b}{crlf}Content-Disposition: form-data; name="pdfs_1"; filename="good.pdf"{crlf}'
                 f'Content-Type: application/pdf{crlf}{crlf}'.encode() + (inputs / "good.pdf").read_bytes() + crlf.encode())
    body = b"".join(parts) + f"--{b}--{crlf}".encode()
    req = urllib.request.Request(base + "/submit", data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}", "Origin": base})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            status, url, text = r.status, r.url, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        status, url, text = e.code, e.url, e.read().decode("utf-8", "replace")
    R["notes"]["raw_probe"] = {"status": status, "url": url, "is_empty_msg": "is empty" in text, "not_pdf_msg": "not a PDF" in text,
                               "job_page": "Job status" in text}
    log("raw probe", R["notes"]["raw_probe"])


@section("s5-results")
def s5(page, base, sid, ctx, dl_dir):
    expect = {0: ("success", True), 1: ("review", True), 2: ("fail", False), 3: ("flaky", True)}
    for pos in range(4):
        kind, has_pwml = expect[pos]
        page.goto(f"{base}/job/{sid}/{pos}")
        shot(page, f"20-result-{pos}-{kind}")
        hrefs = page.eval_on_selector_all("a[href]", "e=>e.map(x=>x.getAttribute('href'))")
        files = [h for h in hrefs if "/file/" in h]
        R["links"][f"result{pos}"] = files
        img = page.locator("img.figure__img")
        if img.count():
            ok = page.evaluate("()=>{const i=document.querySelector('img.figure__img');return i.complete&&i.naturalWidth>0}")
            check(f"S5 item {pos} ({kind}): diagram image renders in the browser", ok, "")
        for h in files:
            r = ctx.request.get(base + h)
            ct, cd, csp = r.headers.get("content-type", ""), r.headers.get("content-disposition", ""), r.headers.get("content-security-policy", "")
            body = r.body()
            exp_ct = ("application/xml" if h.endswith("/pwml") else "image/png" if h.endswith("/image") else
                      "application/json" if h.endswith(".json") else "text/plain" if h.endswith(".txt") else "text/csv" if h.endswith(".csv") else "?")
            ok = r.status == 200 and ct.split(";")[0] == exp_ct and len(body) > 0
            check(f"S5 item {pos} ({kind}) GET {h.split('/file/')[1]}", ok, f"{r.status} {ct} | {cd} | {len(body)} B | csp={csp[:30]}")
        # PWML presence matches the kind
        has_link = any(h.endswith("/file/pwml") for h in files)
        check(f"S5 item {pos} ({kind}): PWML link {'offered' if has_pwml else 'absent'}", has_link == has_pwml, files)
        if has_pwml:
            with page.expect_download() as dl:
                page.click("a.download >> nth=0")
            d = dl.value
            dest = dl_dir / f"item{pos}-{d.suggested_filename}"
            d.save_as(str(dest))
            body = dest.read_text(encoding="utf-8")
            check(f"S5 item {pos} ({kind}): clicking Download PWML saves the right file", body.startswith("<?xml") and "E2E-3 fake output" in body,
                  f"{d.suggested_filename} {dest.stat().st_size} B")
        txt = page.inner_text("main")
        R["notes"][f"result{pos}_text"] = txt[:700]
        if pos == 1:
            check("S5 review_required: warning banner and warnings shown", "Review required before use" in txt and "Fake warning" in txt, "")
        if pos == 2:
            check("S5 failure: plain-language explanation, no PWML", "No PWML was produced" in txt, txt[:200].replace("\n", " "))
        if pos == 3:
            check("S5 retried item: attempts table shows 2 attempts", "retried" in txt.lower() and "Passed" in txt, "")
        # open the collapsed details to screenshot them
        for d_ in page.locator("details.disclosure").all():
            d_.locator("summary").click()
        shot(page, f"21-result-{pos}-{kind}-expanded")
    # error pages
    for path, want in [(f"/job/{sid}/9", 404), (f"/job/{sid}/2/file/pwml", 404), (f"/job/{'0' * 32}", 404), ("/nope", 404),
                       (f"/job/{sid}/0/file/report/..%2F..%2Fx", 404), (f"/job/{sid}/0/file/report/request.json", 404)]:
        r = page.goto(base + path)
        check(f"S5 {path[:60]} -> {want}", r.status == want, r.status)
        if path in ("/nope", f"/job/{sid}/2/file/pwml"):
            shot(page, "22-error-404" if path == "/nope" else "22b-error-404-nofile", full=False)
            check(f"S5 404 page is the styled error page ({path[:30]})", "Page not found" in page.inner_text("main"), page.inner_text("main")[:100])
    r = page.goto(f"{base}/jobs/{sid}")
    check("S5 legacy /jobs/<sid> redirects to the job", page.url.endswith(f"/job/{sid}"), page.url)


@section("s6-static-pages")
def s6(page, base, ctx):
    go(page, base + "/")
    nav = page.eval_on_selector_all("nav.site-nav a", "e=>e.map(x=>[x.innerText.trim(),x.getAttribute('href')])")
    R["notes"]["nav"] = nav
    for text, href in nav:
        go(page, base + "/")
        with page.expect_navigation():
            page.click(f"nav.site-nav >> text={text}")
        cur = page.locator("nav.site-nav a[aria-current=page]").all_inner_texts()
        check(f"S6 nav '{text}' -> {href}", page.url.rstrip("/").endswith(href.rstrip("/")) and cur == [text], f"{page.url} current={cur}")
        name = {"/": "30-home", "/examples": "31-examples", "/help": "32-help", "/about": "33-about"}[href]
        shot(page, name)
        t = page.inner_text("body")
        placeholders = re.findall(r"\[[^\]]*(?:TODO|placeholder|Placeholder)[^\]]*\]", t)
        R["notes"][f"placeholders{href}"] = placeholders
        # every link on the page
        hrefs = page.eval_on_selector_all("a[href]", "e=>e.map(x=>x.getAttribute('href'))")
        bad = []
        ext = []
        for h in sorted(set(hrefs)):
            if h.startswith("#") or h.startswith("mailto:"):
                continue
            u = urljoin(base + href, h)
            if not u.startswith(base):
                ext.append(h)
                continue
            r = ctx.request.get(u)
            if r.status != 200:
                bad.append((h, r.status))
        R["links"][f"page{href}"] = {"external": ext, "bad_internal": bad, "total": len(set(hrefs))}
        check(f"S6 all internal links on {href} return 200", not bad, bad)
    # skip link
    page.goto(base + "/help")
    page.keyboard.press("Tab")
    f = page.evaluate(JS_FOCUS)
    check("S6 first Tab stop is the skip link", f and "skip" in f["text"].lower(), f)
    page.keyboard.press("Enter")
    check("S6 skip link moves to #main", page.url.endswith("#main"), page.url)
    page.click("a.brand")
    check("S6 brand link goes home", page.url.rstrip("/") == base, page.url)


@section("s7-keyboard")
def s7(page, base, labels):
    go(page, base + "/")
    typed_name = typed_org = False
    submitted = False
    for i in range(60):
        page.keyboard.press("Tab")
        f = page.evaluate(JS_FOCUS)
        R["keyboard"].append(f)
        has_ring = (f["outline"] and not f["outline"].startswith("none") and not f["outline"].split()[1].startswith("0")) or (f["shadow"] and f["shadow"] != "none")
        f["ring"] = bool(has_ring)
        if f["id"] in ("name_1",) and not typed_name:
            shot(page, "40-kbd-focus-name", full=False)
            page.keyboard.type("Fake Quick Keyboard", delay=20)
            typed_name = True
        elif f["id"] == "organism_1" and not typed_org:
            page.keyboard.type(labels[0], delay=30)  # Enter here would submit the form, so no popup pick
            page.wait_for_timeout(500)
            typed_org = True
            shot(page, "41-kbd-organism", full=False)
        elif f["tag"] == "BUTTON" and f["type"] == "submit":
            shot(page, "42-kbd-focus-submit", full=False)
            with page.expect_navigation(url=re.compile(r"/job/[0-9a-f]{32}$"), wait_until="load"):
                page.keyboard.press("Enter")
            submitted = True
            R["notes"]["kbd_sid"] = page.url.rsplit("/", 1)[1]
            break
    check("S7 form filled and submitted with keyboard only", submitted and typed_name and typed_org, f"tab stops: {len(R['keyboard'])}")
    noring = [f for f in R["keyboard"] if not f.get("ring")]
    check("S7 every Tab stop shows a focus indicator (outline or shadow)", not noring,
          [(f["tag"], f["id"] or f["name"], f["text"], f["outline"]) for f in noring][:8])
    R["notes"]["kbd_stops"] = [(f["tag"], f["id"] or f["name"], f["type"], f["text"], f.get("ring")) for f in R["keyboard"]]
    # radios: arrow keys
    go(page, base + "/")
    page.focus("#mode-1")
    page.keyboard.press("ArrowRight")
    check("S7 arrow keys switch input mode", page.is_checked("#mode-2"), "")
    shot(page, "43-kbd-radio-focus", full=False)


@section("s7b-pdf-submit")
def s7b(page, base, labels, inputs):
    go(page, base + "/")
    page.click("label[for=mode-3]")
    page.fill("#name_1", "Fake Quick PDF")
    page.fill("#organism_1", labels[0])
    page.set_input_files("#pdfs_1", str(inputs / "good.pdf"))
    shot(page, "44-pdf-form-filled", full=False)
    with page.expect_navigation(wait_until="load"):
        page.click("button[type=submit]")
    t = page.inner_text("main")
    ok = "1 PDF file" in t
    check("S7b real PDF upload accepted, job page shows the PDF", ok, t[:300].replace("\n", " "))
    if not ok:
        shot(page, "45-pdf-upload-failed")
        raw_probe(base, inputs, labels[0])
    shot(page, "45-job-pdf")
    if "/job/" not in page.url:
        return None
    R["notes"]["pdf_sid"] = page.url.rsplit("/", 1)[1]
    return R["notes"]["pdf_sid"]


@section("s8-responsive-dark")
def s8(browser, base, sid):
    cases = [("phone-light", dict(viewport={"width": 390, "height": 844}, device_scale_factor=1, is_mobile=True, has_touch=True, color_scheme="light")),
             ("phone-dark", dict(viewport={"width": 390, "height": 844}, device_scale_factor=1, is_mobile=True, has_touch=True, color_scheme="dark")),
             ("desk-dark", dict(viewport={"width": 1280, "height": 800}, color_scheme="dark")),
             ("desk-light", dict(viewport={"width": 1280, "height": 800}, color_scheme="light"))]
    pages = [("home", "/"), ("job", f"/job/{sid}"), ("res-success", f"/job/{sid}/0"), ("res-review", f"/job/{sid}/1"),
             ("res-fail", f"/job/{sid}/2"), ("help", "/help"), ("examples", "/examples"), ("about", "/about")]
    out = {}
    for cname, opts in cases:
        c = browser.new_context(**opts)
        wire(c)
        STEP[0] = "s8-" + cname
        p = c.new_page()
        p.set_default_timeout(15000)
        for pname, path in pages:
            p.goto(base + path)
            if pname == "home":
                p.click("label[for=mode-3]")
                for n in (2, 3):
                    p.click(f"details.disclosure:has(#name_{n}) summary")
            p.evaluate("document.querySelectorAll('details.disclosure').forEach(d=>d.open=true)")
            ov = p.evaluate(JS_OVERFLOW)
            co = p.evaluate(JS_CONTRAST)
            out[f"{cname}/{pname}"] = {"overflow": ov, "contrast": co}
            if cname in ("phone-light", "phone-dark", "desk-dark") and (pname in ("home", "job", "res-review", "help") or cname == "phone-dark"):
                shot(p, f"50-{cname}-{pname}")
            if cname.startswith("phone"):
                check(f"S8 {cname}/{pname}: no horizontal overflow", ov["scrollW"] <= ov["clientW"] + 1 and ov["nOff"] == 0,
                      f"scrollW={ov['scrollW']} clientW={ov['clientW']} off={ov['off'][:3]}")
            check(f"S8 {cname}/{pname}: text contrast >= 4.5 (3 large)", co["nBad"] == 0, f"{co['nBad']} of {co['checked']} low: {co['bad'][:3]}")
        c.close()
    R["notes"]["s8"] = out
    # dark mode: does the site have a manual toggle?
    return out


@section("s10-rate-limit")
def s10(page, base, labels):
    results = []
    for k in range(1, 4):
        go(page, base + "/")
        page.fill("#name_1", f"Fake Quick Limit {k}")
        page.fill("#organism_1", labels[0])
        resp = {}

        def cap(r):
            if r.request.method == "POST" and r.url.endswith("/submit"):
                resp["status"] = r.status
                resp["retry_after"] = r.headers.get("retry-after", "")
        page.on("response", cap)
        page.click("button[type=submit]")
        page.wait_for_load_state()
        page.remove_listener("response", cap)
        t = page.inner_text("main")
        results.append((k, resp.get("status"), resp.get("retry_after"), t[:160].replace("\n", " ")))
        log("rate try", results[-1])
        if resp.get("status") == 429:
            shot(page, "60-rate-limited")
            check("S10 limit message is friendly and says when to retry", "limit of 5 submissions per day" in t and "try again in about" in t, t[:250].replace("\n", " "))
            check("S10 form keeps values after 429", page.input_value("#name_1") == f"Fake Quick Limit {k}" and page.input_value("#organism_1") == labels[0], "")
            check("S10 429 carries Retry-After", bool(resp.get("retry_after")), resp)
            break
    R["notes"]["rate_tries"] = results
    check("S10 a 429 happened within 3 more tries after 4 earlier slots", any(r[1] == 429 for r in results), results)


def wait_all_done(page, base, sids, limit=90):
    end = time.time() + limit
    pending = list(sids)
    while pending and time.time() < end:
        for s in list(pending):
            try:
                html = page.request.get(f"{base}/job/{s}/status").text()
            except Exception:  # noqa: BLE001
                continue
            if 'hx-get' not in html:
                pending.remove(s)
        time.sleep(1.5)
    return pending


# -- main --------------------------------------------------------------------------------


def install_origin_route(ctx, base):
    """Only records the headers of the first two form POSTs. (An earlier version re-served documents
    through route.fetch to change Referrer-Policy; interception can corrupt file uploads, so it is gone.)"""

    def on_req(r):
        if r.method == "POST" and r.url.endswith("/submit"):
            try:
                h = r.all_headers()
            except Exception:  # noqa: BLE001
                h = r.headers
            R["notes"].setdefault("post_headers", []).append(
                {k: h.get(k) for k in ("origin", "sec-fetch-site", "referer", "host")})
    ctx.on("request", on_req)


# Test shim, NOT a fix: the site sends `Referrer-Policy: no-referrer`, under which Chromium sends
# `Origin: null` on the form POST and the site's CSRF guard rejects it. Once FIX is on, every page the
# test loads gets a <meta name=referrer content=same-origin> so Chromium sends the real Origin.
INJECT_META = """() => { const m = document.createElement('meta'); m.name = 'referrer'; m.content = 'same-origin'; document.head.appendChild(m); }"""


def go(page, url):
    r = page.goto(url)
    if FIX["on"]:
        page.evaluate(INJECT_META)
    return r


def main():
    shutil.rmtree(RUN, ignore_errors=True)
    RUN.mkdir(parents=True)
    (RUN / "inputs").mkdir()
    for f in SHOTS.glob("*.png"):
        f.unlink()
    inputs = RUN / "inputs"
    (inputs / "notes.txt").write_text("this is not a pdf\n", encoding="utf-8")
    make_pdf(inputs / "good.pdf")
    dl_dir = RUN / "downloads"
    dl_dir.mkdir()
    labels = pick_labels()
    R["notes"]["labels"] = labels
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = clean_env(port)
    chk = subprocess.run([PROJ_PY, "-c", "import t2pw,fake_run;print(t2pw.__file__, fake_run.__file__)"], env=env, capture_output=True, text=True, cwd=str(RUN))
    log("t2pw import:", chk.stdout.strip(), chk.stderr.strip()[:200])
    R["notes"]["t2pw_path"] = chk.stdout.strip()

    spawn("uvicorn", [PROJ_PY, "-m", "uvicorn", "--factory", "t2pw.web.app:create_app_from_env", "--host", "127.0.0.1", "--port", str(port)],
          env, EV / "uvicorn.log")

    def start_worker():
        spawn("worker", [PROJ_PY, "-m", "t2pw.web.worker", "--db", env["T2PW_DB"], "--work-root", env["T2PW_WORK_ROOT"],
                         "--run-fn", "fake_run:run_pathway", "--poll", "1", "--timeout", "120"], env, EV / "worker.log")

    try:
        if not wait_http(base + "/about"):
            log("server did not come up")
            R["notes"]["server_up"] = False
            return
        log("server up", base)
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            R["notes"]["browser"] = browser.version
            ctx = browser.new_context(viewport={"width": 1280, "height": 800}, color_scheme="light", accept_downloads=True)
            wire(ctx)
            install_origin_route(ctx, base)
            ctx.set_default_timeout(15000)
            page = ctx.new_page()
            s1(page, base)
            s2(page, base, labels)
            s3(page, base, labels, inputs)
            s4(page, base, labels, start_worker, ctx)
            sid = R["notes"].get("main_sid")
            if sid:
                s5(page, base, sid, ctx, dl_dir)
                s6(page, base, ctx)
                s7(page, base, labels)
                psid = s7b(page, base, labels, inputs)
                s8(browser, base, sid)
                s10(page, base, labels)
                still = wait_all_done(page, base, [x for x in (R["notes"].get("kbd_sid"), psid) if x])
                R["notes"]["still_pending_after_wait"] = still
            ctx.close()
            browser.close()
    finally:
        stop_all()
        (EV / "results.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
        for name in ("uvicorn.log", "worker.log"):
            pass
        log("wrote results.json; checks:", len(R["checks"]), "failed:", sum(1 for c in R["checks"] if not c["ok"]))


if __name__ == "__main__":
    main()
