import json, os, re, socket, sqlite3, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
EV = Path("C:/t/ev-E2E5b"); DL = EV / "dl"; DL.mkdir(parents=True, exist_ok=True)
RUN = Path("C:/t/e2e5-run"); REPO = Path("C:/t/ws-E2E5")
PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
JID = "ea79681c16aa429799bee5c47b24d663"
LEAK = re.compile(r"C:\\|C:/|Users\\Angad|Users/Angad|ws-E2E5|e2e5-run|/tmp/")
R = {}


def dbrow():
    c = sqlite3.connect(f"file:{(RUN/'jobs.sqlite').as_posix()}?mode=ro", uri=True)
    try:
        return c.execute("select status from jobs where id=?", (JID,)).fetchall(), c.execute("select count(*) from jobs").fetchone()
    except Exception as e:
        return repr(e)
    finally:
        c.close()


def get(url):
    try:
        r = urllib.request.urlopen(url, timeout=30); return r.status, r.headers.get("content-type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("content-type"), e.read()
    except Exception as e:
        return repr(e)[:80], None, b""


def scan(b):
    t = b.decode("utf-8", "replace"); hits = [t[max(0, m.start()-30):m.end()+30].replace("\n", " ") for m in LEAK.finditer(t)]
    return len(hits), hits[:3]


s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
base = f"http://127.0.0.1:{p}"
R["db_before"] = dbrow()
keep = ["SystemRoot", "PATH", "TEMP", "TMP", "USERPROFILE", "COMSPEC", "PATHEXT", "WINDIR", "LOCALAPPDATA", "APPDATA"]
env = {k: os.environ[k] for k in keep if k in os.environ}
env.update(PYTHONUTF8="1", PYTHONPATH=str(REPO / "src"), T2PW_DB=str(RUN/"jobs.sqlite"), T2PW_WORK_ROOT=str(RUN/"work"),
           T2PW_BASE_URL=base, T2PW_RATE_KEY_SECRET="dummy-e2e5b-not-a-secret")
fh = open(EV / "uvicorn.log", "wb")
proc = subprocess.Popen([PY, "-m", "uvicorn", "--factory", "t2pw.web.app:create_app_from_env", "--host", "127.0.0.1", "--port", str(p)],
                        cwd=str(RUN), env=env, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT)
try:
    for _ in range(60):
        if get(base + "/about")[0] == 200: break
        time.sleep(1)
    else:
        raise SystemExit("server not up")
    # 1
    page_url = f"{base}/job/{JID}"
    st, ct, html = get(page_url)
    R["page"] = [st, ct, len(html)]
    (DL / "_jobpage.html").write_bytes(html)
    sub = get(f"{base}/job/{JID}/0")
    (DL / "_resultpage.html").write_bytes(sub[2]); R["resultpage"] = [sub[0], sub[1], len(sub[2])]
    pages = {"job": html, "result": sub[2]}
    links = set()
    for h in pages.values():
        for m in re.finditer(r'(?:href|src)="([^"]+)"', h.decode("utf-8", "replace")):
            u = m.group(1).replace("&amp;", "&")
            if re.search(r"/file/|download|\.(pwml|json|svg|png|txt|md|csv|xml|zip)(\?|$)", u, re.I): links.add(u)
    R["links"] = sorted(links); R["trace_in_links"] = any("LEG_TRACE" in l for l in links)
    R["trace_in_pages"] = {k: b"LEG_TRACE" in v for k, v in pages.items()}
    R["downloads"] = []; scans = {}
    for u in sorted(links):
        full = u if u.startswith("http") else base + u
        st, ct, b = get(full)
        name = re.sub(r"[^A-Za-z0-9._-]", "_", u.strip("/"))[-120:]
        if name.endswith("quarantine_report.json"): name = name[:-len("quarantine_report.json")] + "quarantine_report.json.txt"
        (DL / name).write_bytes(b)
        R["downloads"].append([u, st, ct, len(b)]); scans[name] = scan(b)
    # 2
    scans["_jobpage.html"] = scan(html); scans["_resultpage.html"] = scan(sub[2])
    R["leaks"] = scans
    # 3
    R["trace"] = {}
    for path in ["/job/%s/0/file/LEG_TRACE", "/job/%s/0/file/LEG_TRACE.jsonl", "/job/%s/0/file/leg_trace",
                 "/job/%s/file/LEG_TRACE.jsonl", "/static/LEG_TRACE.jsonl", "/job/%s/0/file/..%%2FLEG_TRACE.jsonl"]:
        st, ct, b = get(base + (path % JID if "%s" in path else path))
        R["trace"][path] = [st, b"model_attempt" in b]
    # 4
    R["shots"] = {}
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        for name, full in (("full", True), ("viewport", False)):
            ctx = b.new_context(viewport={"width": 1280, "height": 900}); pg = ctx.new_page()
            pend = set(); failed = []
            pg.on("request", lambda r: pend.add(r.url)); pg.on("requestfinished", lambda r: pend.discard(r.url))
            pg.on("requestfailed", lambda r: (pend.discard(r.url), failed.append((r.url, r.failure))))
            t = time.time(); res = {}
            try:
                pg.goto(page_url, timeout=30000, wait_until="load"); res["load_s"] = round(time.time()-t, 2)
            except Exception as e:
                res["goto_err"] = repr(e)[:100]
            t = time.time()
            try:
                pg.screenshot(path=str(EV / f"{name}.png"), full_page=full, timeout=30000); res["result"] = "pass"
            except Exception as e:
                res["result"] = "timeout/err " + repr(e)[:100]
            res["shot_s"] = round(time.time()-t, 2); res["pending"] = sorted(pend); res["failed"] = failed
            R["shots"][name] = res; ctx.close()
        b.close()
finally:
    if proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        try: proc.wait(10)
        except Exception: pass
    fh.close()
    R["db_after"] = dbrow()
    R["uvicorn_alive"] = proc.poll() is None
    (EV / "results.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    print(json.dumps(R, indent=1, default=str))
