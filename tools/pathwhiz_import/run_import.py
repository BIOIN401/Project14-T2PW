#!/usr/bin/env python3
"""Automated PathWhiz import test for PWML files.

    python tools/pathwhiz_import/run_import.py FILE.pwml [FILE ...] [--list paths.txt]
        [--screenshots A.pwml B.pwml ...] [--out DIR] [--resume] [--down]

Imports each file with the real PathWhiz importer (smpdb lib/pwml_parser.rb) on a clean database,
using a local Docker stack (project t2pw-pwi). Writes DIR/results.jsonl (one row per file),
DIR/summary.json and DIR/summary.md. Files named after --screenshots are also opened in the local
PathWhiz live viewer and saved as PNG (+ SVG) under DIR/screenshots/.

Never touches the owner's smpdb checkout: it works in a clone (tools/pathwhiz_import/work/smpdb).
See README.md in this folder.
"""
import argparse, collections, hashlib, json, os, re, shutil, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = "t2pw-pwi"
COMPOSE = os.path.join(HERE, "docker-compose.yml")
BASE_URL = "http://localhost:3010"
DEFAULT_SRC = os.environ.get("SMPDB_SRC", r"C:/Users/Angad/Desktop/SummerBIOIN/smpdb")
IMAGE = "smpdb-app:latest"          # the owner's already-built image (Ruby 2.2.0, JDK 7)
BUNDLE_VOL = "t2pw-pwi_bundle"      # gems; seeded from the owner's smpdb_bundle_cache volume
OWNER_BUNDLE_VOL = "smpdb_bundle_cache"
PLAYWRIGHT_PY = os.environ.get("PWI_PLAYWRIGHT_PYTHON", r"C:/t/e2e3-venv/Scripts/python.exe")


def sh(cmd, check=True, timeout=None, env=None, quiet=False):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        sys.exit("command failed (%s): %s\n%s" % (r.returncode, " ".join(cmd), (r.stdout + r.stderr)[-1500:]))
    return r


def dc(args, env, **kw):
    return sh(["docker", "compose", "-p", PROJECT, "-f", COMPOSE] + args, env=env, **kw)


def log(*a):
    print("[pwi]", *a, flush=True)


# ----------------------------------------------------------------------------- setup
def ensure_work_copy(src, work):
    """Clone the owner's smpdb (read-only use) and apply its uncommitted tracked changes."""
    if os.path.isdir(os.path.join(work, ".git")):
        return
    log("cloning", src, "->", work)
    os.makedirs(os.path.dirname(work), exist_ok=True)
    sh(["git", "clone", "--local", "--quiet", src, work])
    patch = sh(["git", "-C", src, "diff", "HEAD"]).stdout
    if patch.strip():
        p = os.path.join(os.path.dirname(work), "owner-working-tree.patch")
        open(p, "w", encoding="utf-8", newline="\n").write(patch)
        sh(["git", "-C", work, "apply", p])
        log("applied owner's uncommitted tracked changes; patch sha256",
            hashlib.sha256(patch.encode("utf-8")).hexdigest())


def ensure_config(work):
    open(os.path.join(work, "config", "database.yml"), "w").write(
        "development:\n  adapter: mysql2\n  encoding: utf8\n  database: smpdb_development\n"
        "  username: root\n  password: password\n  host: db\n  port: 3306\n")  # local throwaway
    m = os.path.join(work, "config", "moldb.yml")
    if not os.path.exists(m):
        shutil.copy(os.path.join(work, "config", "moldb.yml.sample"), m)
    shutil.copy(os.path.join(HERE, "pwml_import.rake"), os.path.join(work, "lib", "tasks", "pwml_import.rake"))


def ensure_images_and_volume():
    if not sh(["docker", "image", "inspect", IMAGE], check=False).returncode == 0:
        sys.exit("docker image %s not found. Build it from smpdb/Dockerfile (docker build -t %s .) first." % (IMAGE, IMAGE))
    if sh(["docker", "volume", "inspect", BUNDLE_VOL], check=False).returncode == 0:
        return
    if sh(["docker", "volume", "inspect", OWNER_BUNDLE_VOL], check=False).returncode != 0:
        sys.exit("no gem volume: need docker volume %s (owner's bundle cache) or a populated %s" % (OWNER_BUNDLE_VOL, BUNDLE_VOL))
    log("copying gems from", OWNER_BUNDLE_VOL, "(read-only) into", BUNDLE_VOL)
    sh(["docker", "volume", "create", BUNDLE_VOL])
    sh(["docker", "run", "--rm", "-v", OWNER_BUNDLE_VOL + ":/src:ro", "-v", BUNDLE_VOL + ":/dst",
        "--entrypoint", "bash", IMAGE, "-c", "cp -a /src/. /dst/"], timeout=900)


def app_sh(env, script, **kw):
    return dc(["exec", "-T", "app", "bash", "-c", "cd /app && " + script], env, **kw)


def mysql(env, query):
    r = dc(["exec", "-T", "db", "mysql", "-uroot", "-ppassword", "-N", "-B", "smpdb_development", "-e", query], env, check=False)
    return r.stdout.strip() if r.returncode == 0 else None


def ensure_schema_and_seed(env):
    if mysql(env, "SHOW TABLES LIKE 'users'") != "users":
        log("loading db/schema.rb into the empty database (no production dump is loaded)")
        app_sh(env, "bundle exec rake db:create db:schema:load", timeout=900)
    if mysql(env, "SELECT COUNT(*) FROM visualization_templates") in (None, "0"):
        log("loading visualization_templates / drawable_images seed rows")
        seed = os.path.join(HERE, "seed", "seed_inserts.sql")
        with open(seed, "rb") as f:
            r = subprocess.run(["docker", "compose", "-p", PROJECT, "-f", COMPOSE, "exec", "-T", "db", "mysql", "-uroot",
                                "-ppassword", "--default-character-set=utf8", "smpdb_development"], stdin=f, env=env,
                               capture_output=True)
        if r.returncode != 0:
            sys.exit("seed load failed: " + r.stderr.decode("utf-8", "replace")[-800:])


def server_up():
    try:
        return urllib.request.urlopen(BASE_URL + "/pathwhiz/users/sign_in", timeout=60).status == 200
    except Exception:
        return False


def ensure_server(env):
    if server_up():
        return
    log("starting rails server in the container")
    dc(["exec", "-d", "app", "bash", "-c",
        "cd /app && rm -f tmp/pids/server.pid && bundle exec rails s -b 0.0.0.0 -p 3000 > /out/server.log 2>&1"], env)
    for _ in range(60):
        time.sleep(3)
        if server_up():
            return
    sys.exit("rails server did not come up; see out/server.log")


# ----------------------------------------------------------------------------- import
SECTIONS = {"compounds": "compound", "proteins": "protein", "protein-complexes": "protein-complex",
            "reactions": "reaction", "transports": "transport", "interactions": "interaction",
            "compound-locations": "compound-location", "protein-locations": "protein-location"}


def expected_counts(path):
    """Element counts as written in the PWML, to compare with rows the importer created."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    for sec, item in SECTIONS.items():
        m = re.search(r"<%s>(.*?)</%s>" % (sec, sec), txt, re.S)
        out[sec.replace("-", "_")] = len(re.findall(r"<%s>\s*<id" % item, m.group(1))) if m else 0
    return out


def run_rake(env, staged, timeout):
    """One rake run (one Rails boot) over a list of staged files; yields result rows as they print."""
    names = [n for n, _ in staged]
    open(os.path.join(HERE, "pwml_in", "list.txt"), "w", newline="\n").write("\n".join("/pwml_in/" + n for n in names) + "\n")
    cmd = ["docker", "compose", "-p", PROJECT, "-f", COMPOSE, "exec", "-T", "app", "bash", "-c",
           "cd /app && PWML_LIST=/pwml_in/list.txt bundle exec rake pwml:import 2>&1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True,
                         encoding="utf-8", errors="replace")
    t0, rows = time.time(), []
    try:
        for line in p.stdout:
            if line.startswith("PWML_RESULT "):
                rows.append(json.loads(line[len("PWML_RESULT "):]))
                yield rows[-1]
            elif "rake aborted" in line or "Error" in line and "Elasticsearch" not in line and "original exception" not in line:
                log("rake:", line.strip()[:200])
            if time.time() - t0 > timeout:
                raise TimeoutError("rake batch exceeded %ss" % timeout)
    finally:
        if p.poll() is None:
            p.kill()
            dc(["exec", "-T", "app", "pkill", "-f", "rake pwml:import"], env, check=False)  # container-side only
        p.wait()


def take_screenshot(pw_id, shots_dir, tag):
    os.makedirs(shots_dir, exist_ok=True)
    png, svg = os.path.join(shots_dir, tag + ".png"), os.path.join(shots_dir, tag + ".svg")
    r = subprocess.run([PLAYWRIGHT_PY, os.path.join(HERE, "screenshot.py"), BASE_URL, pw_id, png, svg],
                       capture_output=True, text=True, timeout=500)
    try:
        info = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        info = {"ok": False, "notes": [(r.stdout + r.stderr)[-300:]]}
    return png, svg, info


def cause_key(k, v):
    base = k.rsplit("_", 1)[0] if re.search(r"\d", k.rsplit("_", 1)[-1]) else k
    msg = "; ".join(v) if isinstance(v, list) else str(v)
    return "%s: %s" % (base, msg[:160])


def summarize(rows):
    ok = [r for r in rows if r["success"]]
    clean = [r for r in ok if not r["errors"]]
    causes, files_by_cause = collections.Counter(), collections.defaultdict(set)
    for r in rows:
        for k, v in (r["errors"] or {}).items():
            c = cause_key(k, v)
            causes[c] += 1
            files_by_cause[c].add(r["file"])
        if r.get("parser_exception"):
            c = "PARSER EXCEPTION: " + r["parser_exception"][:160]
            causes[c] += 1
            files_by_cause[c].add(r["file"])
    return {"files": len(rows), "success_true": len(ok), "success_true_no_errors": len(clean),
            "success_false": len(rows) - len(ok),
            "error_causes": [{"cause": c, "count": n, "files": len(files_by_cause[c])} for c, n in causes.most_common()],
            "total_seconds": round(sum(r["seconds"] for r in rows), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--list", help="text file, one PWML path per line")
    ap.add_argument("--screenshots", nargs="*", default=[], help="PWML paths (also given as files) to screenshot")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "run"))
    ap.add_argument("--smpdb-src", default=DEFAULT_SRC, help="owner's smpdb checkout (read only)")
    ap.add_argument("--work", default=os.path.join(HERE, "work", "smpdb"), help="harness clone of smpdb")
    ap.add_argument("--resume", action="store_true", help="skip files already in results.jsonl")
    ap.add_argument("--chunk", type=int, default=12, help="files per rails boot")
    ap.add_argument("--chunk-timeout", type=int, default=3000)
    ap.add_argument("--down", action="store_true", help="docker compose down at the end (keeps volumes and images)")
    a = ap.parse_args()

    paths = [os.path.abspath(f) for f in a.files]
    if a.list:
        paths += [os.path.abspath(l.strip()) for l in open(a.list) if l.strip()]
    shot_set = {os.path.abspath(f) for f in a.screenshots}
    paths += [s for s in shot_set if s not in paths]
    if not paths:
        sys.exit("no PWML files given")
    os.makedirs(a.out, exist_ok=True)
    results_path = os.path.join(a.out, "results.jsonl")
    done = {}
    if a.resume and os.path.exists(results_path):
        done = {json.loads(l)["file"]: 1 for l in open(results_path, encoding="utf-8") if l.strip()}
    todo = [p for p in paths if p not in done]

    work = os.path.abspath(a.work)
    ensure_work_copy(a.smpdb_src, work)
    ensure_config(work)
    ensure_images_and_volume()
    pwml_in = os.path.join(HERE, "pwml_in")
    os.makedirs(pwml_in, exist_ok=True)
    env = dict(os.environ, SMPDB_SRC=work, OUT_DIR=os.path.abspath(a.out), PWML_DIR=pwml_in)
    dc(["up", "-d"], env, timeout=300)
    ensure_schema_and_seed(env)

    stage = {}  # staged name -> original path
    for i, p in enumerate(todo):
        if not os.path.exists(p):
            continue
        n = "f%04d_%s.pwml" % (i, hashlib.sha1(p.encode()).hexdigest()[:8])
        shutil.copy(p, os.path.join(pwml_in, n))
        stage[n] = p

    def record(row):
        orig = stage[os.path.basename(row["file"])]
        row["file"] = orig
        row["expected"] = expected_counts(orig)
        row["error_count"] = len(row.get("errors") or {})
        row.setdefault("screenshot", None)
        return row

    out = open(results_path, "a", encoding="utf-8")

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()
        log("%-7s %5.1fs  %s  errors=%d" % ("OK" if row["success"] else "FAIL", row["seconds"], os.path.basename(row["file"]), row["error_count"]))

    for p in todo:
        if not os.path.exists(p):
            emit({"file": p, "success": False, "errors": {"exception": "file not found"}, "error_count": 1, "seconds": 0,
                  "created_main": {}, "created": {}, "expected": {}, "screenshot": None})

    names = [(n, p) for n, p in stage.items()]
    shots = [(n, p) for n, p in names if p in shot_set]
    rest = [(n, p) for n, p in names if p not in shot_set]
    # screenshot files go one per rake run: the DB holds only the last imported pathway
    for n, p in shots:
        for row in run_rake(env, [(n, p)], a.chunk_timeout):
            row = record(row)
            if row["success"] and row.get("pw_id"):
                ensure_server(env)
                # unique per file: staged name (index + path hash) plus the paper folder, e.g. f0003_ab12cd34_PMC12096016
                tag = re.sub(r"[^A-Za-z0-9_.-]", "_", n[:-5] + "_" + os.path.basename(os.path.dirname(os.path.dirname(p))))[:80]
                png, svg, info = take_screenshot(row["pw_id"], os.path.join(a.out, "screenshots"), tag)
                row["screenshot"] = os.path.relpath(png, a.out).replace("\\", "/")
                row["screenshot_info"] = info
            emit(row)
    for i in range(0, len(rest), a.chunk):
        for row in run_rake(env, rest[i:i + a.chunk], a.chunk_timeout):
            emit(record(row))
    out.close()

    rows = [json.loads(l) for l in open(results_path, encoding="utf-8") if l.strip()]
    s = summarize(rows)
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    md = ["# PathWhiz import summary", "",
          "files %d | imported (success=true) %d | imported with no error messages %d | failed %d | total %.0fs" % (
              s["files"], s["success_true"], s["success_true_no_errors"], s["success_false"], s["total_seconds"]), "",
          "| count | files | cause |", "|---:|---:|---|"]
    md += ["| %d | %d | %s |" % (c["count"], c["files"], c["cause"].replace("|", "/")) for c in s["error_causes"]]
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    log("summary:", json.dumps({k: v for k, v in s.items() if k != "error_causes"}))
    if a.down:
        dc(["down"], env)
    sys.exit(0 if s["success_false"] == 0 else 1)


if __name__ == "__main__":
    main()
