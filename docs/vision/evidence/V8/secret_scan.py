"""Secret scan for V8 evidence, lane record and logs. Run through the .env launcher; prints counts only.

  python C:/t/vis-runs/scratch/vis-extend-db/launch.py <this file> <file or dir> ...
Exit 1 when anything is found (or the DB password is not in the environment to compare against).
"""
import os
import re
import sys
from pathlib import Path

env = {k: os.environ.get(k, "").strip() for k in ("PATHBANK_DB_PASSWORD", "PATHBANK_DB_HOST", "PATHBANK_DB_PORT",
                                                   "PATHBANK_DB_USER", "OPENROUTER_API_KEY")}
pats = {
    "sk-or-": re.compile(r"sk-or-"),
    "bearer": re.compile(r"bearer\s+[a-z0-9._-]{16,}", re.I),
    "api_key=": re.compile(r"api[_-]?key\s*[=:]\s*\S{8,}", re.I),
    "password=": re.compile(r"passw(or)?d\s*[=:]\s*\S+", re.I),
    # URL credentials (scheme://user:pw@host); a bare 'a:b@c' also matched payload element keys
    # ('processes.reactions:<in>><out>@<state>'), which are not credentials
    "url user:pw@host": re.compile(r"://[^/\s:@]+:[^/\s@]+@"),
    # personal data from fetched papers (author e-mail addresses)
    "email": re.compile(r"[\w.+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+"),
}
for k, v in env.items():
    if k == "PATHBANK_DB_PORT" and v:   # a bare number is everywhere; only in a host:port / port= context
        pats[k] = re.compile(r"(?:[:=]\s*|port\W{0,3})" + re.escape(v) + r"(?!\d)", re.I)
    elif len(v) >= 4:
        pats[k] = re.compile(re.escape(v))
files = []
for arg in sys.argv[1:]:
    p = Path(arg)
    files += sorted(x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts) if p.is_dir() else [p]
total = 0
for f in files:
    if f.resolve() == Path(__file__).resolve():
        continue   # this file names the patterns
    text = f.read_text(encoding="utf-8", errors="replace")
    counts = {k: len(p.findall(text)) for k, p in pats.items()}
    total += sum(counts.values())
    if any(counts.values()):
        print(f.name, {k: v for k, v in counts.items() if v})
print("files", len(files), "patterns", sorted(pats), "password in env:", bool(env["PATHBANK_DB_PASSWORD"]))
print("TOTAL", total)
sys.exit(1 if total or not env["PATHBANK_DB_PASSWORD"] else 0)
