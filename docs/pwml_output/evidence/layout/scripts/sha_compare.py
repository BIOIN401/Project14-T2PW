"""Byte identity of replayed PWML files with the sha256 recorded when the reference set was written.

usage: sha_compare.py RESULTS_A.jsonl DIR_B OUT.json
RESULTS_A.jsonl is a replay record (one row per run: idx, pwml_written, sha256), e.g. the P4 d3
results_0.jsonl. The recorded hashes are used, not the committed copies: git's autocrlf rewrites the
checked-out .pwml files with CRLF line ends. Exit 0 only if every written file matches and no extra exists.
"""
import hashlib
import json
import sys
from pathlib import Path

res_a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
want = {}
for line in res_a.read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    if r.get("pwml_written"):
        want[f"{r['idx']}.pwml"] = r["sha256"]
have = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in b.glob("*.pwml")}
diff = sorted(n for n in want if n in have and have[n] != want[n])
res = {"reference": str(res_a), "dir": str(b), "recorded": len(want), "files": len(have),
       "identical": sum(1 for n in want if have.get(n) == want[n]), "different": diff,
       "missing": sorted(set(want) - set(have)), "extra": sorted(set(have) - set(want))}
Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k not in ("reference", "dir")}))
sys.exit(0 if not diff and not res["missing"] and not res["extra"] else 1)
