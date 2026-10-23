"""Stdlib-only: print named entries from a mapping cache (read-only)."""
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
want = sys.argv[2:]
for sect, rows in p.items():
    if not isinstance(rows, dict):
        continue
    for k, v in rows.items():
        if any(w == k or (w.endswith("*") and k.startswith(w[:-1])) for w in want):
            print(sect, k, "=>", json.dumps(v, ensure_ascii=False)[:1800])
            print()
